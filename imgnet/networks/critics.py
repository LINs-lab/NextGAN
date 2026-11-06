"""Trainable critics and the GAN representation wrapper.

Critic preserves TBSM's vector/refine head. ScatteringCritic provides the
GAN scalar/vector head. Both share identical conditioning and spatial blocks;
their output computations and checkpoint parameter names remain distinct.
NaiveCriticHead and ScatteringCriticVariant retain configurable alternatives.
Algorithm objectives, optimizers and updates belong in methods/.
"""

import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


__all__ = [
    "Critic",
    "ScatteringCritic",
    "NaiveCriticHead",
    "ScatteringCriticVariant",
    "RepresentationCritic",
]

OUTPUT_MODES = {"field", "potential"}


class ParameterFreeRMSNorm(nn.Module):
    """RMS normalization without learned affine parameters."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=timesteps.device)
        / half_dim
    )
    args = timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _sincos_embedding_1d(pos, dim, max_period=10000):
    half_dim = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half_dim, dtype=torch.float32, device=pos.device)
        / half_dim
    )
    args = pos.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SpatialTransformerBlock(nn.Module):
    """Self-attention and MLP block for spatial feature tokens."""

    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = ParameterFreeRMSNorm()
        self.qkv = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        )
        self.proj = nn.utils.spectral_norm(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.norm2 = ParameterFreeRMSNorm()
        self.mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim, mlp_dim, bias=False)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(mlp_dim, hidden_dim, bias=False)),
        )

    def forward(self, x):
        h = self.norm1(x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(h)
        x = x + self.mlp(self.norm2(x))
        return x


class _PositionedCritic(nn.Module):
    """Share cached spatial encodings without changing checkpoint parameter names."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_pos_embed_cache", torch.empty(0), persistent=False)
        self._pos_embed_cache_key = None

    def _position_embedding(self, num_tokens, device, dtype):
        key = (num_tokens, device.type, device.index, dtype)
        if self._pos_embed_cache_key == key and self._pos_embed_cache.numel() > 0:
            return self._pos_embed_cache

        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            grid_y, grid_x = torch.meshgrid(
                torch.arange(side, device=device),
                torch.arange(side, device=device),
                indexing="ij",
            )
            pos_y = grid_y.reshape(-1)
            pos_x = grid_x.reshape(-1)
            dim_y = self.hidden_dim // 2
            dim_x = self.hidden_dim - dim_y
            pos_embed = torch.cat(
                [
                    _sincos_embedding_1d(pos_y, dim_y),
                    _sincos_embedding_1d(pos_x, dim_x),
                ],
                dim=-1,
            )
        else:
            pos = torch.arange(num_tokens, device=device)
            pos_embed = _sincos_embedding_1d(pos, self.hidden_dim)

        pos_embed = pos_embed.to(dtype=dtype).unsqueeze(0)
        self._pos_embed_cache = pos_embed
        self._pos_embed_cache_key = key
        return pos_embed


class Critic(_PositionedCritic):
    """Predict the tracked vector from a representation query, class, and timestep."""

    def __init__(
        self,
        num_classes,
        feat_dim,
        hidden_dim=736,
        time_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        use_potential_grad=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(time_dim, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.cond_fuse = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )

        self.x_proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        if use_potential_grad:
            self.refine = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1, bias=False)),
            )
        else:
            self.refine = None

    def forward(self, y, x_cond, t):
        class_labels = y
        query_tokens = x_cond
        timesteps = t
        differentiable_query = query_tokens
        if self.refine is not None and not differentiable_query.requires_grad:
            differentiable_query = query_tokens.clone().detach().requires_grad_(True)
        _, num_tokens, _ = differentiable_query.shape

        class_embedding = self.label_emb(class_labels.reshape(-1).long())
        time_embedding = timestep_embedding(timesteps.reshape(-1) * 1000, self.time_dim)
        projected_time = self.time_mlp(time_embedding)
        conditioning = self.cond_fuse(
            torch.cat([class_embedding, projected_time], dim=-1)
        )

        hidden_tokens = (
            self.x_proj(differentiable_query)
            + self._position_embedding(
                num_tokens, differentiable_query.device, differentiable_query.dtype
            )
            + conditioning.unsqueeze(1)
        )
        for block in self.spatial_blocks:
            hidden_tokens = block(hidden_tokens)
        tracked_vector = self.out_proj(hidden_tokens)
        if self.refine is None:
            return tracked_vector

        # Sum per-token scalar potentials and differentiate the result with
        # respect to the critic query.
        token_potentials = self.refine(tracked_vector + differentiable_query)
        potential = token_potentials.flatten(1).sum(dim=1)
        return torch.autograd.grad(
            outputs=potential.sum(),
            inputs=differentiable_query,
            create_graph=True,
            retain_graph=True,
        )[0]


class ScatteringCritic(_PositionedCritic):
    """Return a vector field or scalar potential for the public GAN objectives."""

    def __init__(
        self,
        num_classes,
        feat_dim,
        hidden_dim=736,
        time_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        output_mode="field",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        if output_mode not in OUTPUT_MODES:
            raise ValueError(
                f"Unknown critic output_mode: {output_mode}; "
                f"expected one of {sorted(OUTPUT_MODES)}"
            )
        self.output_mode = output_mode

        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(time_dim, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.cond_fuse = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )

        self.x_proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.spatial_blocks = nn.ModuleList(
            [
                SpatialTransformerBlock(
                    hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.potential_head = (
            nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False)),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                nn.utils.spectral_norm(nn.Linear(hidden_dim, 1, bias=False)),
            )
            if output_mode != "field"
            else None
        )

    def forward(self, condition, query, time):
        class_labels = condition
        query_tokens = query
        timesteps = time
        _, num_tokens, _ = query_tokens.shape

        class_embedding = self.label_emb(class_labels.reshape(-1).long())
        time_embedding = timestep_embedding(timesteps.reshape(-1) * 1000, self.time_dim)
        projected_time = self.time_mlp(time_embedding)
        conditioning = self.cond_fuse(
            torch.cat([class_embedding, projected_time], dim=-1)
        )

        hidden_tokens = (
            self.x_proj(query_tokens)
            + self._position_embedding(
                num_tokens, query_tokens.device, query_tokens.dtype
            )
            + conditioning.unsqueeze(1)
        )
        for block in self.spatial_blocks:
            hidden_tokens = block(hidden_tokens)
        predicted_field = self.out_proj(hidden_tokens)
        if self.output_mode == "field":
            return predicted_field

        token_potential = self.potential_head(predicted_field)
        return token_potential.mean(dim=1).squeeze(-1)


class NaiveCriticHead(nn.Module):
    """Unconditioned mean-pool + linear critic head with optional spectral norm."""

    def __init__(
        self, num_classes, feat_dim, output_mode="potential", use_spectral_norm=False
    ):
        super().__init__()
        if output_mode != "potential":
            raise ValueError(
                f"NaiveCriticHead only supports output_mode='potential', got {output_mode!r}"
            )
        # With spectral normalization, omit the unconstrained bias to match
        # the scattering heads. The default linear head retains its bias.
        head = nn.Linear(feat_dim, 1, bias=not use_spectral_norm)
        self.head = nn.utils.spectral_norm(head) if use_spectral_norm else head

    def forward(self, condition, query, time):
        # condition (class labels) and time (timestep) are accepted for
        # interface parity with ScatteringCritic.forward but intentionally
        # ignored -- this control has no conditioning of any kind.
        del condition, time
        pooled = query.mean(dim=1)  # (B, N, C) -> (B, C)
        return self.head(pooled).squeeze(-1)


def _maybe_spectral_norm(module, use_spectral_norm):
    return nn.utils.spectral_norm(module) if use_spectral_norm else module


class _SpatialTransformerBlockVariant(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_heads,
        mlp_ratio=4.0,
        use_spectral_norm=True,
        use_cosine_attention=False,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_cosine_attention = use_cosine_attention
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.norm1 = ParameterFreeRMSNorm()
        self.qkv = _maybe_spectral_norm(
            nn.Linear(hidden_dim, hidden_dim * 3, bias=False), use_spectral_norm
        )
        self.proj = _maybe_spectral_norm(
            nn.Linear(hidden_dim, hidden_dim, bias=False), use_spectral_norm
        )
        self.norm2 = ParameterFreeRMSNorm()
        self.mlp = nn.Sequential(
            _maybe_spectral_norm(
                nn.Linear(hidden_dim, mlp_dim, bias=False), use_spectral_norm
            ),
            nn.SiLU(),
            _maybe_spectral_norm(
                nn.Linear(mlp_dim, hidden_dim, bias=False), use_spectral_norm
            ),
        )

    def forward(self, x, needs_double_backward=False):
        # NextGAN differentiates the potential gradient. Use math attention
        # for that query because fused SDPA does not support double backward.
        # Other queries can use the faster available backends.
        h = self.norm1(x)
        B, N, C = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attention_scale = None
        if self.use_cosine_attention:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
            # Unit-length q/k with scale=1 gives cosine-similarity logits.
            attention_scale = 1.0
        if needs_double_backward:
            with sdpa_kernel(SDPBackend.MATH):
                h = F.scaled_dot_product_attention(q, k, v, scale=attention_scale)
        else:
            h = F.scaled_dot_product_attention(q, k, v, scale=attention_scale)
        h = h.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(h)
        x = x + self.mlp(self.norm2(x))
        return x


class ScatteringCriticVariant(_PositionedCritic):
    """Conditional critic with configurable spectral norm and SDPA attention."""

    def __init__(
        self,
        num_classes,
        feat_dim,
        hidden_dim=736,
        time_dim=128,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        output_mode="field",
        use_spectral_norm=True,
        use_cosine_attention=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        if output_mode not in OUTPUT_MODES:
            raise ValueError(
                f"Unknown critic output_mode: {output_mode}; "
                f"expected one of {sorted(OUTPUT_MODES)}"
            )
        self.output_mode = output_mode

        sn = use_spectral_norm
        self.label_emb = nn.Embedding(num_classes, hidden_dim)
        self.time_mlp = nn.Sequential(
            _maybe_spectral_norm(nn.Linear(time_dim, hidden_dim, bias=False), sn),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.cond_fuse = nn.Sequential(
            _maybe_spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim, bias=False), sn),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )

        self.x_proj = nn.Sequential(
            _maybe_spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False), sn),
            ParameterFreeRMSNorm(),
            nn.SiLU(),
        )
        self.spatial_blocks = nn.ModuleList(
            [
                _SpatialTransformerBlockVariant(
                    hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_spectral_norm=sn,
                    use_cosine_attention=use_cosine_attention,
                )
                for _ in range(depth)
            ]
        )
        self.out_proj = nn.Linear(hidden_dim, feat_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        self.potential_head = (
            nn.Sequential(
                _maybe_spectral_norm(nn.Linear(feat_dim, hidden_dim, bias=False), sn),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                _maybe_spectral_norm(nn.Linear(hidden_dim, hidden_dim, bias=False), sn),
                ParameterFreeRMSNorm(),
                nn.SiLU(),
                _maybe_spectral_norm(nn.Linear(hidden_dim, 1, bias=False), sn),
            )
            if output_mode != "field"
            else None
        )

    def forward(self, condition, query, time, needs_double_backward=False):
        # needs_double_backward: True only for NextGAN's interpolated-
        # point query (u_tilde), which the caller differentiates again via
        # torch.autograd.grad(..., create_graph=True) w.r.t. this forward's
        # input -- see _SpatialTransformerBlockVariant.forward's docstring.
        class_labels = condition
        query_tokens = query
        timesteps = time
        _, num_tokens, _ = query_tokens.shape

        class_embedding = self.label_emb(class_labels.reshape(-1).long())
        time_embedding = timestep_embedding(timesteps.reshape(-1) * 1000, self.time_dim)
        projected_time = self.time_mlp(time_embedding)
        conditioning = self.cond_fuse(
            torch.cat([class_embedding, projected_time], dim=-1)
        )

        hidden_tokens = (
            self.x_proj(query_tokens)
            + self._position_embedding(
                num_tokens, query_tokens.device, query_tokens.dtype
            )
            + conditioning.unsqueeze(1)
        )
        for block in self.spatial_blocks:
            hidden_tokens = block(
                hidden_tokens, needs_double_backward=needs_double_backward
            )
        predicted_field = self.out_proj(hidden_tokens)
        if self.output_mode == "field":
            return predicted_field

        token_potential = self.potential_head(predicted_field)
        return token_potential.mean(dim=1).squeeze(-1)


def _distributed_mean(x):
    if not (dist.is_available() and dist.is_initialized()):
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x.div_(dist.get_world_size())


class RepresentationCritic(nn.Module):
    """A frozen image representor followed by a conditional critic energy head."""

    def __init__(
        self,
        representor,
        num_classes,
        critic=None,
        ema_beta=0.999,
        loss_weight=1.0,
        critic_arch="scattering",
    ):
        super().__init__()
        self.representor = representor
        self.loss_weight = float(loss_weight)
        self.beta = float(ema_beta)
        self.critic_arch = critic_arch
        critic_params = {} if critic is None else dict(critic)
        critic_params.setdefault("output_mode", "potential")
        if critic_arch == "scattering":
            self.critic = ScatteringCritic(
                num_classes, feat_dim=representor.feat_dim, **critic_params
            )
        elif critic_arch == "naive":
            self.critic = NaiveCriticHead(
                num_classes, feat_dim=representor.feat_dim, **critic_params
            )
        elif critic_arch == "scattering_variant":
            self.critic = ScatteringCriticVariant(
                num_classes, feat_dim=representor.feat_dim, **critic_params
            )
        else:
            raise ValueError(
                f"Unknown critic_arch={critic_arch!r}; expected one of 'scattering', 'naive', 'scattering_variant'"
            )
        self.register_buffer("mu_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("x2_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("_ema_inited", torch.zeros(1, dtype=torch.bool))

    def _feature_tokens(self, image, requires_grad):
        with torch.set_grad_enabled(requires_grad):
            featmap = self.representor.extract_featmap(image)
            (batch, channels, height, width) = featmap.shape
            return featmap.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

    def _normalize(self, tokens, update_stats):
        if update_stats:
            mu_new = _distributed_mean(tokens.mean(dim=(0, 1)).detach().float())
            x2_new = _distributed_mean((tokens**2).mean(dim=(0, 1)).detach().float())
            if not bool(self._ema_inited.item()):
                self.mu_ema.copy_(mu_new)
                self.x2_ema.copy_(x2_new)
                self._ema_inited.fill_(True)
            else:
                self.mu_ema.lerp_(mu_new, 1.0 - self.beta)
                self.x2_ema.lerp_(x2_new, 1.0 - self.beta)
        channels = tokens.shape[-1]
        std = torch.sqrt(torch.clamp(self.x2_ema - self.mu_ema.square(), min=1e-05))
        normalized = (tokens - self.mu_ema.view(1, 1, channels)) / std.view(
            1, 1, channels
        )
        return F.normalize(normalized, p=2, dim=-1) * math.sqrt(channels)
