import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .modules import (
    precompute_rope_freqs, precompute_rope_freqs_2d,
    apply_rotary_pos_emb, apply_rotary_pos_emb_partial,
    modulate, RMSNorm, get_2d_sincos_pos_embed
)

logger = logging.getLogger("FD_loss")


class _RoPECallable:
    """Apply precomputed RoPE frequencies to JiT attention tensors."""

    def __init__(self, freqs, func):
        self.freqs = freqs
        self.func = func

    def __call__(self, x):
        # Move the token axis next to the batch axis while applying RoPE.
        return self.func(x.transpose(1, 2), self.freqs).transpose(1, 2)


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, input_size=256, patch_size=16, in_chans=3, pca_dim=128, embed_dim=768, bias=True):
        super().__init__()
        input_size = (input_size, input_size)
        patch_size = (patch_size, patch_size)
        self.input_size = input_size
        self.num_patches = (input_size[1] // patch_size[1]) * (input_size[0] // patch_size[0])

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.input_size[0] and W == self.input_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.input_size[0]}*{self.input_size[1]})."
        return self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # Form every timestep-frequency pair explicitly.
        freqs = torch.outer(t.float(), freqs)
        embedding = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)

    def forward(self, labels):
        return self.embedding_table(labels)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = rope(q), rope(k)

        dropout_p = self.attn_drop.p if self.training else 0.
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop=0.0, bias=True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.ffn_dropout(F.silu(x1) * x2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class JiT(nn.Module):
    """Just image Transformer backbone for pixel-space ImageNet generation."""

    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=32,
        in_context_start=8,
        rope_2d=True,
        learned_pe=False,
        use_potential_grad=False,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        # Shares the potential-gradient idea with networks/critics.py's Critic:
        # the trunk produces a scalar potential over the input image, and the
        # backbone's output is grad_x of that potential instead of a direct
        # patch-projection head.
        self.use_potential_grad = bool(use_potential_grad)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        num_patches = self.x_embedder.num_patches
        head_dim = hidden_size // num_heads

        # Use either learned or fixed two-dimensional positional embeddings.
        if learned_pe:
            self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_size) * 0.02)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # Precompute rotary frequencies for the spatial patch tokens.
        if rope_2d:
            self.rope_freqs = precompute_rope_freqs_2d(head_dim, num_patches)
        else:
            self.rope_freqs = precompute_rope_freqs(head_dim, num_patches)

        # Rotate all patch tokens before inserting context, then skip the context prefix.
        self._rope = _RoPECallable(self.rope_freqs, apply_rotary_pos_emb)
        self._rope_with_prefix = _RoPECallable(self.rope_freqs, apply_rotary_pos_emb_partial)

        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        if self.use_potential_grad:
            # Potential mode: pool the trunk's output tokens to a scalar
            # energy and differentiate it w.r.t. the raw input image instead
            # of projecting tokens straight to patch pixels. final_layer is
            # still constructed (and zero-initialized below) so checkpoints
            # remain shape-compatible across genvec/genpot runs, but its
            # parameters receive no gradient in this mode since forward()
            # never calls it.
            self.energy_head = nn.Linear(hidden_size, 1)
        if not learned_pe:
            self.initialize_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[JiT] params: {n_params:.2f}M, depth: {depth}, hidden_size: {hidden_size}")
        logger.info(f"[JiT] input_size: {input_size}, patch_size: {patch_size}, num_patches: {num_patches}")
        logger.info(f"[JiT] rope_2d: {rope_2d}, learned_pe: {learned_pe}, use_potential_grad: {self.use_potential_grad}")

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def _trunk_tokens(self, x, t, y):
        """Run the patch embedding, in-context splice, and attention blocks.

        Returns the post-block spatial tokens (in-context prefix dropped)
        together with the adaLN conditioning vector, i.e. exactly the inputs
        final_layer consumes in direct mode.
        """
        y_emb = self.y_embedder(y)
        c = self.t_embedder(t) + y_emb
        x = self.x_embedder(x) + self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            x = block(x, c, self._rope if i < self.in_context_start else self._rope_with_prefix)

        return x[:, self.in_context_len:], c

    def forward(self, x, t, y):
        if not self.use_potential_grad:
            tokens, c = self._trunk_tokens(x, t, y)
            return self.unpatchify(self.final_layer(tokens, c), self.patch_size)

        # Potential mode: differentiate a scalar energy (mean-pooled over
        # patch tokens) with respect to the raw input image, as in
        # Critic(use_potential_grad=True).
        # torch.enable_grad() keeps this working even when the caller wraps
        # generation in torch.no_grad() (e.g. periodic sample visualization).
        # Math SDPA is required here: differentiating grad_x's own
        # graph during the outer loss.backward() is a double backward through
        # every attention block in _trunk_tokens, which the fused flash/
        # mem-efficient SDPA kernels cannot do.
        # Evaluation may surround generation with inference_mode(). Disable it
        # locally and copy inference tensors so this internal derivative can run.
        with torch.inference_mode(False), torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            query = (
                x if x.requires_grad and not torch.is_inference(x)
                else x.clone().detach().requires_grad_(True)
            )
            time = t.clone() if torch.is_inference(t) else t
            labels = y.clone() if torch.is_inference(y) else y
            tokens, _c = self._trunk_tokens(query, time, labels)
            potential = self.energy_head(tokens.mean(dim=1))
            (grad_x,) = torch.autograd.grad(
                outputs=potential.sum(),
                inputs=query,
                create_graph=True,
                retain_graph=True,
            )
        return grad_x


def JiT_B(patch_size=16, **kwargs):
    return JiT(depth=12, hidden_size=768, num_heads=12,
               bottleneck_dim=128, in_context_len=32, in_context_start=4, patch_size=patch_size, **kwargs)


def JiT_L(patch_size=16, **kwargs):
    return JiT(depth=24, hidden_size=1024, num_heads=16,
               bottleneck_dim=128, in_context_len=32, in_context_start=8, patch_size=patch_size, **kwargs)


def JiT_H(patch_size=16, **kwargs):
    return JiT(depth=32, hidden_size=1280, num_heads=16,
               bottleneck_dim=256, in_context_len=32, in_context_start=10, patch_size=patch_size, **kwargs)


JiT_models = {
    'JiT-B': JiT_B,
    'JiT-L': JiT_L,
    'JiT-H': JiT_H,
}
