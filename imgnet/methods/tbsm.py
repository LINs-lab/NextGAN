"""Original TBSM vector supervision and generator/critic updates."""

import copy
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from networks.critics import Critic


SCATTERING_EPS = 1e-6


def _self_normalized(loss, weight):
    return weight * loss / loss.detach().clamp_min(1e-8)


def _freeze(module):
    return module.requires_grad_(False).eval()


def _distributed_mean(x):
    if not (dist.is_available() and dist.is_initialized()):
        return x
    x = x.clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x.div_(dist.get_world_size())
    return x


class RepresentationScattering(nn.Module):
    """TBSM scattering objective in one frozen representation space."""

    def __init__(
        self,
        representor,
        intra_weight,
        track_weight,
        critic=None,
        ema_beta=0.999,
        loss_weight=1.0,
    ):
        super().__init__()
        self.representor = representor
        self.lam = float(intra_weight)
        self.rho = float(track_weight)
        self.loss_weight = float(loss_weight)
        self.critic_cfg = {} if critic is None else dict(critic)
        self.critic = None
        self.register_buffer("mu_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("x2_ema", torch.zeros(representor.feat_dim))
        self.register_buffer("_ema_inited", torch.zeros(1, dtype=torch.bool))
        self.beta = ema_beta

    def init_critic(self, num_classes):
        if self.rho <= 0 or self.critic is not None:
            return
        self.critic = Critic(
            num_classes, feat_dim=self.mu_ema.numel(), **self.critic_cfg
        )
        self.critic.to(device=self.mu_ema.device)

    def _normalize_bodies(self, x_r, x_p, x_s=None):
        channels = x_r.shape[-1]
        mu_new = x_r.mean(dim=(0, 1)).detach().float()
        x2_new = (x_r**2).mean(dim=(0, 1)).detach().float()
        mu_new = _distributed_mean(mu_new)
        x2_new = _distributed_mean(x2_new)
        if not bool(self._ema_inited.item()):
            self.mu_ema.copy_(mu_new)
            self.x2_ema.copy_(x2_new)
            self._ema_inited.fill_(True)
        else:
            self.mu_ema.lerp_(mu_new, 1.0 - self.beta)
            self.x2_ema.lerp_(x2_new, 1.0 - self.beta)

        var = self.x2_ema - (self.mu_ema**2)
        mu = self.mu_ema.view(1, 1, channels)
        std = torch.sqrt(torch.clamp(var, min=1e-5)).view(1, 1, channels)

        def norm(x):
            if x is None:
                return None
            x = (x - mu) / std
            return F.normalize(x, p=2, dim=-1) * math.sqrt(channels)

        return norm(x_r), norm(x_p), norm(x_s)

    def feature_tokens(self, image, requires_grad=True):
        """Extract spatial tokens from the representation model's final feature map."""
        with torch.set_grad_enabled(requires_grad):
            featmap = self.representor.extract_featmap(image)
        b, c, h, w = featmap.shape
        return featmap.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def compute_three_body_scattering_target(self, x_r, x_p, x_s, v_critic=None):
        """Mix normalized source bearings with the tracked vector and detach the G target."""
        x_p_det = x_p.detach()
        inter_offset = (x_r - x_p_det).flatten(1)
        intra_offset = (x_s - x_p_det).flatten(1)
        inter_radius = torch.linalg.vector_norm(
            inter_offset, ord=2, dim=-1, keepdim=True
        )
        intra_radius = torch.linalg.vector_norm(
            intra_offset, ord=2, dim=-1, keepdim=True
        )
        radius_scale = math.sqrt(x_p.shape[-1] * x_p.shape[-2])
        b_r = inter_offset / (inter_radius + SCATTERING_EPS) * radius_scale
        b_s = intra_offset / (intra_radius + SCATTERING_EPS) * radius_scale
        v_lambda = (b_r - b_s * self.lam).view_as(x_p)
        v_mix = (
            v_lambda
            if v_critic is None
            else v_critic * self.rho + v_lambda * (1 - self.rho)
        )
        return (x_p + v_mix).detach(), v_lambda

    def compute_loss(self, real_img, fake_img, fref_img, y, t):
        x_r = self.feature_tokens(real_img, requires_grad=False)
        x_p = self.feature_tokens(fake_img, requires_grad=True)
        x_s = None
        if self.lam > 0 and fref_img is not None:
            x_s = self.feature_tokens(fref_img, requires_grad=False)
        x_r, x_p, x_s = self._normalize_bodies(x_r, x_p, x_s)
        x_s = x_p.detach() if x_s is None else x_s
        # The critic query interpolates toward real features over [0, 1-lambda].
        alpha = torch.rand(x_p.shape[0], 1, 1, device=x_p.device) * (1 - self.lam)
        x_tilde = alpha * x_r + (1 - alpha) * x_p.detach()
        v_critic = None if self.critic is None else self.critic(y, x_tilde, t)
        x_target, v_lambda = self.compute_three_body_scattering_target(
            x_r, x_p, x_s, v_critic
        )
        raw_critic_loss = F.mse_loss(
            v_lambda if v_critic is None else v_critic, v_lambda
        )
        raw_projectile_loss = F.mse_loss(x_p, x_target)
        with torch.no_grad():
            cos_sim = (
                F.cosine_similarity(x_p.mean(dim=1), x_r.mean(dim=1), dim=-1)
                .mean()
                .item()
            )
        return (
            _self_normalized(raw_projectile_loss, self.loss_weight),
            _self_normalized(raw_critic_loss, self.loss_weight),
            {
                "generator": raw_projectile_loss.detach(),
                "critic": raw_critic_loss.detach(),
                "similarity": cos_sim,
            },
        )


class TBSM(nn.Module):
    """Train a one-step generator with tracked three-body scattering regression."""

    checkpoint_version = 1

    def __init__(
        self,
        generator,
        matchers,
        num_classes,
        gen_lr,
        critic_lr,
        ema_decay,
        t_sampling,
        noise_scale=1.0,
        gen_betas=(0.9, 0.999),
        critic_betas=(0.9, 0.999),
        grad_accum_steps=1,
        critic_steps=1,
    ):
        super().__init__()
        self.generator = generator
        if not isinstance(t_sampling, (list, tuple)):
            raise TypeError(
                "t_sampling must be a list: [] for uniform or values for choice sampling"
            )
        self.t_sampling = tuple(float(value) for value in t_sampling)
        self.noise_scale = float(noise_scale)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        if (
            isinstance(critic_steps, bool)
            or not isinstance(critic_steps, int)
            or critic_steps < 1
        ):
            raise ValueError("critic_steps must be a positive integer")
        self.critic_steps = critic_steps
        self._critic_step_count = 0
        self.matchers = nn.ModuleList(matchers)
        for matcher in self.matchers:
            matcher.init_critic(num_classes)

        self.generator_optimizer = torch.optim.AdamW(
            [p for p in generator.backbone.parameters() if p.requires_grad],
            lr=gen_lr,
            betas=tuple(gen_betas),
        )
        critic_params = [
            p
            for matcher in self.matchers
            if matcher.critic is not None
            for p in matcher.critic.parameters()
        ]
        for p in critic_params:
            p.requires_grad_(True)
        self.critic_optimizer = (
            torch.optim.AdamW(
                critic_params,
                lr=critic_lr,
                betas=tuple(critic_betas),
                weight_decay=0.0,
            )
            if critic_params
            else None
        )

        self.ema_decay = ema_decay
        self._eval_generator = _freeze(copy.deepcopy(generator))

    def set_training_step(self, completed_steps):
        """Restore the outer-minibatch phase used by the update schedule."""
        self._critic_step_count = max(0, int(completed_steps))

    def _ema_update(self):
        with torch.no_grad():
            ema_params = dict(self._eval_generator.named_parameters())
            for name, param in self.generator.named_parameters():
                key = name.replace("backbone.module.", "backbone.")
                if param.requires_grad:
                    ema_params[key].mul_(self.ema_decay).add_(
                        param.detach(), alpha=1.0 - self.ema_decay
                    )

    def step(self, real_x, real_y):
        batch_size = real_x.shape[0]
        device = real_x.device
        self.generator.train()
        self._critic_step_count += 1
        accumulation_index = (self._critic_step_count - 1) % self.grad_accum_steps
        critic_update_index = (self._critic_step_count - 1) // self.grad_accum_steps
        accumulate_generator = (
            critic_update_index % self.critic_steps == self.critic_steps - 1
        )
        step_optimizers = accumulation_index == self.grad_accum_steps - 1

        noise = torch.randn_like(real_x) * self.noise_scale
        if not self.t_sampling:
            t = torch.rand(batch_size, device=device, dtype=real_x.dtype)
        else:
            values = real_x.new_tensor(self.t_sampling)
            t = values[torch.randint(values.numel(), (batch_size,), device=device)]

        t_img = t.view(batch_size, *([1] * (real_x.ndim - 1)))
        # Form the configured paired noisy input; t=1 gives pure noise.
        x_t = real_x * (1.0 - t_img) + noise * t_img

        y = real_y
        with torch.set_grad_enabled(accumulate_generator):
            generated_x = self.generator(x_t, y, t)

        fref_img = None
        if any(matcher.lam > 0 for matcher in self.matchers):
            with torch.no_grad():
                self.generator.eval()
                # The generated source uses independent pure noise at t=1,
                # regardless of the projectile timestep.
                fref_x = self.generator.auxiliary_forward(
                    torch.randn_like(real_x) * self.noise_scale,
                    y,
                    real_x.new_ones(real_x.shape[0]),
                )
                fref_img = self.generator.to_image(fref_x)
                self.generator.train()

        with torch.no_grad():
            real_img = self.generator.to_image(real_x)
        fake_img = self.generator.to_image(generated_x)

        total_gen_loss = real_x.new_tensor(0.0)
        total_critic_loss = real_x.new_tensor(0.0)
        loss_dict = {}
        for index, matcher in enumerate(self.matchers):
            gen_loss, critic_loss, matcher_log = matcher.compute_loss(
                real_img, fake_img, fref_img, y, t
            )
            total_gen_loss = total_gen_loss + gen_loss
            total_critic_loss = total_critic_loss + critic_loss
            suffix = f"_{index}" if len(self.matchers) > 1 else ""
            loss_dict.update(
                {key + suffix: value for key, value in matcher_log.items()}
            )

        # Both optimizers accumulate over grad_accum_steps microbatches. The
        # critic steps at every boundary; the generator steps after every
        # critic_steps critic optimizer steps.
        accum_scale = 1.0 / self.grad_accum_steps
        if accumulate_generator:
            (total_gen_loss * accum_scale).backward(create_graph=False)
        if self.critic_optimizer is not None and total_critic_loss.requires_grad:
            (total_critic_loss * accum_scale).backward(create_graph=False)
        if step_optimizers and self.critic_optimizer is not None:
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
        if step_optimizers and accumulate_generator:
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad(set_to_none=True)
            self._ema_update()
        return loss_dict

    @torch.no_grad()
    def sample(self, y, noise):
        self._eval_generator.eval()
        device = next(self._eval_generator.parameters()).device
        noise = noise.to(device) * self.noise_scale
        x = self._eval_generator(
            noise, y.to(device), t=noise.new_ones(noise.shape[0])
        )
        return self._eval_generator.to_image(x).float().clamp(0, 1)

    def reset_ema(self):
        self._eval_generator = _freeze(copy.deepcopy(self.generator))
