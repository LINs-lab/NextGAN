"""Shared GAN scheduling, optimizer state, sampling and EMA."""

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


def _field_regression_loss(field, displacement):
    """Unit-weight field regression, summed over features and averaged over batch."""
    if field.shape != displacement.shape:
        raise ValueError("Field and displacement must have the same shape")
    residual = field - displacement
    return 0.5 * residual.square().flatten(1).sum(1).mean(), residual


def _freeze(module):
    return module.requires_grad_(False).eval()


@contextmanager
def _freeze_parameters(module):
    """Freeze critic parameters and spectral-normalization state for G's query."""
    params = list(module.parameters())
    states = [param.requires_grad for param in params]
    spectral_modules = [
        (child, child.training)
        for child in module.modules()
        if hasattr(child, "weight_u") and hasattr(child, "weight_v")
    ]
    for param in params:
        param.requires_grad_(False)
    for child, _ in spectral_modules:
        child.training = False
    try:
        yield
    finally:
        for param, state in zip(params, states):
            param.requires_grad_(state)
        for child, training in spectral_modules:
            child.training = training


def _generator_critic(matcher):
    # G needs input derivatives, with no critic gradient reduction/buffer broadcast.
    critic = matcher.critic
    return critic.module if isinstance(critic, DistributedDataParallel) else critic


class GANBase(nn.Module):
    """Shared generator/critic scheduling, optimizer state and EMA."""

    checkpoint_version = 2
    simultaneous_update = False

    def _validate_critic_configuration(self, critics):
        pass

    def __init__(
        self,
        generator,
        critics,
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
        self._validate_critic_configuration(critics)
        if not critics:
            raise ValueError("GAN requires at least one representation critic")
        if not isinstance(t_sampling, (list, tuple)):
            raise TypeError(
                "t_sampling must be a list: [] for uniform or values for choice sampling"
            )
        self.generator = generator
        self.matchers = nn.ModuleList(critics)
        self.t_sampling = tuple(float(value) for value in t_sampling)
        self.noise_scale = float(noise_scale)
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.critic_steps = max(1, int(critic_steps))
        gen_lr = float(gen_lr)
        critic_lr = float(critic_lr)
        if gen_lr <= 0 or critic_lr <= 0:
            raise ValueError("gen_lr and critic_lr must be positive")
        self._completed_steps = 0
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in generator.backbone.parameters() if param.requires_grad],
            lr=gen_lr,
            betas=tuple(gen_betas),
        )
        critic_params = [
            param
            for matcher in self.matchers
            for param in matcher.critic.parameters()
        ]
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=critic_lr,
            betas=tuple(critic_betas),
            weight_decay=0.0,
        )
        self.ema_decay = float(ema_decay)
        self._eval_generator = _freeze(copy.deepcopy(generator))

    def _sample_time(self, real_x):
        if not self.t_sampling:
            return torch.rand(real_x.shape[0], device=real_x.device, dtype=real_x.dtype)
        choices = real_x.new_tensor(self.t_sampling)
        return choices[
            torch.randint(choices.numel(), (real_x.shape[0],), device=real_x.device)
        ]

    def _ema_update(self):
        with torch.no_grad():
            ema_params = dict(self._eval_generator.named_parameters())
            for name, param in self.generator.named_parameters():
                key = name.replace("backbone.module.", "backbone.")
                if param.requires_grad:
                    ema_params[key].mul_(self.ema_decay).add_(
                        param.detach(), alpha=1.0 - self.ema_decay
                    )

    def _prepare_features(self, real_x, real_y, generator_grad):
        """One new sample; extract and normalize each real/fake feature only once."""
        noise = torch.randn_like(real_x) * self.noise_scale
        t = self._sample_time(real_x)
        t_img = t.view(real_x.shape[0], *[1] * (real_x.ndim - 1))
        with torch.set_grad_enabled(generator_grad):
            generated_x = self.generator(
                real_x * (1.0 - t_img) + noise * t_img, real_y, t
            )
            fake_img = self.generator.to_image(generated_x)
        with torch.no_grad():
            real_img = self.generator.to_image(real_x)
        features = []
        for matcher in self.matchers:
            real_feat = matcher._normalize(
                matcher._feature_tokens(real_img, requires_grad=False),
                update_stats=True,
            )
            fake_feat = matcher._normalize(
                matcher._feature_tokens(fake_img, requires_grad=generator_grad),
                update_stats=False,
            )
            features.append((real_feat, fake_feat))
        return features, t

    def _backward_critic(self, features, labels, time):
        loss, logs, context = self._critic_loss(features, labels, time)
        (loss / self.grad_accum_steps).backward()
        return {**logs, "critic_loss": loss.detach()}, context

    def _mean_logs(self, aggregate, logs):
        for key, value in logs.items():
            aggregate[key] = (
                aggregate.get(key, 0.0) + value.detach() / self.grad_accum_steps
            )

    def step(self, next_batch):
        """One G update and K critic updates, each averaging A fresh microbatches."""
        device_type = next(self.generator.backbone.parameters()).device.type
        with torch.autocast(
            device_type=device_type,
            enabled=torch.is_autocast_enabled(device_type),
            cache_enabled=False,
        ):
            return self._step_uncached(next_batch)

    def _step_uncached(self, next_batch):
        self.generator.train()
        # Earlier critic rounds do not retain a generator/encoder backward graph.
        for _ in range(self.critic_steps - 1):
            self.critic_optimizer.zero_grad(set_to_none=True)
            for _ in range(self.grad_accum_steps):
                real_x, labels = next_batch()
                features, time = self._prepare_features(real_x, labels, False)
                self._backward_critic(features, labels, time)
            self.critic_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.generator_optimizer.zero_grad(set_to_none=True)
        logs = {}
        pending = []
        for _ in range(self.grad_accum_steps):
            real_x, labels = next_batch()
            features, time = self._prepare_features(real_x, labels, True)
            critic_logs, context = self._backward_critic(features, labels, time)
            self._mean_logs(logs, critic_logs)
            if self.simultaneous_update:
                # No optimizer has stepped: both gradients use this same critic.
                with _freeze_parameters(self.matchers):
                    loss = self._generator_loss(features, labels, time, context)
                    (loss / self.grad_accum_steps).backward()
                self._mean_logs(logs, {"generator_loss": loss})
            else:
                pending.append((features, labels, time, context))

        if self.simultaneous_update:
            self.generator_optimizer.step()
            self.critic_optimizer.step()
        else:
            self.critic_optimizer.step()
            with _freeze_parameters(self.matchers):
                generator_loss = 0.0
                for features, labels, time, context in pending:
                    loss = self._generator_loss(features, labels, time, context)
                    generator_loss = generator_loss + loss / self.grad_accum_steps
                    self._mean_logs(logs, {"generator_loss": loss})
                # These G forwards preceded the final critic step. One backward
                # over their combined graph lets DDP reduce every contribution.
                generator_loss.backward()
            self.generator_optimizer.step()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self._ema_update()
        self._completed_steps += 1
        logs["loss"] = logs["generator_loss"]
        return logs

    @torch.no_grad()
    def sample(self, y, noise):
        self._eval_generator.eval()
        device = next(self._eval_generator.parameters()).device
        x = self._eval_generator(
            noise.to(device) * self.noise_scale,
            y.to(device),
            t=torch.ones(noise.shape[0], device=device),
        )
        return self._eval_generator.to_image(x).float().clamp(0, 1)

    def reset_ema(self):
        self._eval_generator = _freeze(copy.deepcopy(self.generator))

    def set_training_step(self, completed_steps):
        self._completed_steps = max(0, int(completed_steps))
