import logging
from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import trange

from .model import JiT_models

logger = logging.getLogger("FD_loss")


class JiTDenoiser(nn.Module):
    """JiT backbone and sampling helpers used by the generator adapter."""

    def __init__(
        self,
        input_size=256,
        model_size="base",
        in_channels=3,
        num_classes=1000,
        label_drop_prob=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        t_eps=5e-2,
        noise_scale=1.0,
        rope_2d=True,
        learned_pe=False,
        patch_size=16,
        use_potential_grad=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.label_drop_prob = label_drop_prob
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.in_channels = in_channels
        model_name = {"base": "JiT-B", "large": "JiT-L", "huge": "JiT-H"}[model_size]
        self.net = JiT_models[model_name](
            input_size=input_size, in_channels=in_channels, num_classes=num_classes,
            attn_drop=attn_dropout, proj_drop=proj_dropout,
            rope_2d=rope_2d, learned_pe=learned_pe, patch_size=patch_size,
            use_potential_grad=use_potential_grad,
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        logger.info(f"[JiTDenoiser] params: {n_params:.2f}M, size: {model_size}, input: {input_size}")

    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        return torch.where(drop, torch.full_like(labels, self.num_classes), labels)

    def _backbone_t(self, t):
        # JiT uses the data-at-one time convention.
        return 1.0 - t

    def _forward_with_cfg(self, z, t, labels, cfg, cfg_interval=None):
        t_bb = self._backbone_t(t).flatten()

        x_cond = self.net(z, t_bb, labels)
        v_cond = (z - x_cond) / t.clamp_min(self.t_eps)
        if cfg == 1.0:
            return v_cond

        x_uncond = self.net(z, t_bb, torch.full_like(labels, self.num_classes))
        v_uncond = (z - x_uncond) / t.clamp_min(self.t_eps)

        if cfg_interval is not None:
            low, high = cfg_interval
            mask = (t < high) & ((low == 0) | (t > low))
            cfg = torch.where(mask, cfg, 1.0)
        return v_uncond + cfg * (v_cond - v_uncond)

    def _euler_step(self, z, t, t_next, labels, cfg, cfg_interval=None):
        return z + (t_next - t) * self._forward_with_cfg(z, t, labels, cfg, cfg_interval)

    def _heun_step(self, z, t, t_next, labels, cfg, cfg_interval=None):
        dt = t_next - t
        v1 = self._forward_with_cfg(z, t, labels, cfg, cfg_interval)
        v2 = self._forward_with_cfg(z + dt * v1, t_next, labels, cfg, cfg_interval)
        return z + dt * 0.5 * (v1 + v2)
    
    @torch.no_grad()
    def generate(self, n_samples, labels, cfg=4.0, args=None, verbose=True, z_t=None):
        # Potential-gradient backbones need a local enable_grad() forward;
        # inference_mode would prevent that internal derivative entirely.
        device = labels.device
        if args is None:
            args = SimpleNamespace(
                num_sampling_steps=1, sampling_method="euler", same_noise=False,
                interval_min=0.1, interval_max=1.0,
            )
        num_steps = args.num_sampling_steps
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError("num_sampling_steps must be a positive integer")

        if z_t is None:
            if args.same_noise:
                z = self.noise_scale * torch.randn(1, self.in_channels, self.input_size, self.input_size, device=device)
                z = z.repeat(n_samples, 1, 1, 1)
            else:
                z = self.noise_scale * torch.randn(n_samples, self.in_channels, self.input_size, self.input_size, device=device)
        else:
            z = z_t
        # Integrate from noise at t=1 to data at t=0.
        ts = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        ts = ts.view(-1, *([1] * z.ndim)).expand(-1, n_samples, -1, -1, -1)

        steppers = {"euler": self._euler_step, "heun": self._heun_step}
        stepper = steppers.get(args.sampling_method)
        if stepper is None:
            raise NotImplementedError(f"sampling method {args.sampling_method} not implemented")

        cfg_interval = [self._backbone_t(args.interval_min), self._backbone_t(args.interval_max)]
        cfg_interval[0], cfg_interval[1] = min(cfg_interval[0], cfg_interval[1]), max(cfg_interval[0], cfg_interval[1])
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        pbar = (
            trange(num_steps - 1, desc=f"[Rank{rank}] Denoising (n={n_samples})")
            if n_samples > 32 and verbose
            else range(num_steps - 1)
        )
        for i in pbar:
            z = stepper(z, ts[i], ts[i + 1], labels, cfg, cfg_interval)

        # Use Euler integration for the final step.
        z = self._euler_step(z, ts[-2], ts[-1], labels, cfg, cfg_interval)
        return z


# JiT denoiser presets.
JiTDenoiser_models = {
    "JiT_B": lambda **kw: JiTDenoiser(model_size="base", **kw),
    "JiT_L": lambda **kw: JiTDenoiser(model_size="large", **kw),
    "JiT_H": lambda **kw: JiTDenoiser(model_size="huge", **kw),
}
