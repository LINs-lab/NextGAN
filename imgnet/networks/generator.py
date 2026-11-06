"""Generator wrapper with optional frozen latent-to-image decoding."""

import copy

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


class Generator(nn.Module):
    def __init__(self, backbone, decoder=None):
        super().__init__()
        self.backbone = backbone
        # The frozen decoder maps latent outputs to image space while preserving
        # input gradients. Share its fixed parameters across EMA copies and omit
        # them from optimizer and checkpoint state.
        object.__setattr__(self, "decoder", decoder)

    def __deepcopy__(self, memo):
        copied = type(self)(copy.deepcopy(self.backbone, memo), decoder=self.decoder)
        memo[id(self)] = copied
        return copied

    def forward(self, x, y, t=None):
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        return self.backbone(x, y, t=t)

    def auxiliary_forward(self, x, y, t=None):
        """Generate an independent source without DDP reducer or buffer synchronization."""
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)
        backbone = self.backbone
        if isinstance(backbone, DistributedDataParallel):
            backbone = backbone.module
        return backbone(x, y, t=t)

    def to_image(self, x):
        if self.decoder is not None:
            return self.decoder(x)
        return x * 0.5 + 0.5


class UniversalAdapter(nn.Module):
    """Convert predictions for the shared flow x_t = (1-t) * data + t * noise."""

    def __init__(
        self,
        backbone,
        *,
        source_prediction_type,
        target_prediction_type="x",
    ):
        super().__init__()
        for value in (source_prediction_type, target_prediction_type):
            if value not in ("x", "noise", "velocity"):
                raise ValueError(f"Unknown prediction type: {value}")
        self.backbone = backbone
        self.source_prediction_type = source_prediction_type
        self.target_prediction_type = target_prediction_type

    @staticmethod
    def _view_t(t, x):
        return t.view(t.shape[0], *([1] * (x.ndim - 1)))

    def _x_noise_from_source(self, x_t, t, raw):
        t = self._view_t(t, x_t)
        raw = raw.to(dtype=x_t.dtype)
        eps = torch.finfo(x_t.dtype).eps

        if self.source_prediction_type == "x":
            x0 = raw
            noise = (x_t - (1.0 - t) * x0) / t.clamp_min(eps)
            return x0, noise

        if self.source_prediction_type == "noise":
            noise = raw
            x0 = (x_t - t * noise) / (1.0 - t).clamp_min(eps)
            return x0, noise

        x0 = x_t - t * raw
        noise = x_t + (1.0 - t) * raw
        return x0, noise

    def _target_from_x_noise(self, x0, noise):
        if self.target_prediction_type == "x":
            return x0
        if self.target_prediction_type == "noise":
            return noise
        return noise - x0

    def forward(self, x, y, t=None):
        if t is None:
            t = x.new_zeros(x.shape[0])
        t = t.to(device=x.device, dtype=x.dtype)
        raw = self.backbone(x, y, t=t)
        if self.source_prediction_type == self.target_prediction_type:
            return raw
        x0, noise = self._x_noise_from_source(x, t, raw)
        return self._target_from_x_noise(x0, noise)
