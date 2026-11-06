from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pretrained_path


class SigLIP2Representor(nn.Module):
    """Frozen SigLIP2-B encoder exposing final-layer patch feature maps."""

    def __init__(self, input_size=256, ckpt_path=None):
        super().__init__()
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:
            raise ImportError("SigLIP2Representor requires transformers.") from exc

        self.input_size = int(input_size)
        local_dir = Path(ckpt_path or pretrained_path("siglip2", "siglip2-base-patch16-256"))
        if local_dir.is_file():
            local_dir = local_dir.parent
        if not local_dir.is_dir():
            raise FileNotFoundError(f"SigLIP2 local model directory not found: {local_dir}")
        self.model = SiglipVisionModel.from_pretrained(str(local_dir))
        self.feat_dim = int(self.model.config.hidden_size)

        self.register_buffer("_mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def _preprocess(self, images):
        x = images
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bicubic", align_corners=False)
        return (x - self._mean) / self._std

    def extract_featmap(self, images):
        tokens = self.model(
            self._preprocess(images), output_hidden_states=False
        ).last_hidden_state
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected SigLIP2 tokens with shape [B, N, C], got {tuple(tokens.shape)}")
        b, n, c = tokens.shape
        h = w = int(math.isqrt(n))
        if h * w != n:
            raise RuntimeError(f"Cannot reshape {n} SigLIP2 patch tokens into a square feature map")
        return tokens.transpose(1, 2).reshape(b, c, h, w)
