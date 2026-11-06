import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pretrained_path


class MAERepresentor(nn.Module):
    """Frozen MAE-B encoder exposing final-layer patch-token feature maps."""

    def __init__(self, input_size=224, ckpt_path=None):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "MAERepresentor requires timm to use arch='mae_b'."
            ) from exc

        ckpt_path = ckpt_path or pretrained_path("mae", "vit_base_patch16_224.mae.pth")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"MAE checkpoint not found: {ckpt_path}")
        self.net = timm.create_model("vit_base_patch16_224.mae", pretrained=False, num_classes=0)
        try:
            from safetensors.torch import load_file

            state = load_file(str(ckpt_path), device="cpu")
        except Exception:
            state = torch.load(Path(ckpt_path), map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "module", "encoder"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        self.net.load_state_dict(state, strict=True)
        self.input_size = input_size
        self.feat_dim = self.net.num_features
        self.num_prefix_tokens = getattr(self.net, "num_prefix_tokens", 1)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def _preprocess(self, images):
        x = F.interpolate(images, size=self.input_size, mode="bicubic", align_corners=False)
        return (x - self._mean) / self._std

    def extract_featmap(self, images):
        tokens = self.net.forward_features(self._preprocess(images))
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected MAE ViT tokens with shape [B, N, C], got {tuple(tokens.shape)}")
        tokens = tokens[:, self.num_prefix_tokens:]
        B, N, C = tokens.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise RuntimeError(f"Cannot reshape {N} MAE patch tokens into a square feature map")
        return tokens.transpose(1, 2).reshape(B, C, H, W)
