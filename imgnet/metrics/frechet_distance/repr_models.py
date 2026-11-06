"""Frozen feature extractors for Frechet distance computation.

Supports timm models (DINOv2, CLIP, etc.), ConvNeXt, and InceptionV3.
"""

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F


logger = logging.getLogger("FD_loss")

MODEL_ZOO = Path(os.environ.get("DM_MODEL_ZOO", "/path/to/your/model_zoo"))
os.environ.setdefault("TORCH_HOME", str(MODEL_ZOO))
os.environ.setdefault("HF_HOME", str(MODEL_ZOO))
os.environ.setdefault("HF_HUB_CACHE", str(MODEL_ZOO / "hub"))
os.environ.setdefault("TIMM_HOME", str(MODEL_ZOO / "hub" / "checkpoints"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

def _preprocess(x, mean, std, input_size):
    """[0,1] float -> resize to input_size -> ImageNet normalize."""
    if x.shape[-2] != input_size or x.shape[-1] != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bicubic",
                          align_corners=False, antialias=True)
    return (x - mean) / std


class TimmReprModel(torch.nn.Module):
    """Wraps a timm model as a frozen feature extractor.

    Handles preprocessing: [0, 1] -> resize -> ImageNet normalize.
    Returns ``(cls_token, mean_token)``.
    """

    def __init__(self, model_name: str, device="cuda", input_size: int | None = None):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        logger.info(f"[TimmReprModel] Loading model: {model_name}")
        # dynamic_img_size/pad only supported by ViT-like models
        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        self.feat_dim = self.model.num_features

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        native_size = data_cfg["input_size"][-1]  # (C, H, W) -> W
        if "naflex" in model_name.lower():
            native_size = 256
        if input_size is not None and input_size != native_size:
            self.input_size = input_size
            logger.info(f"[TimmReprModel] Overriding input_size: {native_size} -> {input_size}")
        else:
            self.input_size = native_size

        mean = torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        interpolation = data_cfg.get("interpolation", "bicubic")
        logger.info(
            f"[TimmReprModel] {model_name}: feat_dim={self.feat_dim}, "
            f"input_size={self.input_size}, interpolation={interpolation}, "
            f"mean={data_cfg['mean']}, std={data_cfg['std']}"
        )

    def forward(self, x: torch.Tensor):
        x = _preprocess(x, self.mean, self.std, self.input_size)
        feats = self.model.forward_features(x)
        # CNN models return (B, C, H, W); pool spatially
        if feats.ndim == 4:
            cls_token = feats.mean(dim=[2, 3])
            return cls_token, None
        # ViT models return (B, N, C)
        patch_tokens = feats[:, self.num_prefix_tokens :]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token


def load_repr_model(name: str, device="cuda", input_size: int | None = None):
    """Load a representation feature extractor.

    Each model handles its own input resolution internally based on its
    training configuration (e.g. timm models use ``pretrained_cfg['input_size']``).

    Args:
        name: ``'inception'``, ``'convnext'``, or any timm model name.
        input_size: override the model's native target resolution.

    Returns:
        (model, feat_dim, has_logits, input_size)
    """
    if name == "inception":
        from metrics.utils.perception_util import load_inception

        net = load_inception(device=device)
        return net, 2048, True, 299
    elif name == "convnext":
        net = TimmReprModel("convnextv2_base.fcmae_ft_in22k_in1k", device=device, input_size=224)
        return net, net.feat_dim, False, net.input_size
    else:
        net = TimmReprModel(name, device=device, input_size=input_size)
        return net, net.feat_dim, False, net.input_size


def model_short_name(name: str) -> str:
    """Derive a concise label from a representation model name for logging/metrics."""
    if name in ("inception", "convnext"):
        return name
    low = name.lower()
    if "naflex" in low:
        return "naflex_siglip"
    for keyword in ("dinov2", "dino", "mae", "clip", "siglip"):
        if keyword in low:
            return keyword
    return name.split(".")[0].replace("_", "-")
