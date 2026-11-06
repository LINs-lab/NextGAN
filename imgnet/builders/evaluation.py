"""Build a frozen generator directly from online or EMA checkpoint weights."""

import os

import torch

from config import model_name
from networks.generator import Generator
from .common import build_decoder, build_network


def build_evaluation_generator(cfg, checkpoint, weights="ema"):
    """Load existing checkpoints without training algorithms, critics or optimizers."""
    model_name(cfg)
    if weights not in ("online", "ema"):
        raise ValueError(
            f"evaluation.weights must be 'online' or 'ema', got {weights!r}"
        )
    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Eval checkpoint does not exist: {checkpoint}")

    device = cfg["runtime"]["device"]
    # Full training checkpoints also contain critic/optimizer tensors. Keep
    # those on CPU and transfer only the selected generator into the model.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    step = ckpt["step"]
    if weights == "online":
        selected = ckpt["backbone"]
    else:
        prefix = "_eval_generator.backbone."
        selected = {
            key[len(prefix) :]: value
            for key, value in ckpt["model"].items()
            if key.startswith(prefix)
        }
        if not selected:
            raise ValueError("Checkpoint has no EMA generator weights")

    backbone = build_network(cfg).to(device)
    backbone.load_state_dict(selected, strict=True)
    generator = Generator(backbone, decoder=build_decoder(cfg.get("latent"), device))
    return generator.requires_grad_(False).eval(), step
