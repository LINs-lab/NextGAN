"""Shared assembly for the separate LogisticGAN and NextGAN classes."""

from .common import (
    build_decoder,
    build_representation,
    build_network,
    finish_build,
)

from networks.generator import Generator
from networks.critics import RepresentationCritic
from methods.logistic_gan import LogisticGAN
from methods.nextgan import NextGAN


def build(cfg, *, representations, trainer):
    model_type = {
        "logistic_gan": LogisticGAN,
        "nextgan": NextGAN,
    }[cfg["model"]["name"]]
    device = cfg["runtime"]["device"]
    backbone = build_network(cfg).to(device)
    critics = [
        RepresentationCritic(
            representor=build_representation(encoder),
            num_classes=cfg["data"]["num_classes"],
            **options,
        ).to(device)
        for encoder, options in representations
    ]
    generator = Generator(
        backbone, decoder=build_decoder(cfg.get("latent"), device)
    )
    model = model_type(
        generator=generator,
        critics=critics,
        **trainer,
    )
    return finish_build(model, cfg)
