from .common import (
    build_decoder,
    build_representation,
    build_network,
    finish_build,
)
from networks.generator import Generator
from methods.tbsm import RepresentationScattering, TBSM


def build(cfg, *, representations, trainer):
    device = cfg["runtime"]["device"]
    backbone = build_network(cfg).to(device)
    matchers = [
        RepresentationScattering(
            representor=build_representation(encoder),
            **options,
        ).to(device)
        for encoder, options in representations
    ]
    generator = Generator(
        backbone, decoder=build_decoder(cfg.get("latent"), device)
    )
    model = TBSM(
        generator=generator,
        matchers=matchers,
        num_classes=cfg["data"]["num_classes"],
        **trainer,
    )
    return finish_build(model, cfg)
