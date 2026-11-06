from config import model_name

from .evaluation import build_evaluation_generator
from .common import (
    public_model_params,
    representation_options,
    training_options,
    validate_checkpoint_path,
)

__all__ = ["build_training_model", "build_evaluation_generator"]


def build_training_model(cfg):
    from .tbsm import build as build_tbsm
    from .gan import build as build_gan

    builders = {
        "tbsm": build_tbsm,
        "logistic_gan": build_gan,
        "nextgan": build_gan,
    }
    name = model_name(cfg)
    params = public_model_params(cfg)
    validate_checkpoint_path(cfg)
    representations = representation_options(params, name)
    trainer = training_options(params)
    return builders[name](cfg, representations=representations, trainer=trainer)
