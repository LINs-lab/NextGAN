import torch.nn as nn

from .model import DiT_models


class DiTBackboneAdapter(nn.Module):
    """Expose DiT through the shared generator ``(x, y, t)`` API."""

    def __init__(self, model_name="DiT-XL/2", **kwargs):
        super().__init__()
        self.model = DiT_models[model_name](**kwargs)

    def forward(self, x, y, t=None):
        if t is None:
            t = x.new_zeros(x.shape[0])
        return self.model(x, t, y)


def dit_xl_1(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/1", num_classes=num_classes, **kwargs)


def dit_xl_2(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/2", num_classes=num_classes, **kwargs)


def dit_xl_4(num_classes=1000, **kwargs):
    return DiTBackboneAdapter("DiT-XL/4", num_classes=num_classes, **kwargs)
