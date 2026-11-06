from types import SimpleNamespace

import torch.nn as nn

from .denoiser import JiTDenoiser_models


class JiTBackboneAdapter(nn.Module):
    def __init__(
        self,
        model_name,
        *,
        num_sampling_steps=1,
        sampling_method="euler",
        cfg=1.0,
        interval_min=0.1,
        interval_max=1.0,
        verbose=False,
        **kwargs,
    ):
        super().__init__()
        self.model = JiTDenoiser_models[model_name](**kwargs)
        self.sampling = SimpleNamespace(
            num_sampling_steps=num_sampling_steps,
            sampling_method=sampling_method,
            interval_min=interval_min,
            interval_max=interval_max,
        )
        self.cfg = cfg
        self.verbose = verbose

    def forward(self, x, y, t=None):
        if t is not None:
            labels = self.model.drop_labels(y) if self.training else y
            return self.model.net(x, self.model._backbone_t(t).flatten(), labels)
        return self.model.generate(
            x.shape[0],
            y,
            cfg=self.cfg,
            args=self.sampling,
            verbose=self.verbose,
            z_t=x,
        )


def jit_b(num_classes=1000, **kwargs):
    return JiTBackboneAdapter("JiT_B", num_classes=num_classes, **kwargs)


def jit_l(num_classes=1000, **kwargs):
    return JiTBackboneAdapter("JiT_L", num_classes=num_classes, **kwargs)


def jit_h(num_classes=1000, **kwargs):
    return JiTBackboneAdapter("JiT_H", num_classes=num_classes, **kwargs)
