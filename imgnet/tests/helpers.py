"""Small shared configuration and FP64 GAN fixtures."""

import torch
from torch import nn


class _BiasBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2], dtype=torch.float64))

    def forward(self, x):
        return self.bias.view(1, 1, -1).expand_as(x)


class _Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _BiasBackbone()

    def forward(self, x, labels, time):
        return self.backbone(x)

    def to_image(self, x):
        return x


class _IdentityRepresentor(nn.Module):
    def forward(self, image):
        return image


class _LinearPotential(nn.Module):
    output_mode = "potential"

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))

    def forward(self, labels, tokens, time):
        return (tokens * self.weight).flatten(1).sum(1)


class _Critic(nn.Module):
    critic_arch = "scattering"
    loss_weight = 1.0

    def __init__(self):
        super().__init__()
        self.representor = _IdentityRepresentor()
        self.critic = _LinearPotential()
        self.feature_calls = []
        self.normalization_calls = []

    def _feature_tokens(self, image, requires_grad):
        self.feature_calls.append((image.detach().clone(), requires_grad))
        with torch.set_grad_enabled(requires_grad):
            return self.representor(image)

    def _normalize(self, tokens, update_stats):
        self.normalization_calls.append(update_stats)
        return tokens


class _RecordingSGD(torch.optim.SGD):
    def __init__(self, parameter, lr, events=None, name=None):
        super().__init__([parameter], lr=lr)
        self.parameter = parameter
        self.gradients = []
        self.updated_parameters = []
        self.events = events
        self.name = name

    def step(self, closure=None):
        self.gradients.append(self.parameter.grad.detach().clone())
        if self.events is not None:
            self.events.append(self.name)
        result = super().step(closure)
        self.updated_parameters.append(self.parameter.detach().clone())
        return result


def _fresh_real(index):
    """Equal-size, distinct batches with changing feature directions."""
    return torch.tensor(
        [
            [[1.0 + 0.3 * index, (-1.0) ** index * 0.7]],
            [[-0.4 + 0.1 * index, 0.2 + 0.4 * index]],
        ],
        dtype=torch.float64,
    )


def public_config(name="tbsm"):
    return {
        "model": {
            "name": name,
            "params": {
                "critic": {"hidden_dim": 8, "depth": 1, "num_heads": 2, "time_dim": 8},
                "representations": [{"arch": "resnet18", "input_size": 4}],
                "trainer": {"gen_lr": 0.001, "critic_lr": 0.002},
            },
        },
        "network": {"name": "jit_b", "params": {"input_size": 4}},
        "data": {"num_classes": 2, "resolution": 4},
        "experiment": {"resume": "", "init_weights": ""},
    }
