from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pretrained_path


class ResNetRepresentor(nn.Module):
    def __init__(self, input_size, ckpt_path=None):
        super().__init__()
        import torchvision.models as models

        self.feat_dim = 512
        ckpt_path = ckpt_path or pretrained_path("resnet", "resnet18-f37072fd.pth")
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"ResNet checkpoint not found: {ckpt_path}")
        self.input_size = input_size

        backbone = models.resnet18(weights=None)
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "module", "encoder"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        backbone.load_state_dict(state, strict=True)
        children = list(backbone.children())
        self.features_no_avg = nn.Sequential(*children[:-2])
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def _preprocess(self, images):
        x = F.interpolate(images, size=self.input_size, mode="bilinear", align_corners=False)
        return (x - self._mean) / self._std

    def extract_featmap(self, images):
        return self.features_no_avg(self._preprocess(images))
