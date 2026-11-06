import torch.nn as nn


class SDVAE(nn.Module):
    """Frozen SD-VAE decoder for scaled latents."""

    def __init__(self, model_path, multiplier=0.18215):
        super().__init__()
        from diffusers.models import AutoencoderKL

        self.model = AutoencoderKL.from_pretrained(model_path)
        self.multiplier = float(multiplier)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, latent):
        """Decode a scaled SD-VAE latent to an image in [0, 1]."""
        image = self.model.decode(latent / self.multiplier).sample
        return ((image + 1.0) * 0.5).clamp(0.0, 1.0)
