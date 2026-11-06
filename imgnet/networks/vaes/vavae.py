import torch
import torch.nn as nn


class VAVAE(nn.Module):
    """Frozen VA-VAE (UCGM's vavae_f16d32) mapping between image space and a
    per-channel-normalized latent space.

    Unlike SDVAE's single scalar multiplier, UCGM normalizes each of the 32
    latent channels independently: (latent - mean) / std, using stats
    computed once over a sample of encoded latents (buffers/vaes/stat/
    vavae_f16d32_256.pt in the UCGM release). This class expects a diffusers
    AutoencoderKL checkpoint converted from UCGM's raw vavae_f16d32.ckpt
    plus that stat file's mean/std tensors, so encoding and decoding reproduce the input distribution
    the UCGM TiT-XL checkpoint was trained on.
    """

    def __init__(self, model_path, stat_path, multiplier=1.0):
        super().__init__()
        from diffusers.models import AutoencoderKL

        self.model = AutoencoderKL.from_pretrained(model_path)
        stat = torch.load(stat_path, map_location="cpu", weights_only=True)
        self.register_buffer("latent_mean", stat["mean"].float())
        self.register_buffer("latent_std", stat["std"].float())
        self.multiplier = float(multiplier)
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    def encode(self, image):
        """Encode the posterior mean of an image in [0, 1] as a normalized latent."""
        raw = self.model.encode(image * 2.0 - 1.0).latent_dist.mean
        return (raw - self.latent_mean) / self.latent_std * self.multiplier

    def forward(self, latent):
        """Decode a normalized VA-VAE latent to an image in [0, 1]."""
        raw = latent / self.multiplier * self.latent_std + self.latent_mean
        image = self.model.decode(raw).sample
        return ((image + 1.0) * 0.5).clamp(0.0, 1.0)
