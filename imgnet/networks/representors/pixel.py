import torch.nn as nn


class PixelRepresentor(nn.Module):
    """Represent images as non-overlapping raw-pixel patches.

    No pretrained weights, resizing or channel normalization are applied.
    A patch of size p has 3*p*p features. The critic applies its shared
    running-statistics normalization to these features.
    """

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = int(patch_size)
        self.feat_dim = 3 * self.patch_size * self.patch_size
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def extract_featmap(self, images):
        b, c, h, w = images.shape
        p = self.patch_size
        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"PixelRepresentor: image size {h}x{w} not divisible by patch_size={p}"
            )
        # Unfold into non-overlapping p x p patches, each patch's raw pixels
        # flattened into the channel axis -- the pixel-space analogue of a
        # ViT patch embedding, but with no learned (or pretrained) projection.
        patches = images.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
            b, h // p, w // p, c * p * p
        )
        return patches.permute(0, 3, 1, 2).contiguous()  # (B, feat_dim, H/p, W/p)
