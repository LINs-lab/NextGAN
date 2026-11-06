import os
import hashlib
from glob import glob

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from safetensors import safe_open


class ImageNetLatentDataset(torch.utils.data.Dataset):
    """Load ImageNet latent shards and randomly choose original or flipped latents."""

    def __init__(self, root, multiplier=1.0):
        self.multiplier = float(multiplier)
        self.items = []
        for path in sorted(glob(os.path.join(root, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                count = f.get_slice("labels").get_shape()[0]
            self.items.extend((path, index) for index in range(count))
        if not self.items:
            raise FileNotFoundError(f"No .safetensors latent shards found under {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, offset = self.items[index]
        with safe_open(path, framework="pt", device="cpu") as f:
            key = "latents_flip" if torch.rand(()) < 0.5 else "latents"
            latent = f.get_slice(key)[offset:offset + 1].squeeze(0)
            label = f.get_slice("labels")[offset:offset + 1].squeeze(0)
        return latent * self.multiplier, label


class ImageNetNpyLatentDataset(torch.utils.data.Dataset):
    """Memory-map ImageNet latent arrays stored as NHWC NumPy files."""

    def __init__(self, root, split="train", multiplier=1.0):
        root = os.path.abspath(root)
        self.multiplier = float(multiplier)
        self.latents = np.load(
            os.path.join(root, f"{split}_moments.npy"), mmap_mode="r"
        )
        self.latents_flip = np.load(
            os.path.join(root, f"{split}_moments_flip.npy"), mmap_mode="r"
        )
        self.labels = np.load(
            os.path.join(root, f"{split}_targets.npy"), mmap_mode="r"
        )
        if not (len(self.latents) == len(self.latents_flip) == len(self.labels)):
            raise ValueError("Latent, flipped-latent, and label arrays must have equal length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        source = self.latents_flip if torch.rand(()) < 0.5 else self.latents
        latent = torch.from_numpy(np.array(source[index], copy=True)).float()
        latent = latent.permute(2, 0, 1).contiguous()
        label = int(self.labels[index])
        return latent * self.multiplier, label


class CachedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, *, cache_dir):
        cache_path = self._cache_path(root, cache_dir)
        cached = self._load_cache(cache_path, root)
        if cached is None:
            super().__init__(root=root, transform=transform)
            self._save_cache(cache_path, root)
            return

        self.root = root
        self.transform = transform
        self.target_transform = None
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = cached["classes"]
        self.class_to_idx = cached["class_to_idx"]
        self.samples = cached["samples"]
        self.imgs = self.samples
        self.targets = cached["targets"]

    @staticmethod
    def _class_dir_stats(root):
        stats = []
        if not os.path.isdir(root):
            return stats
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                stat = os.stat(path)
                stats.append((name, stat.st_mtime_ns))
        return stats

    @classmethod
    def _metadata(cls, root):
        return {
            "root": os.path.abspath(root),
            "class_dir_stats": cls._class_dir_stats(root),
        }

    @staticmethod
    def _cache_path(root, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        digest = hashlib.sha1(os.path.abspath(root).encode("utf-8")).hexdigest()[:16]
        return os.path.join(cache_dir, f"imagefolder_{digest}.pt")

    @classmethod
    def _load_cache(cls, cache_path, root):
        if not os.path.isfile(cache_path):
            return None
        try:
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            return cache if cache.get("metadata") == cls._metadata(root) else None
        except Exception:
            return None

    def _save_cache(self, cache_path, root):
        cache = {
            "metadata": self._metadata(root),
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "samples": self.samples,
            "targets": self.targets,
        }
        tmp_path = f"{cache_path}.{os.getpid()}.tmp"
        torch.save(cache, tmp_path)
        os.replace(tmp_path, cache_path)


def _build_cached_image_folder(root, transform, cache_dir):
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        dataset = CachedImageFolder(root=root, transform=transform, cache_dir=cache_dir)
    if distributed:
        dist.barrier()
    if rank != 0:
        dataset = CachedImageFolder(root=root, transform=transform, cache_dir=cache_dir)
    return dataset


def build_dataset(cfg):
    data_cfg = cfg["data"]
    dataset_name = data_cfg["name"]
    resolution = data_cfg["resolution"]
    num_classes = data_cfg["num_classes"]

    if dataset_name == "in1k_latent":
        dataset = ImageNetLatentDataset(
            data_cfg["data_dir"], multiplier=cfg.get("latent", {}).get("multiplier", 1.0)
        )
        return dataset, num_classes

    if dataset_name == "in1k_latent_npy":
        dataset = ImageNetNpyLatentDataset(
            data_cfg["data_dir"],
            split=data_cfg.get("split", "train"),
            multiplier=cfg.get("latent", {}).get("multiplier", 1.0),
        )
        return dataset, num_classes

    if dataset_name == "in1k":
        if resolution >= 256:
            transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])

        raw_data_dir = data_cfg["data_dir"]
        train_dir = os.path.join(raw_data_dir, "train")
        data_dir = train_dir if os.path.exists(train_dir) else raw_data_dir
        cache_dir = data_cfg.get("cache_dir")
        if cache_dir is None:
            # Store the transform-independent ImageFolder index two levels above
            # the run directory so resolutions and runs can reuse it.
            cache_dir = os.path.join(cfg["experiment"]["save_dir"], "..", "..", "dataset_cache")
        cache_dir = os.path.abspath(cache_dir)

        full_dataset = _build_cached_image_folder(data_dir, transform, cache_dir)
        total_classes = len(full_dataset.classes)

        if num_classes < total_classes:
            indices = [
                i for i, (_, lbl) in enumerate(full_dataset.samples)
                if lbl < num_classes
            ]
            dataset = Subset(full_dataset, indices)
        else:
            dataset = full_dataset
            num_classes = total_classes

        return dataset, num_classes

    raise ValueError(f"Unknown dataset: {dataset_name}")
