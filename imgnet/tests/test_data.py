"""ImageFolder caches, latent datasets and latent encoding."""

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from builders import datasets
from PIL import Image
from safetensors.torch import load_file, save_file
from scripts import encode_latents_vavae as encoding
from torch.nn import functional as F
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class DatasetCacheTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "images"
        self.cache_dir = self.directory / "cache"
        for label in range(3):
            folder = self.root / f"n{label:03d}"
            folder.mkdir(parents=True)
            for index in range(2):
                Image.new("RGB", (10, 6), color=(label * 60, index * 50, 20)).save(
                    folder / f"{index}.png"
                )

    def cached(self, transform=None):
        return datasets.CachedImageFolder(
            self.root, transform=transform, cache_dir=self.cache_dir
        )

    def config(self, resolution=256, num_classes=3):
        return {
            "data": {
                "name": "in1k",
                "data_dir": str(self.root),
                "resolution": resolution,
                "num_classes": num_classes,
                "cache_dir": str(self.cache_dir),
            },
            "experiment": {"save_dir": str(self.directory / "run")},
        }

    def test_cache_hit_reuses_index_and_current_transform(self):
        original = self.cached()
        cache_path = datasets.CachedImageFolder._cache_path(self.root, self.cache_dir)
        cache = torch.load(cache_path, weights_only=False)
        self.assertEqual(cache["metadata"]["root"], os.path.abspath(self.root))
        self.assertEqual(len(cache["metadata"]["class_dir_stats"]), 3)
        with patch.object(ImageFolder, "__init__", side_effect=AssertionError("rescan")):
            cached = self.cached(transforms.ToTensor())
        self.assertEqual(cached.samples, original.samples)
        self.assertEqual(cached.targets, original.targets)
        self.assertEqual(cached.class_to_idx, original.class_to_idx)
        self.assertIs(cached.imgs, cached.samples)
        self.assertIsNone(cached.target_transform)
        self.assertFalse(hasattr(cached, "cache_dir"))
        image, label = cached[2]
        self.assertEqual(image.shape, (3, 6, 10))
        self.assertEqual(label, 1)

    def test_class_directory_mtime_invalidates_cached_index(self):
        original = self.cached()
        folder = self.root / "n001"
        previous = folder.stat()
        Image.new("RGB", (10, 6)).save(folder / "new.png")
        os.utime(folder, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
        updated = self.cached()
        self.assertEqual(len(updated), len(original) + 1)
        self.assertIn(str(folder / "new.png"), [path for path, _ in updated.samples])
        with patch.object(ImageFolder, "__init__", side_effect=AssertionError("rescan")):
            self.assertEqual(self.cached().samples, updated.samples)

    def test_corrupt_or_invalid_cache_is_rebuilt(self):
        self.cached()
        cache_path = Path(datasets.CachedImageFolder._cache_path(self.root, self.cache_dir))
        for invalid in (b"not a torch checkpoint", torch.ones(2)):
            with self.subTest(cache_type=type(invalid).__name__):
                if isinstance(invalid, bytes):
                    cache_path.write_bytes(invalid)
                else:
                    torch.save(invalid, cache_path)
                rebuilt = self.cached()
                self.assertEqual(len(rebuilt), 6)
                cache = torch.load(cache_path, weights_only=False)
                self.assertEqual(cache["samples"], rebuilt.samples)
                self.assertEqual(cache["metadata"]["root"], os.path.abspath(self.root))

    def test_first_classes_use_subset_with_original_contiguous_labels(self):
        subset, count = datasets.build_dataset(self.config(num_classes=2))
        self.assertIsInstance(subset, Subset)
        self.assertEqual(count, 2)
        self.assertEqual(subset.indices, [0, 1, 2, 3])
        self.assertEqual([subset[index][1] for index in range(len(subset))], [0, 0, 1, 1])
        full, count = datasets.build_dataset(self.config(num_classes=5))
        self.assertIsInstance(full, datasets.CachedImageFolder)
        self.assertEqual(count, 3)

    def test_pixel_preprocessing_retains_resolution_specific_operations(self):
        for resolution in (128, 256):
            with self.subTest(resolution=resolution):
                dataset, _ = datasets.build_dataset(self.config(resolution=resolution))
                operations = dataset.transform.transforms
                if resolution >= 256:
                    self.assertEqual(operations[0].size, resolution)
                    self.assertIsInstance(operations[1], transforms.CenterCrop)
                    self.assertIsInstance(operations[2], transforms.RandomHorizontalFlip)
                else:
                    self.assertEqual(operations[0].size, (resolution, resolution))
                    self.assertEqual(len(operations), 3)
                image, _ = dataset[0]
                self.assertEqual(image.shape, (3, resolution, resolution))
                self.assertGreaterEqual(image.min().item(), -1.0)
                self.assertLessEqual(image.max().item(), 1.0)

    def test_late_rank_still_joins_barrier_after_rank_zero_created_cache(self):
        folder_type = datasets.CachedImageFolder
        cache_path = Path(folder_type._cache_path(self.root, self.cache_dir))
        for preexisting in (False, True):
            with self.subTest(preexisting=preexisting):
                if cache_path.exists():
                    cache_path.unlink()
                if preexisting:
                    self.cached()
                rank = [0]
                events = []

                def construct(**kwargs):
                    events.append(("build", rank[0], cache_path.is_file()))
                    return folder_type(**kwargs)

                with (
                    patch.object(datasets.dist, "is_initialized", return_value=True),
                    patch.object(datasets.dist, "get_world_size", return_value=2),
                    patch.object(datasets.dist, "get_rank", side_effect=lambda: rank[0]),
                    patch.object(
                        datasets.dist, "barrier", side_effect=lambda: events.append(("barrier", rank[0]))
                    ) as barrier,
                    patch.object(datasets, "CachedImageFolder", side_effect=construct),
                ):
                    first = datasets._build_cached_image_folder(self.root, None, self.cache_dir)
                    self.assertTrue(cache_path.is_file())
                    # Rank 1 starts after rank 0 has already published the cache.
                    rank[0] = 1
                    late = datasets._build_cached_image_folder(self.root, None, self.cache_dir)
                self.assertEqual(barrier.call_count, 2)
                self.assertEqual(
                    events,
                    [("build", 0, preexisting), ("barrier", 0),
                     ("barrier", 1), ("build", 1, True)],
                )
                self.assertEqual(first.samples, late.samples)

    def test_single_process_build_needs_no_collective(self):
        with (
            patch.object(datasets.dist, "is_initialized", return_value=False),
            patch.object(datasets.dist, "get_rank", side_effect=AssertionError("rank")),
            patch.object(datasets.dist, "get_world_size", side_effect=AssertionError("world")),
            patch.object(datasets.dist, "barrier", side_effect=AssertionError("barrier")),
        ):
            dataset = datasets._build_cached_image_folder(self.root, None, self.cache_dir)
        self.assertEqual(len(dataset), 6)

    def test_latent_formats_keep_flip_selection_layout_and_multiplier(self):
        for channels in (4, 12):
            latent_root = self.directory / f"latents-{channels}"
            latent_root.mkdir()
            channels_first = torch.arange(12 * channels, dtype=torch.float32).reshape(
                2, channels, 2, 3
            )
            labels = torch.tensor([1, 2])
            save_file(
                {"latents": channels_first, "latents_flip": -channels_first, "labels": labels},
                str(latent_root / "train.safetensors"),
            )
            channels_last = channels_first.permute(0, 2, 3, 1).numpy()
            np.save(latent_root / "train_moments.npy", channels_last)
            np.save(latent_root / "train_moments_flip.npy", -channels_last)
            np.save(latent_root / "train_targets.npy", labels.numpy())
            for dataset in (
                datasets.ImageNetLatentDataset(str(latent_root), multiplier=2),
                datasets.ImageNetNpyLatentDataset(str(latent_root), multiplier=2),
            ):
                for random, sign in ((0.25, -1), (0.75, 1)):
                    with self.subTest(
                        format=type(dataset).__name__, channels=channels, flipped=sign < 0
                    ):
                        with patch.object(datasets.torch, "rand", return_value=torch.tensor(random)):
                            latent, label = dataset[1]
                        torch.testing.assert_close(latent, sign * 2 * channels_first[1])
                        self.assertEqual(int(label), 2)


class TinyEncoder(torch.nn.Module):
    def encode(self, images):
        return F.avg_pool2d(images, 2)


class LatentEncodingTest(unittest.TestCase):
    def test_shards_preserve_original_flip_pairs_and_subset_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "images"
            for index in range(6):
                path = data / f"class{index // 3}" / f"{index}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                height, width = 11 + index, 15 + index
                pixels = np.arange(height * width * 3).reshape(height, width, 3)
                Image.fromarray(((pixels + index * 19) % 256).astype(np.uint8)).save(path)

            original = ImageFolder(data, transforms.Compose([
                transforms.Resize(8), transforms.CenterCrop(8), transforms.ToTensor(),
            ]))
            flipped = ImageFolder(data, transforms.Compose([
                transforms.Resize(8), transforms.CenterCrop(8),
                transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(),
            ]))
            for limit in (0, 5):
                with self.subTest(limit=limit):
                    output = root / f"latents-{limit}"
                    argv = [
                        "encode_latents_vavae.py", "--data_dir", str(data),
                        "--output_dir", str(output), "--vae_path", "unused",
                        "--stat_path", "unused", "--image_size", "8",
                        "--batch_size", "2", "--shard_size", "3",
                        "--num_workers", "0", "--seed", "73", "--limit", str(limit),
                    ]
                    with (
                        patch("sys.argv", argv),
                        patch.object(encoding, "setup_distributed", return_value=(0, 1, 0)),
                        patch.object(encoding, "VAVAE", return_value=TinyEncoder()),
                        patch("torch.cuda.is_available", return_value=False),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        encoding.main()

                    shards = [load_file(path) for path in sorted(output.glob("*.safetensors"))]
                    self.assertEqual([len(shard["labels"]) for shard in shards],
                                     [4, limit - 4 if limit else 2])
                    indices = (torch.randperm(6, generator=torch.Generator().manual_seed(73))[:limit]
                               if limit else range(6))
                    plain = torch.stack([original[int(i)][0] for i in indices])
                    flip = torch.stack([flipped[int(i)][0] for i in indices])
                    labels = torch.tensor([original.targets[int(i)] for i in indices])
                    torch.testing.assert_close(torch.cat([s["latents"] for s in shards]),
                                               F.avg_pool2d(plain, 2), rtol=0, atol=0)
                    torch.testing.assert_close(torch.cat([s["latents_flip"] for s in shards]),
                                               F.avg_pool2d(flip, 2), rtol=0, atol=0)
                    torch.testing.assert_close(torch.cat([s["labels"] for s in shards]), labels)


if __name__ == "__main__":
    unittest.main()
