"""Encode ImageNet-1K images to VA-VAE latents for latent-space DiT training.

Produces .safetensors shards under <output_dir>/ matching
builders/datasets.py::ImageNetLatentDataset's expected format: each shard
holds "latents", "latents_flip", and "labels" tensors. Per-channel
normalization ((latent - mean) / std, UCGM's convention -- see
networks/vaes/vavae.py's docstring) is applied here at encode time, so the
stored latents are already normalized and ImageNetLatentDataset needs no
multiplier beyond 1.0.

Usage (single node, run under torchrun for multi-GPU):
    torchrun --standalone --nproc_per_node=8 scripts/encode_latents_vavae.py \
        --data_dir /storage/.../imagenet-1k-imagefolder/train \
        --output_dir /storage/.../data/ImageNet-1K/latents/vavae_f16d32/train_256 \
        --vae_path /storage/.../model_zoo/pretrained/vavae/vavae_f16d32 \
        --stat_path /storage/.../model_zoo/pretrained/vavae/vavae_f16d32_256_stat.pt \
        --image_size 256
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from networks.vaes import VAVAE


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--stat_path", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only encode this many images total (small-scale validation).",
    )
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    decoder = VAVAE(
        model_path=args.vae_path,
        stat_path=args.stat_path,
        multiplier=1.0,
    ).to(device)

    dataset = ImageFolder(
        args.data_dir,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
        ]),
    )
    if args.limit > 0:
        # Random indices, not range(limit): ImageFolder orders samples by
        # class-folder name, so the first --limit images would all come
        # from class 0 -- useless for checking that labels actually vary
        # and propagate correctly through the shard.
        generator = torch.Generator().manual_seed(args.seed)
        limit_idx = torch.randperm(len(dataset), generator=generator)[: args.limit].tolist()
        dataset = torch.utils.data.Subset(dataset, limit_idx)

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_images = len(dataset)
    if rank == 0:
        print(
            f"[encode_latents_vavae] total images per rank's loop: {total_images}, world_size={world_size}"
        )

    latents_buf, latents_flip_buf, labels_buf = [], [], []
    shard_idx = 0
    seen = 0

    def flush():
        nonlocal latents_buf, latents_flip_buf, labels_buf, shard_idx
        if not latents_buf:
            return
        latents = torch.cat(latents_buf, dim=0)
        latents_flip = torch.cat(latents_flip_buf, dim=0)
        labels = torch.cat(labels_buf, dim=0)
        shard_path = os.path.join(
            args.output_dir, f"latents_rank{rank:02d}_shard{shard_idx:03d}.safetensors"
        )
        save_file(
            {"latents": latents, "latents_flip": latents_flip, "labels": labels},
            shard_path,
            metadata={"total_size": str(latents.shape[0])},
        )
        print(f"[rank {rank}] saved {shard_path} ({latents.shape[0]} images)")
        latents_buf, latents_flip_buf, labels_buf = [], [], []
        shard_idx += 1

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)

            z = decoder.encode(x).cpu()
            z_flip = decoder.encode(x.flip(-1)).cpu()

            latents_buf.append(z)
            latents_flip_buf.append(z_flip)
            labels_buf.append(y)
            seen += x.shape[0]

            if rank == 0 and batch_idx % 20 == 0:
                print(
                    f"[rank {rank}] processed {seen}/{total_images} images, latent shape {z.shape}"
                )

            if sum(t.shape[0] for t in latents_buf) >= args.shard_size:
                flush()

        flush()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print("[encode_latents_vavae] done")


if __name__ == "__main__":
    main()
