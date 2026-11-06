"""Reference feature extraction and distributed gathering for eval metrics."""

import logging
import os
import time

import torch
import torch.distributed as tdist

logger = logging.getLogger("FD_loss")

@torch.inference_mode()
def extract_ref_features(
    feat_fn,
    ref_dir: str,
    batch_size: int = 64,
    device: str | torch.device = "cuda",
    input_size: int = 256,
) -> torch.Tensor:
    """Extract features from a folder of reference images (distributed-aware).

    All ranks participate in extraction. Results are gathered to rank 0.

    Args:
        feat_fn: callable ``(batch_float_NCHW) -> features_tensor``.
            Input is float [0, 1], already on *device*.
        ref_dir: directory containing PNG/JPG reference images.
        batch_size: images per forward pass.
        device: device to place image batches on.

    Returns:
        ``(N, D)`` float tensor on CPU (rank 0), empty tensor on other ranks.
    """
    distributed = tdist.is_initialized()
    rank = tdist.get_rank() if distributed else 0
    world_size = tdist.get_world_size() if distributed else 1

    from PIL import Image
    import torchvision.transforms.functional as TF

    exts = {"png", "jpg", "jpeg", "webp", "JPEG"}
    # Try flat listing first; fall back to recursive walk (e.g. ImageNet val/)
    files = sorted(f for f in os.listdir(ref_dir) if f.rsplit(".", 1)[-1].lower() in {e.lower() for e in exts})
    if files:
        files = [os.path.join(ref_dir, f) for f in files]
    else:
        files = sorted(
            os.path.join(dp, fn)
            for dp, _, fns in os.walk(ref_dir)
            for fn in fns if fn.rsplit(".", 1)[-1] in exts
        )

    # Shard files across ranks
    per_rank = (len(files) + world_size - 1) // world_size
    start = rank * per_rank
    end = min(start + per_rank, len(files))
    local_files = files[start:end]

    logger.info(f"Extracting P&R ref features: {len(files)} total, "
                f"{len(local_files)} on rank {rank}")

    all_feats: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for i in range(0, len(local_files), batch_size):
        batch_files = local_files[i : i + batch_size]
        imgs = []
        for f in batch_files:
            img = Image.open(f).convert("RGB")
            # Center-crop to square, then resize to input_size
            w, h = img.size
            crop = min(w, h)
            img = TF.center_crop(img, crop)
            img = TF.resize(img, [input_size, input_size], antialias=True)
            imgs.append(TF.pil_to_tensor(img))
        batch = torch.stack(imgs).to(device).float() / 255.0
        all_feats.append(feat_fn(batch).float().cpu())
    local_features = torch.cat(all_feats, dim=0) if all_feats else torch.empty(0)

    elapsed = time.perf_counter() - t0
    logger.info(f"  rank {rank}: {len(local_files)} images in {elapsed:.1f}s")

    if distributed:
        local_features = local_features.to(device)
        features = gather_features(local_features)
        if rank != 0:
            return torch.empty(0)
        features = features.cpu()
    else:
        features = local_features

    return features

def gather_features(local_feats: torch.Tensor) -> torch.Tensor | None:
    """Gather variable-length feature tensors to rank 0.

    Returns the concatenated tensor on rank 0, ``None`` on other ranks.
    """
    world_size = tdist.get_world_size()
    rank = tdist.get_rank()
    device = local_feats.device
    local_n_t = torch.tensor([local_feats.shape[0]], dtype=torch.long, device=device)
    all_n = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    tdist.all_gather(all_n, local_n_t)
    max_n = max(t.item() for t in all_n)
    if local_feats.shape[0] < max_n:
        pad = torch.zeros(max_n - local_feats.shape[0], local_feats.shape[1],
                          dtype=local_feats.dtype, device=device)
        local_feats = torch.cat([local_feats, pad], dim=0)
    if rank == 0:
        gathered = [torch.zeros_like(local_feats) for _ in range(world_size)]
        tdist.gather(local_feats, gather_list=gathered, dst=0)
        trimmed = [g[:n.item()] for g, n in zip(gathered, all_n)]
        return torch.cat(trimmed, dim=0)
    else:
        tdist.gather(local_feats, dst=0)
        return None
