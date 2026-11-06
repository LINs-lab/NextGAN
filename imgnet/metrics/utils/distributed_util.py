import torch
import torch.distributed as dist


def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_enabled() else 1


def broadcast_scalar(value: float, device: str = "cuda") -> float:
    if not is_enabled():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.broadcast(t, src=0)
    return t.item()
