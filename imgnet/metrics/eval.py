#!/usr/bin/env python3
"""Evaluate ImageNet checkpoints with FD, FDr, IS, P&R, and CMMD metrics.

The evaluator builds a one-step generator from a public model YAML configuration and
uses distributed feature extraction for ImageNet evaluation.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np

from config import load_config, model_name
from builders import build_evaluation_generator
from builders.common import input_shape_from_config
from metrics import eval_all_fds
from metrics.utils.distributed_util import get_global_rank, get_world_size

IMGNET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_MODELS = [
    "inception",
    "vit_large_patch16_224.mae",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_large_patch14_clip_224.openai",
    "vit_so400m_patch16_siglip_256.v2_webli",
    "convnext",
]


def _abs_path(path: str, base: Path = IMGNET_ROOT) -> str:
    return str((base / Path(path).expanduser()).resolve())


class EvaluationGenerator:
    def __init__(self, cfg: dict, checkpoint: str, weights: str):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        cfg = {**cfg, "runtime": {**cfg.get("runtime", {}), "device": device}}
        self.noise_shape = input_shape_from_config(cfg)
        self.noise_scale = float(
            cfg["model"].get("params", {}).get("trainer", {}).get("noise_scale", 1.0)
        )
        self.generator, self.current_step = build_evaluation_generator(cfg, checkpoint, weights)

    @torch.inference_mode()
    def generate(self, labels, noise=None):
        if noise is None:
            noise = torch.randn(
                len(labels), *self.noise_shape, device=labels.device, dtype=torch.float32
            )
        noise = noise.to(labels.device) * self.noise_scale
        t = torch.ones(noise.shape[0], device=noise.device, dtype=noise.dtype)
        generated = self.generator(noise, labels, t)
        samples = self.generator.to_image(generated).float().clamp(0, 1)
        # eval_all_fds consumes images in [-1, 1].
        return samples * 2.0 - 1.0


def _latest_checkpoint(save_dir: str, name: str) -> str:
    prefix = f"{name}_step_"
    candidates = (
        path for path in Path(save_dir).glob(f"{prefix}*.pt")
        if path.is_file() and path.stem[len(prefix):].isdecimal()
    )
    latest = max(candidates, key=lambda path: int(path.stem[len(prefix):]), default=None)
    return str(latest) if latest else ""


def _build_eval_args(config_path: str, cfg: dict) -> argparse.Namespace:
    name = model_name(cfg)
    eval_cfg = cfg["evaluation"]
    weights = eval_cfg["weights"]
    if weights not in ("online", "ema"):
        raise ValueError(
            f"evaluation.weights must be 'online' or 'ema', got {weights!r}"
        )
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    save_dir = _abs_path(exp_cfg["save_dir"])
    checkpoint = (
        eval_cfg.get("checkpoint") or exp_cfg.get("resume")
        or _latest_checkpoint(save_dir, name)
    )
    if not checkpoint:
        raise FileNotFoundError(
            f"No checkpoint configured or found for eval config: {config_path}"
        )
    checkpoint = _abs_path(checkpoint)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Eval checkpoint does not exist: {checkpoint}")
    eval_name = eval_cfg.get("name") or Path(checkpoint).stem
    eval_dir = _abs_path(
        eval_cfg.get("save_dir", os.path.join(save_dir, "eval", eval_name))
    )

    return argparse.Namespace(
        checkpoint=checkpoint,
        weights=weights,
        output_dir=eval_dir,
        overwrite=bool(eval_cfg.get("overwrite", True)),
        input_size=int(eval_cfg.get("input_size", data_cfg["resolution"])),
        num_classes=int(eval_cfg.get("num_classes", data_cfg["num_classes"])),
        batch_size=int(eval_cfg.get("batch_size", 256)),
        num_images=int(eval_cfg.get("num_images", 50000)),
        models=eval_cfg.get("models", DEFAULT_EVAL_MODELS),
        seed=int(eval_cfg.get("seed", cfg.get("runtime", {}).get("seed", 1))),
        dtype=str(eval_cfg.get("dtype", "bf16")),
        no_prc=bool(eval_cfg.get("no_prc", False)),
        eval_mmd=bool(eval_cfg.get("eval_mmd", False)),
        prc_models=eval_cfg.get("prc_models"),
        prc_ref_dir=str(
            eval_cfg.get(
                "prc_ref_dir",
                os.environ.get("DM_DATA_ZOO", "/path/to/your/data_zoo")
                + "/imagenet-val-prc",
            )
        ),
        mmd_ref_dir=str(
            eval_cfg.get(
                "mmd_ref_dir",
                os.environ.get("DM_DATA_ZOO", "/path/to/your/data_zoo")
                + "/imagenet/val",
            )
        ),
        prc_k=int(eval_cfg.get("prc_k", 3)),
        prc_batch_size=int(eval_cfg.get("prc_batch_size", 10000)),
    )


def parse_args():
    launcher = argparse.ArgumentParser(description="Evaluate an ImageNet YAML config")
    launcher.add_argument("config", help="Path to an ImageNet YAML config")
    launcher.add_argument(
        "--checkpoint",
        default=None,
        help="Override evaluation.checkpoint from the YAML",
    )
    launch_args = launcher.parse_args()
    config_path = _abs_path(launch_args.config, Path.cwd())
    cfg = load_config(config_path)
    if launch_args.checkpoint:
        cfg.setdefault("evaluation", {})["checkpoint"] = launch_args.checkpoint
    return _build_eval_args(config_path, cfg), cfg


def setup_eval(args):
    # Match training without overriding Flash/Efficient/Math or math-only contexts.
    torch.backends.cuda.enable_cudnn_sdp(False)
    if "RANK" in os.environ and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    rank, world_size = get_global_rank(), get_world_size()
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if rank == 0:
        if args.overwrite and os.path.isdir(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()


def main():
    args, cfg = parse_args()
    setup_eval(args)
    model = EvaluationGenerator(cfg, args.checkpoint, args.weights)
    args.current_step = model.current_step
    eval_all_fds.main_generate(args, model)


if __name__ == "__main__":
    main()
