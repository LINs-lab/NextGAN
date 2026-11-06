import os
import argparse
import random
import time
import json

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

from builders import build_training_model
from builders.common import input_shape_from_config, training_checkpoint
from builders.datasets import build_dataset
from config import load_config, model_name


def next_training_batch(data_iter, dataloader, sampler, epoch):
    try:
        images, labels = next(data_iter)
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
    return images, labels, data_iter, epoch


def setup_distributed():
    # Avoid the pinned runtime's cuDNN SDPA plan failures; retain other SDPA backends.
    torch.backends.cuda.enable_cudnn_sdp(False)
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_deterministic_backend():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def visual_grid_spec(resolution, num_classes):
    """Return fixed class IDs and samples per class for periodic sample grids."""
    if resolution >= 512:
        preferred_classes = [29, 291, 628, 985]
        samples_per_class = 4
    elif resolution >= 256:
        preferred_classes = [42, 281, 517, 867]
        samples_per_class = 4
    else:
        class_count = min(num_classes, 10)
        preferred_classes = (
            torch.linspace(0, num_classes - 1, class_count).long().tolist()
        )
        samples_per_class = 10

    if any(class_id >= num_classes for class_id in preferred_classes):
        class_count = min(len(preferred_classes), num_classes)
        preferred_classes = (
            torch.linspace(0, num_classes - 1, class_count).long().tolist()
        )

    return preferred_classes, samples_per_class


def save_sec_per_step(save_dir, step, steps_done, step_sec_sum):
    """Write the running average step time for the current process invocation."""
    avg_sec_per_step = step_sec_sum / steps_done if steps_done > 0 else 0.0
    speed_path = os.path.join(save_dir, "sec_per_step.json")
    with open(speed_path, "w") as f:
        json.dump(
            {
                "step": step,
                "steps_done": steps_done,
                "total_sec": step_sec_sum,
                "avg_sec_per_step": avg_sec_per_step,
            },
            f,
            indent=2,
        )
    return speed_path, avg_sec_per_step


def main():
    parser = argparse.ArgumentParser(description="Config-driven training script")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to a YAML config under configs/"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    name = model_name(cfg)
    network_cfg = cfg["network"]
    train_cfg = cfg["training"]
    runtime_cfg = cfg["runtime"]

    set_seed(runtime_cfg["seed"])
    if runtime_cfg["deterministic"]:
        configure_deterministic_backend()
    rank, world_size, local_rank = setup_distributed()
    main_process = rank == 0

    os.makedirs(exp_cfg["save_dir"], exist_ok=True)

    device = torch.device(f"cuda:{local_rank}")
    runtime_cfg.update(
        {
            "rank": rank,
            "world_size": world_size,
            "local_rank": local_rank,
            "device": device,
        }
    )
    use_bf16 = train_cfg["bf16"]

    dataset, num_classes = build_dataset(cfg)
    data_cfg["num_classes"] = num_classes
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=runtime_cfg["seed"],
        )
    else:
        sampler = None
    dataloader_generator = torch.Generator()
    dataloader_generator.manual_seed(runtime_cfg["seed"] + rank)

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=(sampler is None),
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
        generator=dataloader_generator,
    )

    method_model, start_step = build_training_model(cfg)
    if main_process:
        model_total, model_trainable = count_parameters(method_model)
        backbone_total, backbone_trainable = count_parameters(
            method_model.generator.backbone
        )
        print(
            "Run config:\n"
            f"  device: {device}\n"
            f"  world_size: {world_size}\n"
            f"  model: {name}\n"
            f"  network: {network_cfg['name']}\n"
            f"  method_model: {model_total / 1e6:.1f}M params ({model_trainable / 1e6:.1f}M trainable)\n"
            f"  generator.backbone: {backbone_total / 1e6:.1f}M params ({backbone_trainable / 1e6:.1f}M trainable)\n"
            f"  dataset: {data_cfg['name']} @ {data_cfg['resolution']}\n"
            f"  bf16: {use_bf16}\n"
            f"  seed: {runtime_cfg['seed']}\n"
            f"  deterministic: {runtime_cfg['deterministic']}\n"
            f"  save_dir: {exp_cfg['save_dir']}"
        )

        vis_classes, vis_cols = visual_grid_spec(data_cfg["resolution"], num_classes)
        vis_batch = len(vis_classes) * vis_cols
        fixed_noise = torch.randn(
            vis_batch, *input_shape_from_config(cfg), device=device
        )
        vis_classes = torch.tensor(vis_classes, device=device, dtype=torch.long)
        fixed_y = torch.repeat_interleave(vis_classes, repeats=vis_cols)

        accumulation = method_model.grad_accum_steps
        step_unit = "microbatches" if name == "tbsm" else "generator updates"
        print(
            f"Beginning {name} training ({train_cfg['total_steps']} {step_unit} from step {start_step}, "
            f"batch_size_per_gpu={train_cfg['batch_size']}, "
            f"batch_size_per_update={train_cfg['batch_size'] * world_size * accumulation})..."
        )

    data_iter = iter(dataloader)
    # Optimizer updates inside a step must invalidate autocast's weight cache.
    autocast_ctx = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16, cache_enabled=False
    )
    epoch = 0

    # Running step-time totals for this process invocation.
    step_sec_sum = 0.0
    step_sec_count = 0

    if main_process and train_cfg["vis_every"] > 0:
        with autocast_ctx:
            generated = method_model.sample(y=fixed_y, noise=fixed_noise)
        path = os.path.join(exp_cfg["save_dir"], f"visual_step_{start_step}.png")
        save_image(generated, path, nrow=vis_cols)
        print(f"  >> Visual grid saved to {path}")

    def next_batch():
        nonlocal data_iter, epoch
        images, labels, data_iter, epoch = next_training_batch(
            data_iter, dataloader, sampler, epoch
        )
        return images.to(device), labels.to(device)

    for step in range(start_step, train_cfg["total_steps"]):
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()

        # A GAN step consumes K*A new microbatches and completes one G update.
        with autocast_ctx:
            if name == "tbsm":
                loss_dict = method_model.step(*next_batch())
            else:
                loss_dict = method_model.step(next_batch)

        torch.cuda.synchronize(device)
        step_sec = time.perf_counter() - step_start

        if not main_process:
            continue

        step_sec_sum += step_sec
        step_sec_count += 1

        if step % train_cfg["log_every"] == 0:
            parts = [f"step_sec={step_sec:.3f}"]
            parts.extend(f"{key}={value:.4f}" for key, value in loss_dict.items())
            print(f"[Step {step:5d}] " + "  ".join(parts))

        completed_step = step + 1
        if train_cfg["vis_every"] > 0 and completed_step % train_cfg["vis_every"] == 0:
            with autocast_ctx:
                generated = method_model.sample(y=fixed_y, noise=fixed_noise)
            path = os.path.join(
                exp_cfg["save_dir"], f"visual_step_{completed_step}.png"
            )
            save_image(generated, path, nrow=vis_cols)
            print(f"  >> Visual grid saved to {path}")

            speed_path, avg_sec_per_step = save_sec_per_step(
                exp_cfg["save_dir"], completed_step, step_sec_count, step_sec_sum
            )
            print(
                f"  >> Global avg sec/step updated to {avg_sec_per_step:.4f} -> {speed_path}"
            )

        if train_cfg["save_every"] > 0 and completed_step % train_cfg["save_every"] == 0:
            ckpt_path = os.path.join(exp_cfg["save_dir"], f"{name}_step_{completed_step}.pt")
            checkpoint = training_checkpoint(method_model, completed_step)
            # Publish checkpoints atomically so a concurrent live evaluator
            # never observes a partially written torch.save file.
            tmp_ckpt_path = f"{ckpt_path}.{os.getpid()}.tmp"
            torch.save(checkpoint, tmp_ckpt_path)
            os.replace(tmp_ckpt_path, ckpt_path)
            print(f"  >> Checkpoint saved to {ckpt_path}")

    if main_process:
        print("Training finished.")
        # Write the final average for this process invocation.
        speed_path, avg_sec_per_step = save_sec_per_step(
            exp_cfg["save_dir"], train_cfg["total_steps"], step_sec_count, step_sec_sum
        )
        print(
            f"  >> Final global avg sec/step: {avg_sec_per_step:.4f} saved to {speed_path}"
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
