"""Distributed training synchronization and optimizer updates under AMP."""

import ast
import json
import multiprocessing
import tempfile
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from methods._gan_base import GANBase
from methods.logistic_gan import LogisticGAN
from methods.nextgan import NextGAN
from networks.critics import RepresentationCritic
from networks.generator import Generator
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, images, labels, t=None):
        return self.conv(images)


class _FrozenRepresentor(nn.Module):
    feat_dim = 3

    def extract_featmap(self, images):
        return F.avg_pool2d(images, 2)


def _parameter_vector(module):
    return torch.cat(
        [parameter.detach().flatten() for parameter in module.parameters()]
    )


def _assert_synchronized(module, name):
    local = _parameter_vector(module)
    if not bool(local.isfinite().all()):
        raise AssertionError(f"{name} has nonfinite parameters")
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    if not all(torch.equal(gathered[0], values) for values in gathered[1:]):
        raise AssertionError(f"{name} parameters diverged between DDP ranks")


def _distributed_worker(rank, rendezvous, output_dir):
    """Own one process group; report failures without hiding child tracebacks."""
    result = {"rank": rank, "passed": False, "cases": []}
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=Path(rendezvous).as_uri(),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=25),
        )
        for method_type in (LogisticGAN, NextGAN):
            # Match initial parameters, but give each rank distinct real batches.
            torch.manual_seed(71)
            critic = RepresentationCritic(
                _FrozenRepresentor(),
                num_classes=3,
                critic={
                    "hidden_dim": 8,
                    "time_dim": 8,
                    "depth": 1,
                    "num_heads": 2,
                    "output_mode": "potential",
                },
            )
            # Exercise a nonzero input Jacobian from the first generator cycle.
            nn.init.normal_(critic.critic.out_proj.weight, std=0.05)
            model = method_type(
                Generator(_TinyBackbone()),
                [critic],
                gen_lr=0.001,
                critic_lr=0.002,
                ema_decay=0.8,
                t_sampling=[1.0],
                grad_accum_steps=2,
                critic_steps=2,
            )
            model.generator.backbone = DistributedDataParallel(model.generator.backbone)
            critic.critic = DistributedDataParallel(critic.critic)
            data_rng = torch.Generator().manual_seed(910 + rank)
            consumed = 0

            def next_batch():
                nonlocal consumed
                images = torch.randn(2, 3, 4, 4, generator=data_rng)
                images = images + 0.1 * consumed + 0.25 * rank
                labels = (torch.arange(2) + consumed + rank) % 3
                consumed += 1
                return images, labels

            for cycle in range(2):
                before = {
                    "generator": _parameter_vector(model.generator),
                    "critic": _parameter_vector(critic.critic),
                    "ema": _parameter_vector(model._eval_generator),
                }
                losses = model.step(next_batch)
                if not all(
                    bool(torch.as_tensor(value).isfinite().all())
                    for value in losses.values()
                ):
                    raise AssertionError(
                        f"{method_type.__name__} returned nonfinite losses"
                    )
                if consumed != (cycle + 1) * 2 * 2:
                    raise AssertionError(
                        "Each critic update must consume two fresh microbatches"
                    )
                for name, module in (
                    ("generator", model.generator),
                    ("critic", critic.critic),
                    ("ema", model._eval_generator),
                ):
                    _assert_synchronized(module, f"{method_type.__name__} {name}")
                    if torch.equal(before[name], _parameter_vector(module)):
                        raise AssertionError(
                            f"{method_type.__name__} {name} did not update"
                        )
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise AssertionError(
                        "Completed generator cycles must not retain gradients"
                    )
            result["cases"].append(
                {"method": method_type.__name__, "cycles": 2, "batches": consumed}
            )
            dist.barrier()
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(
    dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable"
)
class GanDistributedTest(unittest.TestCase):
    def test_fresh_accumulation_keeps_generator_critic_and_ema_synchronized(self):
        # Leave ten seconds of the sixty-second bound for owned-worker cleanup.
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous = str(Path(temporary, "rendezvous"))
            processes = [
                context.Process(
                    target=_distributed_worker, args=(rank, rendezvous, temporary)
                )
                for rank in range(2)
            ]
            deadline = time.monotonic() + 50
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(
                    any(process.is_alive() for process in processes),
                    "Gloo GAN test timed out",
                )
                for rank, process in enumerate(processes):
                    report = Path(temporary, f"rank-{rank}.json")
                    self.assertTrue(
                        report.is_file(),
                        f"Rank {rank} exited {process.exitcode} without a report",
                    )
                    result = json.loads(report.read_text())
                    self.assertTrue(result["passed"], result.get("error", result))
                    self.assertEqual(process.exitcode, 0)
                    self.assertEqual(
                        result["cases"],
                        [
                            {"method": name, "cycles": 2, "batches": 8}
                            for name in ("LogisticGAN", "NextGAN")
                        ],
                    )
            finally:
                # Only terminate children created by this test, including a hung peer.
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    if process.pid is not None:
                        process.join(2)
                        if process.is_alive():
                            process.kill()
                            process.join(2)


def training_context(device, enabled):
    source = Path(__file__).resolve().parents[1] / "main.py"
    tree = ast.parse(source.read_text())
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "method_model"
            and child.func.attr == "step"
            for child in ast.walk(node)
        )
    ]
    assert len(matches) == 1
    context = matches[0].items[0].context_expr
    if isinstance(context, ast.Name):
        context = next(
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == context.id
                for target in node.targets
            )
        )
    expression = ast.Expression(context)
    return eval(
        compile(expression, str(source), "eval"),
        {
            "torch": torch,
            "device": device,
            "use_bf16": enabled,
        },
    )


class TrainingAutocastTest(unittest.TestCase):
    def check_device(self, device, enabled):
        model = torch.nn.Linear(4, 1, bias=False).to(device)
        torch.nn.init.zeros_(model.weight)
        opt = torch.optim.SGD(model.parameters(), lr=0.25)
        fields = []
        with training_context(device, enabled):
            for _ in range(2):
                q = torch.ones(2, 4, device=device, requires_grad=True)
                potential = model(q)
                self.assertEqual(
                    potential.dtype, torch.bfloat16 if enabled else torch.float32
                )
                field = torch.autograd.grad(potential.sum(), q, create_graph=True)[0]
                fields.append(field.detach().clone())
                (0.5 * (field - 1).square().sum(1).mean()).backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        self.assertTrue(torch.equal(fields[0], torch.zeros_like(fields[0])))
        self.assertTrue(
            torch.equal(fields[1], torch.full_like(fields[1], 0.25)),
            "Second D forward reused weights cached before optimizer.step",
        )

    def test_cpu_bf16(self):
        self.check_device(torch.device("cpu"), True)

    def test_cpu_fp32(self):
        self.check_device(torch.device("cpu"), False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bf16(self):
        self.check_device(torch.device("cuda"), True)

    def check_direct_step(self, device, enabled):
        model = torch.nn.Linear(4, 1, bias=False).to(device)
        torch.nn.init.zeros_(model.weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        fields = []
        test = self

        class Runner:
            step = GANBase.step
            generator = SimpleNamespace(backbone=model)

            def _step_uncached(self, next_batch):
                x, _ = next_batch()
                for _ in range(2):
                    q = x.detach().requires_grad_(True)
                    potential = model(q)
                    test.assertEqual(
                        potential.dtype, torch.bfloat16 if enabled else torch.float32
                    )
                    field = torch.autograd.grad(potential.sum(), q, create_graph=True)[
                        0
                    ]
                    fields.append(field.detach().clone())
                    (0.5 * (field - 1).square().sum(1).mean()).backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                return "completed"

        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=enabled, cache_enabled=True
        ):
            self.assertEqual(
                Runner().step(lambda: (torch.ones(2, 4, device=device), None)),
                "completed",
            )
            self.assertTrue(torch.is_autocast_cache_enabled())
            self.assertEqual(torch.is_autocast_enabled(device.type), enabled)
        torch.testing.assert_close(fields[0], torch.zeros_like(fields[0]))
        torch.testing.assert_close(fields[1], torch.full_like(fields[1], 0.25))

    def test_direct_step_cpu_bf16(self):
        self.check_direct_step(torch.device("cpu"), True)

    def test_direct_step_cpu_fp32(self):
        self.check_direct_step(torch.device("cpu"), False)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_direct_step_cuda_bf16(self):
        self.check_direct_step(torch.device("cuda"), True)


if __name__ == "__main__":
    unittest.main()
