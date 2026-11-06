"""Metric computation, CSV output and distributed reference caches."""

import contextlib
import csv
import io
import json
import multiprocessing
import os
import tempfile
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
from metrics import eval_all_fds
from metrics.eval import _build_eval_args
from metrics.frechet_distance.evaluator import extract_ref_features
from PIL import Image


class CPUTorch:
    """Redirect only this GPU pipeline's device/context selection for the test."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(self, *args, **kwargs):
        return torch.device("cpu")

    def autocast(self, *args, **kwargs):
        return contextlib.nullcontext()


class EvaluationPipelineTest(unittest.TestCase):
    def test_csv_rows_preserve_order_rounding_shared_metrics_and_context(self):
        results = {"FID(JiT)": 4.123456789}
        results.update({name: value * 2 for name, value in eval_all_fds.FDR_VALIDATION_FD.items()})
        rows = eval_all_fds._build_csv_rows(
            results, 2.123456789,
            {"FID(ADM)": (0.123456789, 0.987654321)},
            123, {"n": 456, "step": 7, "ema_label": "online"},
            {"FID(JiT)": 3.123456789},
        )
        self.assertEqual([row["model"] for row in rows], list(results))
        self.assertTrue(all(row["n"] == 456 and row["fdr6"] == 2.0 for row in rows))
        self.assertEqual(rows[0]["fd"], 4.123457)
        self.assertIsNone(rows[0]["fdr"])
        for row in rows[:2]:
            self.assertEqual(
                (row["is"], row["precision"], row["recall"], row["cmmd"]),
                (2.1235, 0.123457, 0.987654, 3.123457),
            )
        for row in rows[2:]:
            self.assertEqual(row["fdr"], 2.0)
            self.assertIsNone(row["is"])
            self.assertIsNone(row["precision"])
            self.assertIsNone(row["cmmd"])

    def test_optional_metrics_keep_collective_order_and_rank_zero_values(self):
        features = torch.arange(6).view(3, 2).float()
        for prc, mmd in ((False, False), (True, False), (False, True), (True, True)):
            for rank in (0, 1):
                with self.subTest(prc=prc, mmd=mmd, rank=rank):
                    source_values = ([0.25, 0.75] if prc else []) + ([1.5] if mmd else [])
                    broadcasts = []
                    events = []

                    def gather(local, events=events, rank=rank):
                        events.append("gather")
                        torch.testing.assert_close(local, features)
                        return features if rank == 0 else None

                    def broadcast(value, events=events, broadcasts=broadcasts, source_values=source_values):
                        events.append("broadcast")
                        broadcasts.append(value)
                        return source_values[len(broadcasts) - 1]

                    with (
                        patch.object(eval_all_fds, "torch", CPUTorch()),
                        patch.object(eval_all_fds, "get_global_rank", return_value=rank),
                        patch.object(eval_all_fds, "is_enabled", return_value=True),
                        patch.object(eval_all_fds, "gather_features", side_effect=gather),
                        patch.object(eval_all_fds, "broadcast_scalar", side_effect=broadcast),
                        patch.object(eval_all_fds, "compute_precision_recall", return_value=(0.25, 0.75)) as compute_prc,
                        patch.object(eval_all_fds, "compute_mmd", return_value=1.5) as compute_mmd,
                    ):
                        result = eval_all_fds._reduce_and_compute(
                            [], [], [], {"toy": [features]} if prc or mmd else {},
                            prc_ref_features={"toy": features} if rank == 0 and prc else {},
                            prc_model_names=["toy"] if prc else [],
                            mmd_ref_features={"toy": features} if rank == 0 and mmd else {},
                            mmd_model_names=["toy"] if mmd else [],
                        )
                    self.assertEqual(result, (
                        {}, None, {"toy": (0.25, 0.75)} if prc else {},
                        {"toy": 1.5} if mmd else {},
                    ))
                    self.assertEqual(events, (["gather"] if prc or mmd else []) + ["broadcast"] * len(source_values))
                    self.assertEqual(broadcasts, source_values if rank == 0 else [0.0] * len(source_values))
                    self.assertEqual(compute_prc.call_count, int(rank == 0 and prc))
                    self.assertEqual(compute_mmd.call_count, int(rank == 0 and mmd))

    def test_feature_only_evaluation_does_not_compute_inception_score(self):
        class Features:
            def __call__(self, images):
                return images[:, :2, 0, 0], None

        entry = {
            "name": "feature", "model": Features(), "feat_dim": 2,
            "has_logits": False, "pool_type": "cls",
            "mu_ref": torch.zeros(2, dtype=torch.float64),
            "sigma_ref": torch.eye(2, dtype=torch.float64),
        }
        with (
            patch.object(eval_all_fds, "torch", CPUTorch()),
            patch.object(eval_all_fds, "np_isc", side_effect=AssertionError("unexpected IS")),
        ):
            accumulators = eval_all_fds._make_accumulators([entry], torch.device("cpu"))
            logits = []
            eval_all_fds.accumulate_batch(
                torch.arange(12).reshape(4, 3, 1, 1).float() / 12,
                [entry], accumulators, logits, {},
            )
            results, score, precision_recall, mmd = eval_all_fds._reduce_and_compute(
                [entry], accumulators, logits, {},
                prc_ref_features={}, prc_model_names=[],
                mmd_ref_features={}, mmd_model_names=[],
            )
        self.assertEqual(list(results), ["feature"])
        self.assertTrue(np.isfinite(results["feature"]))
        self.assertIsNone(score)
        self.assertEqual(precision_recall, {})
        self.assertEqual(mmd, {})

    def test_rank_batches_preserve_complete_class_sequence(self):
        class Generator:
            def __init__(self):
                self.labels = []

            def generate(self, labels):
                self.labels.extend(labels.tolist())
                return torch.zeros(len(labels), 3, 2, 2)

        for world_size in (1, 2, 3, 16):
            combined = []
            counts = []
            for rank in range(world_size):
                args = SimpleNamespace(
                    num_images=11, num_classes=3, batch_size=4,
                    enable_amp=False, amp_dtype=torch.float32,
                    prc_k=3, prc_batch_size=10,
                )
                model = Generator()
                with (
                    patch.object(eval_all_fds, "torch", CPUTorch()),
                    patch.object(eval_all_fds, "get_global_rank", return_value=rank),
                    patch.object(eval_all_fds, "get_world_size", return_value=world_size),
                    patch.object(eval_all_fds, "_reduce_and_compute"),
                ):
                    eval_all_fds._generate_and_evaluate(
                        args, model, [],
                        prc_ref_features={}, prc_model_names=[],
                        mmd_ref_features={}, mmd_model_names=[],
                    )
                combined.extend(model.labels)
                counts.append(len(model.labels))
            self.assertEqual(combined, [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_selected_weights_preserve_samples_metrics_and_csv_contract(self):
        class Generator:
            def __init__(self, offset):
                self.offset = offset
                self.batches = []

            def generate(self, labels):
                self.batches.append(labels.tolist())
                x = labels.float() / 4 + self.offset
                images = torch.stack((x, x.square(), torch.zeros_like(x)), dim=1)
                return (images[:, :, None, None] * 2 - 1).expand(len(labels), 3, 2, 2)

        class Features:
            def __call__(self, images):
                features = images[:, :2].mean(dim=(2, 3))
                return features, features

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "nextgan_step_19.pt"
            checkpoint.touch()
            for weights, offset in (("online", 0.1), ("ema", 0.2)):
                cfg = {
                    "model": {"name": "nextgan"},
                    "experiment": {"save_dir": directory},
                    "data": {"resolution": 4, "num_classes": 3},
                    "evaluation": {
                        "weights": weights,
                        "checkpoint": str(checkpoint),
                        "save_dir": str(Path(directory) / weights),
                        "num_images": 11,
                        "batch_size": 4,
                        "models": ["inception"],
                        "no_prc": True,
                    },
                }
                args = _build_eval_args("config.yaml", cfg)
                Path(args.output_dir).mkdir()
                args.current_step = 19
                args.enable_amp = False
                args.amp_dtype = torch.float32
                entry = {
                    "name": "FID(ADM)",
                    "model": Features(),
                    "feat_dim": 2,
                    "has_logits": True,
                    "mu_ref": torch.zeros(2, dtype=torch.float64),
                    "sigma_ref": torch.eye(2, dtype=torch.float64),
                }
                model = Generator(offset)
                with (
                    patch.object(eval_all_fds, "torch", CPUTorch()),
                    patch.object(
                        eval_all_fds, "_load_repr_models", return_value=[entry]
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    eval_all_fds.main_generate(args, model)

                self.assertEqual(model.batches, [[0, 1, 2, 0], [1, 2, 0, 1], [2, 0, 1]])
                with (Path(args.output_dir) / "final_eval_summary.csv").open() as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, eval_all_fds.CSV_FIELDS)
                    rows = list(reader)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(
                    (row["ema_label"], row["step"], row["n"]), (weights, "19", "11")
                )
                self.assertEqual(
                    (
                        row["cfg"],
                        row["interval_min"],
                        row["interval_max"],
                        row["num_sampling_steps"],
                    ),
                    ("1.0", "0.0", "1.0", "1"),
                )
                # Independent closed form of FD against N(0,I) in two dimensions.
                x = (torch.arange(11) % 3).float() / 4 + offset
                features = torch.stack((x, x.square()), dim=1).double().numpy()
                covariance = np.cov(features, rowvar=False)
                trace_sqrt = np.sqrt(
                    np.trace(covariance) + 2 * np.sqrt(np.linalg.det(covariance))
                )
                expected = (
                    np.square(features.mean(0)).sum()
                    + np.trace(covariance)
                    + 2
                    - 2 * trace_sqrt
                )
                self.assertAlmostEqual(float(row["fd"]), expected, places=6)
                self.assertTrue(np.isfinite(float(row["is"])))


class _Features:
    def __call__(self, images):
        cls = images[:, :2, 0, 0]
        return cls, cls.square()


def _cache_paths(directory):
    return {pool: Path(f"{directory}_toy_{pool}.pt") for pool in ("cls", "avg")}


def _reference_cache_worker(rank, rendezvous, temporary, miss_observed):
    result = {"rank": rank, "passed": False, "cases": []}
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
            timeout=timedelta(seconds=10),
        )
        expected = torch.tensor([[25, 50], [75, 100], [125, 150]]).float() / 255
        model = _Features()
        entries = [
            {"name": f"toy_{pool}", "model": model, "feat_dim": 2,
             "has_logits": False, "pool_type": pool, "mu_ref": torch.zeros(2)}
            for pool in ("cls", "avg")
        ]
        for name in ("cache_hit", "published_after_rank_zero_miss", "missing_directory"):
            directory = Path(temporary, name)
            paths = _cache_paths(directory)
            extract_calls = []
            original_exists = os.path.exists

            def exists(path, original_exists=original_exists, name=name, paths=paths):
                available = original_exists(path)
                if rank == 0 and name == "published_after_rank_zero_miss" and str(path) in {
                    str(value) for value in paths.values()
                } and not available:
                    miss_observed.set()
                return available

            def extract_cpu(feat_fn, ref_dir, extract_calls=extract_calls):
                extract_calls.append(ref_dir)
                # CPU Gloo's async writes cannot target inference tensors in
                # Torch 2.5. Exercise the same extraction/gather function body.
                return extract_ref_features.__wrapped__(
                    feat_fn, ref_dir, device="cpu", batch_size=1, input_size=8,
                )

            if rank == 1 and name == "published_after_rank_zero_miss":
                if not miss_observed.wait(8):
                    raise AssertionError("Rank zero did not observe the initial cache miss")
                # A concurrent evaluator can publish a shared cache before a late
                # rank checks it. Rank zero's earlier decision still governs both.
                for path in paths.values():
                    torch.save(torch.full_like(expected, -99), path)
                assert all(path.is_file() for path in paths.values())

            with (
                patch.object(os.path, "exists", side_effect=exists),
                patch.object(eval_all_fds, "extract_ref_features", side_effect=extract_cpu),
                patch.object(torch, "autocast", side_effect=lambda *args, **kwargs: contextlib.nullcontext()),
            ):
                refs, names = eval_all_fds._load_prc_refs(["toy"], entries, str(directory))
            if name == "missing_directory":
                assert refs == {} and names == [] and not extract_calls
            else:
                assert names == ["toy_cls", "toy_avg"]
                assert len(extract_calls) == int(name == "published_after_rank_zero_miss")
                if rank == 0:
                    torch.testing.assert_close(refs["toy_cls"], expected, rtol=0, atol=0)
                    torch.testing.assert_close(refs["toy_avg"], expected.square(), rtol=0, atol=0)
                    for pool, path in paths.items():
                        target = expected if pool == "cls" else expected.square()
                        torch.testing.assert_close(torch.load(path, weights_only=True), target, rtol=0, atol=0)
                else:
                    assert refs == {}, "Nonzero ranks must retain reference placeholders"
            result["cases"].append({"name": name, "extract_calls": len(extract_calls), "names": names})
            dist.barrier()
        result["passed"] = True
    except Exception:  # noqa: BLE001 - Preserve child tracebacks for the parent.
        result["error"] = traceback.format_exc()
    finally:
        Path(temporary, f"rank-{rank}.json").write_text(json.dumps(result))
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo is unavailable")
class EvaluationReferenceCacheTest(unittest.TestCase):
    def test_cache_hit_and_concurrent_publication_follow_rank_zero(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as temporary:
            expected = torch.tensor([[25, 50], [75, 100], [125, 150]]).float() / 255
            for name in ("cache_hit", "published_after_rank_zero_miss"):
                directory = Path(temporary, name)
                directory.mkdir()
                for index, (red, green) in enumerate(((25, 50), (75, 100), (125, 150))):
                    Image.new("RGB", (4, 4), (red, green, 0)).save(directory / f"{index}.png")
                if name == "cache_hit":
                    for pool, path in _cache_paths(directory).items():
                        torch.save(expected if pool == "cls" else expected.square(), path)
            miss_observed = context.Event()
            processes = [
                context.Process(target=_reference_cache_worker, args=(
                    rank, str(Path(temporary, "rendezvous")), temporary, miss_observed,
                )) for rank in range(2)
            ]
            deadline = time.monotonic() + 25
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(max(0, deadline - time.monotonic()))
                self.assertFalse(any(process.is_alive() for process in processes), "Reference-cache Gloo test timed out")
                for rank, process in enumerate(processes):
                    report = Path(temporary, f"rank-{rank}.json")
                    self.assertTrue(report.is_file(), f"Rank {rank} exited {process.exitcode} without a report")
                    result = json.loads(report.read_text())
                    self.assertTrue(result["passed"], result.get("error", result))
                    self.assertEqual(process.exitcode, 0)
                    self.assertEqual([case["extract_calls"] for case in result["cases"]], [0, 1, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    if process.pid is not None:
                        process.join(2)
                        if process.is_alive():
                            process.kill()
                            process.join(2)


if __name__ == "__main__":
    unittest.main()
