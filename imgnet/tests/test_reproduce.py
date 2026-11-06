"""Environment, asset and train/eval reproduction entry points."""

import contextlib
import copy
import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reproduce
import torch
import yaml
from config import load_config
from metrics import eval_all_fds
from metrics.eval import _build_eval_args


class ReproductionTest(unittest.TestCase):
    def test_subprocess_environment_blocks_shell_startup_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            hook = Path(directory) / "startup.sh"
            hook.write_text("export CUBLAS_WORKSPACE_CONFIG=overridden\n")
            with patch.dict(
                "os.environ",
                {
                    "BASH_ENV": str(hook),
                    "ENV": str(hook),
                    "CUBLAS_WORKSPACE_CONFIG": "selected",
                    "PYTHONPATH": "/parent/python",
                },
            ):
                script = 'printf "%s" "$CUBLAS_WORKSPACE_CONFIG"'
                inherited = subprocess.check_output(["bash", "-c", script], text=True)
                self.assertEqual(inherited, "overridden")
                isolated = subprocess.check_output(
                    ["bash", "-c", script],
                    env=reproduce.process_environment(),
                    text=True,
                )
                self.assertEqual(isolated, "selected")
                self.assertNotIn("PYTHONPATH", reproduce.process_environment())
                self.assertEqual(os.environ["BASH_ENV"], str(hook))

    def fixtures(self, root):
        model = root / "models"
        for name in (
            "mae/vit_base_patch16_224.mae.pth",
            "resnet/resnet18-f37072fd.pth",
            "siglip2/siglip2-base-patch16-256/config.json",
        ):
            file = model / "pretrained" / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("{}")
        data = root / "images"
        data.mkdir()
        inception = model / "hub/checkpoints/weights-inception-2015-12-05-6726825d.pth"
        inception.parent.mkdir(parents=True, exist_ok=True)
        inception.write_bytes(b"fixture weights")
        init = root / "initial.pt"
        init.write_bytes(b"initial-test-weights")
        stats = root / "stats.npz"
        stats.write_bytes(b"test-reference-placeholder")
        # Distinct NextGAN configurations exercise the multi-experiment path
        # without introducing another algorithm or latent decoder dependency.
        first_config = reproduce.ROOT / "configs/nextgan_pixel_256.yaml"
        second_config = root / "nextgan_pixel_alternate.yaml"
        second_config.write_text(first_config.read_text())
        argv = [
            "--config",
            str(first_config),
            "--config",
            str(second_config),
            "--model-zoo",
            str(model),
            "--data-zoo",
            str(root),
            "--image-dir",
            str(data),
            "--pixel-init",
            str(init),
            "--runs-dir",
            str(root / "runs"),
            "--nproc",
            "1",
        ]
        return argv, stats, model, data

    def exercise(self, argv, stats, *, bad_metrics=False):
        calls = []
        self.calls = calls

        def run(command, log, env, timeout=None):
            calls.append(list(map(str, command)))
            log.write_text("test execution substitute\n")
            if command[0] == "bash":
                config = Path(command[-1])
                cfg = yaml.safe_load(config.read_text())
                if command[1].endswith("train.sh"):
                    save = Path(cfg["experiment"]["save_dir"])
                    save.mkdir(parents=True, exist_ok=True)
                    step = 4 if cfg["experiment"].get("resume") else 2
                    (save / f"{cfg['model']['name']}_step_{step}.pt").write_bytes(
                        b"test checkpoint"
                    )
                else:
                    args = _build_eval_args(str(config), cfg)
                    summary = Path(args.output_dir) / "final_eval_summary.csv"
                    summary.parent.mkdir(parents=True, exist_ok=True)
                    with summary.open("w") as handle:
                        writer = csv.DictWriter(
                            handle, fieldnames=["model", "step", "ema_label", "fd"]
                        )
                        writer.writeheader()
                        writer.writerow(
                            {
                                "model": "FID(ADM)",
                                "step": 2,
                                "ema_label": "ema",
                                "fd": "nan" if bad_metrics is True else 1.5,
                            }
                        )
                        if bad_metrics == "mixed":
                            writer.writerow(
                                {
                                    "model": "FID(JiT)",
                                    "step": 2,
                                    "ema_label": "ema",
                                    "fd": "nan",
                                }
                            )
            return log

        with (
            patch("reproduce.environment"),
            patch("reproduce.command", side_effect=run),
            patch.dict("os.environ", {}, clear=False),
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "device_count", return_value=1),
            patch.object(
                torch.cuda, "get_device_name", return_value="execution substitute"
            ),
            patch.object(eval_all_fds, "INCEPTION_STATS", [("FID(ADM)", stats)]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return reproduce.main(argv), calls

    def test_defaults_and_dry_run_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = Path(directory) / "not-created"
            args = reproduce.arguments(["--space", "both", "--runs-dir", str(runs)])
            self.assertEqual(
                [p.stem for p in args.config],
                [
                    "nextgan_pixel_256",
                    "nextgan_latent_256",
                ],
            )
            for space in ("pixel", "latent"):
                selected = reproduce.arguments(["--space", space])
                self.assertEqual(
                    [p.stem for p in selected.config], [f"nextgan_{space}_256"]
                )
            with (
                patch("reproduce.environment") as environment,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    reproduce.main(["--dry-run", "--runs-dir", str(runs)]), 0
                )
            environment.assert_not_called()
            self.assertFalse(runs.exists())

    def test_path_overrides_preserve_scientific_settings(self):
        with patch.dict(os.environ, {"DM_MODEL_ZOO": "/caller/models"}):
            cfg = load_config(reproduce.ROOT / "configs/nextgan_latent_256.yaml")
        cfg["evaluation"]["save_dir"] = "/existing-research-results"
        cfg["model"]["params"]["representations"][0]["ckpt_path"] = "weights/custom-mae.pth"
        before = copy.deepcopy(cfg)
        with patch("os.getcwd", return_value="/caller"):
            args = reproduce.arguments(
                [
                    "--space", "latent", "--model-zoo", "models",
                    "--data-zoo", "data", "--image-dir", "images",
                    "--pixel-init", "pixel.pt", "--latent-dir", "latents",
                    "--latent-init", "initial.pt", "--sdvae", "decoder",
                    "--env-dir", "env", "--runs-dir", "runs", "--nproc", "2",
                ]
            )
        for name, path in {
            "env_dir": "env", "runs_dir": "runs", "model_zoo": "models",
            "data_zoo": "data", "image_dir": "images", "pixel_init": "pixel.pt",
            "latent_dir": "latents", "latent_init": "initial.pt", "sdvae": "decoder",
        }.items():
            self.assertEqual(getattr(args, name), Path("/caller") / path)
        output = reproduce.resolved_config(cfg, args, Path("/runs/nextgan"))
        self.assertEqual(cfg, before)
        self.assertEqual(
            output["model"]["params"]["trainer"], before["model"]["params"]["trainer"]
        )
        self.assertEqual(output["latent"]["multiplier"], before["latent"]["multiplier"])
        self.assertEqual(output["network"], before["network"])
        self.assertEqual(output["data"]["data_dir"], "/caller/latents")
        self.assertEqual(output["experiment"]["init_weights"], "/caller/initial.pt")
        self.assertEqual(output["latent"]["decoder_path"], "/caller/decoder")
        self.assertEqual(
            output["model"]["params"]["representations"][0]["ckpt_path"],
            str(reproduce.ROOT / "weights/custom-mae.pth"),
        )
        self.assertEqual(output["evaluation"]["save_dir"], "/runs/nextgan/evaluation")

    def test_evaluation_paths_allow_pixel_config_without_training_params(self):
        cfg = yaml.safe_load(
            (reproduce.ROOT / "configs/nextgan_pixel_256.yaml").read_text()
        )
        cfg["latent"] = None
        cfg["model"].pop("params")
        cfg["evaluation"]["checkpoint"] = "outputs/selected.pt"
        original = copy.deepcopy(cfg)
        args = reproduce.arguments(["--stage", "eval"])
        output = reproduce.resolved_config(cfg, args, Path("/runs/nextgan"))
        self.assertEqual(cfg, original)
        self.assertEqual(
            output["evaluation"]["checkpoint"], str(reproduce.ROOT / "outputs/selected.pt")
        )
        self.assertIsNone(output["latent"])
        self.assertNotIn("params", output["model"])

        cfg["model"]["params"] = {
            "representations": [{"ckpt_path": "/unused-training-weights.pt"}]
        }
        args.model_zoo = Path("/models")
        output = reproduce.resolved_config(cfg, args, Path("/runs/nextgan"))
        self.assertEqual(output["model"]["params"], cfg["model"]["params"])

    def test_eval_asset_check_does_not_parse_training_options(self):
        cfg = {
            "model": {"name": "nextgan", "params": {"old_training_option": True}},
            "evaluation": {"models": []},
        }
        self.assertEqual(reproduce.required_assets(cfg, training=False, evaluating=True), [])
        with self.assertRaises(ValueError):
            reproduce.required_assets(cfg, training=True, evaluating=False)

    def test_selected_configs_train_evaluate_and_record_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            code, calls = self.exercise(argv, stats)
            self.assertEqual(code, 0)
            self.assertEqual(
                [c[1] for c in calls if c[0] == "bash"],
                [
                    "scripts/train.sh",
                    "scripts/eval.sh",
                    "scripts/train.sh",
                    "scripts/eval.sh",
                ],
            )
            report = json.loads((root / "runs/reproduction.json").read_text())
            self.assertEqual(report["status"], "completed")
            self.assertFalse(report["paper_metrics_verified"])
            self.assertFalse(report["dataset_content_hashed"])
            self.assertEqual(
                [r["model"] for r in report["experiments"]], ["nextgan", "nextgan"]
            )
            self.assertTrue(
                all(
                    r["checkpoint_sha256"] and r["metrics"]
                    for r in report["experiments"]
                )
            )
            for experiment in report["experiments"]:
                self.assertEqual(
                    experiment["resolved_config"]["model"]["name"], experiment["model"]
                )
                self.assertEqual(
                    Path(experiment["metric_summary"]).parent,
                    Path(experiment["resolved_config"]["evaluation"]["save_dir"]),
                )
            self.assertIn("scripts/train.sh", report["source"])
            self.assertIn("scripts/eval.sh", report["source"])
            self.assertTrue(
                all(r["metric_summary_sha256"] for r in report["experiments"])
            )

    def test_preflight_rejects_missing_assets_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            (root / "initial.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "initial.pt"):
                self.exercise(argv, stats)
            report = json.loads((root / "runs/reproduction.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(list((root / "runs").rglob("*_step_*.pt")))

    def test_reproduction_rejects_other_methods_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            custom = root / "nextgan_pixel_alternate.yaml"
            config = yaml.safe_load(custom.read_text())
            config["model"]["name"] = "logistic_gan"
            custom.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "model.name nextgan"):
                self.exercise(argv, stats)
            self.assertFalse(any(c[0] == "bash" for c in self.calls))

    def test_eval_only_does_not_require_training_assets_or_overwrite_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, model, data = self.fixtures(root)
            self.exercise(argv, stats)
            old = (root / "runs/reproduction.json").read_bytes()
            shutil.rmtree(model / "pretrained")
            shutil.rmtree(data)
            (root / "initial.pt").unlink()
            _, calls = self.exercise([*argv, "--stage", "eval"], stats)
            self.assertEqual(
                [c[1] for c in calls if c[0] == "bash"],
                ["scripts/eval.sh", "scripts/eval.sh"],
            )
            self.assertEqual((root / "runs/reproduction.json").read_bytes(), old)
            self.assertTrue(list((root / "runs").glob("reproduction-*.json")))

    def test_second_config_is_checked_before_first_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            custom = root / "nextgan_pixel_alternate.yaml"
            config = yaml.safe_load(custom.read_text())
            config["model"]["params"]["representations"][0]["ckpt_path"] = str(
                root / "missing-second.pth"
            )
            custom.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(FileNotFoundError, "missing-second"):
                self.exercise(argv, stats)
            self.assertFalse(any(c[0] == "bash" for c in self.calls))

    def test_nonfinite_metrics_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "finite FD"):
                self.exercise(argv, stats, bad_metrics=True)
            self.assertEqual(
                json.loads((root / "runs/reproduction.json").read_text())["status"],
                "failed",
            )

    def test_one_finite_metric_does_not_hide_another_invalid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "finite FD"):
                self.exercise(argv, stats, bad_metrics="mixed")

    def test_repeated_evaluation_preserves_prior_metrics_when_new_metrics_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            self.exercise(argv, stats)
            manifest = root / "runs/reproduction.json"
            old_report = manifest.read_bytes()
            summaries = {
                path: path.read_bytes()
                for path in (root / "runs").rglob("final_eval_summary.csv")
            }
            self.assertEqual(len(summaries), 2)
            with self.assertRaisesRegex(ValueError, "finite FD"):
                self.exercise([*argv, "--stage", "eval"], stats, bad_metrics=True)
            self.assertEqual(manifest.read_bytes(), old_report)
            for path, contents in summaries.items():
                self.assertEqual(path.read_bytes(), contents)

    def test_resume_preserves_old_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv, stats, _, _ = self.fixtures(root)
            self.exercise(argv, stats)
            self.exercise([*argv, "--resume", "--stage", "train"], stats)
            for name in ("nextgan_pixel_256", "nextgan_pixel_alternate"):
                save = root / "runs" / name / "training"
                self.assertTrue((save / "nextgan_step_2.pt").is_file())
                self.assertTrue((save / "nextgan_step_4.pt").is_file())


if __name__ == "__main__":
    unittest.main()
