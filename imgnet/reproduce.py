#!/usr/bin/env python3
"""Install the environment, train NextGAN, and evaluate its checkpoints.

Run --help first. Dataset/model assets must be supplied locally. The supplied
top-level YAMLs are starting templates; configs/reproduced contains verified
DiT recipes. Match the assets and evaluation protocol documented in README.md.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
REPRESENTATION_WEIGHTS = {
    "mae_b": ("mae", "vit_base_patch16_224.mae.pth"),
    "resnet18": ("resnet", "resnet18-f37072fd.pth"),
    "siglip2_b": ("siglip2", "siglip2-base-patch16-256"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def process_environment():
    """Keep the selected venv independent of inherited shell startup hooks."""
    excluded = {"PYTHONPATH", "PYTHONHOME", "BASH_ENV", "ENV"}
    return {
        key: value
        for key, value in os.environ.items()
        if key not in excluded and not key.startswith("BASH_FUNC_")
    }


def command(argv, log, env, timeout=None):
    if log.exists():
        log = log.with_name(
            log.stem + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + log.suffix
        )
    print(f"Running: {shlex.join(map(str, argv))}\nLog: {log}", flush=True)
    with log.open("w") as handle:
        subprocess.run(
            list(map(str, argv)),
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout,
        )
    return log


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", choices=("pixel", "latent", "both"), default="pixel")
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        help="Custom final YAML; repeat for multiple experiments. Overrides --space.",
    )
    parser.add_argument(
        "--stage", choices=("all", "setup", "check", "train", "eval"), default="all"
    )
    parser.add_argument("--env-dir", type=Path, default=ROOT / ".venv-reproduce")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=ROOT / "reproductions" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"),
    )
    parser.add_argument(
        "--model-zoo", type=Path, default=os.environ.get("DM_MODEL_ZOO")
    )
    parser.add_argument("--data-zoo", type=Path, default=os.environ.get("DM_DATA_ZOO"))
    for name in ("image-dir", "latent-dir", "pixel-init", "latent-init", "sdvae"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--nproc", type=int, help="Override runtime.nproc_per_node explicitly"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume latest checkpoint in --runs-dir"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Verify an existing pinned environment",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without installing/writing/training",
    )
    args = parser.parse_args(argv)
    if args.nproc is not None and args.nproc < 1:
        parser.error("--nproc must be positive")
    for name in (
        "env_dir", "runs_dir", "model_zoo", "data_zoo", "image_dir",
        "latent_dir", "pixel_init", "latent_init", "sdvae",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    configs = args.config or [
        ROOT / "configs" / f"nextgan_{space}_256.yaml"
        for space in (("pixel", "latent") if args.space == "both" else (args.space,))
    ]
    args.config = [path.expanduser().resolve() for path in configs]
    if len({path.stem for path in args.config}) != len(args.config):
        parser.error("Config basenames must be unique within one reproduction")
    for path in args.config:
        if not path.is_file():
            parser.error(f"Config not found: {path}")
    return args


def environment(args):
    """Bootstrap using stdlib only; import Torch only after switching environments."""
    if sys.version_info[:2] < (3, 10) or sys.version_info[:2] > (3, 12):
        raise RuntimeError("Use Python 3.10–3.12")
    if (
        os.environ.get("SIMGAN_REPRO_ENV") == str(args.env_dir)
        and Path(sys.prefix).resolve() == args.env_dir
    ):
        return
    if args.env_dir.exists() and not (args.env_dir / "pyvenv.cfg").is_file():
        raise RuntimeError(f"Refusing to replace a non-venv directory: {args.env_dir}")
    if not args.env_dir.exists():
        if args.skip_install:
            raise FileNotFoundError(
                f"--skip-install requires an existing venv: {args.env_dir}"
            )
        venv.EnvBuilder(with_pip=True).create(args.env_dir)
    python = args.env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    env = process_environment()
    if not args.skip_install:
        command(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--progress-bar",
                "off",
                "--disable-pip-version-check",
                "-r",
                ROOT / "requirements.txt",
            ],
            args.runs_dir / "install.log",
            env,
            timeout=1200,
        )
    command(
        [python, "-m", "pip", "check"], args.runs_dir / "pip-check.log", env, timeout=60
    )
    # Preserve platform driver paths, with the isolated wheel's libraries first.
    libraries = sorted(args.env_dir.glob("lib/python*/site-packages/torch/lib"))
    libraries += sorted(args.env_dir.glob("lib/python*/site-packages/nvidia/*/lib"))
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        dict.fromkeys([*map(str, libraries), *existing])
    )
    env["SIMGAN_REPRO_ENV"] = str(args.env_dir)
    # Preserve the timestamped run directory across re-exec.
    argv = [
        *sys.argv[1:],
        "--env-dir",
        str(args.env_dir),
        "--runs-dir",
        str(args.runs_dir),
    ]
    os.execve(
        str(python),
        [str(python), "-E", "-s", str(Path(__file__).resolve()), *argv],
        env,
    )


def check_versions():
    expected = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            expected[name] = version
    actual = {name: importlib.metadata.version(name) for name in expected}
    wrong = {
        name: actual[name]
        for name in expected
        if actual[name].split("+")[0] != expected[name]
    }
    if wrong:
        raise RuntimeError(f"Pinned dependencies differ: {wrong}")
    return actual


def resolved_config(cfg, args, run_dir):
    cfg = copy.deepcopy(cfg)
    # YAML paths are relative to imgnet/; CLI paths are already absolute.
    for section, key in [
        ("data", "data_dir"),
        ("experiment", "init_weights"),
        ("experiment", "resume"),
        ("latent", "decoder_path"),
        ("latent", "stat_path"),
        ("evaluation", "checkpoint"),
    ]:
        value = (cfg.get(section) or {}).get(key)
        if value:
            cfg[section][key] = str((ROOT / Path(value).expanduser()).resolve())
    if args.nproc is not None:
        cfg["runtime"]["nproc_per_node"] = args.nproc
    latent = cfg.get("latent") is not None
    data = args.latent_dir if latent else args.image_dir
    init = args.latent_init if latent else args.pixel_init
    if data is not None:
        cfg["data"]["data_dir"] = str(data)
    if init is not None:
        cfg["experiment"]["init_weights"] = str(init)
    if latent and args.sdvae is not None:
        if cfg["latent"].get("decoder_type", "sdvae") != "sdvae":
            raise ValueError("--sdvae applies only to SD-VAE configurations")
        cfg["latent"]["decoder_path"] = str(args.sdvae)
    cfg["experiment"]["save_dir"] = str(run_dir / "training")
    cfg.setdefault("evaluation", {})["save_dir"] = str(run_dir / "evaluation")
    for rep in cfg["model"].get("params", {}).get("representations", []):
        if rep.get("ckpt_path"):
            rep["ckpt_path"] = str((ROOT / Path(rep["ckpt_path"]).expanduser()).resolve())
    return cfg


def required_assets(cfg, training, evaluating):
    assets = []
    if training:
        from builders.common import (
            public_model_params,
            representation_options,
            training_options,
        )
        from networks.representors.utils import pretrained_path

        params = public_model_params(cfg)
        representation_options(params, cfg["model"]["name"])
        training_options(params)
        assets.append(("training data", Path(cfg["data"]["data_dir"]), False))
        exp = cfg["experiment"]
        selected = exp.get("resume") or exp.get("init_weights")
        if not selected:
            raise ValueError(
                "Paper reproduction requires initialization or resume weights"
            )
        assets.append(("resume/initialization", Path(selected), True))
        for rep in params["representations"]:
            if rep["arch"] != "pixel":
                path = rep.get("ckpt_path") or pretrained_path(
                    *REPRESENTATION_WEIGHTS[rep["arch"]]
                )
                assets.append((rep["arch"], Path(path), True))
    if cfg.get("latent") is not None:
        assets.append(("decoder", Path(cfg["latent"]["decoder_path"]), True))
        if cfg["latent"].get("decoder_type") == "vavae":
            assets.append(("VA-VAE statistics", Path(cfg["latent"]["stat_path"]), True))
    if evaluating:
        from metrics import eval_all_fds
        from metrics.eval import DEFAULT_EVAL_MODELS

        models = cfg.get("evaluation", {}).get("models", DEFAULT_EVAL_MODELS)
        if "inception" in models:
            from metrics.utils.perception_util import _inception_ckpt_path

            assets.append(("Inception weights", _inception_ckpt_path(), True))
            found = [path for _, path in eval_all_fds.INCEPTION_STATS if path.is_file()]
            if not found:
                raise FileNotFoundError(
                    "Prepare FID reference statistics under DM_DATA_ZOO/fid_stats or DM_FID_STATS_DIR"
                )
            assets.extend(("FID reference", path, True) for path in found)
    missing = [
        f"{label}: {path}"
        for label, path, _ in assets
        if "/path/to/" in str(path) or not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing assets (see ASSETS.md):\n" + "\n".join(missing)
        )
    return assets


def main(argv=None):
    args = arguments(argv)
    if args.dry_run:
        print(
            "No installation, writes or training in --dry-run. Asset compatibility is not verified."
        )
        print(
            f"Environment: {args.env_dir}\nStage: {args.stage}\nRuns: {args.runs_dir}"
        )
        if not args.skip_install:
            print(
                shlex.join(
                    [
                        str(args.env_dir / "bin/python"),
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(ROOT / "requirements.txt"),
                    ]
                )
            )
        for config in args.config:
            run = args.runs_dir / config.stem
            print(f"Config: {config}")
            if args.stage in ("all", "train"):
                print(
                    shlex.join(["bash", "scripts/train.sh", str(run / "config.yaml")])
                )
            if args.stage in ("all", "eval"):
                print(shlex.join(["bash", "scripts/eval.sh", str(run / "config.yaml")]))
        return 0
    if args.stage != "setup" and args.model_zoo is None:
        raise ValueError("Set DM_MODEL_ZOO or pass --model-zoo")
    if args.stage in ("all", "check", "eval") and args.data_zoo is None:
        raise ValueError("Set DM_DATA_ZOO or pass --data-zoo")
    environment(args)
    versions = check_versions()
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    env = process_environment()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    for name, value in [
        ("DM_MODEL_ZOO", args.model_zoo),
        ("DM_DATA_ZOO", args.data_zoo),
    ]:
        if value is not None:
            env[name] = str(value)
            os.environ[name] = env[name]
    report = {
        "status": "preparing",
        "versions": versions,
        "source": {},
        "experiments": [],
        "paper_metrics_verified": False,
        "dataset_content_hashed": False,
        "resume_limitation": "RNG state and data cursor are not restored by training checkpoints",
    }
    for directory in ("builders", "methods", "networks", "metrics", "scripts"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".sh"):
                report["source"][str(path.relative_to(ROOT))] = sha256(path)
    for name in ("main.py", "config.py", "reproduce.py", "requirements.txt"):
        report["source"][name] = sha256(ROOT / name)
    manifest = args.runs_dir / "reproduction.json"
    if manifest.exists():
        manifest = args.runs_dir / (
            "reproduction-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json"
        )
    write_json(manifest, report)
    report["environment_log"] = str(
        command(
            [sys.executable, "-m", "pip", "freeze"],
            args.runs_dir / "environment.txt",
            env,
            timeout=60,
        )
    )
    if args.stage == "setup":
        report["status"] = "environment_ready"
        write_json(manifest, report)
        return 0
    try:
        import torch
        import yaml
        from config import load_config, model_name
        from metrics.eval import _latest_checkpoint, _build_eval_args

        if args.stage in ("all", "train", "eval"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Training/evaluation requires CUDA; --stage check can inspect assets on CPU"
                )
            report["cuda"] = torch.version.cuda
            report["gpus"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
        hashes = {}
        training = args.stage in ("all", "check", "train")
        evaluating = args.stage in ("all", "check", "eval")
        prepared = []
        for original in args.config:
            run = args.runs_dir / original.stem
            run.mkdir(parents=True, exist_ok=True)
            config_path = run / "config.yaml"
            reuse = (args.resume or args.stage == "eval") and config_path.is_file()
            cfg = load_config(config_path if reuse else original)
            name = model_name(cfg)
            if name != "nextgan":
                raise ValueError(
                    "This entry point targets model.name nextgan"
                )
            cfg = resolved_config(cfg, args, run)
            if (
                args.stage in ("all", "train")
                and cfg["runtime"]["nproc_per_node"] > torch.cuda.device_count()
            ):
                raise ValueError(
                    "runtime.nproc_per_node exceeds visible GPUs; set --nproc for this machine"
                )
            save_dir = Path(cfg["experiment"]["save_dir"])
            if args.resume:
                latest = _latest_checkpoint(str(save_dir), name)
                if not latest:
                    raise FileNotFoundError(f"No checkpoint to resume under {save_dir}")
                cfg["experiment"]["resume"] = latest
            if (
                training
                and args.stage != "check"
                and save_dir.exists()
                and any(save_dir.iterdir())
                and not cfg["experiment"].get("resume")
            ):
                raise FileExistsError(
                    f"Existing training output: {save_dir}; use --resume or a new --runs-dir"
                )
            if args.stage == "eval":
                latest = _latest_checkpoint(str(save_dir), name)
                checkpoint = latest or cfg.get("evaluation", {}).get("checkpoint")
                if not checkpoint or not Path(checkpoint).is_file():
                    raise FileNotFoundError(
                        f"No evaluation checkpoint for {original.name}; select the previous --runs-dir"
                    )
                cfg["evaluation"]["checkpoint"] = str(
                    Path(checkpoint).resolve()
                )
            assets = required_assets(cfg, training, evaluating)
            artifact = {
                "config_source": str(original),
                "resolved_config": cfg,
                "model": name,
                "status": "checked",
                "assets": [],
            }
            for label, path, hash_contents in assets:
                entry = {"label": label, "path": str(path), "files": {}}
                files = (
                    [path]
                    if path.is_file()
                    else sorted(p for p in path.rglob("*") if p.is_file())
                    if hash_contents
                    else []
                )
                for file in files:
                    key = str(file.resolve())
                    if key not in hashes:
                        hashes[key] = sha256(file)
                    entry["files"][key] = hashes[key]
                artifact["assets"].append(entry)
            report["experiments"].append(artifact)
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            artifact["resolved_config_sha256"] = sha256(config_path)
            write_json(manifest, report)
            prepared.append((cfg, run, config_path, artifact))
        # Check every selected experiment before starting any expensive training.
        for cfg, run, config_path, artifact in prepared:
            name = cfg["model"]["name"]
            save_dir = Path(cfg["experiment"]["save_dir"])
            if args.stage in ("all", "train"):
                artifact["status"] = "training"
                write_json(manifest, report)
                artifact["train_log"] = str(
                    command(
                        ["bash", "scripts/train.sh", config_path],
                        run / "train.log",
                        env,
                    )
                )
                checkpoint = _latest_checkpoint(str(save_dir), name)
                if not checkpoint:
                    raise FileNotFoundError(
                        f"Training wrote no numeric checkpoint under {save_dir}"
                    )
                cfg["evaluation"]["checkpoint"] = checkpoint
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                artifact["resolved_config_sha256"] = sha256(config_path)
                artifact["checkpoint"] = checkpoint
                artifact["checkpoint_sha256"] = sha256(checkpoint)
            if args.stage in ("all", "eval"):
                evaluation_dir = Path(cfg["evaluation"]["save_dir"])
                if evaluation_dir.exists():
                    # The evaluator clears its output directory before running.
                    cfg["evaluation"]["save_dir"] = str(
                        evaluation_dir.with_name(
                            evaluation_dir.name
                            + "-"
                            + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                        )
                    )
                    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    artifact["resolved_config_sha256"] = sha256(config_path)
                artifact["status"] = "evaluating"
                write_json(manifest, report)
                checkpoint = Path(cfg["evaluation"]["checkpoint"])
                artifact["checkpoint"] = str(checkpoint)
                if artifact.get("checkpoint_sha256") is None:
                    artifact["checkpoint_sha256"] = sha256(checkpoint)
                artifact["eval_log"] = str(
                    command(
                        ["bash", "scripts/eval.sh", config_path], run / "eval.log", env
                    )
                )
                summary = (
                    Path(_build_eval_args(str(config_path), cfg).output_dir)
                    / "final_eval_summary.csv"
                )
                if not summary.is_file():
                    raise FileNotFoundError(
                        "Evaluator returned without a metric summary"
                    )
                with summary.open() as handle:
                    rows = list(csv.DictReader(handle))
                finite = []
                for row in rows:
                    try:
                        finite.append(math.isfinite(float(row.get("fd", "nan"))))
                    except (TypeError, ValueError):
                        finite.append(False)
                if not finite or not all(finite):
                    raise ValueError(
                        "Evaluator returned without a finite FD metric for every row"
                    )
                artifact["metric_summary"] = str(summary)
                artifact["metric_summary_sha256"] = sha256(summary)
                artifact["metrics"] = rows
            artifact["status"] = "checked" if args.stage == "check" else "completed"
            write_json(manifest, report)
        report["status"] = "checked" if args.stage == "check" else "completed"
    except BaseException as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(manifest, report)
        raise
    write_json(manifest, report)
    print(f"Reproduction record: {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        subprocess.SubprocessError,
    ) as error:
        print(f"Reproduction failed: {error}", file=sys.stderr)
        raise SystemExit(1)
