"""YAML loading with explicit environment-variable substitution."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


_ENV_TOKEN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def _expand_string(value: str):
    names = [match.group(1) or match.group(2) for match in _ENV_TOKEN.finditer(value)]
    missing = sorted({name for name in names if name not in os.environ})
    if missing:
        raise KeyError(
            "Config references unset environment variable(s): " + ", ".join(missing)
        )
    expanded = os.path.expandvars(value)
    if _ENV_TOKEN.fullmatch(value):
        # Preserve numeric/bool/list types for fields such as nproc_per_node.
        return yaml.safe_load(expanded)
    return expanded


def expand_config_env(obj):
    if isinstance(obj, dict):
        return {key: expand_config_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_config_env(value) for value in obj]
    if isinstance(obj, str):
        return _expand_string(obj)
    return obj


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return expand_config_env(raw)


def model_name(cfg):
    """Validate and return the single public algorithm selector."""
    if "method" in cfg:
        raise ValueError("Legacy method configuration is removed; use model.name")
    name = cfg["model"]["name"]
    if name not in ("tbsm", "logistic_gan", "nextgan"):
        raise ValueError(
            f"Unknown model.name={name!r}; expected tbsm, logistic_gan, nextgan"
        )
    return name
