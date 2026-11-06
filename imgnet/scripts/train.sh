#!/bin/bash
set -euo pipefail
: "${DM_MODEL_ZOO:?Set DM_MODEL_ZOO to the model-cache directory before training}"
export TORCH_HOME="${TORCH_HOME:-$DM_MODEL_ZOO}"
export HF_HOME="${HF_HOME:-$DM_MODEL_ZOO}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$DM_MODEL_ZOO/hub}"
export TIMM_HOME="${TIMM_HOME:-$DM_MODEL_ZOO/hub/checkpoints}"
if [ "$#" -ne 1 ]; then
    echo "Usage: bash scripts/train.sh configs/xxx.yaml"
    exit 1
fi
config="$1"
if [ ! -f "$config" ]; then
    echo "Config not found: $config"
    exit 1
fi
launch_settings="$(python3 - "$config" <<'PY'
import sys
from config import load_config

cfg = load_config(sys.argv[1])
runtime = cfg["runtime"]
nproc = runtime["nproc_per_node"]
deterministic = runtime["deterministic"]
if type(nproc) is not int or nproc < 1:
    raise ValueError("runtime.nproc_per_node must be a positive integer")
if not isinstance(deterministic, bool):
    raise TypeError("runtime.deterministic must be a bool")
# Build a cold ImageFolder index before workers start waiting at NCCL barriers.
if nproc > 1 and cfg["data"]["name"] == "in1k":
    from builders.datasets import build_dataset
    dataset, _ = build_dataset(cfg)
    print(f"ImageNet index ready: {len(dataset)} images", file=sys.stderr)
print(nproc, str(deterministic).lower())
PY
)"
read -r nproc_per_node deterministic <<< "$launch_settings"
if [ "$deterministic" = "true" ]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
if [ "$nproc_per_node" -eq 1 ]; then
    # This standalone process must not join the scheduler's process group.
    unset RANK WORLD_SIZE LOCAL_RANK MASTER_ADDR MASTER_PORT
    python3 main.py --config "$config"
else
    # Scheduler-neutral single-node launch. A multi-node environment should
    # invoke main.py through its own torchrun/rendezvous wrapper instead.
    # Failing fast keeps a crashed worker group from silently restarting.
    torchrun --standalone --nproc_per_node="$nproc_per_node" --max-restarts=0 \
        main.py --config "$config"
fi
