: "${DM_MODEL_ZOO:?Set DM_MODEL_ZOO to the model-cache directory before evaluation}"
# metrics/eval_all_fds.py reads DM_DATA_ZOO (and DM_FID_STATS_DIR, which
# defaults to $DM_DATA_ZOO/fid_stats) to locate the reference FID stats
# (.npz files). Without this, it silently falls back to the placeholder
# "/path/to/your/data_zoo", which either skips FID(JiT) or crashes with
# FileNotFoundError for the other representation-space FIDs (mae/dinov2/
# clip/siglip2/convnext) -- the eval otherwise "succeeds" with no fd values.
: "${DM_DATA_ZOO:?Set DM_DATA_ZOO to the directory containing FID reference stats}"
export TORCH_HOME="${TORCH_HOME:-$DM_MODEL_ZOO}"
export HF_HOME="${HF_HOME:-$DM_MODEL_ZOO}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$DM_MODEL_ZOO/hub}"
export TIMM_HOME="${TIMM_HOME:-$DM_MODEL_ZOO/hub/checkpoints}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800

set -euo pipefail

# Scheduler-neutral single-node evaluation. Multi-node environments should
# supply their own torchrun/rendezvous wrapper around `python -m metrics.eval`.
torchrun --standalone --nproc_per_node="${EVAL_NPROC_PER_NODE:-1}" --max-restarts=0 -m metrics.eval "$@"
