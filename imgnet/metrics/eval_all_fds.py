"""Distributed metrics for the selected one-step ImageNet generator."""

import csv
import datetime
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from metrics.frechet_distance.metrics import (
    compute_fid as np_fid,
    compute_isc as np_isc,
    compute_mmd,
    compute_precision_recall,
)
from metrics.frechet_distance.repr_models import load_repr_model, model_short_name
from metrics.utils.distributed_util import (
    broadcast_scalar, get_global_rank, get_world_size, is_enabled,
)
from metrics.frechet_distance.evaluator import extract_ref_features, gather_features

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

DATA_ZOO = Path(os.environ.get("DM_DATA_ZOO", "/path/to/your/data_zoo"))
FID_STATS_DIR = Path(os.environ.get("DM_FID_STATS_DIR", DATA_ZOO / "fid_stats"))

INCEPTION_STATS = [
    ("FID(JiT)",     FID_STATS_DIR / "jit_in256_stats.npz"),
    ("FID(ADM)",     FID_STATS_DIR / "guided_diffusion_stats.npz"),
]

TARGET_SIZE = 256

def get_start_end_indices(total_samples, num_processes, rank):
    base, remainder = divmod(total_samples, num_processes)
    start_idx = rank * base + min(rank, remainder)
    return start_idx, start_idx + base + (rank < remainder)

# Per-model target size overrides (model name -> target size).
# Models not listed here use the global TARGET_SIZE.
DEFAULT_TARGET_SIZES = {
    "vit_large_patch16_224.mae": 224,
    "vit_so400m_patch16_siglip_256.v2_webli": 224,
}

# Validation-set raw FD values used to convert raw FD to FDr.
# FDr-X = rawFD-X / valFD-X. FDr-6 is the arithmetic mean over these six
# representation spaces.
FDR_VALIDATION_FD = {
    "FID(ADM)": 1.68,
    "convnext": 56.87,
    "dinov2_cls": 14.19,
    "mae_cls": 0.04,
    "siglip_cls": 0.60,
    "clip_cls": 5.60,
}

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# Core: accumulate features, reduce, compute FD
# ---------------------------------------------------------------------------

def _make_accumulators(repr_models, device):
    """Create accumulators, sharing them for entries with the same (model, pool_type)."""
    accumulators = []
    acc_map = {}  # (model_id, pool_type) -> accumulator dict
    for repr_entry in repr_models:
        key = (id(repr_entry["model"]), repr_entry.get("pool_type", "cls"))
        if key in acc_map:
            accumulators.append(acc_map[key])  # shared reference
        else:
            dim = repr_entry["feat_dim"]
            acc = {
                "feat_sum": torch.zeros(dim, dtype=torch.float64, device=device),
                "feat_outer": torch.zeros(dim, dim, dtype=torch.float64, device=device),
                "count": 0,
            }
            acc_map[key] = acc
            accumulators.append(acc)
    return accumulators


@torch.inference_mode()
def accumulate_batch(images, repr_models, accumulators, inception_logits,
                      local_prc_feats):
    """Extract features from one batch and accumulate into sufficient stats.

    images must be in [0, 1] range (model-ready).  Caches forward passes per
    model object so that cls/avg entries sharing the same model don't trigger
    redundant computation.  Accumulators are shared for entries with the same
    (model, pool_type), so updates happen only once.
    """
    fwd_cache = {}  # id(model) -> (output_0, output_1)
    updated = set()  # accumulator ids already updated this batch

    for i, repr_entry in enumerate(repr_models):
        model = repr_entry["model"]
        model_id = id(model)
        if model_id not in fwd_cache:
            if repr_entry["has_logits"]:
                fwd_cache[model_id] = model(images)
            else:
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    fwd_cache[model_id] = model(images)

        out_0, out_1 = fwd_cache[model_id]

        if repr_entry["has_logits"]:
            feats = out_0
        else:
            pool_type = repr_entry.get("pool_type", "cls")
            feats = out_0 if pool_type == "cls" else out_1

        if repr_entry["name"] in local_prc_feats:
            local_prc_feats[repr_entry["name"]].append(feats.float().cpu())

        # Only accumulate once per shared accumulator
        acc_id = id(accumulators[i])
        if acc_id not in updated:
            feats64 = feats.double()
            accumulators[i]["feat_sum"].add_(feats64.sum(0))
            accumulators[i]["feat_outer"].addmm_(feats64.T, feats64)
            accumulators[i]["count"] += feats.shape[0]
            updated.add(acc_id)

    # Collect inception logits once per batch (all inception entries share the model)
    for repr_entry in repr_models:
        if repr_entry["has_logits"]:
            _, logits = fwd_cache[id(repr_entry["model"])]
            inception_logits.append(logits.cpu())
            break


def _reduce_and_compute(repr_models, accumulators, inception_logits,
                        local_feat_lists, *,
                        prc_ref_features, prc_model_names,
                        mmd_ref_features, mmd_model_names,
                        prc_k=3, prc_batch_size=5000):
    """Reduce across ranks, compute FD / IS / P&R / CMMD.

    ``local_feat_lists`` collects per-sample features for the union of
    P&R and CMMD models.  Each metric uses its own reference features.
    """
    rank = get_global_rank()
    device = torch.device("cuda")
    distributed = is_enabled()
    prc_model_names = set(prc_model_names)
    mmd_model_names = set(mmd_model_names)
    feat_model_names = prc_model_names | mmd_model_names

    # Reduce each unique accumulator once, then compute FD per entry
    reduced = {}  # id(acc) -> (mu, sigma) or None (non-rank-0)
    results = {}
    for i, repr_entry in enumerate(repr_models):
        acc = accumulators[i]
        acc_id = id(acc)

        if acc_id not in reduced:
            if distributed:
                for key in ("feat_sum", "feat_outer"):
                    dist.reduce(acc[key], dst=0, op=dist.ReduceOp.SUM)
                count_t = torch.tensor([acc["count"]], dtype=torch.long, device=device)
                dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
                total = count_t.item()
            else:
                total = acc["count"]

            if rank == 0:
                feat_sum = acc["feat_sum"].cpu().numpy()
                feat_outer = acc["feat_outer"].cpu().numpy()
                mu = (feat_sum / total).astype(np.float64)
                sigma = ((feat_outer - np.outer(feat_sum, feat_sum) / total)
                         / (total - 1)).astype(np.float64)
                reduced[acc_id] = (mu, sigma)
            else:
                reduced[acc_id] = None

        if rank == 0:
            mu, sigma = reduced[acc_id]
            fd_val = float(np_fid(mu, sigma,
                                  repr_entry["mu_ref"].cpu().numpy(),
                                  repr_entry["sigma_ref"].cpu().numpy()))
        else:
            fd_val = 0.0
        results[repr_entry["name"]] = broadcast_scalar(fd_val)

    # Inception Score
    is_val = None
    if inception_logits:
        local_logits = torch.cat(inception_logits, dim=0).to(device)
        if distributed:
            all_logits = gather_features(local_logits)
            is_val = float(np_isc(all_logits.cpu())[0]) if rank == 0 else 0.0
            is_val = broadcast_scalar(is_val)
        else:
            is_val, _ = np_isc(local_logits.cpu())

    # Per-sample metrics (P&R, CMMD) — require gathered features + reference.
    # NOTE: feat_model_names is the same on all ranks; ref dicts are only
    # populated on rank 0.  Guard on feat_model_names (not ref dicts) to
    # ensure all ranks participate in the collective gather/broadcast.
    prc_results: dict[str, tuple[float, float]] = {}
    mmd_results: dict[str, float] = {}
    for feat_name in sorted(feat_model_names):
        if not local_feat_lists.get(feat_name):
            continue
        local_feats = torch.cat(local_feat_lists.pop(feat_name), dim=0).to(device)
        all_gen = (gather_features(local_feats)
                   if distributed else local_feats)
        del local_feats

        do_prc = feat_name in prc_model_names
        do_mmd = feat_name in mmd_model_names
        prec = rec = mmd_val = 0.0

        if rank == 0:
            gen_gpu = all_gen.float()
            del all_gen
            metrics_str = []

            if do_prc:
                prc_ref = prc_ref_features[feat_name].to(device=device, dtype=torch.float32)
                logger.info(f"P&R ({feat_name}): ref={prc_ref.shape[0]}, "
                            f"gen={gen_gpu.shape[0]}, k={prc_k}")
                prec, rec = compute_precision_recall(prc_ref, gen_gpu, k=prc_k,
                                                     batch_size=prc_batch_size)
                del prc_ref
                metrics_str += [f"Precision={prec:.4f}", f"Recall={rec:.4f}"]
            if do_mmd:
                mmd_ref = mmd_ref_features[feat_name].to(device=device, dtype=torch.float32)
                mmd_val = compute_mmd(mmd_ref, gen_gpu)
                del mmd_ref
                metrics_str.append(f"CMMD={mmd_val:.4f}")
            del gen_gpu
            logger.info(f"  {feat_name}: {'  '.join(metrics_str)}")
        else:
            del all_gen

        if do_prc:
            prc_results[feat_name] = (
                broadcast_scalar(prec),
                broadcast_scalar(rec),
            )
        if do_mmd:
            mmd_results[feat_name] = broadcast_scalar(mmd_val)
        torch.cuda.empty_cache()

    return results, is_val, prc_results, mmd_results


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def _generate_and_evaluate(args, model, repr_models, *,
                           prc_ref_features, prc_model_names,
                           mmd_ref_features, mmd_model_names):
    """Generate images and compute FD/IS/P&R/CMMD."""
    world_size, rank = get_world_size(), get_global_rank()
    device = torch.device("cuda")
    feat_model_names = sorted(set(prc_model_names) | set(mmd_model_names))

    start_idx, end_idx = get_start_end_indices(args.num_images, world_size, rank)
    local_n = end_idx - start_idx
    rank_classes = np.arange(start_idx, end_idx, dtype=np.int64) % args.num_classes

    accumulators = _make_accumulators(repr_models, device)
    inception_logits: list[torch.Tensor] = []
    local_feat_lists: dict[str, list[torch.Tensor]] = {n: [] for n in feat_model_names}

    generated = 0
    while generated < local_n:
        b = min(args.batch_size, local_n - generated)
        y = torch.from_numpy(rank_classes[generated:generated + b]).long().to(device)
        with torch.autocast("cuda", enabled=args.enable_amp, dtype=args.amp_dtype):
            images = model.generate(y)
            images = ((images + 1) / 2).clamp(0, 1)

        accumulate_batch(images, repr_models, accumulators, inception_logits,
                          local_feat_lists)

        del images
        generated += b

    torch.cuda.empty_cache()

    return _reduce_and_compute(
        repr_models, accumulators, inception_logits, local_feat_lists,
        prc_ref_features=prc_ref_features, prc_model_names=prc_model_names,
        mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_model_names,
        prc_k=args.prc_k, prc_batch_size=args.prc_batch_size,
    )


# ---------------------------------------------------------------------------
# Repr model & P&R setup
# ---------------------------------------------------------------------------

def _load_repr_models(model_names, input_size):
    """Load repr models and reference stats. Returns list of repr-entry dicts."""
    rank = get_global_rank()
    repr_models = []
    if "inception" in model_names and not any(path.is_file() for _, path in INCEPTION_STATS):
        paths = ", ".join(str(path) for _, path in INCEPTION_STATS)
        raise FileNotFoundError(
            f"Inception FID requires at least one reference statistics file: {paths}. "
            "Set DM_DATA_ZOO or DM_FID_STATS_DIR to the prepared statistics directory."
        )
    for name in model_names:
        if rank == 0:
            logger.info(f"Loading repr model '{name}' ...")
        repr_model, feat_dim, has_logits, ts = load_repr_model(
            name, input_size=DEFAULT_TARGET_SIZES.get(name, TARGET_SIZE),
        )

        if name == "inception":
            # Inception uses multiple reference datasets with custom labels
            for label, stats_path in INCEPTION_STATS:
                if not stats_path.is_file():
                    logger.warning(f"  Skipping {label}: {stats_path} not found")
                    continue
                ref = np.load(stats_path)
                repr_models.append({
                    "name": label, "model": repr_model, "feat_dim": feat_dim,
                    "has_logits": has_logits,
                    "mu_ref": torch.tensor(ref["mu"], device="cuda", dtype=torch.float64),
                    "sigma_ref": torch.tensor(ref["sigma"], device="cuda", dtype=torch.float64),
                    "pool_type": "cls",
                })
                logger.info(f"  '{label}': feat_dim={feat_dim}")
        else:
            safe_name = name.replace("/", "_").replace(".", "_")
            stats_input_size = 256 if input_size == 512 else input_size
                
            stats_path = FID_STATS_DIR / f"{safe_name}_in{stats_input_size}_t{ts}_stats.npz"
            short = model_short_name(name)
            ref = np.load(stats_path)

            pools = [("cls", "mu", "sigma")]
            has_avg = "avg_mu" in ref
            if has_avg:
                pools.append(("avg", "avg_mu", "avg_sigma"))

            for pool, mu_key, sig_key in pools:
                label = f"{short}_{pool}" if has_avg else short
                repr_models.append({
                    "name": label, "model": repr_model, "feat_dim": feat_dim,
                    "has_logits": has_logits,
                    "mu_ref": torch.tensor(ref[mu_key], device="cuda", dtype=torch.float64),
                    "sigma_ref": torch.tensor(ref[sig_key], device="cuda", dtype=torch.float64),
                    "pool_type": pool,
                })
                logger.info(f"  '{label}': feat_dim={feat_dim}, pool={pool}")
    return repr_models


def _load_prc_refs(prc_model_names, repr_models, prc_ref_dir):
    """Load P&R reference features using all GPUs.

    For dual-output models (cls + avg), runs one forward pass and caches
    both outputs to avoid redundant computation.
    """
    rank = get_global_rank()
    prc_ref_features, prc_names = {}, []
    # Track which models we've already extracted: model_id -> {"cls": T, "avg": T|None}
    extracted = {}

    for prc_raw in prc_model_names:
        short = model_short_name(prc_raw)
        if prc_raw == "inception":
            inception_labels = {label for label, _ in INCEPTION_STATS}
            match = next((entry for entry in repr_models
                          if entry["name"] in inception_labels), None)
            if match is None:
                logger.warning(f"P&R model '{prc_raw}' not in models; skipping")
                continue
            entries_to_register = [(match, "inception", "cls")]
        else:
            entries_to_register = []
            bare_entry = next((e for e in repr_models if e["name"] == short), None)
            if bare_entry is not None:
                entries_to_register.append((bare_entry, short, "cls"))
            else:
                for pool in ("cls", "avg"):
                    entry = next((e for e in repr_models
                                  if e["name"] == f"{short}_{pool}"), None)
                    if entry is not None:
                        entries_to_register.append((entry, f"{short}_{pool}", pool))
            if not entries_to_register:
                logger.warning(f"P&R model '{prc_raw}' not in models; skipping")
                continue

        repr_model = entries_to_register[0][0]["model"]
        model_id = id(repr_model)

        if model_id not in extracted:
            has_logits = entries_to_register[0][0]["has_logits"]
            feat_dim = entries_to_register[0][0]["feat_dim"]
            has_dual = len(entries_to_register) > 1

            # Check if all individual caches exist — if so, load from cache
            cache_paths = {
                pool: f"{prc_ref_dir.rstrip('/')}_{suffix}.pt"
                for _, suffix, pool in entries_to_register
            }
            ref_available = False
            all_cached = False
            if rank == 0:
                ref_available = os.path.isdir(prc_ref_dir)
                all_cached = ref_available and all(
                    os.path.exists(cp) for cp in cache_paths.values()
                )
            if is_enabled():
                # Shared caches may be published while a late rank joins.
                # All ranks must follow rank 0 into extraction or the cache path.
                cache_status = torch.tensor(
                    [ref_available, all_cached], dtype=torch.long,
                    device=entries_to_register[0][0]["mu_ref"].device,
                )
                dist.broadcast(cache_status, src=0)
                ref_available, all_cached = cache_status.tolist()
            if not ref_available:
                logger.warning(f"No P&R ref dir for {short}; skipping")
                continue

            if all_cached and rank == 0:
                result = {}
                for pool, cp in cache_paths.items():
                    result[pool] = torch.load(cp, map_location="cpu", weights_only=True)
                    logger.info(f"Loaded cached P&R ref from {cp} ({result[pool].shape})")
                extracted[model_id] = result
            elif all_cached:
                extracted[model_id] = {pool: None for pool in cache_paths}
            else:
                # One forward pass for both cls and avg
                def both_fn(x, m=repr_model, logits=has_logits, dual=has_dual):
                    if logits:
                        return m(x)[0]
                    with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                        out0, out1 = m(x)
                    if dual and out1 is not None:
                        return torch.cat([out0, out1], dim=1)
                    return out0

                all_feats = extract_ref_features(both_fn, prc_ref_dir)

                if rank == 0 and all_feats.numel() > 0:
                    if has_dual and all_feats.shape[1] > feat_dim:
                        cls_feats = all_feats[:, :feat_dim]
                        avg_feats = all_feats[:, feat_dim:]
                    else:
                        cls_feats = all_feats
                        avg_feats = None
                    extracted[model_id] = {"cls": cls_feats, "avg": avg_feats}
                    # Cache individually
                    for _, suffix, pool in entries_to_register:
                        f = extracted[model_id].get(pool)
                        if f is not None:
                            cp = cache_paths[pool]
                            os.makedirs(os.path.dirname(cp) or ".", exist_ok=True)
                            torch.save(f, cp)
                            logger.info(f"Cached P&R ref ({suffix}): {f.shape} -> {cp}")
                else:
                    extracted[model_id] = {"cls": None, "avg": None}

        for match, _, pool in entries_to_register:
            prc_name = match["name"]
            prc_names.append(prc_name)
            if rank == 0 and extracted.get(model_id, {}).get(pool) is not None:
                prc_ref_features[prc_name] = extracted[model_id][pool]

    return prc_ref_features, prc_names


# ---------------------------------------------------------------------------
# CSV & summary output
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "model", "fd", "fdr", "fdr6", "is", "precision", "recall", "cmmd", "n",
    "ema_label", "cfg", "step", "interval_min", "interval_max",
    "num_sampling_steps", "timestamp",
]


def _append_csv(csv_path, rows):
    """Append list of dicts to CSV."""
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _compute_fdr6(results):
    vals = []
    for name in FDR_VALIDATION_FD:
        if name not in results:
            return None
        vals.append(results[name] / FDR_VALIDATION_FD[name])
    return float(np.mean(vals))


def _build_csv_rows(results, is_val, prc_results, num_images, context,
                    mmd_results):
    """Build one CSV row per representation, sharing Inception's IS/P&R/CMMD."""
    def _round(v, n):
        return round(v, n) if v is not None else None

    inception_labels = {label for label, _ in INCEPTION_STATS}
    inception_prc = next(((p, r) for name, (p, r) in prc_results.items()
                          if name in inception_labels), (None, None))
    inception_mmd = next((v for name, v in mmd_results.items()
                          if name in inception_labels), None)
    fdr6 = _compute_fdr6(results)
    rows = []
    for rname, fd_val in results.items():
        is_inception = rname in inception_labels
        prec, rec = inception_prc if is_inception else prc_results.get(rname, (None, None))
        mmd_val = inception_mmd if is_inception else mmd_results.get(rname, None)
        fdr = fd_val / FDR_VALIDATION_FD[rname] if rname in FDR_VALIDATION_FD else None
        rows.append({
            "model": rname, "fd": _round(fd_val, 6),
            "fdr": _round(fdr, 6), "fdr6": _round(fdr6, 6),
            "n": num_images,
            "is": _round(is_val if is_inception else None, 4),
            "precision": _round(prec, 6), "recall": _round(rec, 6),
            "cmmd": _round(mmd_val, 6),
            **context,
        })
    return rows


def _format_table_value(value):
    return value if value not in (None, "") else "-"


SUMMARY_METRIC_COLUMNS = [
    ("FID(JiT)", "FID(JiT)", "fd"),
    ("FID(ADM)", "FID(ADM)", "fd"),
    ("IS", "FID(JiT)", "is"),
    ("FDr6", "FID(JiT)", "fdr6"),
    ("MAE(cls)", "mae_cls", "fd"),
    ("MAE(avg)", "mae_avg", "fd"),
    ("DINOv2(cls)", "dinov2_cls", "fd"),
    ("DINOv2(avg)", "dinov2_avg", "fd"),
    ("CLIP(cls)", "clip_cls", "fd"),
    ("CLIP(avg)", "clip_avg", "fd"),
    ("SigLIP(cls)", "siglip_cls", "fd"),
    ("SigLIP(avg)", "siglip_avg", "fd"),
    ("ConvNeXt", "convnext", "fd"),
]


def _print_summary_table(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    groups = {}
    for row in rows:
        key = (row["step"], row["cfg"], row["ema_label"], row["n"])
        groups.setdefault(key, {})[row["model"]] = row

    table_rows = []
    for (step, cfg, ema, n), by_model in groups.items():
        out = {"step": step, "cfg": cfg, "ema": ema, "n": n}
        for col, model_name, field in SUMMARY_METRIC_COLUMNS:
            out[col] = by_model.get(model_name, {}).get(field, "")
        table_rows.append(out)

    columns = ["step", "cfg", "ema", "n"] + [c[0] for c in SUMMARY_METRIC_COLUMNS]
    widths = {
        col: max(len(col), *(len(_format_table_value(row.get(col))) for row in table_rows))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in table_rows:
        print("  ".join(_format_table_value(row.get(col)).ljust(widths[col])
                        for col in columns))


def main_generate(args, model):
    """Evaluate the already selected online or EMA generator once."""
    rank = get_global_rank()
    repr_models = _load_repr_models(args.models, args.input_size)
    prc_models = args.prc_models or args.models

    prc_ref_features, prc_names = {}, []
    if not args.no_prc:
        prc_ref_features, prc_names = _load_prc_refs(
            prc_models, repr_models, args.prc_ref_dir)

    mmd_ref_features, mmd_names = {}, []
    if args.eval_mmd:
        mmd_ref_features, mmd_names = _load_prc_refs(
            prc_models, repr_models, args.mmd_ref_dir)

    csv_path = os.path.join(args.output_dir, "final_eval_summary.csv")
    results, is_val, prc_results, mmd_results = _generate_and_evaluate(
        args, model, repr_models,
        prc_ref_features=prc_ref_features, prc_model_names=prc_names,
        mmd_ref_features=mmd_ref_features, mmd_model_names=mmd_names,
    )
    # Preserve the CSV contract for the fixed one-step, unguided generator.
    context = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
        "step": args.current_step, "ema_label": args.weights,
        "cfg": 1.0, "interval_min": 0.0, "interval_max": 1.0,
        "num_sampling_steps": 1,
    }
    csv_rows = _build_csv_rows(results, is_val, prc_results,
                               args.num_images, context, mmd_results)
    if rank == 0:
        _append_csv(csv_path, csv_rows)
        _print_summary_table(csv_path)
    if is_enabled():
        dist.destroy_process_group()
