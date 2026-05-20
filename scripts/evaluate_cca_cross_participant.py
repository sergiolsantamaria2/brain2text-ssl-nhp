#!/usr/bin/env python3
"""
Phase B2 — Cross-participant CCA: T12 ↔ T15.

Tests whether the cross-session stability shown in B1 (within T15) also holds
across participants. For each encoder condition we have a *pair* of models
(one trained on T12, one on T15) sharing the transformer + final_norm weights;
only the day-specific read-ins differ. We then compute CCA between every T12
val session and every T15 val session.

Conditions:
  - random_init : paired random-init decoders, identical shared weights, only
                  day-specific read-ins differ (n_days=24 vs 45).
  - ft_no_ssl   : best TFS-no-SSL ckpts (T15: from B1; T12: Phase A control).
  - ft_ssl      : best AR-binary-soma SSL ckpts (T15 seed10 from B1; T12
                  Phase A SSL seed10).
  - ft_ssl_shuf : ft_ssl with the T15 member temporally permuted.
  - raw_input   : NO encoder. PCA(10) directly on the 512-dim raw features
                  (TX+SBP). Control for B1 too — tells us how much of the CCA
                  signal is already in the input.

This script ALSO regenerates B1 outputs to include the raw_input within-T15
condition so the published B1 table is consistent with B2.

Outputs:
  ${OUTPUT_DIR}/analysis/cca_phase_b2/
    embeddings/{encoder}/{participant}_{session_id}.npy
    results/cca_pairs_b2.csv, summary_b2.csv
    figures/cca_distribution_b2.png, cca_decay_b2.png, cca_combined_b1_b2.png
  ${OUTPUT_DIR}/analysis/cca_phase_b1/results/cca_pairs.csv (rewritten with raw_input)
  ${OUTPUT_DIR}/analysis/cca_phase_b1/figures/cca_*.png (regenerated)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from brain2text.evaluation.cca import (  # noqa: E402
    DEFAULT_ARCH,
    all_pairs_cca,
    build_decoder,
    build_paired_random_decoders,
    cross_dataset_pairs_cca,
    extract_raw_session_features,
    extract_session_embeddings,
    load_finetuned_decoder,
)
from brain2text.data.dataset import (  # noqa: E402
    BrainToTextDataset,
    train_test_split_indicies,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cca_phase_b2")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # T15
    p.add_argument("--t15-config", default="configs/baselines/gru_defaults.yaml")
    p.add_argument("--t15-ckpt-no-ssl",
                   default="${OUTPUT_DIR}/ft_nosssl/transformer_no_ssl_hd512/checkpoint/best_checkpoint")
    p.add_argument("--t15-ckpt-ssl",
                   default="${OUTPUT_DIR}/ssl_study_ft250k/ft250k_ar_binary_soma_epoch400/checkpoint/best_checkpoint")
    # T12
    p.add_argument("--t12-config", default="configs/experiments/06_phase_a_t12/dataset_t12.yaml",
                   help="Config providing dataset.dataset_dir + dataset.sessions for T12. Layered on top of t15-config for non-dataset keys.")
    p.add_argument("--t12-ckpt-no-ssl",
                   default="${OUTPUT_DIR}/phase_a_t12/ft_t12_no_ssl_seed10/checkpoint/best_checkpoint")
    p.add_argument("--t12-ckpt-ssl",
                   default="${OUTPUT_DIR}/phase_a_t12/ft_t12_ssl_ar_binary_soma_seed10/checkpoint/best_checkpoint")
    # Outputs
    p.add_argument("--out-root-b2", default="${OUTPUT_DIR}/analysis/cca_phase_b2")
    p.add_argument("--out-root-b1", default="${OUTPUT_DIR}/analysis/cca_phase_b1")
    # CCA params (kept identical to B1 for comparability)
    p.add_argument("--m-pca", type=int, default=10)
    p.add_argument("--k-top", type=int, default=4)
    p.add_argument("--k-decay", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--random-init-seed", type=int, default=42,
                   help="Shared seed for the paired random-init decoders (T12 and T15).")
    p.add_argument("--reuse-embeddings", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Use only 2 sessions per dataset to verify the pipeline end-to-end.")
    p.add_argument("--skip-b1-update", action="store_true",
                   help="Don't regenerate B1 outputs with raw_input.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def build_val_loader(cfg, batch_size: int, dry_run: bool = False):
    """Build a val DataLoader for whichever dataset cfg points to.
    Mirrors rnn_trainer's filter (keep sessions with non-empty data_train.hdf5)
    so n_days matches what the checkpoint was trained with.
    """
    dataset_dir = cfg["dataset"]["dataset_dir"]
    sessions = list(cfg["dataset"]["sessions"])
    if dry_run:
        sessions = sessions[:2]

    import h5py
    keep_sessions: List[str] = []
    for s in sessions:
        train_fp = os.path.join(dataset_dir, s, "data_train.hdf5")
        if not os.path.exists(train_fp):
            logger.warning(f"  skip (no train file): {s}")
            continue
        try:
            with h5py.File(train_fp, "r") as f:
                if len(f.keys()) == 0:
                    logger.warning(f"  skip (empty train): {s}")
                    continue
        except Exception as e:
            logger.warning(f"  skip (train read error {e}): {s}")
            continue
        keep_sessions.append(s)

    val_paths = [os.path.join(dataset_dir, s, "data_val.hdf5") for s in keep_sessions]
    sessions_with_val = sum(1 for p in val_paths if os.path.exists(p))
    logger.info(f"  n_days={len(keep_sessions)}  with_val={sessions_with_val}  (dataset_dir={dataset_dir})")

    _, val_trials = train_test_split_indicies(
        file_paths=val_paths,
        test_percentage=1,
        seed=int(cfg["dataset"].get("seed", 10)),
        bad_trials_dict=None,
    )

    feature_subset = cfg["dataset"].get("feature_subset", None)
    val_dataset = BrainToTextDataset(
        trial_indicies=val_trials,
        split="test",
        days_per_batch=None,
        n_batches=None,
        batch_size=batch_size,
        random_seed=int(cfg["dataset"].get("seed", 10)),
        feature_subset=feature_subset,
    )
    loader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0, pin_memory=True)
    day_to_session = {i: s for i, s in enumerate(keep_sessions)}
    return loader, day_to_session, len(keep_sessions)


def merge_t12_config(base_cfg, t12_cfg_path: str):
    """t12_cfg_path only overrides dataset.dataset_dir + dataset.sessions."""
    t12 = OmegaConf.load(t12_cfg_path)
    merged = OmegaConf.merge(base_cfg, t12)
    return merged


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def emb_dir(out_root: Path, encoder: str) -> Path:
    d = out_root / "embeddings" / encoder
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_embeddings(out_root: Path, encoder: str, participant: str,
                    embeddings: Dict[int, np.ndarray],
                    day_to_session: Dict[int, str]) -> None:
    d = emb_dir(out_root, encoder)
    for day_idx, Z in embeddings.items():
        sid = day_to_session.get(day_idx, f"day{day_idx:03d}")
        np.save(d / f"{participant}_{sid}.npy", Z.astype(np.float32, copy=False))


# ---------------------------------------------------------------------------
# Per-encoder driver
# ---------------------------------------------------------------------------

def run_encoder_extraction(name: str, factory_t12: Callable, factory_t15: Callable,
                           loader_t12, loader_t15,
                           day_to_session_t12, day_to_session_t15,
                           device, out_root: Path, reuse: bool) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    cache = emb_dir(out_root, name)

    def _try_load(participant: str, day_to_session: Dict[int, str]) -> Optional[Dict[int, np.ndarray]]:
        files = sorted(cache.glob(f"{participant}_*.npy"))
        if not files:
            return None
        sid_to_day = {v: k for k, v in day_to_session.items()}
        out: Dict[int, np.ndarray] = {}
        for f in files:
            sid = f.stem[len(participant) + 1:]
            if sid not in sid_to_day:
                continue
            out[sid_to_day[sid]] = np.load(f)
        return out or None

    if reuse:
        emb_a = _try_load("t12", day_to_session_t12)
        emb_b = _try_load("t15", day_to_session_t15)
        if emb_a and emb_b:
            logger.info(f"[{name}] reusing cached embeddings (t12={len(emb_a)}, t15={len(emb_b)})")
            return emb_a, emb_b

    logger.info(f"[{name}] building T12 model + extracting")
    model = factory_t12()
    emb_t12 = extract_session_embeddings(model, loader_t12, device)
    save_embeddings(out_root, name, "t12", emb_t12, day_to_session_t12)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[{name}] building T15 model + extracting")
    model = factory_t15()
    emb_t15 = extract_session_embeddings(model, loader_t15, device)
    save_embeddings(out_root, name, "t15", emb_t15, day_to_session_t15)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"[{name}] T12: {len(emb_t12)} sessions; T15: {len(emb_t15)} sessions")
    return emb_t12, emb_t15


# ---------------------------------------------------------------------------
# Aggregation + figures
# ---------------------------------------------------------------------------

def write_csv(rows: List[dict], path: Path, k_max: int) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["encoder", "day_i", "day_j"] + [f"cc_{i+1}" for i in range(k_max)] + ["mean_top4"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def compute_summary(rows: List[dict], encoders: List[str]) -> dict:
    out = {}
    for name in encoders:
        vals = np.array(
            [r["mean_top4"] for r in rows if r["encoder"] == name and r["mean_top4"] == r["mean_top4"]],
            dtype=np.float64,
        )
        out[name] = {
            "n_pairs": int(vals.size),
            "mean_top4_mean": float(vals.mean()) if vals.size else float("nan"),
            "mean_top4_std": float(vals.std()) if vals.size else float("nan"),
            "mean_top4_median": float(np.median(vals)) if vals.size else float("nan"),
        }
    return out


def write_summary_csv(summary: dict, encoders: List[str], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "n_pairs", "mean_top4_mean", "mean_top4_std", "mean_top4_median"])
        for name in encoders:
            s = summary[name]
            w.writerow([name, s["n_pairs"], s["mean_top4_mean"], s["mean_top4_std"], s["mean_top4_median"]])


def make_distribution_figure(rows, encoders, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data, labels = [], []
    for name in encoders:
        vals = [r["mean_top4"] for r in rows if r["encoder"] == name and r["mean_top4"] == r["mean_top4"]]
        if vals:
            data.append(vals); labels.append(name)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylabel("Mean top-4 canonical correlation")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_decay_figure(rows, encoders, k_decay: int, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in encoders:
        ccs = [[r.get(f"cc_{kk+1}", np.nan) for kk in range(k_decay)] for r in rows if r["encoder"] == name]
        if not ccs:
            continue
        arr = np.array(ccs, dtype=np.float64)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / max(1, np.sqrt(arr.shape[0]))
        x = np.arange(1, k_decay + 1)
        ax.plot(x, mean, marker="o", label=name)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlabel("k (canonical component)")
    ax.set_ylabel("CC_k (mean ± SEM)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, k_decay + 1))
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_combined_figure(b1_rows, b2_rows, encoders, out_path: Path):
    """Headline figure: B1 (within-T15) vs B2 (cross-participant) side by side per encoder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(encoders)
    x = np.arange(n)
    width = 0.38

    def stats(rows, name):
        vals = np.array([r["mean_top4"] for r in rows if r["encoder"] == name and r["mean_top4"] == r["mean_top4"]],
                        dtype=np.float64)
        return (vals.mean() if vals.size else np.nan,
                vals.std() / np.sqrt(max(1, vals.size)) if vals.size else np.nan)

    b1_means, b1_sems = zip(*[stats(b1_rows, e) for e in encoders])
    b2_means, b2_sems = zip(*[stats(b2_rows, e) for e in encoders])

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width/2, b1_means, width, yerr=b1_sems, label="B1: within T15", capsize=4)
    ax.bar(x + width/2, b2_means, width, yerr=b2_sems, label="B2: cross T12↔T15", capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(encoders, rotation=15)
    ax.set_ylabel("Mean top-4 canonical correlation (mean ± SEM)")
    ax.set_title("Cross-session vs cross-participant representational alignment")
    ax.set_ylim(0, max(0.25, max(list(b1_means) + list(b2_means)) * 1.25))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_b2 = Path(args.out_root_b2).resolve(); out_b2.mkdir(parents=True, exist_ok=True)
    out_b1 = Path(args.out_root_b1).resolve(); out_b1.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ---- Configs ----
    base_cfg = OmegaConf.load(args.t15_config)
    t12_cfg = merge_t12_config(base_cfg, args.t12_config)

    # ---- Loaders ----
    logger.info("Building T15 val loader")
    loader_t15, d2s_t15, n_t15 = build_val_loader(base_cfg, args.batch_size, dry_run=args.dry_run)
    logger.info("Building T12 val loader")
    loader_t12, d2s_t12, n_t12 = build_val_loader(t12_cfg, args.batch_size, dry_run=args.dry_run)
    logger.info(f"n_days T12={n_t12}  T15={n_t15}")

    # ---- Encoder factories ----
    def random_pair():
        # Build paired random decoders (shared transformer weights, per-participant read-ins).
        m_t12, m_t15 = build_paired_random_decoders(
            n_days_a=n_t12, n_days_b=n_t15, seed=args.random_init_seed
        )
        return m_t12, m_t15

    # Use closures so we don't keep both models alive at once.
    _random_pair_holder = {"models": None}

    def factory_random_t12():
        if _random_pair_holder["models"] is None:
            _random_pair_holder["models"] = random_pair()
        m = _random_pair_holder["models"][0]
        # Detach the sibling so it can be re-built freshly when needed.
        return m

    def factory_random_t15():
        if _random_pair_holder["models"] is None:
            _random_pair_holder["models"] = random_pair()
        m = _random_pair_holder["models"][1]
        return m

    encoder_specs = [
        ("random_init",
         factory_random_t12,
         factory_random_t15),
        ("ft_no_ssl",
         lambda: load_finetuned_decoder(args.t12_ckpt_no_ssl, n_days=n_t12, strict=False),
         lambda: load_finetuned_decoder(args.t15_ckpt_no_ssl, n_days=n_t15, strict=False)),
        ("ft_ssl",
         lambda: load_finetuned_decoder(args.t12_ckpt_ssl, n_days=n_t12, strict=False),
         lambda: load_finetuned_decoder(args.t15_ckpt_ssl, n_days=n_t15, strict=False)),
    ]

    # ---- Extract embeddings ----
    embeddings: Dict[str, Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]] = {}
    for name, fa, fb in encoder_specs:
        embeddings[name] = run_encoder_extraction(
            name, fa, fb, loader_t12, loader_t15, d2s_t12, d2s_t15,
            device, out_b2, args.reuse_embeddings,
        )

    # ---- Raw input control (no model) ----
    logger.info("[raw_input] extracting raw features T12")
    raw_t12 = extract_raw_session_features(loader_t12)
    save_embeddings(out_b2, "raw_input", "t12", raw_t12, d2s_t12)
    logger.info("[raw_input] extracting raw features T15")
    raw_t15 = extract_raw_session_features(loader_t15)
    save_embeddings(out_b2, "raw_input", "t15", raw_t15, d2s_t15)
    embeddings["raw_input"] = (raw_t12, raw_t15)

    # ---- B2 cross-dataset CCA ----
    k_max = max(args.k_decay, args.k_top)
    rows_b2: List[dict] = []
    for name in ["random_init", "ft_no_ssl", "ft_ssl", "raw_input"]:
        emb_a, emb_b = embeddings[name]
        logger.info(f"[B2 {name}] CCA on {len(emb_a)}×{len(emb_b)} cross pairs")
        results = cross_dataset_pairs_cca(emb_a, emb_b, encoder_name=name,
                                           m_pca=args.m_pca, k=k_max,
                                           shuffle=False, rng_seed=args.seed)
        rows_b2.extend(r.as_row(k_max=k_max) for r in results)

    # ft_ssl_shuf: same SSL embeddings, T15 side temporally permuted.
    emb_a, emb_b = embeddings["ft_ssl"]
    logger.info("[B2 ft_ssl_shuf] CCA shuffled control")
    results = cross_dataset_pairs_cca(emb_a, emb_b, encoder_name="ft_ssl_shuf",
                                       m_pca=args.m_pca, k=k_max,
                                       shuffle=True, rng_seed=args.seed + 1)
    rows_b2.extend(r.as_row(k_max=k_max) for r in results)

    encoders_b2 = ["random_init", "ft_no_ssl", "ft_ssl", "ft_ssl_shuf", "raw_input"]
    write_csv(rows_b2, out_b2 / "results" / "cca_pairs_b2.csv", k_max=k_max)
    summary_b2 = compute_summary(rows_b2, encoders_b2)
    write_summary_csv(summary_b2, encoders_b2, out_b2 / "results" / "summary_b2.csv")
    make_distribution_figure(rows_b2, encoders_b2, out_b2 / "figures" / "cca_distribution_b2.png",
                              "Cross-participant CCA (T12 ↔ T15)")
    make_decay_figure(rows_b2, encoders_b2, args.k_decay,
                      out_b2 / "figures" / "cca_decay_b2.png",
                      "Cross-participant CC decay")

    # ---- B1 update: regenerate with raw_input (within-T15 pairs over the new raw inputs) ----
    if not args.skip_b1_update:
        logger.info("[B1 update] computing within-T15 raw_input pairs and regenerating B1 outputs")

        b1_csv = out_b1 / "results" / "cca_pairs.csv"
        rows_b1: List[dict] = []
        if b1_csv.exists():
            import csv as _csv
            with open(b1_csv, newline="") as f:
                reader = _csv.DictReader(f)
                for r in reader:
                    rec = {"encoder": r["encoder"], "day_i": int(r["day_i"]), "day_j": int(r["day_j"])}
                    for kk in range(k_max):
                        v = r.get(f"cc_{kk+1}", "")
                        rec[f"cc_{kk+1}"] = float(v) if v not in (None, "", "nan") else np.nan
                    rec["mean_top4"] = float(r["mean_top4"]) if r["mean_top4"] not in (None, "", "nan") else np.nan
                    rows_b1.append(rec)
            # Drop any pre-existing raw_input rows so we don't double up across reruns.
            rows_b1 = [r for r in rows_b1 if r["encoder"] != "raw_input"]
        else:
            logger.warning(f"  B1 CSV not found at {b1_csv}; emitting raw_input only")

        results = all_pairs_cca(raw_t15, encoder_name="raw_input",
                                 m_pca=args.m_pca, k=k_max,
                                 shuffle=False, rng_seed=args.seed)
        rows_b1.extend(r.as_row(k_max=k_max) for r in results)

        write_csv(rows_b1, b1_csv, k_max=k_max)
        encoders_b1 = ["random_init", "ft_no_ssl", "ft_ssl", "ft_ssl_shuf", "raw_input"]
        summary_b1 = compute_summary(rows_b1, encoders_b1)
        write_summary_csv(summary_b1, encoders_b1, out_b1 / "results" / "summary.csv")
        make_distribution_figure(rows_b1, encoders_b1,
                                  out_b1 / "figures" / "cca_distribution.png",
                                  "Within-T15 CCA (B1)")
        make_decay_figure(rows_b1, encoders_b1, args.k_decay,
                          out_b1 / "figures" / "cca_decay.png",
                          "Within-T15 CC decay (B1)")

        # Combined headline figure (B1 + B2).
        make_combined_figure(rows_b1, rows_b2, encoders_b1,
                              out_b2 / "figures" / "cca_combined_b1_b2.png")

    # ---- Stdout report ----
    logger.info("=" * 70)
    logger.info("Phase B2 — Mean top-4 CC per encoder (cross-participant T12↔T15)")
    logger.info("=" * 70)
    for name in encoders_b2:
        s = summary_b2[name]
        logger.info(f"  {name:14s}  n={s['n_pairs']:5d}  mean={s['mean_top4_mean']:.4f}  "
                    f"std={s['mean_top4_std']:.4f}  median={s['mean_top4_median']:.4f}")
    if not args.skip_b1_update:
        logger.info("-" * 70)
        logger.info("Phase B1 — Mean top-4 CC per encoder (within T15)")
        logger.info("-" * 70)
        for name in ["random_init", "ft_no_ssl", "ft_ssl", "ft_ssl_shuf", "raw_input"]:
            s = summary_b1.get(name, {})
            if not s:
                continue
            logger.info(f"  {name:14s}  n={s['n_pairs']:5d}  mean={s['mean_top4_mean']:.4f}  "
                        f"std={s['mean_top4_std']:.4f}  median={s['mean_top4_median']:.4f}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
