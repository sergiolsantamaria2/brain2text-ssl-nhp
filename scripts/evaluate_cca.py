#!/usr/bin/env python3
"""
Phase B1 — Cross-session CCA on T15 (offline analysis).

Three encoder conditions plus a shuffled control:
  - random_init   : TransformerDecoder built from scratch (no checkpoint).
  - ft_no_ssl     : best TFS finetuning checkpoint (no SSL pretraining).
  - ft_ssl        : best AR-binary-soma → finetuning checkpoint (PER 0.0918 seed10).
  - ft_ssl_shuf   : ft_ssl with one member of each pair temporally permuted.

Outputs (under --out-root):
  embeddings/<encoder>/<session_id>.npy   (T_total, embed_dim) float32
  results/cca_pairs.csv                   one row per (encoder, day_i, day_j)
  results/summary.csv                     mean top-4 CC per encoder
  figures/cca_distribution.png            boxplot
  figures/cca_decay.png                   CC_k vs k

Usage (cluster, single GPU, no training):
  python scripts/run_cca_phase_b1.py \
      --ckpt-no-ssl  trained_models/tfs_postfix/h03_log_resffn_deep_cosine/checkpoint/best_model.pt \
      --ckpt-ssl     ${OUTPUT_DIR}/ssl_study_ft250k_multiseed/ft250k_ar_binary_soma_epoch400_seed10/checkpoint/best_model.pt \
      --base-config  configs/baselines/gru_defaults.yaml \
      --out-root     ${OUTPUT_DIR}/analysis/cca_phase_b1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Make `src/` importable when running as a plain script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from brain2text.evaluation.cca import (  # noqa: E402
    DEFAULT_ARCH,
    all_pairs_cca,
    build_decoder,
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
logger = logging.getLogger("cca_phase_b1")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", default="configs/baselines/gru_defaults.yaml",
                   help="Base config providing dataset.dataset_dir + dataset.sessions.")
    p.add_argument("--ckpt-no-ssl", required=True,
                   help="best_model.pt for the TFS no-SSL baseline.")
    p.add_argument("--ckpt-ssl", required=True,
                   help="best_model.pt for the AR-binary-soma SSL+FT run (seed 10, ft 250k).")
    p.add_argument("--out-root", default="${OUTPUT_DIR}/analysis/cca_phase_b1")
    p.add_argument("--m-pca", type=int, default=10, help="PCA dims before CCA (Gallego et al. use ~10).")
    p.add_argument("--k-top", type=int, default=4, help="Top-k canonical correlations to summarize.")
    p.add_argument("--k-decay", type=int, default=10, help="Number of CCs to track for the decay plot.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default=None, help="cuda / cpu / cuda:0 etc. Auto-detected if omitted.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse-embeddings", action="store_true",
                   help="If set, load existing .npy files instead of re-running encoders.")
    p.add_argument("--limit-sessions", type=int, default=0,
                   help="If >0, only use the first N sessions (debugging).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_val_loader(cfg, batch_size: int):
    """Build a T15 val DataLoader from the base config (mirrors rnn_trainer)."""
    dataset_dir = cfg["dataset"]["dataset_dir"]
    sessions = list(cfg["dataset"]["sessions"])

    # Replicate the exact filter that rnn_trainer applies at train time so n_days
    # matches the checkpoint: keep sessions whose data_train.hdf5 exists and is
    # non-empty. Sessions without a val file produce empty val trial lists in
    # train_test_split_indicies and are silently absent from the loader — but
    # they still occupy their day index, so day_indices stay consistent with
    # what the checkpoint was trained with.
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

    val_file_paths = [os.path.join(dataset_dir, s, "data_val.hdf5") for s in keep_sessions]

    sessions_with_val = sum(1 for p in val_file_paths if os.path.exists(p))
    logger.info(f"  using {len(keep_sessions)} sessions (n_days), "
                f"{sessions_with_val} have val data")

    _, val_trials = train_test_split_indicies(
        file_paths=val_file_paths,
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
    loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    # day -> human session id for reporting.
    day_to_session = {i: s for i, s in enumerate(keep_sessions)}
    return loader, day_to_session, len(keep_sessions)


# ---------------------------------------------------------------------------
# Embedding I/O
# ---------------------------------------------------------------------------

def embeddings_dir(out_root: Path, encoder_name: str) -> Path:
    d = out_root / "embeddings" / encoder_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_embeddings(out_root: Path, encoder_name: str,
                    embeddings: Dict[int, np.ndarray],
                    day_to_session: Dict[int, str]) -> None:
    d = embeddings_dir(out_root, encoder_name)
    for day_idx, Z in embeddings.items():
        sid = day_to_session.get(day_idx, f"day{day_idx:03d}")
        np.save(d / f"{sid}.npy", Z.astype(np.float32, copy=False))
    with open(d / "_index.json", "w") as f:
        json.dump({str(k): day_to_session.get(k, str(k)) for k in embeddings}, f, indent=2)


def load_embeddings(out_root: Path, encoder_name: str,
                    day_to_session: Dict[int, str]) -> Dict[int, np.ndarray]:
    d = embeddings_dir(out_root, encoder_name)
    sid_to_day = {v: k for k, v in day_to_session.items()}
    out: Dict[int, np.ndarray] = {}
    for npy in sorted(d.glob("*.npy")):
        sid = npy.stem
        if sid not in sid_to_day:
            logger.warning(f"  unknown session id in cache: {sid}")
            continue
        out[sid_to_day[sid]] = np.load(npy)
    return out


# ---------------------------------------------------------------------------
# Per-encoder run
# ---------------------------------------------------------------------------

def get_or_extract(name: str,
                   model_factory,
                   loader,
                   device,
                   out_root: Path,
                   day_to_session: Dict[int, str],
                   reuse: bool) -> Dict[int, np.ndarray]:
    cache = embeddings_dir(out_root, name)
    if reuse and any(cache.glob("*.npy")):
        logger.info(f"[{name}] reusing cached embeddings from {cache}")
        emb = load_embeddings(out_root, name, day_to_session)
        if emb:
            return emb
        logger.warning(f"[{name}] cache empty, re-extracting")

    logger.info(f"[{name}] building model and extracting embeddings")
    model = model_factory()
    emb = extract_session_embeddings(model, loader, device)
    save_embeddings(out_root, name, emb, day_to_session)
    sizes = {day_to_session.get(d, d): Z.shape for d, Z in emb.items()}
    logger.info(f"[{name}] saved {len(emb)} sessions; first 3 shapes: {list(sizes.items())[:3]}")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


# ---------------------------------------------------------------------------
# Aggregation + figures
# ---------------------------------------------------------------------------

def write_results_csv(rows: List[dict], path: Path, k_max: int) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["encoder", "day_i", "day_j"] + [f"cc_{i+1}" for i in range(k_max)] + ["mean_top4"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def make_figures(rows: List[dict], out_dir: Path, encoders_in_order: List[str], k_decay: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Boxplot of mean top-4 CC per encoder ---
    data = []
    labels = []
    for name in encoders_in_order:
        vals = [r["mean_top4"] for r in rows if r["encoder"] == name and r["mean_top4"] == r["mean_top4"]]
        if not vals:
            continue
        data.append(vals)
        labels.append(name)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Mean top-4 canonical correlation")
    ax.set_title("Cross-session representational alignment (T15 val)")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cca_distribution.png", dpi=150)
    plt.close(fig)

    # --- Decay: CC_k vs k, averaged across pairs ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in encoders_in_order:
        ccs = []
        for r in rows:
            if r["encoder"] != name:
                continue
            vec = [r.get(f"cc_{kk+1}", np.nan) for kk in range(k_decay)]
            ccs.append(vec)
        if not ccs:
            continue
        arr = np.array(ccs, dtype=np.float64)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / max(1, np.sqrt(arr.shape[0]))
        x = np.arange(1, k_decay + 1)
        ax.plot(x, mean, marker="o", label=name)
        ax.fill_between(x, mean - sem, mean + sem, alpha=0.2)
    ax.set_xlabel("k (canonical component)")
    ax.set_ylabel("CC_k (mean ± SEM across pairs)")
    ax.set_title("Decay of canonical correlations")
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, k_decay + 1))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "cca_decay.png", dpi=150)
    plt.close(fig)


def write_summary(rows: List[dict], path: Path, encoders_in_order: List[str]) -> dict:
    import csv
    summary = {}
    for name in encoders_in_order:
        vals = np.array(
            [r["mean_top4"] for r in rows if r["encoder"] == name and r["mean_top4"] == r["mean_top4"]],
            dtype=np.float64,
        )
        summary[name] = {
            "n_pairs": int(vals.size),
            "mean_top4_mean": float(vals.mean()) if vals.size else float("nan"),
            "mean_top4_std": float(vals.std()) if vals.size else float("nan"),
            "mean_top4_median": float(np.median(vals)) if vals.size else float("nan"),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "n_pairs", "mean_top4_mean", "mean_top4_std", "mean_top4_median"])
        for name in encoders_in_order:
            s = summary[name]
            w.writerow([name, s["n_pairs"], s["mean_top4_mean"], s["mean_top4_std"], s["mean_top4_median"]])
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # ---- Config / data ----
    cfg = OmegaConf.load(args.base_config)
    if args.limit_sessions > 0:
        cfg["dataset"]["sessions"] = list(cfg["dataset"]["sessions"])[: args.limit_sessions]

    loader, day_to_session, n_days = build_val_loader(cfg, args.batch_size)
    logger.info(f"n_days = {n_days}")

    # ---- Encoders ----
    def make_random():
        return build_decoder(n_days=n_days)

    # strict=False: head/out are not used for z extraction (we stop at final_norm),
    # so checkpoints with different head configs (e.g. head_type=none vs resffn) load fine.
    def make_no_ssl():
        return load_finetuned_decoder(args.ckpt_no_ssl, n_days=n_days, strict=False)

    def make_ssl():
        return load_finetuned_decoder(args.ckpt_ssl, n_days=n_days, strict=False)

    encoders_in_order = ["random_init", "ft_no_ssl", "ft_ssl"]
    factories = {
        "random_init": make_random,
        "ft_no_ssl": make_no_ssl,
        "ft_ssl": make_ssl,
    }

    embeddings_per_encoder: Dict[str, Dict[int, np.ndarray]] = {}
    for name in encoders_in_order:
        embeddings_per_encoder[name] = get_or_extract(
            name, factories[name], loader, device, out_root, day_to_session, args.reuse_embeddings
        )

    # ---- CCA ----
    rows: List[dict] = []
    k_max = max(args.k_decay, args.k_top)

    for name in encoders_in_order:
        logger.info(f"[{name}] computing CCA over all pairs")
        results = all_pairs_cca(
            embeddings_per_encoder[name],
            encoder_name=name,
            m_pca=args.m_pca,
            k=k_max,
            shuffle=False,
            rng_seed=args.seed,
        )
        for r in results:
            rows.append(r.as_row(k_max=k_max))

    # Shuffled control: same SSL+FT embeddings, one member permuted in time.
    logger.info("[ft_ssl_shuf] computing shuffled-control CCA")
    shuf_results = all_pairs_cca(
        embeddings_per_encoder["ft_ssl"],
        encoder_name="ft_ssl_shuf",
        m_pca=args.m_pca,
        k=k_max,
        shuffle=True,
        rng_seed=args.seed + 1,
    )
    for r in shuf_results:
        rows.append(r.as_row(k_max=k_max))

    encoders_full = encoders_in_order + ["ft_ssl_shuf"]

    # ---- Persist ----
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"

    write_results_csv(rows, results_dir / "cca_pairs.csv", k_max=k_max)
    summary = write_summary(rows, results_dir / "summary.csv", encoders_full)
    make_figures(rows, figures_dir, encoders_full, k_decay=args.k_decay)

    # ---- Stdout report ----
    logger.info("=" * 60)
    logger.info("Phase B1 — Mean top-4 canonical correlation per encoder")
    logger.info("=" * 60)
    for name in encoders_full:
        s = summary[name]
        logger.info(f"  {name:14s}  n={s['n_pairs']:4d}  "
                    f"mean={s['mean_top4_mean']:.4f}  "
                    f"std={s['mean_top4_std']:.4f}  "
                    f"median={s['mean_top4_median']:.4f}")
    logger.info("=" * 60)
    logger.info(f"results CSV: {results_dir / 'cca_pairs.csv'}")
    logger.info(f"summary CSV: {results_dir / 'summary.csv'}")
    logger.info(f"figures:     {figures_dir}")


if __name__ == "__main__":
    main()
