#!/usr/bin/env python3
"""Preprocess Brain-to-Text '24 (participant T12, Willett 2023 Dryad Sep 2023) .mat files
to the per-session HDF5 schema consumed by BrainToTextDataset.

Input layout (on cluster):
    {raw_dir}/competitionData/
        train/*.mat
        test/*.mat
        competitionHoldOut/*.mat   (ignored)
Each .mat is one session/day with fields:
    sentenceText : (N,) str        — transcription per trial
    tx1..tx4     : (N,) object     — each trial (T_i, 256) threshold-crossings
    spikePow     : (N,) object     — each trial (T_i, 256) spike-band power
    blockIdx     : (N,) uint8      — block index per trial

Output layout (matches T15 hdf5_data_final):
    {out_dir}/{session}/data_train.hdf5
    {out_dir}/{session}/data_val.hdf5
where {session} is e.g. "t12.2022.04.28" and each hdf5 contains groups
"trial_{t:04d}" with:
    input_features : (T, 512) float32   — concat(tx1, spikePow), z-scored
                                          per-channel within each block
                                          (Willett 2023 readme hint; also
                                          matches the unit-variance scale
                                          assumed by white_noise_std=0.8)
    seq_class_ids  : (L,) int32         — phoneme IDs (LOGIT_TO_PHONEME order)
    transcription  : (500,) uint8       — ASCII, zero-padded
    attrs: n_time_steps, seq_len, block_num, trial_num, session, sentence_label

Run on the cluster:
    python scripts/preprocess_b2t24.py \
        --raw-dir  ${DATA_DIR}/b2t24_raw \
        --out-dir  ${DATA_DIR}/b2t24_hdf5 \
        [--max-sessions 2]   # dry-run
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import scipy.io as sio

# Phoneme ID map used by the training pipeline (BLANK=0, 39 phonemes=1..39, SIL=40).
# This matches LOGIT_TO_PHONEME in evaluate_model_helpers.py; the trainer emits 41
# logits in exactly this order (sil_index=40 in configs/rnn_args.yaml).
PHONEMES = [
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B",  "CH", "D",  "DH",
    "EH", "ER", "EY", "F",  "G",
    "HH", "IH", "IY", "JH", "K",
    "L",  "M",  "N",  "NG", "OW",
    "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",
    "W",  "Y",  "Z",  "ZH",
]
PHONEME_TO_ID: Dict[str, int] = {p: i + 1 for i, p in enumerate(PHONEMES)}
SIL_ID = 40
BLANK_ID = 0
assert len(PHONEMES) == 39, "Expected 39 phonemes + BLANK + SIL = 41 classes"

TRANSCRIPTION_MAXLEN = 500  # matches max_seq_elements in rnn_args.yaml
ZSCORE_EPS = 1e-6           # avoid /0 on dead channels

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("preprocess_b2t24")


def _ensure_nltk_resources() -> None:
    """Make sure g2p_en's NLTK dependencies are present.

    g2p_en uses nltk.pos_tag, which on recent nltk looks up the renamed
    'averaged_perceptron_tagger_eng' resource. The auto-download that
    g2p_en triggers only fetches the old-named resource, so we pull both.
    """
    import nltk
    for kind, name in (
        ("taggers", "averaged_perceptron_tagger_eng"),
        ("taggers", "averaged_perceptron_tagger"),
        ("corpora", "cmudict"),
    ):
        try:
            nltk.data.find(f"{kind}/{name}")
        except LookupError:
            log.info(f"Downloading nltk resource: {name}")
            try:
                nltk.download(name, quiet=True)
            except Exception as e:
                log.warning(f"  download of {name} failed: {e}")


def _session_name_from_mat(path: Path) -> str:
    """Derive 't12.YYYY.MM.DD' from a .mat filename stem.

    Willett 2023 Dryad names files like ``t12.2022.04.28_sentences.mat`` or
    ``t12.2022.04.28.mat``; we strip anything after the date.
    """
    stem = path.stem
    m = re.match(r"(t12\.\d{4}\.\d{2}\.\d{2})", stem)
    if m is None:
        raise ValueError(
            f"Cannot parse T12 session name from '{path.name}'. "
            f"Expected prefix like 't12.YYYY.MM.DD'."
        )
    return m.group(1)


def _remove_punctuation(sentence: str) -> str:
    sentence = re.sub(r"[^a-zA-Z\- \']", "", sentence)
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    sentence = " ".join(sentence.split())
    return sentence


def _text_to_phoneme_ids(text: str, g2p) -> List[int]:
    """Convert a sentence to class IDs using g2p_en + the training logit order."""
    clean = _remove_punctuation(text)
    ids: List[int] = []
    if len(clean) == 0:
        ids.append(SIL_ID)
        return ids

    for tok in g2p(clean):
        if tok == " ":
            ids.append(SIL_ID)
            continue
        # strip stress digits and keep alpha-only tokens
        tok = re.sub(r"[0-9]", "", tok)
        if re.match(r"[A-Z]+$", tok):
            pid = PHONEME_TO_ID.get(tok)
            if pid is not None:
                ids.append(pid)
    ids.append(SIL_ID)  # trailing silence (matches T15 convention)
    return ids


def _load_mat(path: Path) -> dict:
    """Load a .mat file with cell arrays simplified to numpy object arrays."""
    mat = sio.loadmat(str(path), simplify_cells=True)
    for req in ("sentenceText", "tx1", "spikePow", "blockIdx"):
        if req not in mat:
            raise KeyError(
                f"{path.name}: missing required field '{req}'. "
                f"Available keys: {sorted(k for k in mat if not k.startswith('__'))}"
            )
    return mat


def _extract_trials(mat: dict) -> List[dict]:
    """Return a list of per-trial dicts with features/text/block_idx."""
    sentences = mat["sentenceText"]
    tx1 = mat["tx1"]
    spikepow = mat["spikePow"]
    block_idx = mat["blockIdx"]

    sentences = np.atleast_1d(sentences)
    tx1 = np.atleast_1d(tx1)
    spikepow = np.atleast_1d(spikepow)
    block_idx = np.atleast_1d(block_idx).ravel()

    n = len(sentences)
    if not (len(tx1) == len(spikepow) == len(block_idx) == n):
        raise ValueError(
            f"Length mismatch: sentences={n}, tx1={len(tx1)}, "
            f"spikePow={len(spikepow)}, blockIdx={len(block_idx)}"
        )

    trials: List[dict] = []
    for i in range(n):
        txi = np.asarray(tx1[i], dtype=np.float32)
        sbi = np.asarray(spikepow[i], dtype=np.float32)
        if txi.ndim != 2 or sbi.ndim != 2:
            raise ValueError(f"Trial {i}: expected 2D features, got tx1={txi.shape}, spikePow={sbi.shape}")
        if txi.shape != sbi.shape:
            raise ValueError(f"Trial {i}: tx1/spikePow shape mismatch: {txi.shape} vs {sbi.shape}")
        if txi.shape[1] != 256:
            raise ValueError(f"Trial {i}: expected 256 channels, got {txi.shape[1]}")

        feats = np.concatenate([txi, sbi], axis=1).astype(np.float32)  # (T, 512)
        if not np.isfinite(feats).all():
            raise ValueError(f"Trial {i}: non-finite values in features")

        text = str(sentences[i]).strip()
        trials.append({
            "features": feats,
            "text": text,
            "block_num": int(block_idx[i]),
            "trial_num": i,
        })
    return trials


def _blockwise_zscore_inplace(trials: List[dict]) -> dict:
    """Per-block, per-channel z-score the features of each trial in place.

    The B2T '24 readme explicitly recommends this: "Use the blockIdx variable
    to perform blockwise z-scoring of the data to remove drifts in the feature
    means which can be quite severe." It also matches the scale assumed by our
    augmentation HPs (white_noise_std=0.8 expects unit variance).

    For each block, concatenate all trials along time, compute the per-channel
    mean and std across that concatenation, then apply (x - mean) / std to
    every trial in the block. tx1 and spikePow channels are normalized
    independently because they share the same per-channel treatment.
    """
    from collections import defaultdict
    idx_by_block: Dict[int, List[int]] = defaultdict(list)
    for i, tr in enumerate(trials):
        idx_by_block[tr["block_num"]].append(i)

    diag = {"n_blocks": len(idx_by_block), "n_dead_channels_max": 0}
    for block_num, trial_idxs in idx_by_block.items():
        stacked = np.concatenate([trials[i]["features"] for i in trial_idxs], axis=0)  # (sum_T, 512)
        mean = stacked.mean(axis=0, keepdims=True)
        std = stacked.std(axis=0, keepdims=True)
        # Track dead channels (std == 0) for diagnostics; we still normalize with eps
        n_dead = int((std <= 0).sum())
        diag["n_dead_channels_max"] = max(diag["n_dead_channels_max"], n_dead)
        std_safe = np.where(std > ZSCORE_EPS, std, 1.0)
        for i in trial_idxs:
            trials[i]["features"] = ((trials[i]["features"] - mean) / std_safe).astype(np.float32)
    return diag


def _encode_transcription(text: str) -> np.ndarray:
    """ASCII-encode text into a fixed-length uint8 array, zero-padded."""
    buf = np.zeros(TRANSCRIPTION_MAXLEN, dtype=np.uint8)
    b = text.encode("ascii", errors="ignore")[: TRANSCRIPTION_MAXLEN - 1]
    buf[: len(b)] = np.frombuffer(b, dtype=np.uint8)
    return buf


def _write_session_split(
    out_path: Path,
    session: str,
    trials: List[dict],
    g2p,
) -> dict:
    """Write one {split}.hdf5 for a session and return a small stats dict."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"n_trials": 0, "T_min": None, "T_max": None, "L_max": 0}

    with h5py.File(out_path, "w") as f:
        for t_idx, tr in enumerate(trials):
            feats = tr["features"]  # (T, 512)
            T, C = feats.shape
            assert C == 512

            pid_list = _text_to_phoneme_ids(tr["text"], g2p)
            seq_ids = np.asarray(pid_list, dtype=np.int32)
            if len(seq_ids) == 0:
                # Should not happen — _text_to_phoneme_ids always appends SIL —
                # but guard anyway to avoid writing empty labels that break CTC.
                log.warning(f"{session} trial {t_idx}: empty phoneme sequence, skipping")
                continue

            g = f.create_group(f"trial_{t_idx:04d}")
            g.create_dataset("input_features", data=feats, dtype="float32", compression="gzip")
            g.create_dataset("seq_class_ids", data=seq_ids, dtype="int32")
            g.create_dataset("transcription", data=_encode_transcription(tr["text"]), dtype="uint8")

            g.attrs["n_time_steps"] = int(T)
            g.attrs["seq_len"] = int(len(seq_ids))
            g.attrs["block_num"] = int(tr["block_num"])
            g.attrs["trial_num"] = int(tr["trial_num"])
            g.attrs["session"] = session
            g.attrs["sentence_label"] = tr["text"]

            stats["n_trials"] += 1
            stats["T_min"] = T if stats["T_min"] is None else min(stats["T_min"], T)
            stats["T_max"] = T if stats["T_max"] is None else max(stats["T_max"], T)
            stats["L_max"] = max(stats["L_max"], int(len(seq_ids)))

    return stats


def _collect_session_mats(raw_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Return {session: {"train": Path|None, "val": Path|None}}.

    Walks competitionData/train and competitionData/test. The train split from
    the raw dataset becomes data_train.hdf5; the test split (held-out trials
    from the *same* sessions) becomes data_val.hdf5.
    """
    comp = raw_dir / "competitionData"
    if not comp.is_dir():
        raise FileNotFoundError(f"Missing {comp}")

    def _scan(sub: str) -> Dict[str, Path]:
        d = comp / sub
        out: Dict[str, Path] = {}
        if not d.is_dir():
            log.warning(f"{d} not found — skipping")
            return out
        for mat in sorted(d.glob("*.mat")):
            sess = _session_name_from_mat(mat)
            if sess in out:
                raise RuntimeError(f"Duplicate session '{sess}' in {d}: {mat} and {out[sess]}")
            out[sess] = mat
        return out

    train_mats = _scan("train")
    test_mats = _scan("test")

    all_sessions = sorted(set(train_mats) | set(test_mats))
    grouped: Dict[str, Dict[str, Path]] = {}
    for s in all_sessions:
        grouped[s] = {"train": train_mats.get(s), "val": test_mats.get(s)}
    return grouped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Path to the Brain-to-Text '24 .mat release (must contain competitionData/train and competitionData/test).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Destination directory for the per-session HDF5 output.",
    )
    p.add_argument("--max-sessions", type=int, default=0, help=">0 to process only the first N sessions (dry-run).")
    p.add_argument("--sessions", nargs="*", default=None, help="Optional explicit list of session names to process.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing per-session hdf5 files.")
    p.add_argument(
        "--write-dataset-config",
        type=Path,
        default=None,
        help="If set, after preprocessing write a YAML with dataset.sessions / dataset_dir / "
             "dataset_probability_val for use by Phase A training configs.",
    )
    args = p.parse_args()

    try:
        from g2p_en import G2p
    except ImportError:
        log.error("g2p_en not installed. `pip install g2p_en`.")
        return 2

    _ensure_nltk_resources()

    log.info(f"raw_dir = {args.raw_dir}")
    log.info(f"out_dir = {args.out_dir}")

    sessions = _collect_session_mats(args.raw_dir)
    if args.sessions:
        sessions = {s: sessions[s] for s in args.sessions if s in sessions}
    if args.max_sessions > 0:
        sessions = dict(list(sessions.items())[: args.max_sessions])

    if not sessions:
        log.error("No sessions to process.")
        return 1

    log.info(f"Processing {len(sessions)} sessions")
    g2p = G2p()

    summary: List[str] = []
    for sess, mats in sessions.items():
        sess_dir = args.out_dir / sess
        log.info(f"--- {sess} ---")
        for split_name, hdf_name in (("train", "data_train.hdf5"), ("val", "data_val.hdf5")):
            mat_path = mats[split_name]
            out_path = sess_dir / hdf_name
            if mat_path is None:
                log.info(f"  {split_name}: no .mat for this session, skipping")
                continue
            if out_path.exists() and not args.overwrite:
                log.info(f"  {split_name}: {out_path} exists, skipping (use --overwrite)")
                continue
            mat = _load_mat(mat_path)
            trials = _extract_trials(mat)
            zs = _blockwise_zscore_inplace(trials)
            stats = _write_session_split(out_path, sess, trials, g2p)
            log.info(
                f"  {split_name}: {stats['n_trials']} trials "
                f"T_range=[{stats['T_min']},{stats['T_max']}] L_max={stats['L_max']} "
                f"blocks={zs['n_blocks']} dead_ch≤{zs['n_dead_channels_max']} → {out_path}"
            )
            summary.append(f"{sess}/{split_name}: n={stats['n_trials']}")

    log.info("Summary:")
    for line in summary:
        log.info(f"  {line}")
    log.info(f"Done. Sessions processed: {len(sessions)}")

    if args.write_dataset_config is not None:
        # Only list sessions that actually have a data_train.hdf5 on disk.
        final_sessions = [
            s for s in sorted(sessions.keys())
            if (args.out_dir / s / "data_train.hdf5").exists()
        ]
        if not final_sessions:
            log.warning("No sessions with data_train.hdf5 — skipping dataset config write.")
        else:
            args.write_dataset_config.parent.mkdir(parents=True, exist_ok=True)
            with args.write_dataset_config.open("w") as fh:
                fh.write("# Auto-generated by scripts/preprocess_b2t24.py — DO NOT EDIT BY HAND\n")
                fh.write("# Source: Willett 2023 Dryad (B2T '24, participant T12, Sep 2023 release)\n")
                fh.write("dataset:\n")
                fh.write(f"  dataset_dir: {args.out_dir}\n")
                fh.write("  sessions:\n")
                for s in final_sessions:
                    fh.write(f"  - {s}\n")
                fh.write("  dataset_probability_val:\n")
                for _ in final_sessions:
                    fh.write("  - 1\n")
            log.info(f"Wrote dataset config with {len(final_sessions)} sessions → {args.write_dataset_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
