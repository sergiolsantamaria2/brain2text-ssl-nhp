"""
Phase B1 — Cross-session representational stability via CCA on T15.

Procedure follows Gallego, Perich, Miller, Solla, Pandarinath (2018) — bioRxiv
447441v3 ("Long-term stability of cortical population dynamics underlying
consistent behavior"): per-session latent matrices, PCA to a low-dimensional
subspace, then CCA between session pairs. Top-k canonical correlations
quantify representational alignment across sessions.

Encoder conditions compared:
  1. random_init   — TransformerDecoder built from scratch (architecture only).
  2. ft_no_ssl     — best TFS finetuning checkpoint (no SSL).
  3. ft_ssl        — best AR-binary-soma SSL → finetuning checkpoint.
  4. ft_ssl_shuf   — control: ft_ssl with one member of each pair temporally
                     shuffled. Expected to collapse to ~0 (sanity check).

The "z" embedding is the output of the shared transformer stack + final_norm,
i.e. exactly the representation that is transferred from SSL pretraining (the
post-block LayerNorm output, before the optional ResFFN head and the phoneme
classifier). Day-specific input layers and patch_embed run normally so that
session-conditioned read-in is applied — without it, cross-session CCA would
be biased by mismatched input projections.

Usage as CLI: see scripts/run_cca_phase_b1.py (this module is import-only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from brain2text.models.transformer import TransformerDecoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder construction / loading
# ---------------------------------------------------------------------------

DEFAULT_ARCH = dict(
    neural_dim=512,
    n_classes=41,
    rnn_dropout=0.2,
    input_dropout=0.2,
    n_layers=7,
    patch_size=5,
    patch_stride=5,
    embed_dim=384,
    n_heads=6,
    head_dim=512,
    attn_dropout=0.4,
    head_type="resffn",
    head_num_blocks=1,
    head_norm="layernorm",
    head_dropout=0.1,
    head_activation="gelu",
    input_speckle_p=0.0,
    time_mask_ratio=0.0,
)


def build_decoder(n_days: int, arch: Optional[dict] = None) -> TransformerDecoder:
    """Construct a TransformerDecoder with the canonical architecture. No checkpoint loaded."""
    cfg = dict(DEFAULT_ARCH)
    if arch:
        cfg.update(arch)
    return TransformerDecoder(n_days=n_days, ssl_checkpoint=None, **cfg)


def _strip_compile_prefix(state: dict) -> dict:
    """torch.compile saves keys as `_orig_mod.<name>`. Strip them if present."""
    if not state:
        return state
    if any(k.startswith("_orig_mod.") for k in state):
        return {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    return state


def load_finetuned_decoder(
    ckpt_path: str,
    n_days: int,
    arch: Optional[dict] = None,
    strict: bool = True,
) -> TransformerDecoder:
    """Load a TransformerDecoder from a finetuning checkpoint.

    With strict=False, only keys that exist in the model AND have matching shape
    are copied. This tolerates checkpoints with different optional-head configs
    (e.g. head_type=none vs resffn), which is fine because z extraction stops
    at final_norm and never invokes the head/out projections.
    """
    model = build_decoder(n_days=n_days, arch=arch)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = _strip_compile_prefix(state)

    if strict:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"State dict mismatch loading {ckpt_path}: "
                f"missing={missing[:5]}{'...' if len(missing)>5 else ''} "
                f"unexpected={unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
            )
        logger.info(f"Loaded {ckpt_path} (strict, {len(state)} tensors)")
        return model

    # Permissive: keep keys that match name + shape; report what was kept/skipped.
    own = model.state_dict()
    matched, mismatched, missing = {}, [], []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            matched[k] = v
        else:
            mismatched.append(k)
    for k in own:
        if k not in state:
            missing.append(k)

    own.update(matched)
    model.load_state_dict(own, strict=True)
    logger.info(
        f"Loaded {ckpt_path} (permissive): {len(matched)} matched, "
        f"{len(mismatched)} skipped (extra/shape), {len(missing)} missing in ckpt. "
        f"Critical (blocks/final_norm/day/patch_embed) MUST be in matched."
    )
    # Sanity: the components we actually use for z extraction must all be present.
    required_prefixes = ("blocks.", "final_norm.", "day_weights", "day_biases", "patch_embed.")
    not_loaded = [k for k in own if k.startswith(required_prefixes) and k not in matched]
    if not_loaded:
        raise RuntimeError(
            f"Critical encoder weights not loaded from {ckpt_path}: {not_loaded[:8]}"
        )
    return model


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_batch(
    model: TransformerDecoder,
    features: torch.Tensor,
    day_indices: torch.Tensor,
    n_time_steps: torch.Tensor,
) -> List[np.ndarray]:
    """
    Run model up to and including final_norm, return per-trial embeddings as a
    list of (T_patches_t, embed_dim) np arrays. Patch-level lengths are derived
    from each trial's actual n_time_steps (padding stripped).

    Mirrors TransformerDecoder.forward but stops after final_norm and skips
    the ResFFN head and the phoneme classifier.
    """
    model.eval()
    B, T, C = features.shape
    device = features.device

    day_ids = day_indices.long()
    W = model.day_weights.index_select(0, day_ids)
    b = model.day_biases.index_select(0, day_ids)
    x = torch.einsum("btd,bdk->btk", features, W) + b
    x = model.day_layer_activation(x)
    # day_layer_dropout is a no-op in eval()

    ps, st = model.patch_size, model.patch_stride
    x = x.unfold(1, ps, st)                 # (B, n_patches, C, ps)
    x = x.permute(0, 1, 3, 2).contiguous()  # (B, n_patches, ps, C)
    n_patches_full = x.shape[1]
    x = x.reshape(B, n_patches_full, ps * C)

    x = model.patch_embed(x)
    for block in model.blocks:
        x = block(x)
    x = model.final_norm(x)                 # (B, n_patches, embed_dim)

    # Per-trial valid patch length: floor((T_t - ps) / st) + 1, clipped.
    patch_lens = ((n_time_steps.long() - ps) // st + 1).clamp(min=0, max=n_patches_full)

    out: List[np.ndarray] = []
    x_cpu = x.detach().to(torch.float32).cpu().numpy()
    for i in range(B):
        L = int(patch_lens[i].item())
        out.append(x_cpu[i, :L, :])
    return out


def extract_session_embeddings(
    model: TransformerDecoder,
    val_loader: Iterable[dict],
    device: torch.device,
) -> Dict[int, np.ndarray]:
    """
    Run encoder over the full val_loader and concatenate embeddings per session.

    Returns a dict {day_idx: ndarray of shape (T_total, embed_dim)} where T_total
    is the sum of patch lengths across all val trials of that session.
    """
    model = model.to(device).eval()
    chunks: Dict[int, List[np.ndarray]] = {}

    for batch in val_loader:
        features = batch["input_features"].to(device, non_blocking=True)
        day_indices = batch["day_indicies"].to(device, non_blocking=True)
        n_time_steps = batch["n_time_steps"]

        z_list = encode_batch(model, features, day_indices, n_time_steps)
        days = day_indices.detach().cpu().numpy()
        for i, z in enumerate(z_list):
            d = int(days[i])
            chunks.setdefault(d, []).append(z)

    return {d: np.concatenate(parts, axis=0) for d, parts in chunks.items() if parts}


# ---------------------------------------------------------------------------
# CCA
# ---------------------------------------------------------------------------

def _pca_reduce(Z: np.ndarray, m: int) -> np.ndarray:
    """Center + project onto the top-m principal components (no sklearn dep)."""
    Zc = Z - Z.mean(axis=0, keepdims=True)
    # SVD on centered data; pick top-m right singular vectors.
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    m_eff = min(m, Vt.shape[0])
    return Zc @ Vt[:m_eff].T  # (T, m)


def cca_correlations(
    Z_i: np.ndarray,
    Z_j: np.ndarray,
    m_pca: int = 10,
    k: int = 10,
) -> np.ndarray:
    """
    Compute top-k canonical correlations between two latent matrices via the
    QR/SVD method (Gallego et al., 2018, Methods).

    Steps:
      1. PCA-reduce each Z to m_pca components (centered).
      2. QR decompose each reduced matrix: Z = QR.
      3. SVD of Q_i^T @ Q_j; singular values are the canonical correlations.

    Returns an ndarray of length min(k, m_pca) sorted in descending order,
    clipped to [0, 1].
    """
    if Z_i.shape[0] != Z_j.shape[0]:
        T = min(Z_i.shape[0], Z_j.shape[0])
        Z_i = Z_i[:T]
        Z_j = Z_j[:T]

    Z_i_red = _pca_reduce(Z_i, m_pca)
    Z_j_red = _pca_reduce(Z_j, m_pca)

    Q_i, _ = np.linalg.qr(Z_i_red)
    Q_j, _ = np.linalg.qr(Z_j_red)

    M = Q_i.T @ Q_j
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s[:k]


def shuffled_cca(
    Z_i: np.ndarray,
    Z_j: np.ndarray,
    m_pca: int = 10,
    k: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Same as cca_correlations but the second matrix is temporally permuted."""
    rng = rng if rng is not None else np.random.default_rng(0)
    T = min(Z_i.shape[0], Z_j.shape[0])
    Z_i = Z_i[:T]
    perm = rng.permutation(T)
    Z_j_shuf = Z_j[:T][perm]
    return cca_correlations(Z_i, Z_j_shuf, m_pca=m_pca, k=k)


# ---------------------------------------------------------------------------
# Pair iteration
# ---------------------------------------------------------------------------

@dataclass
class CCAPairResult:
    encoder: str
    day_i: int
    day_j: int
    correlations: np.ndarray  # length k

    def as_row(self, k_max: int = 10) -> dict:
        row = {"encoder": self.encoder, "day_i": self.day_i, "day_j": self.day_j}
        for kk in range(k_max):
            row[f"cc_{kk + 1}"] = float(self.correlations[kk]) if kk < len(self.correlations) else np.nan
        row["mean_top4"] = float(np.mean(self.correlations[:4])) if len(self.correlations) >= 4 else np.nan
        return row


def all_pairs_cca(
    embeddings: Dict[int, np.ndarray],
    encoder_name: str,
    m_pca: int = 10,
    k: int = 10,
    shuffle: bool = False,
    rng_seed: int = 0,
) -> List[CCAPairResult]:
    """Compute CCA for every (i, j) with i < j over the session embeddings."""
    days = sorted(embeddings.keys())
    rng = np.random.default_rng(rng_seed)
    results: List[CCAPairResult] = []
    for ii in range(len(days)):
        for jj in range(ii + 1, len(days)):
            di, dj = days[ii], days[jj]
            Z_i, Z_j = embeddings[di], embeddings[dj]
            if shuffle:
                cc = shuffled_cca(Z_i, Z_j, m_pca=m_pca, k=k, rng=rng)
            else:
                cc = cca_correlations(Z_i, Z_j, m_pca=m_pca, k=k)
            results.append(CCAPairResult(encoder_name, di, dj, cc))
    return results


# ---------------------------------------------------------------------------
# Cross-dataset (e.g. T12 ↔ T15) pairs
# ---------------------------------------------------------------------------

def cross_dataset_pairs_cca(
    embeddings_a: Dict[int, np.ndarray],
    embeddings_b: Dict[int, np.ndarray],
    encoder_name: str,
    m_pca: int = 10,
    k: int = 10,
    shuffle: bool = False,
    rng_seed: int = 0,
    day_offset_b: int = 10000,
) -> List[CCAPairResult]:
    """
    Cartesian product of session pairs across two datasets (every i in A vs
    every j in B). day_offset_b is added to B's day indices in the result rows
    so they don't collide with A's in downstream CSVs.
    """
    days_a = sorted(embeddings_a.keys())
    days_b = sorted(embeddings_b.keys())
    rng = np.random.default_rng(rng_seed)
    results: List[CCAPairResult] = []
    for di in days_a:
        for dj in days_b:
            Z_i, Z_j = embeddings_a[di], embeddings_b[dj]
            if shuffle:
                cc = shuffled_cca(Z_i, Z_j, m_pca=m_pca, k=k, rng=rng)
            else:
                cc = cca_correlations(Z_i, Z_j, m_pca=m_pca, k=k)
            results.append(CCAPairResult(encoder_name, di, dj + day_offset_b, cc))
    return results


# ---------------------------------------------------------------------------
# Raw input control (no encoder)
# ---------------------------------------------------------------------------

def extract_raw_session_features(val_loader, feature_subset=None) -> Dict[int, np.ndarray]:
    """
    Concatenate raw input_features per session (no encoder). Used as a control
    for B1/B2: tells us how much of the cross-session alignment is already in
    the input vs. introduced/preserved by the model.

    Returns {day_idx: ndarray (T_total, n_features)}.
    """
    chunks: Dict[int, List[np.ndarray]] = {}
    for batch in val_loader:
        feats = batch["input_features"].numpy()           # (B, T, C)
        days = batch["day_indicies"].numpy()
        n_t = batch["n_time_steps"].numpy()
        for i in range(feats.shape[0]):
            L = int(n_t[i])
            x = feats[i, :L, :]
            if feature_subset is not None:
                x = x[:, feature_subset]
            chunks.setdefault(int(days[i]), []).append(x.astype(np.float32, copy=False))
    return {d: np.concatenate(parts, axis=0) for d, parts in chunks.items() if parts}


# ---------------------------------------------------------------------------
# Reproducible random init (same transformer weights for T12 and T15)
# ---------------------------------------------------------------------------

def build_paired_random_decoders(
    n_days_a: int,
    n_days_b: int,
    seed: int = 42,
    arch: Optional[dict] = None,
) -> Tuple[TransformerDecoder, TransformerDecoder]:
    """
    Build two random-init TransformerDecoders that share identical transformer +
    patch_embed + final_norm + head + out weights, differing only in the
    day-specific read-ins (different n_days).

    Approach: build A with the seed, then build B with the seed (so its random
    state matches up to the day-layer creation), and finally COPY all
    non-day parameters from A into B. This guarantees the only difference is
    the per-participant read-in, which is the right control for the
    Phase B2 random_init condition.
    """
    torch.manual_seed(seed)
    model_a = build_decoder(n_days=n_days_a, arch=arch)

    torch.manual_seed(seed)
    model_b = build_decoder(n_days=n_days_b, arch=arch)

    # Overwrite shared (non-day) weights of B with A's so they match exactly.
    a_state = model_a.state_dict()
    b_state = model_b.state_dict()
    for k, v in a_state.items():
        if k in ("day_weights", "day_biases"):
            continue
        if k in b_state and b_state[k].shape == v.shape:
            b_state[k] = v.clone()
    model_b.load_state_dict(b_state, strict=True)
    return model_a, model_b
