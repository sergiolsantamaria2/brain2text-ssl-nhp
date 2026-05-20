#!/usr/bin/env python3
"""
GRU SSL Pretraining Model — Causal Next-Step Prediction
=========================================================

Architecture:
    Input (B, T, C_subject)
    → Patch (size=14, stride=4) → (B, n_patches, C_subject × 14)
    → SubjectInputProj[sid]: Linear(C_subject × 14, gru_input_size)
    → GRU: 5 layers, 768 hidden, unidirectional (causal)
    → SubjectPredHead[sid]: Linear(768, K × C_subject × 14)
    → MSE loss vs actual future patches

Design:
    - GRU layers + h0 are SHARED across all subjects → transfer to finetuning
    - SubjectInputProj handles different channel counts per subject
    - SubjectPredHead predicts K future patches in the subject's native space
    - gru_input_size = 7168 (= 512 × 14) matches finetuning GRU input exactly

Transfer to finetuning:
    Only gru.* and h0 weights transfer. In finetuning, the day-specific affine
    (512→512) + patching (512×14=7168) replaces SubjectInputProj, and the CTC
    head replaces SubjectPredHead. The GRU sees the same input dimensionality.

Why K=3 predict steps:
    With patch_size=14 and stride=4, consecutive patches overlap by 10 bins.
    Predicting only 1 step ahead would be ~71% overlap (too easy). At K=3,
    the target is 12 bins ahead — only 2/14 bins overlap with the current
    patch, forcing the GRU to learn genuine temporal dynamics.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUSSLPretrainModel(nn.Module):
    """GRU model for causal SSL pretraining with next-step prediction."""

    def __init__(
        self,
        gru_input_size: int = 7168,
        n_units: int = 768,
        n_layers: int = 5,
        rnn_dropout: float = 0.3,
        patch_size: int = 14,
        patch_stride: int = 4,
        n_predict_steps: int = 3,
        subject_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.gru_input_size = gru_input_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.n_predict_steps = n_predict_steps

        # ---- Shared GRU (transfers to finetuning) ----
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=n_units,
            num_layers=n_layers,
            dropout=rnn_dropout if n_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Learnable initial hidden state (also transfers)
        self.h0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, 1, n_units))
        )

        # ---- Subject-specific layers (NOT transferred) ----
        self.input_projs = nn.ModuleDict()
        self.pred_heads = nn.ModuleDict()

        if subject_channels:
            for subject_id, n_channels in subject_channels.items():
                self.register_subject(subject_id, n_channels)

    def register_subject(self, subject_id: str, n_channels: int):
        """Register a new subject with per-subject projection and prediction head."""
        key = self._sanitize_key(subject_id)
        if key not in self.input_projs:
            patch_dim = n_channels * self.patch_size
            self.input_projs[key] = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, self.gru_input_size),
            )
            self.pred_heads[key] = nn.Sequential(
                nn.LayerNorm(self.n_units),
                nn.Linear(self.n_units, self.n_predict_steps * patch_dim),
            )

    def _sanitize_key(self, subject_id: str) -> str:
        return subject_id.replace(".", "_").replace("-", "_")

    @property
    def n_shared_params(self) -> int:
        n = sum(p.numel() for p in self.gru.parameters())
        n += self.h0.numel()
        return n

    @property
    def n_subject_params(self) -> int:
        n = sum(p.numel() for p in self.input_projs.parameters())
        n += sum(p.numel() for p in self.pred_heads.parameters())
        return n

    def forward(self, x: torch.Tensor, subject_id: str) -> dict:
        """
        Causal SSL forward: patch → project → GRU → predict future patches.

        Args:
            x: (B, T, C) raw neural data
            subject_id: Subject identifier

        Returns:
            dict with loss, predictions, raw_patches
        """
        B, T, C = x.shape
        key = self._sanitize_key(subject_id)
        patch_dim = C * self.patch_size

        # ---- Patching (same as finetuning GRU) ----
        x_unf = x.unfold(1, self.patch_size, self.patch_stride)  # (B, n_patches, C, ps)
        n_patches = x_unf.shape[1]
        x_unf = x_unf.permute(0, 1, 3, 2).contiguous()  # (B, n_patches, ps, C)
        raw_patches = x_unf.reshape(B, n_patches, patch_dim)  # (B, n_patches, ps*C)

        # ---- Subject projection → shared GRU ----
        gru_input = self.input_projs[key](raw_patches)  # (B, n_patches, gru_input_size)
        h0 = self.h0.expand(self.n_layers, B, self.n_units).contiguous()
        gru_output, _ = self.gru(gru_input, h0)  # (B, n_patches, n_units)

        # ---- Prediction head ----
        pred_flat = self.pred_heads[key](gru_output)
        predictions = pred_flat.reshape(B, n_patches, self.n_predict_steps, patch_dim)

        # ---- Loss: MSE on K future patches ----
        K = self.n_predict_steps
        total_loss = torch.tensor(0.0, device=x.device)
        total_count = 0

        for step in range(K):
            max_t = n_patches - step - 1
            if max_t <= 0:
                continue
            pred = predictions[:, :max_t, step, :]
            target = raw_patches[:, step + 1 : step + 1 + max_t, :]
            total_loss = total_loss + F.mse_loss(pred, target, reduction="sum")
            total_count += pred.numel()

        loss = total_loss / max(total_count, 1)

        return {
            "loss": loss,
            "predictions": predictions,
            "raw_patches": raw_patches,
        }

    def get_gru_state(self) -> dict:
        """
        Extract GRU + h0 weights for transfer to finetuning.

        Returns dict with keys: 'gru.weight_ih_l0', 'gru.weight_hh_l0', ..., 'h0'
        """
        state = {}
        for name, param in self.named_parameters():
            if name.startswith("gru.") or name == "h0":
                state[name] = param.data.clone()
        return state


# ==============================================================================
# R² metric
# ==============================================================================

def compute_r2_gru(
    predictions: torch.Tensor,
    raw_patches: torch.Tensor,
    n_predict_steps: int,
) -> float:
    """
    R² for causal prediction quality.

    Args:
        predictions: (B, n_patches, K, patch_dim)
        raw_patches: (B, n_patches, patch_dim)
        n_predict_steps: K
    """
    with torch.no_grad():
        all_pred = []
        all_tgt = []
        n_patches = raw_patches.shape[1]

        for step in range(n_predict_steps):
            max_t = n_patches - step - 1
            if max_t <= 0:
                continue
            all_pred.append(predictions[:, :max_t, step, :].reshape(-1))
            all_tgt.append(raw_patches[:, step + 1 : step + 1 + max_t, :].reshape(-1))

        if not all_pred:
            return 0.0

        pred_flat = torch.cat(all_pred).float()
        tgt_flat = torch.cat(all_tgt).float()

        ss_res = ((pred_flat - tgt_flat) ** 2).sum()
        ss_tot = ((tgt_flat - tgt_flat.mean()) ** 2).sum()
        return (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()


# ==============================================================================
# Factory
# ==============================================================================

def build_gru_ssl_model(
    subject_channels: Dict[str, int],
    gru_input_size: int = 7168,
    n_units: int = 768,
    n_layers: int = 5,
    rnn_dropout: float = 0.3,
    patch_size: int = 14,
    patch_stride: int = 4,
    n_predict_steps: int = 3,
) -> GRUSSLPretrainModel:
    """Build GRU SSL pretraining model with all subjects registered."""
    model = GRUSSLPretrainModel(
        gru_input_size=gru_input_size,
        n_units=n_units,
        n_layers=n_layers,
        rnn_dropout=rnn_dropout,
        patch_size=patch_size,
        patch_stride=patch_stride,
        n_predict_steps=n_predict_steps,
        subject_channels=subject_channels,
    )

    print(f"GRU SSL Pretraining Model:")
    print(f"  Shared params (GRU + h0):  {model.n_shared_params:,}")
    print(f"  Subject params:            {model.n_subject_params:,}")
    print(f"  Total params:              {model.n_shared_params + model.n_subject_params:,}")
    print(f"  GRU: {n_layers} layers, {n_units} units, input={gru_input_size}")
    print(f"  Patch: size={patch_size}, stride={patch_stride}")
    print(f"  Predict: {n_predict_steps} steps ahead")
    print(f"  Subjects: {list(subject_channels.keys())}")

    return model


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == "__main__":
    subjects = {"monkey_A": 96, "monkey_B": 192, "monkey_C": 353}
    model = build_gru_ssl_model(subjects)

    for sid, n_ch in subjects.items():
        x = torch.randn(4, 500, n_ch)
        out = model(x, sid)
        r2 = compute_r2_gru(out["predictions"], out["raw_patches"], model.n_predict_steps)
        print(f"  {sid}: loss={out['loss'].item():.4f}, R²={r2:.4f}, "
              f"patches={out['raw_patches'].shape[1]}")

    print("\nGRU SSL model test passed!")
