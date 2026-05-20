#!/usr/bin/env python3
"""
Transformer Decoder for CTC Phoneme Decoding
==============================================

Wraps the SSL Transformer encoder for finetuning with CTC loss.
Designed to plug into rnn_trainer.py as a drop-in replacement for GRUDecoder.

Architecture (following BIT paper):
  Input (B, T, 512) → Day-specific linear → PatchEmbed → Transformer → Linear → Logits

Key design decisions:
  - Day-specific input layers handle session non-stationarity (same as GRU baseline)
  - PatchEmbed groups patch_size bins and projects to embed_dim
  - No masking during finetuning (BIT paper Section 3.1)
  - Output is phoneme logits via linear head
  - Can load pretrained transformer blocks from SSL checkpoint

SSL checkpoint compatibility:
  - SSL pretraining used 256 channels (TX only) with subject-specific PatchEmbed
  - Finetuning uses 512 channels (TX+SBP) with day-specific input layers
  - Only the shared transformer blocks + final_norm are transferred
  - PatchEmbed and output head are always initialized fresh

Usage:
  Set model.decoder_type: "transformer" in your YAML config.
"""

import math
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain2text.models.gru import ResidualFFNBlock, speckle_mask
from brain2text.models.ssl_transformer import generate_contiguous_mask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import TransformerBlock from the SSL module
# ---------------------------------------------------------------------------
from brain2text.models.ssl_transformer import TransformerBlock


class TransformerDecoder(nn.Module):
    """
    Transformer-based phoneme decoder compatible with rnn_trainer.py.

    Same forward signature as GRUDecoder:
        forward(features, day_indices) -> logits  # (B, T', n_classes)
    """

    def __init__(
        self,
        neural_dim: int = 512,
        n_units: int = 768,        # API compat, not used
        n_days: int = 1,
        n_classes: int = 41,
        rnn_dropout: float = 0.2,
        input_dropout: float = 0.2,
        n_layers: int = 7,
        patch_size: int = 5,
        patch_stride: int = 5,
        # Transformer-specific
        embed_dim: int = 384,
        n_heads: int = 6,
        head_dim: Optional[int] = None,  # None → embed_dim // n_heads; 512 → BIT Table 10
        ff_dim: Optional[int] = None,
        attn_dropout: float = 0.4,
        ssl_checkpoint: Optional[str] = None,
        head_type: str = "none",
        head_num_blocks: int = 0,
        head_norm: str = "layernorm",
        head_dropout: float = 0.1,
        head_activation: str = "gelu",
        input_speckle_p: float = 0.0,
        input_speckle_mode: str = "feature",
        # Time masking (SpecAugment-style contiguous span masking, BIT paper)
        time_mask_ratio: float = 0.0,
        time_mask_max_span: int = 15,
        **kwargs,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_days = n_days
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride > 0 else patch_size
        self.time_mask_ratio = float(time_mask_ratio)
        self.time_mask_max_span = int(time_mask_max_span)

        # ---- Day-specific input layers (same as GRU baseline) ----
        self.day_weights = nn.Parameter(
            torch.randn(n_days, neural_dim, neural_dim) * (1.0 / math.sqrt(neural_dim))
        )
        self.day_biases = nn.Parameter(torch.zeros(n_days, 1, neural_dim))
        self.day_layer_activation = nn.Softsign()
        self.day_layer_dropout = nn.Dropout(input_dropout)

        # ---- Patch Embedding ----
        patch_input_dim = neural_dim * patch_size
        self.patch_embed = nn.Sequential(
            nn.LayerNorm(patch_input_dim),
            nn.Linear(patch_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

       # --- NUEVO: Speckle Masking ---
        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)

        # ---- Shared Transformer Encoder ----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=rnn_dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # --- NUEVO: Post-Transformer Head ---
        ht = str(head_type).lower()
        if ht == "none" or head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=embed_dim,
                    norm_type=head_norm,
                    dropout=head_dropout,
                    activation=head_activation,
                )
                for _ in range(head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={head_type}. Use: none, resffn.")

        # ---- Phoneme classification head ----
        self.out = nn.Linear(embed_dim, n_classes)

        # ---- Load SSL pretrained weights ----
        if ssl_checkpoint:
            self._load_ssl_weights(ssl_checkpoint)

    def _load_ssl_weights(self, checkpoint_path: str):
        """
        Load pretrained transformer blocks from SSL checkpoint.
        Only loads blocks.* and final_norm.* (shared transformer).
        Does NOT load patch embeddings, mask token, or reversed patch embeddings.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(f"SSL checkpoint not found: {checkpoint_path}. Training from scratch.")
            return

        logger.info(f"Loading SSL pretrained weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "encoder_state" in ckpt:
            state = ckpt["encoder_state"]
        else:
            state = ckpt

        my_state = self.state_dict()
        loaded = 0
        skipped = 0

        for ssl_key, ssl_param in state.items():
            if not (ssl_key.startswith("blocks.") or ssl_key.startswith("final_norm.")):
                continue
            if ssl_key in my_state and my_state[ssl_key].shape == ssl_param.shape:
                my_state[ssl_key] = ssl_param
                loaded += 1
            else:
                skipped += 1

        self.load_state_dict(my_state, strict=True)
        logger.info(f"  SSL: {loaded} tensors loaded, {skipped} skipped")
        if skipped > 0:
            raise RuntimeError(
                f"SSL weight loading: {skipped} tensors had shape mismatches and were skipped! "
                "This usually means head_dim differs between SSL pretraining and finetuning. "
                "Check that ssl_pretrain.yaml and transformer_with_ssl.yaml use the same head_dim."
            )

    def forward(
        self,
        features: torch.Tensor,
        day_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features:    (B, T, 512) neural features
            day_indices: (B,) session index per sample
        Returns:
            logits: (B, T', n_classes)
        """
        B, T, C = features.shape

        # ---- Day-specific affine ----
        day_ids = day_indices.long()
        W = self.day_weights.index_select(0, day_ids)
        b = self.day_biases.index_select(0, day_ids)
        x = torch.einsum("btd,bdk->btk", features, W) + b
        x = self.day_layer_activation(x)
        x = self.day_layer_dropout(x)

        # ---- Patch extraction ----
        ps = self.patch_size
        st = self.patch_stride
        x = x.unfold(1, ps, st)                 # (B, n_patches, C, ps)
        x = x.permute(0, 1, 3, 2).contiguous()  # (B, n_patches, ps, C)
        n_patches = x.shape[1]
        x = x.reshape(B, n_patches, ps * C)      # (B, n_patches, ps*C)

        # --- NUEVO: Speckle Masking (entrenamiento) ---
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)

        # ---- Patch embed ----
        x = self.patch_embed(x)

        # ---- Time masking (training only, SpecAugment-style) ----
        if self.training and self.time_mask_ratio > 0:
            mask = generate_contiguous_mask(
                n_patches=n_patches,
                mask_ratio=self.time_mask_ratio,
                max_span=self.time_mask_max_span,
                batch_size=B,
                device=x.device,
            )  # (B, n_patches) bool, True = masked
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        # ---- Transformer ----
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # --- NUEVO: Aplicar el Post-Head ---
        x = self.head(x)

        # ---- Classification head ----
        logits = self.out(x)
        return logits
