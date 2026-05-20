#!/usr/bin/env python3
"""
GRU + Frozen SSL Transformer Decoder
=====================================

Combines a frozen SSL Transformer as a parallel feature extractor with the
standard GRU decoder. The SSL encoder extracts fixed representations that are
concatenated with the raw (day-affine) features before GRU processing.

Architecture:
  Input (B, T, 512) → Day-specific affine
    Branch 1: raw features          → (B, T, 512)
    Branch 2: SSL Transformer (frozen) → (B, T, 384)
  → Concat → (B, T, 896)
  → Patch (size=14, stride=4) → GRU (5 layers, 768 units) → head → Linear → logits

SSL Transformer details:
  - 7 TransformerBlocks, embed_dim=384, 6 heads (same as pretraining)
  - Internal patch_size=5, stride=5 (same as pretraining)
  - Only the first 256 channels (threshold crossings) are fed to the SSL encoder,
    matching the pretraining setup (T15 subject, 256-channel TX array)
  - blocks.*, final_norm.*, and patch_embed are all loaded from the checkpoint
  - Entire SSL subnetwork is frozen (requires_grad=False) and kept in eval mode
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain2text.models.ssl_transformer import TransformerBlock

from brain2text.models.gru import ResidualFFNBlock, speckle_mask

logger = logging.getLogger(__name__)


class GRUWithSSLDecoder(nn.Module):
    """
    GRU decoder augmented with a frozen SSL Transformer feature extractor.

    Same forward signature as GRUDecoder:
        forward(features, day_indicies) -> logits  # (B, T', n_classes)
    """

    # SSL Transformer hyperparameters (must match the pretraining config)
    _SSL_EMBED_DIM = 384
    _SSL_N_HEADS = 6
    _SSL_HEAD_DIM = 512   # BIT Table 10: must match ssl_pretrain.yaml head_dim
    _SSL_N_BLOCKS = 7
    _SSL_PATCH_SIZE = 5
    _SSL_PATCH_STRIDE = 5
    # Pretraining used 256-channel TX data (threshold crossings only).
    # We slice x[:, :, :_SSL_N_CHANNELS] before feeding the frozen encoder.
    _SSL_N_CHANNELS = 256

    def __init__(
        self,
        neural_dim: int = 512,
        n_units: int = 768,
        n_days: int = 1,
        n_classes: int = 41,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_layers: int = 5,
        patch_size: int = 14,
        patch_stride: int = 4,
        # Post-GRU head
        head_type: str = "none",
        head_num_blocks: int = 0,
        head_norm: str = "none",
        head_dropout: float = 0.0,
        head_activation: str = "gelu",
        # Speckled masking
        input_speckle_p: float = 0.0,
        input_speckle_mode: str = "feature",
        # SSL checkpoint path
        ssl_checkpoint: Optional[str] = None,
    ):
        super().__init__()

        self.neural_dim = int(neural_dim)
        self.n_units = int(n_units)
        self.n_days = int(n_days)
        self.n_classes = int(n_classes)
        self.n_layers = int(n_layers)
        self.patch_size = int(patch_size)
        self.patch_stride = int(patch_stride)
        self.rnn_dropout = float(rnn_dropout)
        self.input_dropout = float(input_dropout)
        self.head_type = str(head_type)
        self.head_num_blocks = int(head_num_blocks)
        self.head_norm = str(head_norm)
        self.head_dropout = float(head_dropout)
        self.head_activation = str(head_activation)
        self.input_speckle_p = float(input_speckle_p)
        self.input_speckle_mode = str(input_speckle_mode)

        # ---- Day-specific affine (identical to GRUDecoder) ----
        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.Parameter(
            torch.eye(self.neural_dim).unsqueeze(0).repeat(self.n_days, 1, 1)
        )  # (n_days, D, D)
        self.day_biases = nn.Parameter(torch.zeros(self.n_days, self.neural_dim))  # (n_days, D)
        self.day_layer_dropout = nn.Dropout(self.input_dropout)

        # ---- SSL Transformer (frozen feature extractor) ----
        # Input to patch_embed is 256 channels (threshold crossings only),
        # matching the pretraining checkpoint: 256 * patch_size = 256*5 = 1280.
        ssl_patch_input_dim = self._SSL_N_CHANNELS * self._SSL_PATCH_SIZE
        self.ssl_patch_embed = nn.Sequential(
            nn.LayerNorm(ssl_patch_input_dim),
            nn.Linear(ssl_patch_input_dim, self._SSL_EMBED_DIM),
            nn.LayerNorm(self._SSL_EMBED_DIM),
        )

        self.ssl_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self._SSL_EMBED_DIM,
                n_heads=self._SSL_N_HEADS,
                head_dim=self._SSL_HEAD_DIM,  # must match SSL checkpoint
                ff_dim=None,
                dropout=0.0,
                attn_dropout=0.0,
            )
            for _ in range(self._SSL_N_BLOCKS)
        ])
        self.ssl_final_norm = nn.LayerNorm(self._SSL_EMBED_DIM)

        # Load blocks.* and final_norm.* from checkpoint
        if ssl_checkpoint:
            self._load_ssl_weights(ssl_checkpoint)

        # Freeze the entire SSL subnetwork
        for param in self.ssl_patch_embed.parameters():
            param.requires_grad = False
        for param in self.ssl_blocks.parameters():
            param.requires_grad = False
        for param in self.ssl_final_norm.parameters():
            param.requires_grad = False

        # ---- GRU ----
        combined_dim = self.neural_dim + self._SSL_EMBED_DIM  # 512 + 384 = 896
        gru_input_size = combined_dim * self.patch_size if self.patch_size > 0 else combined_dim

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Learnable initial hidden state
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

        # ---- Optional post-GRU head ----
        ht = self.head_type.lower()
        if ht == "none" or self.head_num_blocks <= 0:
            self.head = nn.Identity()
        elif ht in ("resffn", "ffn"):
            self.head = nn.Sequential(*[
                ResidualFFNBlock(
                    d=self.n_units,
                    norm_type=self.head_norm,
                    dropout=self.head_dropout,
                    activation=self.head_activation,
                )
                for _ in range(self.head_num_blocks)
            ])
        else:
            raise ValueError(f"Unknown head_type={self.head_type!r}. Use: none, resffn.")

        # ---- Output projection ----
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_ssl_weights(self, checkpoint_path: str) -> None:
        """
        Load shared transformer weights from an SSL checkpoint.

        Maps checkpoint keys:
          blocks.*                      → ssl_blocks.*
          final_norm.*                  → ssl_final_norm.*
          patch_embeds.<subject>.proj.* → ssl_patch_embed.*

        For patch_embed, prefers the "t15" subject (256-ch TX array); falls back
        to the first checkpoint subject whose Linear weight matches shape
        (embed_dim, 256 * patch_size).  Primate subjects have different channel
        counts and are skipped.
        """
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            logger.warning(
                f"SSL checkpoint not found: {checkpoint_path}. "
                "Transformer weights will use random initialisation."
            )
            return

        logger.info(f"Loading SSL transformer weights from: {checkpoint_path}")
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

        # --- blocks.* and final_norm.* ---
        for ssl_key, ssl_param in state.items():
            if ssl_key.startswith("blocks."):
                my_key = "ssl_" + ssl_key       # blocks.X.* → ssl_blocks.X.*
            elif ssl_key.startswith("final_norm."):
                my_key = "ssl_" + ssl_key       # final_norm.* → ssl_final_norm.*
            else:
                continue

            if my_key in my_state and my_state[my_key].shape == ssl_param.shape:
                my_state[my_key] = ssl_param
                loaded += 1
            else:
                skipped += 1

        # --- patch_embed: find a subject with 256-channel compatible weights ---
        subjects = sorted({
            k.split(".")[1]
            for k in state.keys()
            if k.startswith("patch_embeds.")
        })

        subject_key = None
        # Prefer t15 (case-insensitive)
        for s in subjects:
            if s.lower() == "t15":
                subject_key = s
                break
        # Fallback: first subject whose Linear weight matches (384, 1280)
        if subject_key is None:
            expected_shape = (self._SSL_EMBED_DIM, self._SSL_N_CHANNELS * self._SSL_PATCH_SIZE)
            for s in subjects:
                w_key = f"patch_embeds.{s}.proj.1.weight"
                if w_key in state and tuple(state[w_key].shape) == expected_shape:
                    subject_key = s
                    break

        if subject_key is not None:
            prefix = f"patch_embeds.{subject_key}.proj."
            pe_loaded = 0
            pe_skipped = 0
            for ssl_key, ssl_param in state.items():
                if not ssl_key.startswith(prefix):
                    continue
                # patch_embeds.<subj>.proj.N.attr → ssl_patch_embed.N.attr
                my_key = "ssl_patch_embed." + ssl_key[len(prefix):]
                if my_key in my_state and my_state[my_key].shape == ssl_param.shape:
                    my_state[my_key] = ssl_param
                    pe_loaded += 1
                else:
                    pe_skipped += 1
            loaded += pe_loaded
            skipped += pe_skipped
            logger.info(
                f"  SSL patch_embed loaded from subject: {subject_key!r} "
                f"({pe_loaded} tensors, {pe_skipped} skipped)"
            )
        else:
            logger.warning(
                "  No compatible patch_embeds found in checkpoint "
                f"(available subjects: {subjects}). ssl_patch_embed uses random init."
            )

        self.load_state_dict(my_state, strict=True)
        logger.info(f"  SSL total: {loaded} tensors loaded, {skipped} skipped")

    # ------------------------------------------------------------------
    # Keep SSL subnetwork in eval mode at all times
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl_patch_embed.eval()
        for block in self.ssl_blocks:
            block.eval()
        self.ssl_final_norm.eval()
        return self

    # ------------------------------------------------------------------
    # SSL feature extraction
    # ------------------------------------------------------------------

    def _ssl_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frozen SSL features from x.

        Args:
            x: (B, T, 512) — post day-affine activations, detached
        Returns:
            (B, T, 384) — interpolated back to the original temporal resolution
        """
        B, T, _ = x.shape
        ps = self._SSL_PATCH_SIZE
        st = self._SSL_PATCH_STRIDE

        # Slice to threshold-crossing channels only (matches pretraining)
        x_ssl = x[:, :, : self._SSL_N_CHANNELS]            # (B, T, 256)
        C = self._SSL_N_CHANNELS

        # Patch extraction: (B, T, 256) → (B, n_patches, ps*256)
        x_p = x_ssl.unfold(1, ps, st)                      # (B, n_patches, 256, ps)
        n_patches = x_p.shape[1]
        x_p = x_p.permute(0, 1, 3, 2).contiguous()        # (B, n_patches, ps, 256)
        x_p = x_p.view(B, n_patches, ps * C)               # (B, n_patches, ps*256)

        # Patch embed → transformer → norm
        tokens = self.ssl_patch_embed(x_p)                 # (B, n_patches, 384)
        for block in self.ssl_blocks:
            tokens = block(tokens)
        tokens = self.ssl_final_norm(tokens)               # (B, n_patches, 384)

        # Interpolate from patch resolution back to T timesteps
        tokens = tokens.permute(0, 2, 1)                   # (B, 384, n_patches)
        tokens = F.interpolate(tokens, size=T, mode="linear", align_corners=False)
        tokens = tokens.permute(0, 2, 1)                   # (B, T, 384)

        return tokens

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor, day_indicies: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:     (B, T, 512) neural features
            day_indicies: (B,) session index per sample
        Returns:
            logits: (B, T', n_classes)
        """
        B, T, _ = features.shape

        # ---- Day-specific affine ----
        day_ids = day_indicies.view(-1).long()
        W = self.day_weights.index_select(0, day_ids)              # (B, D, D)
        b = self.day_biases.index_select(0, day_ids).unsqueeze(1)  # (B, 1, D)
        x = torch.einsum("btd,bdk->btk", features, W) + b
        x = self.day_layer_activation(x)
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # ---- Branch 1: raw features ----
        raw = x  # (B, T, 512)

        # ---- Branch 2: frozen SSL Transformer ----
        # Detach to prevent any gradient from flowing through the SSL path
        ssl_feats = self._ssl_encode(x.detach())  # (B, T, 384)

        # ---- Concatenate branches ----
        x = torch.cat([raw, ssl_feats], dim=-1)   # (B, T, 896)

        # ---- Patch extraction (size=14, stride=4) ----
        if self.patch_size > 0:
            ps = self.patch_size
            st = self.patch_stride
            x = x.unfold(1, ps, st)                        # (B, T', 896, ps)
            x = x.permute(0, 1, 3, 2).contiguous()         # (B, T', ps, 896)
            x = x.view(B, x.shape[1], ps * x.shape[3])     # (B, T', ps*896)

        # ---- Speckled masking (training only) ----
        if self.training and self.input_speckle_p > 0:
            x = speckle_mask(x, self.input_speckle_p, self.input_speckle_mode)

        # ---- GRU ----
        states = self.h0.expand(self.n_layers, B, self.n_units).contiguous()
        output, _ = self.gru(x, states)            # (B, T', n_units)

        # ---- Optional post-GRU head ----
        output = self.head(output)

        # ---- Logits ----
        return self.out(output)                    # (B, T', n_classes)
