#!/usr/bin/env python3
"""
AR Binary SSL Pretraining — SpikeGPT-style
=============================================

Autoregressive pretraining with binary cross-entropy loss on binarized spikes.
Inspired by the SpikeGPT approach (spike-data-wrapper/spike_gpt_allen_ibl.py).

Key differences from masked reconstruction (BIT-style):
  - BCE loss on binarized spikes instead of MSE on continuous z-scored data
  - Channel masking (30% hidden) instead of temporal masking
  - Causal attention for next-step prediction
  - Dual loss: ar_visible (next-step) + ar_hidden (same-step cross-channel)

Architecture:
  - Input: (B, T, C) z-scored → binarize (x > 0) → mask channels → patch
  - PatchEmbed[subject]: (B, n_patches, C * patch_size) → (B, n_patches, embed_dim)
  - Transformer blocks (causal attention) × depth
  - OutputHead[subject]: (B, n_patches, embed_dim) → (B, n_patches, C * patch_size)
  - Sigmoid → BCE loss against binarized targets

Shared weights (blocks.* + final_norm.*) transfer to finetuning identically
to the masked/causal pretraining models.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain2text.models.ssl_transformer import (
    TransformerBlock,
    PatchEmbedding,
)


class ARBinaryOutputHead(nn.Module):
    """Subject-specific output: embed_dim → C * patch_size (logits for BCE)."""

    def __init__(self, n_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_channels * patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, embed_dim)
        Returns:
            logits: (B, n_patches, C * patch_size)
        """
        return self.proj(x)


class ARBinarySSLTransformerEncoder(nn.Module):
    """
    SpikeGPT-style AR pretraining with BCE on binarized spikes.

    Forward pass:
      1. Binarize input: (x > 0).float()
      2. Channel masking: zero out channel_mask_ratio of channels (random per batch)
      3. Patch embedding → transformer (causal) → output head
      4. Dual BCE loss:
         - ar_visible: predict NEXT patch's visible channels
         - ar_hidden: predict SAME patch's hidden channels
    """

    def __init__(
        self,
        embed_dim: int = 384,
        n_heads: int = 6,
        head_dim: Optional[int] = None,
        depth: int = 7,
        ff_dim: Optional[int] = None,
        dropout: float = 0.2,
        attn_dropout: float = 0.4,
        patch_size: int = 5,
        channel_mask_ratio: float = 0.3,
        subject_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.channel_mask_ratio = channel_mask_ratio

        # ---- Shared Transformer (CAUSAL) ----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                is_causal=True,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # ---- Subject-Specific Layers ----
        self.patch_embeds = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()

        if subject_channels:
            for subject_id, n_channels in subject_channels.items():
                self.register_subject(subject_id, n_channels)

    def register_subject(self, subject_id: str, n_channels: int):
        """Register a new subject with its channel count."""
        key = self._get_subject_key(subject_id)
        if key not in self.patch_embeds:
            self.patch_embeds[key] = PatchEmbedding(
                n_channels, self.patch_size, self.embed_dim
            )
            self.output_heads[key] = ARBinaryOutputHead(
                n_channels, self.patch_size, self.embed_dim
            )

    def _get_subject_key(self, subject_id: str) -> str:
        return subject_id.replace(".", "_").replace("-", "_")

    @property
    def n_shared_params(self) -> int:
        return (sum(p.numel() for p in self.blocks.parameters())
                + sum(p.numel() for p in self.final_norm.parameters()))

    @property
    def n_subject_params(self) -> int:
        return (sum(p.numel() for p in self.patch_embeds.parameters())
                + sum(p.numel() for p in self.output_heads.parameters()))

    def get_encoder_state(self) -> dict:
        """Get shared encoder weights for transfer to finetuning.
        Same keys as SSLTransformerEncoder: blocks.* and final_norm.*"""
        state = {}
        for name, param in self.named_parameters():
            if name.startswith("blocks.") or name.startswith("final_norm."):
                state[name] = param.data
        return state

    def forward(
        self,
        x: torch.Tensor,
        subject_id: str,
        return_loss: bool = True,
    ) -> dict:
        """
        AR Binary SSL forward pass.

        Args:
            x: (B, T, C) neural data (z-scored, continuous values)
            subject_id: Subject identifier
            return_loss: If True, compute and return losses

        Returns:
            dict with:
                'loss': total BCE loss (ar_visible + ar_hidden)
                'ar_visible_loss': BCE for next-step visible channel prediction
                'ar_hidden_loss': BCE for same-step hidden channel prediction
                'latent': (B, n_patches, embed_dim) transformer output
                'accuracy': fraction of correct binary predictions
                'spike_fraction': fraction of bins with spikes in input
        """
        B, T, C = x.shape
        n_patches = T // self.patch_size
        key = self._get_subject_key(subject_id)

        # 1. Binarize: spike present = 1, no spike = 0
        x_binary = (x > 0).float()

        # 2. Channel masking: random subset hidden (same mask for all samples in batch)
        n_hidden = max(1, int(self.channel_mask_ratio * C))
        perm = torch.randperm(C, device=x.device)
        hidden_idx = perm[:n_hidden].sort().values
        visible_idx = perm[n_hidden:].sort().values

        # 3. Zero out hidden channels in input
        x_input = x_binary.clone()
        x_input[:, :, hidden_idx] = 0.0

        # 4. Patch embedding
        tokens = self.patch_embeds[key](x_input)  # (B, n_patches, embed_dim)

        # 5. Causal transformer
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)  # (B, n_patches, embed_dim)

        # 6. Output head → logits for all channels
        logits_flat = self.output_heads[key](tokens)  # (B, n_patches, C * patch_size)
        # Reshape to (B, n_patches, patch_size, C) for per-channel loss
        logits = logits_flat.reshape(B, n_patches, self.patch_size, C)

        result = {"latent": tokens}

        if return_loss:
            # Target: binarized spikes in patch form
            target = x_binary.reshape(B, n_patches, self.patch_size, C)

            # ar_visible: from patch t, predict patch t+1's visible channels
            if n_patches > 1:
                ar_visible_loss = F.binary_cross_entropy_with_logits(
                    logits[:, :-1, :, visible_idx],   # (B, P-1, ps, n_vis)
                    target[:, 1:, :, visible_idx],
                )
            else:
                ar_visible_loss = torch.tensor(0.0, device=x.device)

            # ar_hidden: from patch t, predict patch t's hidden channels
            # (model never saw these — must infer from visible channels via transformer)
            ar_hidden_loss = F.binary_cross_entropy_with_logits(
                logits[:, :, :, hidden_idx],   # (B, P, ps, n_hid)
                target[:, :, :, hidden_idx],
            )

            result["loss"] = ar_visible_loss + ar_hidden_loss
            result["ar_visible_loss"] = ar_visible_loss
            result["ar_hidden_loss"] = ar_hidden_loss

            # Diagnostic metrics (no grad needed)
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                result["accuracy"] = (preds == target).float().mean().item()
                result["spike_fraction"] = x_binary.mean().item()

        return result


def build_ar_binary_ssl_model(
    subject_channels: Dict[str, int],
    embed_dim: int = 384,
    n_heads: int = 6,
    head_dim: Optional[int] = None,
    depth: int = 7,
    ff_dim: Optional[int] = None,
    patch_size: int = 5,
    channel_mask_ratio: float = 0.3,
    dropout: float = 0.2,
    attn_dropout: float = 0.4,
) -> ARBinarySSLTransformerEncoder:
    """Build AR Binary SSL model with all subjects registered."""
    model = ARBinarySSLTransformerEncoder(
        embed_dim=embed_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        depth=depth,
        ff_dim=ff_dim,
        dropout=dropout,
        attn_dropout=attn_dropout,
        patch_size=patch_size,
        channel_mask_ratio=channel_mask_ratio,
        subject_channels=subject_channels,
    )

    print(f"AR Binary SSL Transformer:")
    print(f"  Shared params:  {model.n_shared_params:,}")
    print(f"  Subject params: {model.n_subject_params:,}")
    print(f"  Total params:   {model.n_shared_params + model.n_subject_params:,}")
    print(f"  Subjects: {list(subject_channels.keys())}")
    print(f"  Channel mask ratio: {channel_mask_ratio}")
    return model
