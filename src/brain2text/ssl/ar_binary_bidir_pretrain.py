#!/usr/bin/env python3
"""
AR Binary SSL — Bidirectional, dual-loss ablation
=====================================================

Architectural ablation of the dual AR binary SSL (ar_binary_pretrain.py).
Same channel-masking recipe and same dual loss (ar_visible + ar_hidden), but
the transformer uses bidirectional self-attention (is_causal=False).

Closes the attention x loss matrix:

    Causal     + dual           = Technique 4 (winning configuration)
    Causal     + ar_hidden only = (no tiene sentido sin causalidad)
    Bidir      + ar_hidden only = ablation 4-ab (degrades downstream)
    Bidir      + dual           = this ablation (4-ab2)

Hypothesis: under bidirectional attention, ar_visible (predicting patch t+1
from patch t) is trivially solvable via forward-looking attention
toward position t+1 itself; the model learns to copy the next
patch en lugar de inferir estructura predictiva. Si esto ocurre,
ar_visible_loss should drop quickly to ~0 while ar_hidden_loss
stays high, and the encoder should degrade FT as badly as 4-ab or worse.

Shared weights (blocks.* + final_norm.*) transfer to finetuning identically
to the other transformer SSL pretraining models.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain2text.models.ssl_transformer import (
    TransformerBlock,
    PatchEmbedding,
)
from brain2text.ssl.ar_binary_pretrain import ARBinaryOutputHead


class ARBinaryBidirSSLTransformerEncoder(nn.Module):
    """
    Bidirectional AR-binary SSL with the dual BCE loss (ar_visible +
    ar_hidden). Identical to ARBinarySSLTransformerEncoder except
    is_causal=False in every TransformerBlock.
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

        # ---- Shared Transformer (BIDIRECTIONAL) ----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                is_causal=False,
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
        Bidirectional AR Binary SSL forward pass with dual BCE loss.

        Args:
            x: (B, T, C) neural data (z-scored, continuous)
            subject_id: Subject identifier
            return_loss: If True, compute and return losses

        Returns:
            dict with:
                'loss': total BCE loss (ar_visible + ar_hidden)
                'ar_visible_loss': BCE for next-step visible channel prediction
                'ar_hidden_loss':  BCE for same-step hidden channel prediction
                'latent': (B, n_patches, embed_dim) transformer output
                'accuracy': fraction of correct binary predictions
                'spike_fraction': fraction of bins with spikes in input
        """
        B, T, C = x.shape
        n_patches = T // self.patch_size
        key = self._get_subject_key(subject_id)

        # 1. Binarize
        x_binary = (x > 0).float()

        # 2. Channel masking (same mask for all samples in batch)
        n_hidden = max(1, int(self.channel_mask_ratio * C))
        perm = torch.randperm(C, device=x.device)
        hidden_idx = perm[:n_hidden].sort().values
        visible_idx = perm[n_hidden:].sort().values

        # 3. Zero hidden channels in input
        x_input = x_binary.clone()
        x_input[:, :, hidden_idx] = 0.0

        # 4. Patch embedding
        tokens = self.patch_embeds[key](x_input)

        # 5. Bidirectional transformer
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)

        # 6. Output head → logits for all channels
        logits_flat = self.output_heads[key](tokens)
        logits = logits_flat.reshape(B, n_patches, self.patch_size, C)

        result = {"latent": tokens}

        if return_loss:
            target = x_binary.reshape(B, n_patches, self.patch_size, C)

            if n_patches > 1:
                ar_visible_loss = F.binary_cross_entropy_with_logits(
                    logits[:, :-1, :, visible_idx],
                    target[:, 1:, :, visible_idx],
                )
            else:
                ar_visible_loss = torch.tensor(0.0, device=x.device)

            ar_hidden_loss = F.binary_cross_entropy_with_logits(
                logits[:, :, :, hidden_idx],
                target[:, :, :, hidden_idx],
            )

            result["loss"] = ar_visible_loss + ar_hidden_loss
            result["ar_visible_loss"] = ar_visible_loss
            result["ar_hidden_loss"] = ar_hidden_loss

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                result["accuracy"] = (preds == target).float().mean().item()
                result["spike_fraction"] = x_binary.mean().item()

        return result


def build_ar_binary_bidir_ssl_model(
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
) -> ARBinaryBidirSSLTransformerEncoder:
    model = ARBinaryBidirSSLTransformerEncoder(
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

    print(f"AR Binary Bidirectional (dual loss) SSL Transformer:")
    print(f"  Shared params:  {model.n_shared_params:,}")
    print(f"  Subject params: {model.n_subject_params:,}")
    print(f"  Total params:   {model.n_shared_params + model.n_subject_params:,}")
    print(f"  Subjects: {list(subject_channels.keys())}")
    print(f"  Channel mask ratio: {channel_mask_ratio}")
    return model
