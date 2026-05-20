#!/usr/bin/env python3
"""
Contrastive Temporal SSL Pretraining — wav2vec 2.0 / SimCLR style
====================================================================

Intra-modal contrastive learning over neural patch embeddings.

  - Positive pair  : temporally adjacent patches within the same trial
                     (delta = 1 patch = 100 ms)
  - Negative pairs : patches from other trials in the batch (any time)
                     Same-trial / different-time entries are NOT negatives;
                     they are masked out of the candidate set.
  - Loss           : symmetric InfoNCE with learnable temperature (CLIP style)

Architecture:
  - Subject-specific patch embed (same as the other SSL variants)
  - Shared transformer (bidirectional, NOT causal)
  - Shared final_norm
  - Shared MLP projector  embed_dim → 256 → 128, L2-normalized
  - Loss on the L2-normalized 128-dim projections

Shared weights (blocks.* + final_norm.*) transfer to finetuning identically
to the masked / AR-binary / causal variants.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain2text.models.ssl_transformer import (
    TransformerBlock,
    PatchEmbedding,
)


# ==============================================================================
# Projector
# ==============================================================================

class ContrastiveProjector(nn.Module):
    """Shared MLP projector: embed_dim -> hidden_dim -> out_dim, L2-normalized."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
        Returns:
            z: (B, T, out_dim), L2-normalized along last dim
        """
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x, dim=-1)


# ==============================================================================
# InfoNCE (symmetric, intra-trial temporal)
# ==============================================================================

def info_nce_temporal_loss(
    z: torch.Tensor,
    log_tau: torch.Tensor,
    tau_min: float = 0.01,
    tau_max: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Symmetric InfoNCE with intra-trial temporal positives and cross-trial negatives.

    For each anchor (trial=i, patch=t) with t in [0, P-2]:
      - positive  : (i, t+1)
      - negatives : all (j, t') with j != i  (any t')
    Same-trial entries other than the positive are MASKED OUT (excluded from
    both numerator and denominator).

    Loss is computed in both directions (anchor->positive, positive->anchor)
    and averaged (CLIP / SimCLR convention).

    Args:
        z       : (B, P, D) L2-normalized embeddings
        log_tau : scalar parameter; temperature = clamp(exp(log_tau), tau_min, tau_max)
        tau_min : lower temperature clip
        tau_max : upper temperature clip

    Returns:
        dict with:
            'loss'     : scalar InfoNCE loss
            'accuracy' : fraction of anchors whose positive is the argmax candidate
            'tau'      : current effective temperature (float)
    """
    B, P, D = z.shape
    device = z.device

    # Need at least 2 patches (positive pair) and 2 trials (negative pool)
    if P < 2 or B < 2:
        zero = z.new_zeros((), dtype=torch.float32).requires_grad_(True)
        return {"loss": zero, "accuracy": 0.0, "tau": float("nan")}

    # Force float32 throughout the InfoNCE computation. Under AMP, matmul would
    # otherwise downcast to fp16 and masked_fill(-1e9) would overflow (fp16 max
    # ~65504).
    with torch.cuda.amp.autocast(enabled=False):
        tau = log_tau.float().exp().clamp(tau_min, tau_max)

        z_flat = z.reshape(B * P, D).float()
        sim = (z_flat @ z_flat.T) / tau  # (BP, BP), fp32

        flat_idx = torch.arange(B * P, device=device)
        trial_idx = flat_idx // P
        pos_in_trial = flat_idx % P

        anchor_mask = pos_in_trial < P - 1
        anchor_indices = torch.where(anchor_mask)[0]   # B*(P-1) entries
        positive_indices = anchor_indices + 1
        n_anchors = anchor_indices.shape[0]

        anchor_trial = trial_idx[anchor_indices]
        arange_n = torch.arange(n_anchors, device=device)

        # ---- Direction 1: anchor -> positive ----
        logits1 = sim[anchor_indices]                                         # (n_anchors, BP)
        same_trial1 = trial_idx.unsqueeze(0) == anchor_trial.unsqueeze(1)     # (n_anchors, BP)
        pos_mask1 = torch.zeros_like(same_trial1)
        pos_mask1[arange_n, positive_indices] = True
        invalid1 = same_trial1 & ~pos_mask1
        logits1 = logits1.masked_fill(invalid1, -1e9)
        loss1 = F.cross_entropy(logits1, positive_indices)

        # ---- Direction 2: positive -> anchor ----
        pos_trial = trial_idx[positive_indices]   # equals anchor_trial
        logits2 = sim[positive_indices]
        same_trial2 = trial_idx.unsqueeze(0) == pos_trial.unsqueeze(1)
        pos_mask2 = torch.zeros_like(same_trial2)
        pos_mask2[arange_n, anchor_indices] = True
        invalid2 = same_trial2 & ~pos_mask2
        logits2 = logits2.masked_fill(invalid2, -1e9)
        loss2 = F.cross_entropy(logits2, anchor_indices)

        loss = 0.5 * (loss1 + loss2)

        with torch.no_grad():
            pred = logits1.argmax(dim=-1)
            accuracy = (pred == positive_indices).float().mean().item()

    return {
        "loss": loss,
        "accuracy": accuracy,
        "tau": float(tau.item()),
    }


# ==============================================================================
# Encoder
# ==============================================================================

class ContrastiveSSLTransformerEncoder(nn.Module):
    """Bidirectional transformer + shared projector + InfoNCE loss."""

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
        proj_hidden_dim: int = 256,
        proj_out_dim: int = 128,
        tau_init: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        subject_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.tau_min = tau_min
        self.tau_max = tau_max

        # ---- Shared transformer (BIDIRECTIONAL) ----
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

        # ---- Shared projector ----
        self.projector = ContrastiveProjector(embed_dim, proj_hidden_dim, proj_out_dim)

        # ---- Learnable temperature (stored as log to keep it positive) ----
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init), dtype=torch.float32))

        # ---- Subject-specific patch embeds ----
        self.patch_embeds = nn.ModuleDict()
        if subject_channels:
            for subject_id, n_channels in subject_channels.items():
                self.register_subject(subject_id, n_channels)

    def register_subject(self, subject_id: str, n_channels: int):
        key = self._get_subject_key(subject_id)
        if key not in self.patch_embeds:
            self.patch_embeds[key] = PatchEmbedding(
                n_channels, self.patch_size, self.embed_dim
            )

    def _get_subject_key(self, subject_id: str) -> str:
        return subject_id.replace(".", "_").replace("-", "_")

    @property
    def n_shared_params(self) -> int:
        n = sum(p.numel() for p in self.blocks.parameters())
        n += sum(p.numel() for p in self.final_norm.parameters())
        n += sum(p.numel() for p in self.projector.parameters())
        n += self.log_tau.numel()
        return n

    @property
    def n_subject_params(self) -> int:
        return sum(p.numel() for p in self.patch_embeds.parameters())

    def get_encoder_state(self) -> dict:
        """Return only blocks.* and final_norm.* — same contract as other SSL variants."""
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
        Args:
            x          : (B, T, C) z-scored neural data
            subject_id : subject identifier
            return_loss: if True, compute InfoNCE loss

        Returns:
            dict with 'latent', 'projection', and (if return_loss) 'loss',
            'accuracy', 'tau'.
        """
        key = self._get_subject_key(subject_id)
        tokens = self.patch_embeds[key](x)            # (B, P, embed_dim)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.final_norm(tokens)              # (B, P, embed_dim)
        z = self.projector(tokens)                    # (B, P, out_dim), L2-normalized

        result = {"latent": tokens, "projection": z}

        if return_loss:
            metrics = info_nce_temporal_loss(z, self.log_tau, self.tau_min, self.tau_max)
            result["loss"] = metrics["loss"]
            result["accuracy"] = metrics["accuracy"]
            result["tau"] = metrics["tau"]

        return result


def build_contrastive_ssl_model(
    subject_channels: Dict[str, int],
    embed_dim: int = 384,
    n_heads: int = 6,
    head_dim: Optional[int] = None,
    depth: int = 7,
    ff_dim: Optional[int] = None,
    patch_size: int = 5,
    proj_hidden_dim: int = 256,
    proj_out_dim: int = 128,
    tau_init: float = 0.1,
    tau_min: float = 0.01,
    tau_max: float = 1.0,
    dropout: float = 0.2,
    attn_dropout: float = 0.4,
) -> ContrastiveSSLTransformerEncoder:
    model = ContrastiveSSLTransformerEncoder(
        embed_dim=embed_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        depth=depth,
        ff_dim=ff_dim,
        dropout=dropout,
        attn_dropout=attn_dropout,
        patch_size=patch_size,
        proj_hidden_dim=proj_hidden_dim,
        proj_out_dim=proj_out_dim,
        tau_init=tau_init,
        tau_min=tau_min,
        tau_max=tau_max,
        subject_channels=subject_channels,
    )

    print(f"Contrastive SSL Transformer:")
    print(f"  Shared params:  {model.n_shared_params:,}")
    print(f"  Subject params: {model.n_subject_params:,}")
    print(f"  Total params:   {model.n_shared_params + model.n_subject_params:,}")
    print(f"  Subjects: {list(subject_channels.keys())}")
    print(f"  Projector: {embed_dim} -> {proj_hidden_dim} -> {proj_out_dim}")
    print(f"  tau_init={tau_init}, tau range=[{tau_min}, {tau_max}]")
    return model


# ==============================================================================
# Self-tests (Step 1d)
# ==============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print("contrastive_pretrain.py — self-tests")
    print("=" * 60)

    subjects = {"soma": 96, "T15": 256}

    # ---- Test (a)+(c): training step reduces loss & gradient flows ----
    print("\n[Test 1] Training reduces loss + grad flows to transformer")
    model = build_contrastive_ssl_model(subjects)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(100):
        x = torch.randn(8, 50, 96)  # (B=8, T=50, C=96), patch_size=5 -> P=10
        out = model(x, "soma")
        optimizer.zero_grad()
        out["loss"].backward()
        optimizer.step()
        losses.append(out["loss"].item())
    print(f"  initial loss: {losses[0]:.4f}")
    print(f"  final loss  : {losses[-1]:.4f}")
    print(f"  accuracy    : {out['accuracy']:.4f}")
    print(f"  tau         : {out['tau']:.4f}")
    assert losses[-1] < losses[0], "Loss must decrease over training"

    # ---- Test (c): grad reaches transformer blocks ----
    print("\n[Test 2] Gradient norm in transformer blocks > 0")
    model = build_contrastive_ssl_model(subjects)
    x = torch.randn(8, 50, 96)
    out = model(x, "soma")
    out["loss"].backward()
    block_grad_sq = 0.0
    for _, p in model.blocks.named_parameters():
        if p.grad is not None:
            block_grad_sq += p.grad.norm().item() ** 2
    block_grad_norm = block_grad_sq ** 0.5
    print(f"  block grad norm: {block_grad_norm:.4f}")
    assert block_grad_norm > 0, "Gradient must flow to transformer blocks"

    # ---- Test (b): random data + untrained model -> loss near log(N) ----
    print("\n[Test 3] Untrained model on random data: loss near log(N)")
    model = build_contrastive_ssl_model(subjects)
    model.eval()
    with torch.no_grad():
        x = torch.randn(8, 50, 96)
        out = model(x, "soma")
        # n_candidates per anchor = 1 (positive) + (B-1)*P = 1 + 7*10 = 71
        n_cand = 1 + (8 - 1) * 10
        log_n = math.log(n_cand)
        print(f"  loss          : {out['loss'].item():.4f}")
        print(f"  log({n_cand})       : {log_n:.4f}")
        # Tolerance ~1.5 nats: model is untrained so similarities are random
        assert abs(out["loss"].item() - log_n) < 1.5, \
            f"Untrained loss should be near log({n_cand})={log_n:.4f}, got {out['loss'].item():.4f}"

    # ---- Test 4: encoder state contract (only blocks/final_norm) ----
    print("\n[Test 4] get_encoder_state() returns only blocks.* and final_norm.*")
    state = model.get_encoder_state()
    for k in state.keys():
        assert k.startswith("blocks.") or k.startswith("final_norm."), \
            f"Unexpected key in encoder state: {k}"
    print(f"  {len(state)} tensors saved, all blocks.* / final_norm.*")

    # ---- Test 5: temperature stays in clip range under extreme log_tau ----
    print("\n[Test 5] Temperature clipping")
    model = build_contrastive_ssl_model(subjects, tau_init=0.1, tau_min=0.01, tau_max=1.0)
    model.log_tau.data = torch.tensor(math.log(10.0))   # would give tau=10.0
    out = model(torch.randn(4, 50, 96), "soma")
    print(f"  log_tau pushed high -> tau = {out['tau']:.4f}  (expected 1.0)")
    assert out["tau"] <= 1.0 + 1e-6
    model.log_tau.data = torch.tensor(math.log(0.0001))  # would give tau=1e-4
    out = model(torch.randn(4, 50, 96), "soma")
    print(f"  log_tau pushed low  -> tau = {out['tau']:.4f}  (expected 0.01)")
    assert out["tau"] >= 0.01 - 1e-6

    print("\n[OK] All contrastive_pretrain.py self-tests passed")
