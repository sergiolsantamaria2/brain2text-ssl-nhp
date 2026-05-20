#!/usr/bin/env python3
"""
BIT-style Transformer Encoder for SSL Pretraining
===================================================

Architecture (BIT paper Table 10):
  - Patch Embedding: (T, C) → group Tpatch bins → LayerNorm → Linear → LayerNorm
  - Transformer: 7 blocks, 384 embed_dim, 6 heads, RoPE, bidirectional
  - Masking: contiguous spans, mask_ratio=0.5, max_span=15 patches
  - Reconstruction: reversed patch embed → MSE loss on original data

Subject-specific components:
  - PatchEmbed (read-in): per subject, projects C×patch_size → embed_dim
  - ReversedPatchEmbed (read-out): per subject, projects embed_dim → C×patch_size
  These are stored in ModuleDicts and selected by subject_id during forward.

Shared components:
  - Transformer blocks (self-attention + FFN)
  - Learnable mask token
  - RoPE positional embeddings

For finetuning:
  - Remove masking module
  - Replace read-out with linear phoneme classifier
  - Add CTC loss
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Rotary Position Embedding (RoPE)
# ==============================================================================

class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (Su et al., 2024).
    Applied to query and key tensors in attention.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos/sin tables
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],  # (T, dim)
            self.sin_cached[:seq_len],  # (T, dim)
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, 
    k: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    q, k: (B, n_heads, T, head_dim)
    cos, sin: (T, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# Transformer Components
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional bidirectional mask."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.is_causal = is_causal
        # head_dim=None → embed_dim // n_heads (standard)
        # head_dim=512  → expanded Q/K/V projection (BIT Table 10)
        self.head_dim = head_dim if head_dim is not None else (embed_dim // n_heads)
        self.inner_dim = n_heads * self.head_dim
        
        # --- FIX: Usar self.head_dim en lugar de head_dim ---
        self.scale = self.head_dim ** -0.5
        # ----------------------------------------------------
        
        self.qkv = nn.Linear(embed_dim, 3 * self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
        # Use self.head_dim consistently here
        # RoPE (initialized with reasonable max, auto-extends)
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len=512)
        # --------------------------------------------
    
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, embed_dim)
            attn_mask: (T, T) or None. For bidirectional, this is None.
        Returns:
            out: (B, T, embed_dim)
        """
        B, T, _ = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (B, T, 3 * inner_dim)
        # Reshape to separate attention heads
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim) -> Pone primero 2, luego 0, luego 3...
        q, k, v = qkv.unbind(0)  # each (B, n_heads, T, head_dim). Separo
        
        # Apply RoPE
        cos, sin = self.rope(T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        # Use PyTorch's efficient implementation when available
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None if self.is_causal else attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=self.is_causal,
        )
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, self.inner_dim)  # (B, T, inner_dim)
        out = self.out_proj(out)
        out = self.proj_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Standard transformer FFN: Linear → GELU → Dropout → Linear → Dropout."""
    
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        self.net = nn.Sequential( #nn.Sequential encadena capas en orden
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual."""
    
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        ff_dim: Optional[int] = None,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        is_causal: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, n_heads, head_dim, attn_dropout, dropout, is_causal
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
    
    def forward(
        self, 
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attn_mask) # x + Attention(Norm(x))
        x = x + self.ffn(self.norm2(x)) # x + FFN(Norm(x))
        return x


# ==============================================================================
# Subject-Specific Patch Embedding
# ==============================================================================

class PatchEmbedding(nn.Module):
    """
    Subject-specific patch embedding module.
    
    Converts (B, T, C) → (B, T/patch_size, embed_dim)
    by grouping patch_size consecutive time bins and projecting.
    
    Architecture: LayerNorm → Linear(C × patch_size → embed_dim) → LayerNorm
    """
    
    def __init__(self, n_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.input_dim = n_channels * patch_size
        
        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) where T is divisible by patch_size
        Returns:
            patches: (B, T/patch_size, embed_dim)
        """
        B, T, C = x.shape
        assert T % self.patch_size == 0, f"T={T} not divisible by patch_size={self.patch_size}"
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"
        
        # Reshape: (B, T, C) → (B, n_patches, patch_size * C)
        n_patches = T // self.patch_size
        x = x.reshape(B, n_patches, self.patch_size * C)
        
        return self.proj(x)


class ReversedPatchEmbedding(nn.Module):
    """
    Subject-specific reversed patch embedding (decoder for reconstruction).
    
    Converts (B, n_patches, embed_dim) → (B, T, C)
    """
    
    def __init__(self, n_channels: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.output_dim = n_channels * patch_size
        
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_patches, embed_dim)
        Returns:
            reconstructed: (B, T, C)
        """
        B, n_patches, _ = x.shape
        x = self.proj(x)  # (B, n_patches, patch_size * C)
        x = x.reshape(B, n_patches * self.patch_size, self.n_channels)
        return x


# ==============================================================================
# Contiguous Span Masking
# ==============================================================================

def generate_contiguous_mask(
    n_patches: int,
    mask_ratio: float,
    max_span: int,
    batch_size: int = 1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate contiguous span masks for temporal patches.
    
    Masked patches form contiguous spans of variable length (up to max_span),
    maintaining consistent overall masking ratio.
    
    Args:
        n_patches: Number of temporal patches
        mask_ratio: Target fraction of patches to mask
        max_span: Maximum length of a contiguous mask span
        batch_size: Number of masks to generate
        device: Device for the output tensor
    
    Returns:
        mask: (batch_size, n_patches) bool tensor, True = masked
    """
    n_to_mask = int(n_patches * mask_ratio)
    masks = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        masked_count = 0
        # Keep adding spans until we reach the target
        while masked_count < n_to_mask:
            remaining = n_to_mask - masked_count
            span_len = min(
                torch.randint(1, max_span + 1, (1,)).item(),
                remaining,
            )
            # Random start position
            max_start = n_patches - span_len
            if max_start <= 0:
                start = 0
            else:
                start = torch.randint(0, max_start + 1, (1,)).item()
            
            masks[b, start : start + span_len] = True
            masked_count = masks[b].sum().item()
    
    return masks


# ==============================================================================
# Main SSL Transformer Model
# ==============================================================================

class SSLTransformerEncoder(nn.Module):
    """
    BIT-style Transformer Encoder for SSL pretraining.
    
    Architecture:
      Input (B, T, C) → PatchEmbed[subject] → (B, n_patches, embed_dim)
        → Mask some patches with learnable mask token
        → Transformer blocks × depth
        → ReversedPatchEmbed[subject] → (B, T, C)
        → MSE loss against original input
    
    The transformer (shared) learns general neural representations.
    PatchEmbed/ReversedPatchEmbed (per-subject) handle different electrode counts.
    """
    
    def __init__(
        self,
        # Transformer
        embed_dim: int = 384,
        n_heads: int = 6,
        head_dim: Optional[int] = None,  # None → embed_dim // n_heads; 512 → BIT Table 10
        depth: int = 7,
        ff_dim: Optional[int] = None,  # Default: 4 * embed_dim
        dropout: float = 0.2,
        attn_dropout: float = 0.4,
        # Patch
        patch_size: int = 5,
        # Masking
        mask_ratio: float = 0.5,
        max_mask_span: int = 15,
        # Denoising mode: no temporal masking, add noise, reconstruct clean
        denoising_noise_std: float = 0.0,
        # Channel masking mode: zero random channels, reconstruct them (MSE)
        channel_mask_ratio: float = 0.0,
        # Subject configs: {subject_id: n_channels}
        subject_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.depth = depth
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.max_mask_span = max_mask_span
        self.denoising_noise_std = denoising_noise_std
        self.channel_mask_ratio = channel_mask_ratio

        # ---- Shared Transformer ----
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                ff_dim=ff_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # ---- Subject-Specific Layers ----
        self.patch_embeds = nn.ModuleDict()
        self.reversed_patch_embeds = nn.ModuleDict()
        
        if subject_channels:
            for subject_id, n_channels in subject_channels.items():
                self.register_subject(subject_id, n_channels)
    
    def register_subject(self, subject_id: str, n_channels: int):
        """Register a new subject with its channel count."""
        # Sanitize key for ModuleDict (no dots allowed)
        key = subject_id.replace(".", "_").replace("-", "_")
        
        if key not in self.patch_embeds:
            self.patch_embeds[key] = PatchEmbedding(
                n_channels, self.patch_size, self.embed_dim
            )
            self.reversed_patch_embeds[key] = ReversedPatchEmbedding(
                n_channels, self.patch_size, self.embed_dim
            )
    
    def _get_subject_key(self, subject_id: str) -> str:
        """Convert subject_id to valid ModuleDict key."""
        return subject_id.replace(".", "_").replace("-", "_")
    
    @property
    def n_shared_params(self) -> int:
        """Count shared (non-subject-specific) parameters."""
        shared = sum(p.numel() for p in self.blocks.parameters())
        shared += sum(p.numel() for p in self.final_norm.parameters())
        shared += self.mask_token.numel()
        return shared
    
    @property
    def n_subject_params(self) -> int:
        """Count subject-specific parameters."""
        total = sum(p.numel() for p in self.patch_embeds.parameters())
        total += sum(p.numel() for p in self.reversed_patch_embeds.parameters())
        return total
    
    def encode(
        self,
        x: torch.Tensor,
        subject_id: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode neural data to latent representations.
        
        Args:
            x: (B, T, C) neural data
            subject_id: Subject identifier
            mask: (B, n_patches) bool tensor, True = masked. If None, no masking.
        
        Returns:
            latent: (B, n_patches, embed_dim)
        """
        key = self._get_subject_key(subject_id)
        
        # Patch embedding
        tokens = self.patch_embeds[key](x)  # (B, n_patches, embed_dim)
        
        # Apply masking
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(tokens)  # (B, n_patches, embed_dim)
            mask_token = self.mask_token.expand_as(tokens)
            tokens = torch.where(mask_expanded, mask_token, tokens)
        
        # Transformer
        for block in self.blocks:
            tokens = block(tokens)
        
        tokens = self.final_norm(tokens)
        
        return tokens
    
    def decode(
        self,
        latent: torch.Tensor,
        subject_id: str,
    ) -> torch.Tensor:
        """
        Decode latent representations to reconstructed neural data.
        
        Args:
            latent: (B, n_patches, embed_dim)
            subject_id: Subject identifier
        
        Returns:
            reconstructed: (B, T, C)
        """
        key = self._get_subject_key(subject_id)
        return self.reversed_patch_embeds[key](latent)
    
    def forward(
        self,
        x: torch.Tensor,
        subject_id: str,
        return_loss: bool = True,
        mask_ratio: Optional[float] = None,
    ) -> dict:
        """
        Full SSL forward pass: encode with masking → decode → MSE loss.
        
        Args:
            x: (B, T, C) neural data (z-scored)
            subject_id: Subject identifier
            return_loss: If True, compute and return MSE loss
            mask_ratio: Override default mask_ratio (set to 0 for inference)
        
        Returns:
            dict with:
                'loss': scalar MSE loss (if return_loss)
                'reconstructed': (B, T, C) reconstructed data
                'latent': (B, n_patches, embed_dim) latent representations
                'mask': (B, n_patches) mask used
        """
        B, T, C = x.shape
        n_patches = T // self.patch_size
        mr = mask_ratio if mask_ratio is not None else self.mask_ratio

        # Determine mode: denoising, channel masking, or temporal masking
        use_denoising = self.denoising_noise_std > 0 and self.training
        use_channel_mask = self.channel_mask_ratio > 0 and self.training

        x_target = x  # Always reconstruct original clean input
        x_enc = x      # Input to encoder (may be noisy or channel-masked)
        mask = None
        channel_hidden_idx = None

        if use_denoising:
            # Denoising mode: add noise, no temporal masking
            x_enc = x + torch.randn_like(x) * self.denoising_noise_std
        elif use_channel_mask:
            # Channel masking mode: zero random channels, no temporal masking
            n_hide = max(1, int(self.channel_mask_ratio * C))
            perm = torch.randperm(C, device=x.device)
            channel_hidden_idx = perm[:n_hide].sort().values
            x_enc = x.clone()
            x_enc[:, :, channel_hidden_idx] = 0.0
        else:
            # Standard temporal masking
            if mr > 0 and self.training:
                mask = generate_contiguous_mask(
                    n_patches, mr, self.max_mask_span, B, x.device
                )

        # Encode
        latent = self.encode(x_enc, subject_id, mask)

        # Decode
        reconstructed = self.decode(latent, subject_id)

        result = {
            "reconstructed": reconstructed,
            "latent": latent,
            "mask": mask,
        }

        # Compute loss
        if return_loss:
            if channel_hidden_idx is not None:
                # Channel mask mode: MSE only on hidden channels
                diff = (reconstructed[:, :, channel_hidden_idx] - x_target[:, :, channel_hidden_idx]) ** 2
                loss = diff.mean()
            elif mask is not None:
                # Temporal mask mode: MSE only on masked patches
                mask_bins = mask.unsqueeze(-1).expand(B, n_patches, self.patch_size)
                mask_bins = mask_bins.reshape(B, T)
                mask_bins = mask_bins.unsqueeze(-1).expand(B, T, C)

                diff = (reconstructed - x_target) ** 2
                loss = (diff * mask_bins.float()).sum() / mask_bins.float().sum().clamp(min=1)
            else:
                # Denoising or no-mask mode: MSE on full data
                loss = F.mse_loss(reconstructed, x_target)

            result["loss"] = loss

        return result
    
    def get_encoder_state(self) -> dict:
        """
        Get state dict of shared encoder (for transfer to finetuning).
        Excludes subject-specific layers and mask token.
        """
        state = {}
        for name, param in self.named_parameters():
            if name.startswith("blocks.") or name.startswith("final_norm."):
                state[name] = param.data
        return state


# ==============================================================================
# R² metric for validation
# ==============================================================================

def compute_r2(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute R² (coefficient of determination) for reconstruction quality.
    
    Args:
        predicted: (B, T, C) reconstructed data
        target: (B, T, C) original data
        mask: (B, n_patches) bool if only computing on masked patches
    
    Returns:
        r2: scalar R² value
    """
    with torch.no_grad():
        if mask is not None:
            B, T, C = target.shape
            patch_size = T // mask.shape[1]
            mask_bins = mask.unsqueeze(-1).expand(B, mask.shape[1], patch_size)
            mask_bins = mask_bins.reshape(B, T).unsqueeze(-1).expand(B, T, C)
            
            pred_flat = predicted[mask_bins].float()
            tgt_flat = target[mask_bins].float()
        else:
            pred_flat = predicted.reshape(-1).float()
            tgt_flat = target.reshape(-1).float()
        
        ss_res = ((pred_flat - tgt_flat) ** 2).sum()
        ss_tot = ((tgt_flat - tgt_flat.mean()) ** 2).sum()
        
        r2 = 1.0 - (ss_res / ss_tot.clamp(min=1e-8))
        return r2.item()


# ==============================================================================
# Model factory
# ==============================================================================

def build_ssl_model(
    subject_channels: Dict[str, int],
    embed_dim: int = 384,
    n_heads: int = 6,
    head_dim: Optional[int] = None,
    depth: int = 7,
    ff_dim: Optional[int] = None,
    patch_size: int = 5,
    mask_ratio: float = 0.5,
    max_mask_span: int = 15,
    denoising_noise_std: float = 0.0,
    channel_mask_ratio: float = 0.0,
    dropout: float = 0.2,
    attn_dropout: float = 0.4,
) -> SSLTransformerEncoder:
    """
    Build SSL transformer with all subjects registered.

    Args:
        subject_channels: {subject_id: n_channels}
        head_dim: per-head Q/K/V dimension. None → embed_dim // n_heads.
                  Set 512 to replicate BIT Table 10.
        denoising_noise_std: If > 0, use denoising mode (add noise, no temporal masking)
        channel_mask_ratio: If > 0, use channel masking mode (zero channels, no temporal masking)
        ... other model hyperparameters

    Returns:
        model: SSLTransformerEncoder
    """
    model = SSLTransformerEncoder(
        embed_dim=embed_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        depth=depth,
        ff_dim=ff_dim,
        dropout=dropout,
        attn_dropout=attn_dropout,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        max_mask_span=max_mask_span,
        denoising_noise_std=denoising_noise_std,
        channel_mask_ratio=channel_mask_ratio,
        subject_channels=subject_channels,
    )
    
    print(f"SSL Transformer Encoder:")
    print(f"  Shared params:  {model.n_shared_params:,}")
    print(f"  Subject params: {model.n_subject_params:,}")
    print(f"  Total params:   {model.n_shared_params + model.n_subject_params:,}")
    print(f"  Subjects: {list(subject_channels.keys())}")
    
    return model


# ==============================================================================
# Causal Prediction SSL
# ==============================================================================

class CausalPredictionHead(nn.Module):
    """
    Subject-specific multi-step prediction head.
    From each position's embedding, predicts the next N patches in parallel.

    Input: (B, n_patches, embed_dim)
    Output: (B, n_patches, N, C * patch_size)
    """

    def __init__(self, n_channels: int, patch_size: int, embed_dim: int, n_predict_steps: int):
        super().__init__()
        self.n_predict_steps = n_predict_steps
        self.patch_dim = n_channels * patch_size

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_predict_steps * self.patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        out = self.proj(x)  # (B, T, N * patch_dim)
        return out.reshape(B, T, self.n_predict_steps, self.patch_dim)


class CausalSSLTransformerEncoder(nn.Module):
    """
    Causal transformer for next-N-step prediction SSL pretraining.

    Same shared transformer as SSLTransformerEncoder but:
    - Uses causal (autoregressive) attention
    - No masking — causality prevents information leakage
    - Prediction head outputs N future patches per position
    - Loss: MSE between predicted and actual future patches

    The shared transformer blocks (blocks.* and final_norm.*) have
    identical architecture to the masked version, so checkpoints are
    interchangeable for downstream finetuning.
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
        n_predict_steps: int = 3,
        subject_channels: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.n_predict_steps = n_predict_steps

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
        self.prediction_heads = nn.ModuleDict()

        if subject_channels:
            for subject_id, n_channels in subject_channels.items():
                self.register_subject(subject_id, n_channels)

    def register_subject(self, subject_id: str, n_channels: int):
        key = subject_id.replace(".", "_").replace("-", "_")
        if key not in self.patch_embeds:
            self.patch_embeds[key] = PatchEmbedding(
                n_channels, self.patch_size, self.embed_dim
            )
            self.prediction_heads[key] = CausalPredictionHead(
                n_channels, self.patch_size, self.embed_dim, self.n_predict_steps
            )

    def _get_subject_key(self, subject_id: str) -> str:
        return subject_id.replace(".", "_").replace("-", "_")

    @property
    def n_shared_params(self) -> int:
        shared = sum(p.numel() for p in self.blocks.parameters())
        shared += sum(p.numel() for p in self.final_norm.parameters())
        return shared

    @property
    def n_subject_params(self) -> int:
        total = sum(p.numel() for p in self.patch_embeds.parameters())
        total += sum(p.numel() for p in self.prediction_heads.parameters())
        return total

    def encode(self, x: torch.Tensor, subject_id: str) -> torch.Tensor:
        """Encode with causal attention. No masking needed."""
        key = self._get_subject_key(subject_id)
        tokens = self.patch_embeds[key](x)  # (B, n_patches, embed_dim)

        for block in self.blocks:
            tokens = block(tokens)

        return self.final_norm(tokens)

    def forward(self, x: torch.Tensor, subject_id: str, return_loss: bool = True) -> dict:
        """
        Causal SSL forward: encode → predict future patches → MSE loss.

        The prediction target for position t is the actual patch at
        positions t+1, t+2, ..., t+N, taken from the raw input.
        """
        B, T, C = x.shape
        n_patches = T // self.patch_size
        N = self.n_predict_steps
        key = self._get_subject_key(subject_id)

        # Raw patches as targets (before transformer)
        raw_patches = x.reshape(B, n_patches, self.patch_size * C)

        # Encode with causal attention
        latent = self.encode(x, subject_id)  # (B, n_patches, embed_dim)

        # Predict N future patches from each position
        predictions = self.prediction_heads[key](latent)  # (B, n_patches, N, C*ps)

        result = {
            "latent": latent,
            "predictions": predictions,
        }

        if return_loss:
            total_loss = torch.tensor(0.0, device=x.device)
            total_count = 0

            for step in range(N):
                max_t = n_patches - step - 1
                if max_t <= 0:
                    continue
                pred = predictions[:, :max_t, step, :]
                target = raw_patches[:, step + 1:step + 1 + max_t, :]
                total_loss = total_loss + F.mse_loss(pred, target, reduction="sum")
                total_count += pred.numel()

            result["loss"] = total_loss / max(total_count, 1)

        return result

    def get_encoder_state(self) -> dict:
        """Get shared encoder weights (same format as masked version)."""
        state = {}
        for name, param in self.named_parameters():
            if name.startswith("blocks.") or name.startswith("final_norm."):
                state[name] = param.data
        return state


def compute_r2_causal(
    predictions: torch.Tensor,
    raw_patches: torch.Tensor,
    n_predict_steps: int,
) -> float:
    """
    R² for causal prediction quality.

    Args:
        predictions: (B, n_patches, N, patch_dim) predicted future patches
        raw_patches: (B, n_patches, patch_dim) actual patches from input
        n_predict_steps: N
    """
    with torch.no_grad():
        all_pred: List[torch.Tensor] = []
        all_tgt: List[torch.Tensor] = []
        n_patches = raw_patches.shape[1]

        for step in range(n_predict_steps):
            max_t = n_patches - step - 1
            if max_t <= 0:
                continue
            all_pred.append(predictions[:, :max_t, step, :].reshape(-1))
            all_tgt.append(raw_patches[:, step + 1:step + 1 + max_t, :].reshape(-1))

        if not all_pred:
            return 0.0

        pred_flat = torch.cat(all_pred).float()
        tgt_flat = torch.cat(all_tgt).float()

        ss_res = ((pred_flat - tgt_flat) ** 2).sum()
        ss_tot = ((tgt_flat - tgt_flat.mean()) ** 2).sum()
        return (1.0 - ss_res / ss_tot.clamp(min=1e-8)).item()


def build_causal_ssl_model(
    subject_channels: Dict[str, int],
    embed_dim: int = 384,
    n_heads: int = 6,
    head_dim: Optional[int] = None,
    depth: int = 7,
    ff_dim: Optional[int] = None,
    patch_size: int = 5,
    n_predict_steps: int = 3,
    dropout: float = 0.2,
    attn_dropout: float = 0.4,
) -> CausalSSLTransformerEncoder:
    """Build causal SSL transformer with all subjects registered."""
    model = CausalSSLTransformerEncoder(
        embed_dim=embed_dim,
        n_heads=n_heads,
        head_dim=head_dim,
        depth=depth,
        ff_dim=ff_dim,
        dropout=dropout,
        attn_dropout=attn_dropout,
        patch_size=patch_size,
        n_predict_steps=n_predict_steps,
        subject_channels=subject_channels,
    )

    print(f"Causal SSL Transformer Encoder:")
    print(f"  Shared params:  {model.n_shared_params:,}")
    print(f"  Subject params: {model.n_subject_params:,}")
    print(f"  Total params:   {model.n_shared_params + model.n_subject_params:,}")
    print(f"  Predict steps:  {n_predict_steps}")
    print(f"  Subjects: {list(subject_channels.keys())}")
    return model


# ==============================================================================
# Quick test
# ==============================================================================

if __name__ == "__main__":
    # Test with dummy data
    subjects = {
        "T15": 256,
        "monkey_A": 96,
        "monkey_B": 192,
    }
    
    model = build_ssl_model(subjects)
    
    # Test forward pass for each subject
    for sid, n_ch in subjects.items():
        x = torch.randn(4, 500, n_ch)  # (B, T, C) — 500 bins = 10s
        out = model(x, sid)
        
        print(f"\n{sid} (C={n_ch}):")
        print(f"  Input:         {list(x.shape)}")
        print(f"  Reconstructed: {list(out['reconstructed'].shape)}")
        print(f"  Latent:        {list(out['latent'].shape)}")
        print(f"  Loss:          {out['loss'].item():.4f}")
        if out["mask"] is not None:
            print(f"  Mask ratio:    {out['mask'].float().mean():.2f}")
        
        r2 = compute_r2(out["reconstructed"], x, out["mask"])
        print(f"  R²:            {r2:.4f}")
    
    # Test inference mode (no masking)
    model.eval()
    x = torch.randn(2, 500, 256)
    with torch.no_grad():
        out = model(x, "T15", mask_ratio=0.0)
    print(f"\nInference (no mask): loss={out['loss'].item():.4f}, mask={out['mask']}")
    
    print("\nModel test passed!")
