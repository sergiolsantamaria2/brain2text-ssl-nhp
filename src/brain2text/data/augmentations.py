import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution (make padding robust across torch versions)
    k = int(gaussKernel.shape[-1])

    if isinstance(padding, str):
        pad_mode = padding.lower()
        if pad_mode == "same":
            pad = k // 2
        elif pad_mode == "valid":
            pad = 0
        else:
            raise ValueError(f"Unknown padding='{padding}'. Use 'same' or 'valid' or an int.")
    else:
        pad = int(padding)

    smoothed = F.conv1d(inputs, gaussKernel, padding=pad, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]

import torch

def masked_batch_zscore(x: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    x: (B,T,C), lengths: (B,)
    Normalizes each channel using mean/std computed over valid (non-padded) time steps across the batch.
    """
    B, T, C = x.shape
    t = torch.arange(T, device=x.device).unsqueeze(0)                  # (1,T)
    mask = (t < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)        # (B,T,1)

    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)                        # (1,)
    mean = (x * mask).sum(dim=(0, 1)) / denom                          # (C,)
    var = ((x - mean) * mask).pow(2).sum(dim=(0, 1)) / denom           # (C,)
    std = (var + eps).sqrt()
    return (x - mean) / std


def masked_instance_zscore(x: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Per-sample z-score: normalize each (sample, channel) over valid time steps.
    """
    B, T, C = x.shape
    t = torch.arange(T, device=x.device).unsqueeze(0)
    mask = (t < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)        # (B,T,1)

    denom = mask.sum(dim=1).clamp_min(1.0)                             # (B,1)
    mean = (x * mask).sum(dim=1) / denom                               # (B,C)
    var = ((x - mean.unsqueeze(1)) * mask).pow(2).sum(dim=1) / denom   # (B,C)
    std = (var + eps).sqrt()
    return (x - mean.unsqueeze(1)) / std


def apply_feature_norm(x: torch.Tensor, lengths: torch.Tensor, norm: str, eps: float = 1e-5) -> torch.Tensor:
    norm = (norm or "none").lower()
    if norm == "none":
        return x
    if norm == "batch_zscore":
        return masked_batch_zscore(x, lengths, eps=eps)
    if norm == "instance_zscore":
        return masked_instance_zscore(x, lengths, eps=eps)
    raise ValueError(f"Unknown feature_norm='{norm}'. Use: none, batch_zscore, instance_zscore.")
