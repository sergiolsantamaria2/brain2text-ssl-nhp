"""
Diphone utilities for Brain-to-Text decoding.

Based on DCoND (Divide-Conquer Neural Decoder) from:
"Brain-to-text decoding with context-aware neural representations and large language models"
Li et al., 2024 (1st place Brain-to-Text '24)

Key insight: Neural signals encode transitions between phonemes (diphones),
not just individual phonemes. Decoding diphones and marginalizing to phonemes
reduces PER from 16.62% to 15.34%.

IMPORTANT: This dataset uses 1-indexed phonemes:
- Index 0 = CTC blank (never in ground truth labels)
- Indices 1-40 = 40 phonemes (with 40 = SIL)

Diphone encoding:
- 40 phonemes × 40 phonemes = 1600 diphone classes
- Plus 1 blank token = 1601 total classes
- Diphone index = (prev_phoneme-1) * 40 + (current_phoneme-1)  [0-1599]
- Blank is always the last index (1600)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


# Standard number of phonemes (excluding CTC blank)
N_PHONEMES = 40
N_DIPHONES = N_PHONEMES * N_PHONEMES  # 1600
N_DIPHONE_CLASSES = N_DIPHONES + 1     # 1601 (including blank)
DIPHONE_BLANK_IDX = N_DIPHONES         # 1600

# Dataset uses 1-indexed phonemes: 1-40 are real phonemes, 0 is CTC blank
PHONEME_MIN = 1
PHONEME_MAX = 40
CTC_BLANK_PHONEME = 0  # CTC blank in phoneme space
SIL_PHONEME = 40       # Silence phoneme (used as initial context)


def phoneme_to_diphone_index(prev_phoneme: int, curr_phoneme: int) -> int:
    """
    Convert a phoneme pair to a diphone index.
    
    Args:
        prev_phoneme: Previous phoneme index (1-40)
        curr_phoneme: Current phoneme index (1-40)
    
    Returns:
        Diphone index (0-1599)
    """
    assert PHONEME_MIN <= prev_phoneme <= PHONEME_MAX, f"Invalid prev_phoneme: {prev_phoneme}"
    assert PHONEME_MIN <= curr_phoneme <= PHONEME_MAX, f"Invalid curr_phoneme: {curr_phoneme}"
    # Convert to 0-indexed for calculation
    prev_adj = prev_phoneme - 1  # 1→0, 40→39
    curr_adj = curr_phoneme - 1
    return prev_adj * N_PHONEMES + curr_adj


def diphone_index_to_phonemes(diphone_idx: int) -> Tuple[int, int]:
    """
    Convert a diphone index back to phoneme pair.
    
    Args:
        diphone_idx: Diphone index (0-1599)
    
    Returns:
        Tuple of (prev_phoneme, curr_phoneme) in 1-indexed form (1-40)
    """
    assert 0 <= diphone_idx < N_DIPHONES, f"Invalid diphone_idx: {diphone_idx}"
    prev_adj = diphone_idx // N_PHONEMES
    curr_adj = diphone_idx % N_PHONEMES
    # Convert back to 1-indexed
    return prev_adj + 1, curr_adj + 1


def phoneme_sequence_to_diphones(
    phoneme_seq: List[int],
    ctc_blank: int = CTC_BLANK_PHONEME,
    initial_context: int = SIL_PHONEME,
) -> List[int]:
    """
    Convert a phoneme sequence to a diphone sequence.
    
    Args:
        phoneme_seq: List of phoneme indices (1-40 for real phonemes, 0 for CTC blank)
        ctc_blank: Index of CTC blank in phoneme space (default 0)
        initial_context: Phoneme to use as initial "previous" context (default SIL=40)
    
    Returns:
        List of diphone indices (0-1599 for real diphones, 1600 for blank)
    
    Example:
        phoneme_seq = [36, 17, 8, 40]  # Example phoneme sequence
        # With SIL as initial context:
        # diphones = [(SIL,36), (36,17), (17,8), (8,40)]
        # = [(40,36), (36,17), (17,8), (8,40)]
        # = [39*40+35, 35*40+16, 16*40+7, 7*40+39]
        # = [1595, 1416, 647, 319]
    """
    if len(phoneme_seq) == 0:
        return []
    
    diphone_seq = []
    prev_phoneme = initial_context  # Start with SIL as context
    
    for phoneme in phoneme_seq:
        if phoneme == ctc_blank:
            # CTC blank stays as diphone blank
            diphone_seq.append(DIPHONE_BLANK_IDX)
        else:
            # Validate phoneme is in valid range
            if not (PHONEME_MIN <= phoneme <= PHONEME_MAX):
                raise ValueError(f"Invalid phoneme index: {phoneme}, expected {PHONEME_MIN}-{PHONEME_MAX}")
            # Convert to diphone
            diphone_idx = phoneme_to_diphone_index(prev_phoneme, phoneme)
            diphone_seq.append(diphone_idx)
            prev_phoneme = phoneme
    
    return diphone_seq


def marginalize_diphone_logits(
    diphone_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Marginalize diphone logits to phoneme logits.
    
    This is the "conquer" step of DCoND: we sum over all possible
    previous phonemes to get the probability of the current phoneme.
    
    P(current_phoneme) = Σ_{prev} P(prev, current)
    
    In log space: log P(current) = logsumexp_{prev} log P(prev, current)
    
    Args:
        diphone_logits: [batch, time, 1601] raw logits from model
        temperature: Temperature for softmax (default 1.0)
    
    Returns:
        [batch, time, 41] phoneme logits (40 phonemes + blank)
        Output indices: 0 = CTC blank, 1-40 = phonemes (matching original dataset format)
    """
    batch, time, n_classes = diphone_logits.shape
    assert n_classes == N_DIPHONE_CLASSES, f"Expected {N_DIPHONE_CLASSES} classes, got {n_classes}"
    
    # Separate diphone logits and blank logit
    diphone_part = diphone_logits[:, :, :N_DIPHONES]  # [batch, time, 1600]
    blank_logit = diphone_logits[:, :, N_DIPHONES:]   # [batch, time, 1]
    
    # Apply temperature
    if temperature != 1.0:
        diphone_part = diphone_part / temperature
        blank_logit = blank_logit / temperature
    
    # Reshape to [batch, time, n_prev=40, n_curr=40]
    diphone_reshaped = diphone_part.reshape(batch, time, N_PHONEMES, N_PHONEMES)
    
    # Marginalize over previous phoneme (logsumexp over dim=2)
    # This gives us log P(current_phoneme) for phonemes 0-39 (internal indexing)
    phoneme_logits_internal = torch.logsumexp(diphone_reshaped, dim=2)  # [batch, time, 40]
    
    # Output format: [blank, phoneme_1, phoneme_2, ..., phoneme_40]
    # blank_logit is for CTC blank (index 0 in output)
    # phoneme_logits_internal[..., i] is for phoneme i+1 in original indexing
    phoneme_logits = torch.cat([blank_logit, phoneme_logits_internal], dim=-1)  # [batch, time, 41]
    
    return phoneme_logits


def create_diphone_to_phoneme_matrix() -> torch.Tensor:
    """
    Create a sparse marginalization matrix for efficient diphone→phoneme conversion.
    
    Returns:
        [1601, 41] matrix M where phoneme_logits = diphone_logits @ M
        (after appropriate log-space handling)
    """
    M = torch.zeros(N_DIPHONE_CLASSES, N_PHONEMES + 1)
    
    # Each diphone (prev, curr) contributes to phoneme curr+1 (1-indexed output)
    for prev in range(N_PHONEMES):
        for curr in range(N_PHONEMES):
            diphone_idx = prev * N_PHONEMES + curr
            M[diphone_idx, curr + 1] = 1.0  # curr+1 because output index 0 is blank
    
    # Blank maps to blank (index 0)
    M[DIPHONE_BLANK_IDX, 0] = 1.0
    
    return M


class DiphoneConverter:
    """
    Helper class for diphone conversion in training and inference.
    
    Usage:
        converter = DiphoneConverter()
        
        # Training: convert phoneme labels to diphone labels
        diphone_labels = converter.phonemes_to_diphones(phoneme_labels)
        
        # Inference: convert diphone logits to phoneme logits
        phoneme_logits = converter.marginalize(diphone_logits)
    """
    
    def __init__(
        self,
        n_phonemes: int = N_PHONEMES,
        ctc_blank: int = CTC_BLANK_PHONEME,
        initial_context: int = SIL_PHONEME,
    ):
        self.n_phonemes = n_phonemes
        self.n_diphones = n_phonemes * n_phonemes
        self.n_diphone_classes = self.n_diphones + 1
        self.diphone_blank_idx = self.n_diphones
        self.ctc_blank = ctc_blank
        self.initial_context = initial_context
    
    def phonemes_to_diphones_sequence(self, phoneme_seq: List[int]) -> List[int]:
        """Convert single phoneme sequence to diphone sequence."""
        return phoneme_sequence_to_diphones(
            phoneme_seq, 
            ctc_blank=self.ctc_blank,
            initial_context=self.initial_context,
        )
    
    def phonemes_to_diphones_batch(
        self, 
        phoneme_labels: torch.Tensor,
        label_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert batch of phoneme labels to diphone labels.
        
        Args:
            phoneme_labels: [batch, max_len] phoneme indices (1-40, with 0 for padding/blank)
            label_lengths: [batch] actual lengths
        
        Returns:
            diphone_labels: [batch, max_len] diphone indices
            label_lengths: [batch] (unchanged, same lengths)
        """
        batch_size, max_len = phoneme_labels.shape
        device = phoneme_labels.device
        
        diphone_labels = torch.full(
            (batch_size, max_len), 
            fill_value=self.diphone_blank_idx,
            dtype=phoneme_labels.dtype,
            device=device
        )
        
        for b in range(batch_size):
            length = int(label_lengths[b].item())
            seq = phoneme_labels[b, :length].tolist()
            diphone_seq = self.phonemes_to_diphones_sequence(seq)
            
            for t, d in enumerate(diphone_seq):
                if t < max_len:
                    diphone_labels[b, t] = d
        
        return diphone_labels, label_lengths
    
    def marginalize(
        self, 
        diphone_logits: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Marginalize diphone logits to phoneme logits."""
        return marginalize_diphone_logits(diphone_logits, temperature)
    
    @property
    def num_classes(self) -> int:
        """Number of diphone classes (for model output layer)."""
        return self.n_diphone_classes


# Quick sanity check
if __name__ == "__main__":
    print("=== Diphone Utils Test (1-indexed phonemes) ===")
    
    # Test basic conversion with 1-indexed phonemes
    print("\n1. Basic phoneme→diphone conversion:")
    phonemes = [36, 17, 8, 40]  # Example from dataset
    diphones = phoneme_sequence_to_diphones(phonemes)
    print(f"   Phonemes: {phonemes}")
    print(f"   Diphones: {diphones}")
    
    # Verify reverse
    print("\n2. Diphone→phoneme reverse:")
    for d in diphones:
        if d < N_DIPHONES:
            prev, curr = diphone_index_to_phonemes(d)
            print(f"   Diphone {d} → (prev={prev}, curr={curr})")
    
    # Test marginalization
    print("\n3. Marginalization test:")
    batch, time = 2, 10
    fake_logits = torch.randn(batch, time, N_DIPHONE_CLASSES)
    phoneme_logits = marginalize_diphone_logits(fake_logits)
    print(f"   Diphone logits shape: {fake_logits.shape}")
    print(f"   Phoneme logits shape: {phoneme_logits.shape}")
    assert phoneme_logits.shape == (batch, time, 41), "Output should be 41 classes"
    
    # Test converter class
    print("\n4. DiphoneConverter class:")
    converter = DiphoneConverter()
    print(f"   n_phonemes: {converter.n_phonemes}")
    print(f"   n_diphone_classes: {converter.num_classes}")
    
    # Test batch conversion
    print("\n5. Batch label conversion:")
    phoneme_labels = torch.tensor([
        [36, 17, 8, 40, 0],  # 0 is padding/unused
        [22, 25, 29, 0, 0],
    ])
    label_lengths = torch.tensor([4, 3])
    diphone_labels, _ = converter.phonemes_to_diphones_batch(phoneme_labels, label_lengths)
    print(f"   Phoneme labels:\n   {phoneme_labels}")
    print(f"   Diphone labels:\n   {diphone_labels}")
    
    # Verify diphone indices are in valid range
    valid_diphones = diphone_labels[diphone_labels != converter.diphone_blank_idx]
    assert valid_diphones.max() < N_DIPHONES, f"Diphone index out of range: {valid_diphones.max()}"
    assert valid_diphones.min() >= 0, f"Negative diphone index: {valid_diphones.min()}"
    print(f"   All diphone indices in valid range [0, {N_DIPHONES-1}] ✓")
    
    print("\n=== All tests passed! ===")
