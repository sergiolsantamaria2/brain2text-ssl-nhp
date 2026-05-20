#!/usr/bin/env python3
"""
SSL Pretraining Dataset for BIT-style Masked Autoencoder
=========================================================

Loads preprocessed neural data from multiple subjects (NHP + Human) for 
self-supervised pretraining via masked reconstruction.

Design decisions following BIT paper (2511.21740v2):
  - Subject-specific patch embeddings → single-subject batches
  - Random window sampling from continuous sessions
  - Data augmentation: white noise, constant offset, Gaussian smoothing
  - Masking is done in the model, NOT here
  - Weighted sampling proportional to total data per subject

Data sources:
  - NHP: {nhp_dir}/{subject_id}/session_XXXX.h5 → neural_data (T, C)
  - Human T15: {human_dir}/{session_date}/ → per-trial input_features (T, 512)
    For pretraining, we concatenate trials into continuous sessions and use
    only thresholded spikes (first 256 channels) since SBP isn't available
    for most NHP data.

Usage:
  dataset = SSLPretrainDataset(config)
  loader = DataLoader(dataset, batch_size=None, num_workers=4)
  
  for batch in loader:
      # batch['neural_data']: (B, window_len, C)  
      # batch['subject_id']: str
      # batch['n_channels']: int
      ...

Author: SSL Pretraining Pipeline
"""

import glob
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class SSLDataConfig:
    """Configuration for SSL pretraining dataset."""
    
    # Data paths
    nhp_pretrain_dir: str = "${DATA_DIR}/nhp_pretrain"
    human_data_dir: str = "${DATA_DIR}/hdf5_data_final"
    
    # Human data settings
    include_human: bool = False
    human_sessions: Optional[List[str]] = None  # None = all sessions
    human_use_only_tx: bool = True  # Use only thresholded spikes (first 256 of 512)
    # If False, uses all 512 features (TX + SBP)
    
    # Window sampling
    window_bins: int = 500  # Number of 20ms bins per window (500 = 10s)
    # Must be divisible by patch_size (5) → 500/5 = 100 patches
    patch_size: int = 5  # Time bins per patch (BIT default)
    min_session_bins: int = 100  # Skip sessions shorter than this
    
    # Batching
    batch_size: int = 64  #64 windows (it corresponds to 64 windows x 500 bins(20ms each) x 192 channels(electrodes))
    
    # Data transforms
    log_transform: bool = False  # sign(x) * log1p(|x|), applied BEFORE augmentation

    # Data augmentation (BIT Table 10)
    white_noise_std: float = 0.2
    constant_offset_std: float = 0.05
    gaussian_smooth_std: float = 2.0  # Gaussian smoothing kernel width
    gaussian_smooth_kernel_size: int = 11  # Must be odd
    
    # Sampling
    n_batches_per_epoch: int = 1000  # Since data is sampled randomly
    val_fraction: float = 0.1  # Fraction of sessions held out for validation
    seed: int = 42
    
    # Exclude subjects with too few data
    min_subject_hours: float = 0.1  # Minimum hours to include a subject
    
    # Workers
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        assert self.window_bins % self.patch_size == 0, \
            f"window_bins ({self.window_bins}) must be divisible by patch_size ({self.patch_size})"


# ==============================================================================
# Subject Registry — tracks all available subjects and their sessions
# ==============================================================================

@dataclass
class SessionInfo:
    """Metadata for a single session."""
    filepath: str
    n_bins: int
    n_channels: int
    subject_id: str
    source: str  # "nhp" or "human"


@dataclass
class SubjectInfo: # A subject-specific patch embedding is used because subjects have different electrode counts
    """Metadata for a subject (group of sessions with same channel count)."""
    subject_id: str
    n_channels: int
    sessions: List[SessionInfo] = field(default_factory=list)
    total_bins: int = 0
    total_hours: float = 0.0
    source: str = "nhp"


def discover_nhp_subjects(
    nhp_dir: str,
    min_session_bins: int = 100,
) -> Dict[str, SubjectInfo]:
    """
    Discover all NHP subjects from the preprocessed directory.
    
    Reads metadata from each session_XXXX.h5 file without loading full data.
    """
    subjects = {}
    
    if not os.path.isdir(nhp_dir):
        logger.warning(f"NHP directory not found: {nhp_dir}")
        return subjects
    
    for subject_dir in sorted(os.listdir(nhp_dir)): # Iterate over per-subject directories under the NHP root
        subject_path = os.path.join(nhp_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        
        # Find all session files
        session_files = sorted(glob.glob(os.path.join(subject_path, "session_*.h5")))
        if not session_files:
            continue
        
        sessions = []
        n_channels = None
        total_bins = 0
        
        for sf in session_files:
            try:
                with h5py.File(sf, "r") as f:
                    if "neural_data" not in f:
                        continue
                    shape = f["neural_data"].shape  # (T, C) Solo forma, no los datos
                    n_bins = shape[0]
                    nc = shape[1]
                    
                    if n_bins < min_session_bins:
                        continue
                    
                    if n_channels is None:
                        n_channels = nc
                    elif nc != n_channels:
                        logger.warning(
                            f"Channel mismatch in {subject_dir}: "
                            f"expected {n_channels}, got {nc} in {os.path.basename(sf)}"
                        )
                        continue
                    
                    sessions.append(SessionInfo(
                        filepath=sf,
                        n_bins=n_bins,
                        n_channels=nc,
                        subject_id=subject_dir,
                        source="nhp",
                    ))
                    total_bins += n_bins
                    
            except Exception as e:
                logger.error(f"Error reading {sf}: {e}")
                continue
        
        if sessions and n_channels is not None:   # Returned to the caller
            subjects[subject_dir] = SubjectInfo(
                subject_id=subject_dir,
                n_channels=n_channels,
                sessions=sessions,
                total_bins=total_bins,
                total_hours=total_bins * 0.02 / 3600,  # 20ms bins
                source="nhp",
            )
    
    return subjects


def discover_human_subjects( # Human datasets are organized by trial (individual utterances)
    human_dir: str,
    sessions_filter: Optional[List[str]] = None,
    use_only_tx: bool = True,
    min_session_bins: int = 100,
) -> Dict[str, SubjectInfo]:
    """
    Discover human T15 data from the competition HDF5 format.
    
    The competition data is stored as per-trial HDF5 files with 
    input_features of shape (T_trial, 512). For pretraining, we treat
    each session date as a "session" and will concatenate trials on-the-fly.
    
    Args:
        human_dir: Path to hdf5_data_final/
        sessions_filter: List of session names to include (e.g., ["t15.2023.08.11"])
        use_only_tx: If True, use only first 256 channels (thresholded spikes)
        min_session_bins: Minimum bins to include a session
    """
    subjects = {}
    
    if not os.path.isdir(human_dir):
        logger.warning(f"Human data directory not found: {human_dir}")
        return subjects
    
    # Find all session HDF5 files (one per date)
    session_files = sorted(glob.glob(os.path.join(human_dir, "*.hdf5")))
    if not session_files:
        # Try looking for date-named directories with HDF5 inside
        session_files = sorted(glob.glob(os.path.join(human_dir, "**/*.hdf5"), recursive=True))
    
    if not session_files:
        logger.warning(f"No HDF5 files found in {human_dir}")
        return subjects
    
    n_channels = 256 if use_only_tx else 512
    subject_id = "T15"
    sessions = []
    total_bins = 0
    
    for sf in session_files:
        basename = os.path.splitext(os.path.basename(sf))[0]
        
        # Filter sessions if specified
        if sessions_filter and basename not in sessions_filter:
            continue
        
        try:
            with h5py.File(sf, "r") as f:
                # Count total bins across all trials in this session
                session_bins = 0
                trial_keys = [k for k in f.keys() if k.startswith("trial_")]
                
                for tk in trial_keys:
                    if "input_features" in f[tk]:
                        shape = f[tk]["input_features"].shape
                        session_bins += shape[0]
                
                if session_bins < min_session_bins:
                    continue
                
                sessions.append(SessionInfo(
                    filepath=sf,
                    n_bins=session_bins,
                    n_channels=n_channels,
                    subject_id=subject_id,
                    source="human",
                ))
                total_bins += session_bins
                
        except Exception as e:
            logger.error(f"Error reading {sf}: {e}")
            continue
    
    if sessions:
        subjects[subject_id] = SubjectInfo(
            subject_id=subject_id,
            n_channels=n_channels,
            sessions=sessions,
            total_bins=total_bins,
            total_hours=total_bins * 0.02 / 3600,
            source="human",
        )
    
    return subjects


# ==============================================================================
# Data loading helpers
# ==============================================================================

def load_nhp_session(filepath: str) -> np.ndarray:
    """Load a single NHP session. Returns (T, C) float32."""
    with h5py.File(filepath, "r") as f:
        return f["neural_data"][:].astype(np.float32) # carga todo en memoria. el [:] carga todos los datos


def load_human_session(
    filepath: str, 
    use_only_tx: bool = True,
    split: str = "train",
) -> np.ndarray:
    """
    Load a human session by concatenating all trials.
    
    For SSL pretraining, we don't need trial boundaries or labels.
    We concatenate all trials into one continuous array.
    
    Args:
        filepath: Path to session HDF5 file
        use_only_tx: If True, use only first 256 channels
        split: Which split to load ('train', 'val', 'test', or 'all')
    
    Returns:
        (T_total, C) float32 array
    """
    chunks = []
    with h5py.File(filepath, "r") as f:
        trial_keys = sorted([k for k in f.keys() if k.startswith("trial_")])
        
        for tk in trial_keys:
            # Check if this trial belongs to the right split
            if split != "all" and "partition" in f[tk].attrs:
                trial_split = f[tk].attrs["partition"]
                if isinstance(trial_split, bytes):
                    trial_split = trial_split.decode()
                # For SSL pretraining, use all data (train + val + test)
                # BIT paper A.4: "all neural data can be incorporated"
            
            if "input_features" in f[tk]:
                data = f[tk]["input_features"][:].astype(np.float32)
                if use_only_tx:
                    data = data[:, :256]  # First 256 = thresholded spikes
                chunks.append(data)
    
    if chunks:
        return np.concatenate(chunks, axis=0)
    return np.zeros((0, 256 if use_only_tx else 512), dtype=np.float32)


# ==============================================================================
# Data Augmentation (BIT paper)
# ==============================================================================

def gaussian_smooth_1d(data: np.ndarray, kernel_std: float, kernel_size: int) -> np.ndarray:
    """
    Apply Gaussian smoothing along time axis.
    
    Args:
        data: (T, C) array
        kernel_std: Standard deviation of Gaussian kernel
        kernel_size: Size of the kernel (must be odd)
    
    Returns:
        smoothed: (T, C) array
    """
    if kernel_std <= 0 or kernel_size <= 1:
        return data
    
    # Create Gaussian kernel
    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / kernel_std) ** 2)
    kernel /= kernel.sum()
    
    # Apply via convolution along time axis for each channel
    # Use numpy for efficiency (avoid per-channel loop)
    from scipy.ndimage import convolve1d
    smoothed = convolve1d(data, kernel, axis=0, mode='reflect')
    
    return smoothed


def augment_window(
    window: np.ndarray,
    white_noise_std: float = 0.2,
    constant_offset_std: float = 0.05,
    smooth_std: float = 2.0,
    smooth_kernel_size: int = 11,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply BIT-style data augmentation to a neural data window.
    
    Args:
        window: (T, C) float32 array (already z-scored)
        white_noise_std: Std of additive Gaussian noise
        constant_offset_std: Std of per-channel constant offset
        smooth_std: Gaussian smoothing kernel width
        smooth_kernel_size: Smoothing kernel size
        rng: NumPy random generator
    
    Returns:
        augmented: (T, C) float32 array
    """
    if rng is None:
        rng = np.random.default_rng()
    
    T, C = window.shape
    aug = window.copy()
    
    # 1. Gaussian smoothing (applied first, before noise)
    if smooth_std > 0:
        aug = gaussian_smooth_1d(aug, smooth_std, smooth_kernel_size)
    
    # 2. Constant offset per channel
    if constant_offset_std > 0:
        offset = rng.normal(0, constant_offset_std, size=(1, C)).astype(np.float32)
        aug += offset
    
    # 3. Additive white noise
    if white_noise_std > 0:
        noise = rng.normal(0, white_noise_std, size=(T, C)).astype(np.float32)
        aug += noise
    
    return aug


# ==============================================================================
# SSL Pretraining Dataset
# ==============================================================================

class SSLPretrainDataset(Dataset):
    """
    Dataset for SSL pretraining with subject-specific batching.
    
    Each __getitem__ returns a FULL BATCH of windows from a single subject.
    This enables subject-specific patch embeddings without cross-subject padding.
    
    Sampling strategy:
      1. Sample a subject proportional to its total data duration
      2. Sample batch_size random windows from that subject's sessions
      3. Apply data augmentation
      4. Return batch
    """
    
    def __init__(
        self,
        config: SSLDataConfig,
        split: str = "train",
    ):
        self.config = config
        self.split = split
        self.rng = np.random.default_rng(config.seed if split == "val" else None)
        
        # Discover all subjects
        logger.info("Discovering NHP subjects...")
        self.subjects: Dict[str, SubjectInfo] = discover_nhp_subjects(
            config.nhp_pretrain_dir,
            config.min_session_bins,
        )
        
        if config.include_human:
            logger.info("Discovering human subjects...")
            human_subjects = discover_human_subjects(
                config.human_data_dir,
                config.human_sessions,
                config.human_use_only_tx,
                config.min_session_bins,
            )
            self.subjects.update(human_subjects)
        
        # Filter by minimum hours
        filtered = {}
        for sid, subj in self.subjects.items():
            if subj.total_hours >= config.min_subject_hours:
                filtered[sid] = subj
            else:
                logger.debug(f"Excluding {sid}: {subj.total_hours:.3f}h < {config.min_subject_hours}h")
        self.subjects = filtered
        
        # Split sessions into train/val per subject
        self._split_sessions()
        
        # Compute sampling weights (proportional to total bins)
        self._compute_weights()
        
        # Preload all sessions for this split into RAM
        self._session_cache: Dict[str, np.ndarray] = {}
        self._preload_sessions()

        # Log summary
        total_hours = sum(s.total_hours for s in self.subjects.values())
        n_subjects = len(self.subjects)
        logger.info(f"SSL Dataset ({split}): {n_subjects} subjects, {total_hours:.1f}h total")
        for sid, subj in sorted(self.subjects.items()):
            n_sess = len(self._get_split_sessions(sid))
            logger.info(
                f"  {sid}: {n_sess} sessions, {subj.n_channels} ch, "
                f"{subj.total_hours:.2f}h"
            )
    
    def _split_sessions(self):
        """Split sessions into train/val for each subject."""
        self.train_sessions: Dict[str, List[SessionInfo]] = {}
        self.val_sessions: Dict[str, List[SessionInfo]] = {}
        
        split_rng = np.random.default_rng(self.config.seed)
        
        for sid, subj in self.subjects.items():
            sessions = list(subj.sessions)
            split_rng.shuffle(sessions)
            
            n_val = max(1, int(len(sessions) * self.config.val_fraction))
            
            if len(sessions) <= 2:
                # Too few sessions — use all for train, last for val
                self.train_sessions[sid] = sessions
                self.val_sessions[sid] = sessions[-1:]
            else:
                self.val_sessions[sid] = sessions[:n_val]
                self.train_sessions[sid] = sessions[n_val:]
    
    def _get_split_sessions(self, subject_id: str) -> List[SessionInfo]:
        """Get sessions for current split."""
        if self.split == "train":
            return self.train_sessions.get(subject_id, [])
        else:
            return self.val_sessions.get(subject_id, [])
    
    def _compute_weights(self):
        """Compute subject sampling weights proportional to data amount."""
        self.subject_ids = []
        self.subject_weights = []
        
        for sid, subj in self.subjects.items():
            sessions = self._get_split_sessions(sid)
            if not sessions:
                continue
            total_bins = sum(s.n_bins for s in sessions)
            if total_bins < self.config.window_bins:
                continue
            self.subject_ids.append(sid)
            self.subject_weights.append(total_bins)
        
        # Normalize to probabilities
        total = sum(self.subject_weights)
        self.subject_weights = [w / total for w in self.subject_weights]
        
        logger.info(f"Sampling weights ({self.split}):")
        for sid, w in zip(self.subject_ids, self.subject_weights):
            logger.info(f"  {sid}: {w:.4f}")
    
    def __len__(self):
        return self.config.n_batches_per_epoch
    
    def _preload_sessions(self):
        """Load all sessions for the current split into RAM at startup."""
        logger.info(f"Preloading all {self.split} sessions into RAM...")
        for sid in self.subject_ids:
            for session in self._get_split_sessions(sid):
                if session.filepath in self._session_cache:
                    continue
                if session.source == "nhp":
                    data = load_nhp_session(session.filepath)
                else:  # human
                    data = load_human_session(
                        session.filepath,
                        use_only_tx=self.config.human_use_only_tx,
                        split="all",
                    )
                self._session_cache[session.filepath] = data
                logger.debug(f"  Loaded {os.path.basename(session.filepath)}: {data.shape}")

        n_sessions = len(self._session_cache)
        total_elements = sum(a.size for a in self._session_cache.values())
        total_gb = total_elements * 4 / (1024 ** 3)
        logger.info(
            f"Session cache ready ({self.split}): "
            f"{n_sessions} sessions, ~{total_gb:.2f} GB RAM"
        )


    def _sample_window(
        self, 
        session_data: np.ndarray,
    ) -> np.ndarray:
        """
        Sample a random window from session data.
        
        Args:
            session_data: (T, C) array
            
        Returns:
            window: (window_bins, C) array
        """
        T = session_data.shape[0]
        W = self.config.window_bins
        
        if T <= W:
            # Session too short — pad with zeros
            padded = np.zeros((W, session_data.shape[1]), dtype=np.float32)
            padded[:T] = session_data
            return padded
        
        # Random start position
        start = self.rng.integers(0, T - W)
        return session_data[start:start + W].copy()
    
    def __getitem__(self, idx) -> dict:
        """
        Get a batch of windows from a single subject.
        
        Returns:
            dict with:
                'neural_data': (batch_size, window_bins, n_channels) float32 tensor
                'subject_id': str
                'n_channels': int
                'valid_mask': (batch_size, window_bins) bool tensor — True where data is real
        """
        # 1. Sample subject
        subject_idx = self.rng.choice(
            len(self.subject_ids), 
            p=self.subject_weights,
        )
        subject_id = self.subject_ids[subject_idx]
        subj = self.subjects[subject_id]
        sessions = self._get_split_sessions(subject_id)
        
        # 2. Build session pool with weights
        session_weights = np.array([s.n_bins for s in sessions], dtype=np.float64)
        session_weights /= session_weights.sum()
        
        # 3. Sample batch_size windows
        batch_windows = []
        batch_valid = []
        
        for _ in range(self.config.batch_size):
            # Pick a session (weighted by length)
            sess_idx = self.rng.choice(len(sessions), p=session_weights)
            session = sessions[sess_idx]
            
            # Load from RAM cache
            session_data = self._session_cache[session.filepath]
            window = self._sample_window(session_data)
            
            # Track validity (for padded short sessions)
            T_real = min(session_data.shape[0], self.config.window_bins)
            valid = np.zeros(self.config.window_bins, dtype=bool)
            valid[:T_real] = True

            # Log-transform BEFORE augmentation (must match finetuning pipeline)
            if self.config.log_transform:
                window = np.sign(window) * np.log1p(np.abs(window))

            # Apply augmentation (only during training)
            if self.split == "train":
                window = augment_window(
                    window,
                    white_noise_std=self.config.white_noise_std,
                    constant_offset_std=self.config.constant_offset_std,
                    smooth_std=self.config.gaussian_smooth_std,
                    smooth_kernel_size=self.config.gaussian_smooth_kernel_size,
                    rng=self.rng,
                )
            
            batch_windows.append(window)
            batch_valid.append(valid)
        
        # Stack into tensors
        neural_data = torch.from_numpy(np.stack(batch_windows, axis=0))  # (B, T, C)
        valid_mask = torch.from_numpy(np.stack(batch_valid, axis=0))  # (B, T)
        
        return {
            "neural_data": neural_data,
            "subject_id": subject_id,
            "n_channels": subj.n_channels,
            "valid_mask": valid_mask,
        }


# ==============================================================================
# Convenience factory
# ==============================================================================

def create_ssl_dataloaders(
    config: SSLDataConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders for SSL pretraining.
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = SSLPretrainDataset(config, split="train")
    val_dataset = SSLPretrainDataset(config, split="val")
    
    # batch_size=None because __getitem__ already returns full batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        shuffle=True,  # Shuffles the batch indices
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,  # Sequential for reproducibility
        pin_memory=config.pin_memory,
    )
    
    return train_loader, val_loader


# ==============================================================================
# CLI test
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s] %(message)s")
    
    config = SSLDataConfig(
        n_batches_per_epoch=10,  # Small for testing
        batch_size=4,
    )
    
    # Parse optional overrides
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            key = key.lstrip("-").replace("-", "_")
            if hasattr(config, key):
                field_type = type(getattr(config, key))
                setattr(config, key, field_type(val))
    
    print(f"\nConfig: {config}\n")
    
    train_loader, val_loader = create_ssl_dataloaders(config)
    
    print(f"\n{'='*60}")
    print("Testing train loader...")
    print(f"{'='*60}")
    
    for i, batch in enumerate(train_loader):
        print(
            f"Batch {i}: subject={batch['subject_id']}, "
            f"shape={list(batch['neural_data'].shape)}, "
            f"channels={batch['n_channels']}, "
            f"dtype={batch['neural_data'].dtype}, "
            f"range=[{batch['neural_data'].min():.2f}, {batch['neural_data'].max():.2f}]"
        )
        if i >= 4:
            break
    
    print(f"\n{'='*60}")
    print("Testing val loader...")
    print(f"{'='*60}")
    
    for i, batch in enumerate(val_loader):
        print(
            f"Batch {i}: subject={batch['subject_id']}, "
            f"shape={list(batch['neural_data'].shape)}, "
            f"channels={batch['n_channels']}"
        )
        if i >= 2:
            break
    
    print("\nDataloader test passed!")
