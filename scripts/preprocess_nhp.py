#!/usr/bin/env python3
"""
NHP Neural Data Preprocessing Pipeline for SSL Pretraining
===========================================================

Processes NHP datasets (NWB format) into a unified HDF5 format suitable for 
cross-species SSL pretraining following the BIT paper (2511.21740v2) methodology.

Preprocessing steps per BIT Appendix A.3:
  1. Resample neural activity into 20ms time bins (if not already)
  2. Z-score across days to mitigate non-stationarity
  3. Dead channel detection: <2 dead → interpolate; >2 dead → exclude
  4. Use only thresholded spikes for pretraining (SBP unavailable in most NHP)

Output format:
  {output_dir}/{subject_id}/
      session_{idx:04d}.h5
          - neural_data: (T, C) float32  — binned spike counts, z-scored
          - metadata attrs: subject_id, dataset, session_file, n_electrodes, 
                            duration_s, bin_size_ms, dead_channels

Usage:
  python src/brain2text/ssl/preprocess_nhp.py --config configs/nhp_preprocess.yaml
  python src/brain2text/ssl/preprocess_nhp.py --datasets 000070 000121 001201

"""

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# Optional imports
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# Configuration
# ==============================================================================

@dataclass # Auto-generates the constructor. Per-dataset configuration is centralized here.
class DatasetConfig:
    """Configuration for a single NHP dataset."""
    dataset_id: str
    name: str
    data_type: str  # "spike_times" | "binned_tc" | "mat_spikes" (future)
    brain_area: str
    n_electrodes_expected: int
    in_bit_paper: bool
    root_path: str = ""
    # Dead channel policy
    max_dead_channels: int = 2
    min_spikes_alive: int = 10


@dataclass 
class PipelineConfig: 
    """Global pipeline configuration."""
    monkey_data_root: str = "${DATA_DIR}/monkey_data"
    output_dir: str = "${DATA_DIR}/nhp_pretrain"
    bin_size_ms: float = 20.0
    # Z-score settings
    zscore_across_sessions: bool = True
    zscore_eps: float = 1e-6
    # Dead channel policy (from BIT paper A.3)
    max_dead_channels: int = 2
    min_spikes_alive: int = 10  # units with <10 spikes considered dead
    interpolate_dead: bool = True  # if True, replace dead with temporal mean
    # Processing
    min_session_duration_s: float = 30.0  # skip very short sessions
    log_level: str = "INFO"

# Dataset registry: maps DANDI ID (or local key) to its configuration.
# Dataset registry — maps DANDI IDs to their configurations
DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "000070": DatasetConfig(
        dataset_id="000070",
        name="Churchland et al. 2012",
        data_type="spike_times",
        brain_area="M1",
        n_electrodes_expected=192,
        in_bit_paper=True,
    ),
    "000121": DatasetConfig(
        dataset_id="000121",
        name="Even-Chen et al. 2019",
        data_type="spike_times",
        brain_area="PMd/M1",
        n_electrodes_expected=192,
        in_bit_paper=True,
    ),
    "000128": DatasetConfig(
        dataset_id="000128",
        name="MC_Maze (Churchland/Kaufman)",
        data_type="spike_times",
        brain_area="M1/PMd",
        n_electrodes_expected=192,
        in_bit_paper=True,
    ),
    "000129": DatasetConfig(
        dataset_id="000129",
        name="MC_RTT (O'Doherty/NLB)",
        data_type="spike_times",
        brain_area="M1",
        n_electrodes_expected=96,
        in_bit_paper=True,
    ),
    "000688": DatasetConfig(
        dataset_id="000688",
        name="Perich et al. 2018",
        data_type="spike_times",
        brain_area="M1/PMd",
        n_electrodes_expected=192,
        in_bit_paper=True,
    ),
    "000941": DatasetConfig(
        dataset_id="000941",
        name="FALCON Benchmark (Karpowicz 2024)",
        data_type="spike_times",
        brain_area="M1",
        n_electrodes_expected=64,
        in_bit_paper=False,
    ),
    "001201": DatasetConfig(
        dataset_id="001201",
        name="LINK (Temmar et al. 2025)",
        data_type="binned_tc",
        brain_area="M1",
        n_electrodes_expected=96,
        in_bit_paper=False,
    ),
    # ---- .mat datasets (not on DANDI) ----
    "odoherty_2017": DatasetConfig(
        dataset_id="odoherty_2017",
        name="O'Doherty et al. 2017",
        data_type="mat_spikes",
        brain_area="M1",
        n_electrodes_expected=192,
        in_bit_paper=True,
    ),
    "chowdhury_2020": DatasetConfig(
        dataset_id="chowdhury_2020",
        name="Chowdhury et al. 2020",
        data_type="mat_spikes",
        brain_area="Area2/S1",
        n_electrodes_expected=96,  # probe shows 96 (BIT paper says 92)
        in_bit_paper=True,
    ),
    "ma_2023": DatasetConfig(
        dataset_id="ma_2023",
        name="Ma et al. 2023",
        data_type="mat_spikes",
        brain_area="M1",
        n_electrodes_expected=96,
        in_bit_paper=True,
    ),
}

# Datasets to SKIP
SKIP_DATASETS = {"001174"}  # Calcium imaging — incompatible modality


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(level: str = "INFO") -> logging.Logger: #Simple logging
    logger = logging.getLogger("nhp_preprocess")
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


# ==============================================================================
# NWB Processing: spike_times → binned spike counts
# ==============================================================================

def bin_spike_times(
    spike_times: np.ndarray, # all spike timestamps concatenated
    spike_times_index: np.ndarray, # cumulative index mapping spikes to units
    session_duration: float, # total session duration in seconds
    bin_size_s: float = 0.02,
) -> np.ndarray: # returns shape (T, C)
    """
    Bin spike times into fixed-width time bins for all units.
    
    Args:
        spike_times: Flat array of all spike timestamps across all units
        spike_times_index: Cumulative index array (unit i has spikes from 
                          spike_times_index[i-1] to spike_times_index[i])
        session_duration: Total session duration in seconds
        bin_size_s: Bin width in seconds (default 0.02 = 20ms)
    
    Returns:
        binned: (n_bins, n_units) array of spike counts per bin
    """
    n_units = len(spike_times_index)
    n_bins = int(np.ceil(session_duration / bin_size_s))
    bin_edges = np.arange(0, n_bins + 1) * bin_size_s
    
    binned = np.zeros((n_bins, n_units), dtype=np.float32)
    
    prev_idx = 0
    for unit_i in range(n_units):
        end_idx = spike_times_index[unit_i]
        unit_spikes = spike_times[prev_idx:end_idx] # extract spike times of this unit via the cumulative index
        
        if len(unit_spikes) > 0:
            # np.histogram is efficient for binning
            counts, _ = np.histogram(unit_spikes, bins=bin_edges) # bin spike timestamps into 20 ms windows
            binned[:, unit_i] = counts
        
        prev_idx = end_idx
    
    return binned # Returns a (n_bins, n_units) table of spike counts


def get_session_duration_nwb(f: h5py.File) -> float:
    """Extract session duration from an NWB file using multiple strategies."""
    
    # Strategy 1: trials stop_time
    if "intervals" in f and "trials" in f["intervals"]:
        trials = f["intervals/trials"]
        if "stop_time" in trials:
            return float(np.max(trials["stop_time"][:]))
    
    # Strategy 2: max spike time
    if "units" in f and "spike_times" in f["units"]:
        st = f["units/spike_times"]
        if len(st) > 0:
            return float(st[-1]) + 0.02  # add one bin
    
    # Strategy 3: acquisition timestamps
    if "acquisition" in f:
        for key in f["acquisition"]:
            if "timestamps" in f[f"acquisition/{key}"]:
                ts = f[f"acquisition/{key}/timestamps"]
                if len(ts) > 0:
                    return float(ts[-1])
    
    # Strategy 4: analysis timestamps (for 001201-type)
    if "analysis" in f:
        for key in f["analysis"]:
            if "timestamps" in f[f"analysis/{key}"]:
                ts = f[f"analysis/{key}/timestamps"]
                if len(ts) > 0:
                    return float(ts[-1])
    
    return 0.0


def process_spike_times_nwb( #Lee un archivo NWB concreto
    filepath: str,
    bin_size_s: float = 0.02,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Process a single NWB file with spike_times data.
    
    Returns:
        (binned_data, metadata) or None if file is invalid
    """
    if logger is None:
        logger = logging.getLogger("nhp_preprocess")
    
    try:
        with h5py.File(filepath, "r") as f:
            # Check for units table
            if "units" not in f or "spike_times" not in f["units"]:
                logger.warning(f"  No spike_times in {os.path.basename(filepath)}")
                return None
            
            spike_times = f["units/spike_times"][:]
            spike_times_index = f["units/spike_times_index"][:]
            n_units = len(spike_times_index)
            
            if n_units == 0 or len(spike_times) == 0:
                logger.warning(f"  Empty units in {os.path.basename(filepath)}")
                return None
            
            # Get session duration
            duration = get_session_duration_nwb(f)
            if duration <= 0:
                logger.warning(f"  Could not determine duration for {os.path.basename(filepath)}")
                return None
            
            # Bin spikes
            binned = bin_spike_times(spike_times, spike_times_index, duration, bin_size_s)
            
            metadata = {
                "n_units": n_units,
                "n_bins": binned.shape[0],
                "duration_s": duration,
                "bin_size_ms": bin_size_s * 1000,
                "source_file": os.path.basename(filepath),
            }
            
            return binned, metadata
            
    except Exception as e:
        logger.error(f"  Error processing {filepath}: {e}")
        return None


def process_binned_tc_nwb(
    filepath: str,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Process a single NWB file with pre-binned ThresholdCrossings (e.g., 001201).
    
    Returns:
        (binned_data, metadata) or None if file is invalid
    """
    if logger is None:
        logger = logging.getLogger("nhp_preprocess")
    
    try:
        with h5py.File(filepath, "r") as f:
            # Look for ThresholdCrossings in analysis/
            tc_path = None
            for candidate in [
                "analysis/ThresholdCrossings/data",
                "acquisition/ThresholdCrossings/data",
            ]:
                if candidate in f:
                    tc_path = candidate
                    break
            
            if tc_path is None:
                logger.warning(f"  No ThresholdCrossings in {os.path.basename(filepath)}")
                return None
            
            data = f[tc_path][:].astype(np.float32)  # (T, C)
            
            # Get timestamps to verify bin size
            ts_path = tc_path.replace("/data", "/timestamps")
            if ts_path in f:
                ts = f[ts_path][:10]
                if len(ts) >= 2:
                    bin_ms = (ts[1] - ts[0]) * 1000
                else:
                    bin_ms = 20.0
                duration = float(f[ts_path][-1])
            else:
                bin_ms = 20.0
                duration = data.shape[0] * 0.02
            
            metadata = {
                "n_units": data.shape[1],
                "n_bins": data.shape[0],
                "duration_s": duration,
                "bin_size_ms": bin_ms,
                "source_file": os.path.basename(filepath),
            }
            
            return data, metadata
            
    except Exception as e:
        logger.error(f"  Error processing {filepath}: {e}")
        return None


# ==============================================================================
# Dead channel detection and handling (BIT Appendix A.3)
# ==============================================================================

def detect_dead_channels(
    binned_data: np.ndarray,
    min_spikes: int = 10,
) -> np.ndarray:
    """
    Detect dead channels based on total spike count.
    
    A channel is "dead" if it has fewer than `min_spikes` total spikes
    across the entire session.
    
    Args:
        binned_data: (T, C) binned spike counts
        min_spikes: Minimum total spikes to be considered alive
    
    Returns:
        dead_mask: (C,) boolean array, True = dead
    """
    total_spikes = np.sum(binned_data, axis=0)
    return total_spikes < min_spikes #True o False, siendo true los canales muertos


def handle_dead_channels(
    binned_data: np.ndarray,
    dead_mask: np.ndarray,
    interpolate: bool = True,
) -> np.ndarray:
    """
    Handle dead channels by interpolation with temporal mean.
    
    Per BIT A.3: if <2 dead channels, interpolate with mean neural 
    activity over time.
    
    Args:
        binned_data: (T, C) array
        dead_mask: (C,) boolean, True = dead
        interpolate: if True, replace dead channels with temporal mean
    
    Returns:
        processed_data: (T, C) array with dead channels handled
    """
    if not interpolate or not np.any(dead_mask):
        return binned_data
    
    data = binned_data.copy() # avoid in-place modification of the input
    # Temporal mean of alive channels
    alive_mask = ~dead_mask # invert the dead-channel mask
    if np.any(alive_mask):
        temporal_mean = np.mean(data[:, alive_mask], axis=1, keepdims=True) # per-bin mean across living channels
        # Broadcast to dead channels
        data[:, dead_mask] = temporal_mean # assign that mean to the dead channels
    
    return data


# ==============================================================================
# Z-scoring across sessions (BIT Appendix A.3)
# ==============================================================================

def compute_zscore_stats(
    all_session_data: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std across all sessions for a subject.
    
    Args:
        all_session_data: List of (T_i, C) arrays, all same C
    
    Returns:
        global_mean: (C,) per-channel mean
        global_std: (C,) per-channel std
    """
    # Concatenate all sessions along time axis
    concatenated = np.concatenate(all_session_data, axis=0)  # (T_total, C) Todas las sesiones de un sujeto concatenadas
    global_mean = np.mean(concatenated, axis=0)
    global_std = np.std(concatenated, axis=0)
    return global_mean, global_std


def apply_zscore(
    data: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply z-scoring: (x - mean) / (std + eps)."""
    return (data - mean) / (std + eps)
# Compute z-score stats per subject across all sessions, then apply standard z-score

# ==============================================================================
# Subject extraction from NWB filenames
# ==============================================================================

def extract_subject_from_nwb(filepath: str) -> str:
    """
    Extract subject ID from NWB filename or path.
    
    Examples:
        sub-Jenkins_ses-xxx.nwb → Jenkins
        sub-Monkey-N_ses-xxx.nwb → Monkey-N
        sub-C_ses-CO-xxx.nwb → C
    """
    basename = os.path.basename(filepath)
    if basename.startswith("sub-"):
        # Extract subject between "sub-" and next "_"
        parts = basename.split("_")
        subject = parts[0].replace("sub-", "")
        return subject
    return "unknown"
# Z-score per subject; group files by subject identifier extracted from filename

# ==============================================================================
# MAT file processing (O'Doherty, Chowdhury, Ma)
# ==============================================================================

def process_mat_spikes(
    filepath: str,
    dataset_id: str,
    bin_size_s: float = 0.02,
    logger: Optional[logging.Logger] = None,
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Process a .mat file containing spike data.
    
    Handles both HDF5 v7.3 .mat (h5py) and classic .mat (scipy.io).
    Tries multiple field name conventions for spike data.
    """
    if logger is None:
        logger = logging.getLogger("nhp_preprocess")
    
    basename = os.path.basename(filepath)
    
    # Field name candidates (ordered by likelihood)
    # O'Doherty uses "spikes", Chowdhury likely uses "neural" or trial structs,
    # Ma likely uses spike times or binned counts
    SPIKE_FIELDS = [
        "spikes", "spike_times", "unit_spikes", "spikeRaster",
        "neural", "S", "units", "spike_counts_per_unit",
        "Spikes", "SpikeData", "sp", "spk",
    ]
    BINNED_FIELDS = [
        "spike_counts", "binned_spikes", "neural_data", "firing_rates",
        "FR", "Z", "spikeCounts", "binned", "rates", "tc",
        "ThresholdCrossings", "tx", "S1_spikes", "M1_spikes",
    ]
    TIME_FIELDS = ["t", "time", "timeframe", "bin_times", "timestamps", "T"]
    
    # Try HDF5 format first (MATLAB v7.3)
    try:
        with h5py.File(filepath, "r") as f:
            result = _extract_from_h5_mat(
                f, basename, SPIKE_FIELDS, BINNED_FIELDS, TIME_FIELDS,
                bin_size_s, logger
            )
            if result is not None:
                return result
    except Exception:
        pass
    
    # Try scipy for classic .mat
    if HAS_SCIPY:
        try:
            result = _extract_from_scipy_mat(
                filepath, basename, SPIKE_FIELDS, BINNED_FIELDS, TIME_FIELDS,
                bin_size_s, logger
            )
            if result is not None:
                return result
        except Exception as e:
            logger.debug(f"  scipy fallback failed for {basename}: {e}")
    
    logger.warning(f"  Could not extract spikes from {basename}")
    return None


def _rebin_to_target(
    data: np.ndarray,
    source_bin_ms: float,
    target_bin_ms: float,
) -> np.ndarray:
    """
    Rebin data from source_bin_ms to target_bin_ms by summing adjacent bins.
    
    E.g., 1ms bins → 20ms bins: sum every 20 consecutive bins.
    data: (T_source, C)
    Returns: (T_target, C)
    """
    ratio = int(round(target_bin_ms / source_bin_ms))
    if ratio <= 1:
        return data
    
    # Truncate to multiple of ratio
    n_keep = (data.shape[0] // ratio) * ratio
    data = data[:n_keep]
    
    # Reshape and sum
    T_new = data.shape[0] // ratio
    C = data.shape[1]
    return data.reshape(T_new, ratio, C).sum(axis=1)


def _extract_from_h5_mat(
    f: h5py.File,
    basename: str,
    spike_fields: list,
    binned_fields: list,
    time_fields: list,
    bin_size_s: float,
    logger: logging.Logger,
) -> Optional[Tuple[np.ndarray, dict]]:
    """Extract spike data from HDF5-based .mat file."""
    top_keys = list(f.keys())
    target_bin_ms = bin_size_s * 1000
    
    # ==========================================================
    # Strategy 0: xds group (Chowdhury/Ma format)
    # Contains spike_counts prebinned at 1ms, needs rebinning
    # ==========================================================
    if "xds" in f and isinstance(f["xds"], h5py.Group):
        xds = f["xds"]
        
        # Read prebinned spike counts
        if "spike_counts" in xds and isinstance(xds["spike_counts"], h5py.Dataset):
            sc = xds["spike_counts"]
            
            # Get original bin width
            orig_bin_ms = 1.0  # default
            if "bin_width" in xds and isinstance(xds["bin_width"], h5py.Dataset):
                orig_bin_ms = float(xds["bin_width"][0, 0]) * 1000  # s → ms
            
            # Read data — shape is (C, T) in MATLAB convention, need (T, C)
            logger.debug(f"  xds/spike_counts: shape={sc.shape}, orig_bin={orig_bin_ms}ms")
            binned = sc[:].astype(np.float32)
            if binned.shape[0] < binned.shape[1]:
                binned = binned.T  # (T, C)
            
            # Get duration from time_frame
            duration = binned.shape[0] * (orig_bin_ms / 1000.0)
            if "time_frame" in xds and isinstance(xds["time_frame"], h5py.Dataset):
                tf = xds["time_frame"][:].flatten()
                if len(tf) > 1:
                    duration = float(tf[-1] - tf[0])
            
            # Rebin to target (e.g., 1ms → 20ms)
            if abs(orig_bin_ms - target_bin_ms) > 0.5:
                binned = _rebin_to_target(binned, orig_bin_ms, target_bin_ms)
                logger.debug(
                    f"  Rebinned {orig_bin_ms}ms → {target_bin_ms}ms: "
                    f"({binned.shape[0]}, {binned.shape[1]})"
                )
            
            return binned, {
                "n_units": binned.shape[1],
                "n_bins": binned.shape[0],
                "duration_s": duration,
                "bin_size_ms": target_bin_ms,
                "source_file": basename,
                "original_bin_ms": orig_bin_ms,
            }
        
        # Fallback: xds/spikes (raw spike times)
        if "spikes" in xds and isinstance(xds["spikes"], h5py.Dataset):
            spikes_obj = xds["spikes"]
            if spikes_obj.dtype == h5py.ref_dtype:
                n_channels = max(spikes_obj.shape)
                
                # Get duration from meta or time_frame
                duration = 0
                if "time_frame" in xds:
                    tf = xds["time_frame"][:].flatten()
                    if len(tf) > 1:
                        duration = float(tf[-1] - tf[0])
                elif "meta" in xds and "duration" in xds["meta"]:
                    duration = float(xds["meta"]["duration"][0, 0])
                
                if duration <= 0:
                    max_t = 0
                    for i in range(min(n_channels, 5)):  # sample a few
                        try:
                            ref = spikes_obj[i, 0] if spikes_obj.shape[0] >= spikes_obj.shape[1] else spikes_obj[0, i]
                            st = f[ref][:].flatten()
                            if len(st) > 0:
                                max_t = max(max_t, float(np.max(st)))
                        except Exception:
                            pass
                    duration = max_t + bin_size_s
                
                if duration > 0:
                    t_start = 0
                    if "time_frame" in xds:
                        tf = xds["time_frame"][:].flatten()
                        if len(tf) > 0:
                            t_start = float(tf[0])
                    
                    n_bins = int(np.ceil(duration / bin_size_s))
                    bin_edges = np.arange(0, n_bins + 1) * bin_size_s + t_start
                    binned = np.zeros((n_bins, n_channels), dtype=np.float32)
                    
                    for ch in range(n_channels):
                        try:
                            ref = spikes_obj[ch, 0] if spikes_obj.shape[0] >= spikes_obj.shape[1] else spikes_obj[0, ch]
                            st = f[ref][:].flatten()
                            if len(st) > 0:
                                counts, _ = np.histogram(st, bins=bin_edges)
                                binned[:, ch] = counts
                        except Exception:
                            pass
                    
                    return binned, {
                        "n_units": n_channels, "n_bins": n_bins,
                        "duration_s": duration, "bin_size_ms": target_bin_ms,
                        "source_file": basename,
                    }
    
    # ==========================================================
    # Strategy 1: Pre-binned data at top level
    # ==========================================================
    for field in binned_fields:
        if field in f and isinstance(f[field], h5py.Dataset) and f[field].ndim == 2:
            binned = f[field][:].astype(np.float32)
            if binned.shape[0] < binned.shape[1]:
                binned = binned.T
            duration = binned.shape[0] * bin_size_s
            for tf in time_fields:
                if tf in f and isinstance(f[tf], h5py.Dataset):
                    arr = f[tf][:].flatten()
                    if len(arr) > 1:
                        duration = float(arr[-1] - arr[0])
                        break
            return binned, {
                "n_units": binned.shape[1], "n_bins": binned.shape[0],
                "duration_s": duration, "bin_size_ms": bin_size_s * 1000,
                "source_file": basename,
            }
    
    # ==========================================================
    # Strategy 2: Spike times as cell array (object references)
    # (O'Doherty format: top-level 'spikes' + 't')
    # Handles both 1D refs (N_channels,) and 2D refs (N_sorts, N_channels)
    # For 2D: row 0 = unsorted/all spikes (what we want)
    # ==========================================================
    for field in spike_fields:
        if field not in f:
            continue
        spikes_obj = f[field]
        if not (isinstance(spikes_obj, h5py.Dataset) and spikes_obj.dtype == h5py.ref_dtype):
            continue
        
        # Determine layout
        sp_shape = spikes_obj.shape
        if len(sp_shape) == 2:
            # 2D: (n_sorts, n_channels) — e.g., O'Doherty (5, 96)
            # Row 0 = unsorted/all spikes
            n_channels = sp_shape[1]
            spike_row = 0
            logger.debug(f"  {field}: 2D refs {sp_shape}, using row {spike_row} (unsorted)")
        else:
            # 1D: (n_channels,)
            n_channels = max(sp_shape)
            spike_row = None
        
        def _get_ref(ch_idx):
            """Get reference for channel, handling 1D and 2D layouts."""
            if spike_row is not None:
                return spikes_obj[spike_row, ch_idx]
            else:
                # 1D — try direct indexing
                if spikes_obj.ndim == 1:
                    return spikes_obj[ch_idx]
                else:
                    return spikes_obj[ch_idx, 0] if sp_shape[0] > sp_shape[1] else spikes_obj[0, ch_idx]
        
        # Get duration from time field
        duration = 0
        for tf in time_fields:
            if tf in f and isinstance(f[tf], h5py.Dataset):
                arr = f[tf][:].flatten()
                if len(arr) > 1:
                    duration = float(arr[-1] - arr[0])
                    break
        
        if duration <= 0:
            max_t = 0
            for i in range(min(n_channels, 10)):
                try:
                    ref = _get_ref(i)
                    st = f[ref][:].flatten()
                    if len(st) > 0:
                        max_t = max(max_t, float(np.max(st)))
                except Exception:
                    pass
            duration = max_t + bin_size_s
        
        if duration <= 0:
            return None
        
        t_start = 0
        for tf in time_fields:
            if tf in f and isinstance(f[tf], h5py.Dataset):
                arr = f[tf][:].flatten()
                if len(arr) > 0:
                    t_start = float(arr[0])
                    break
        
        n_bins = int(np.ceil(duration / bin_size_s))
        bin_edges = np.arange(0, n_bins + 1) * bin_size_s + t_start
        binned = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        for ch in range(n_channels):
            try:
                ref = _get_ref(ch)
                st = f[ref][:].flatten()
                if len(st) > 0:
                    counts, _ = np.histogram(st, bins=bin_edges)
                    binned[:, ch] = counts
            except Exception:
                pass
        
        return binned, {
            "n_units": n_channels, "n_bins": n_bins,
            "duration_s": duration, "bin_size_ms": bin_size_s * 1000,
            "source_file": basename,
        }
    
    logger.debug(f"  No recognized spike data in {basename} (h5). Keys: {top_keys}")
    return None


def _extract_from_scipy_mat(
    filepath: str,
    basename: str,
    spike_fields: list,
    binned_fields: list,
    time_fields: list,
    bin_size_s: float,
    logger: logging.Logger,
) -> Optional[Tuple[np.ndarray, dict]]:
    """Extract spike data from classic MATLAB .mat file via scipy."""
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=True)
    mat_keys = [k for k in mat.keys() if not k.startswith("__")]
    
    # Build a flat lookup: check top-level AND one-level-deep struct fields
    # This handles datasets where spikes are inside a struct, e.g. data.spikes
    flat_lookup = {}
    for k in mat_keys:
        flat_lookup[k] = mat[k]
        # If it's a struct (has dtype.names), expose its fields too
        v = mat[k]
        if isinstance(v, np.ndarray) and v.dtype.names:
            for fname in v.dtype.names:
                try:
                    fval = v[fname].flatten()[0]
                    flat_lookup[f"{k}.{fname}"] = fval
                    # Also add just the field name (for generic matching)
                    if fname not in flat_lookup:
                        flat_lookup[fname] = fval
                except Exception:
                    pass
    
    target_bin_ms = bin_size_s * 1000
    
    # Strategy 1: Pre-binned data (possibly at different bin size)
    for field in binned_fields:
        val = flat_lookup.get(field)
        if val is not None and isinstance(val, np.ndarray) and val.ndim == 2:
            binned = val.astype(np.float32)
            if binned.shape[0] < binned.shape[1]:
                binned = binned.T
            
            # Detect source bin size from metadata
            source_bin_ms = target_bin_ms  # assume target by default
            for bs_name in ["bin_size", "trial_data.bin_size", "binSize", "dt"]:
                bs_val = flat_lookup.get(bs_name)
                if bs_val is not None:
                    try:
                        bs_float = float(np.asarray(bs_val).flatten()[0])
                        if bs_float < 1:  # seconds → ms
                            source_bin_ms = bs_float * 1000
                        else:
                            source_bin_ms = bs_float
                        logger.debug(f"  Detected source bin_size: {source_bin_ms}ms")
                        break
                    except Exception:
                        pass
            
            # Rebin if needed
            if abs(source_bin_ms - target_bin_ms) > 0.5:
                binned = _rebin_to_target(binned, source_bin_ms, target_bin_ms)
                logger.debug(
                    f"  Rebinned {source_bin_ms}ms → {target_bin_ms}ms: "
                    f"({binned.shape[0]}, {binned.shape[1]})"
                )
            
            duration = binned.shape[0] * bin_size_s
            for tf in time_fields:
                tv = flat_lookup.get(tf)
                if tv is not None:
                    t = np.asarray(tv).flatten()
                    if len(t) > 1:
                        duration = float(t[-1] - t[0])
                        break
            return binned, {
                "n_units": binned.shape[1], "n_bins": binned.shape[0],
                "duration_s": duration, "bin_size_ms": target_bin_ms,
                "source_file": basename,
                "original_bin_ms": source_bin_ms,
            }
    
    # Strategy 2: Spike times as object array (cell array in MATLAB)
    for field in spike_fields:
        val = flat_lookup.get(field)
        if val is None:
            continue
        if not isinstance(val, np.ndarray):
            continue
        
        # Object array: each element is a vector of spike times
        if val.dtype == object:
            spikes_flat = val.flatten()
            n_channels = len(spikes_flat)
        # 2D numeric array where columns are units (pre-sorted spike raster)
        elif val.ndim == 2 and val.dtype in (np.float64, np.float32, np.int32):
            # This is likely already binned or a raster — treat as binned
            binned = val.astype(np.float32)
            if binned.shape[0] < binned.shape[1]:
                binned = binned.T
            duration = binned.shape[0] * bin_size_s
            return binned, {
                "n_units": binned.shape[1], "n_bins": binned.shape[0],
                "duration_s": duration, "bin_size_ms": bin_size_s * 1000,
                "source_file": basename,
            }
        else:
            continue
        
        # Get duration from time field
        duration = 0
        t_start = 0
        for tf in time_fields:
            tv = flat_lookup.get(tf)
            if tv is not None:
                t = np.asarray(tv).flatten()
                if len(t) > 1:
                    t_start = float(t[0])
                    duration = float(t[-1] - t[0])
                    break
        
        if duration <= 0:
            all_max = []
            all_min = []
            for ch_sp in spikes_flat:
                if ch_sp is not None and hasattr(ch_sp, '__len__') and len(ch_sp) > 0:
                    arr = np.asarray(ch_sp).flatten()
                    all_max.append(float(np.max(arr)))
                    all_min.append(float(np.min(arr)))
            if all_max:
                t_start = min(all_min) if all_min else 0
                duration = max(all_max) - t_start + bin_size_s
        
        if duration <= 0:
            continue
        
        n_bins = int(np.ceil(duration / bin_size_s))
        bin_edges = np.arange(0, n_bins + 1) * bin_size_s + t_start
        binned = np.zeros((n_bins, n_channels), dtype=np.float32)
        
        for ch in range(n_channels):
            ch_sp = spikes_flat[ch]
            if ch_sp is not None and hasattr(ch_sp, '__len__') and len(ch_sp) > 0:
                counts, _ = np.histogram(np.asarray(ch_sp).flatten(), bins=bin_edges)
                binned[:, ch] = counts
        
        return binned, {
            "n_units": n_channels, "n_bins": n_bins,
            "duration_s": duration, "bin_size_ms": bin_size_s * 1000,
            "source_file": basename,
        }
    
    # Strategy 3: Trial-based struct arrays (common in Chowdhury/Ma)
    # Look for arrays of structs where each element has a 'spikes' or 'neural' field
    for k in mat_keys:
        val = mat[k]
        if not isinstance(val, np.ndarray):
            continue
        if val.dtype.names is None:
            continue
        
        # Check if this struct has neural data fields
        neural_field = None
        for candidate in spike_fields + binned_fields:
            if candidate in val.dtype.names:
                neural_field = candidate
                break
        if neural_field is None:
            continue
        
        # It's a trial struct — concatenate neural data across trials
        logger.info(f"  Found trial-based struct '{k}' with field '{neural_field}'")
        
        all_trial_data = []
        trials = val.flatten()
        for trial in trials:
            try:
                tdata = trial[neural_field]
                if isinstance(tdata, np.ndarray) and tdata.ndim >= 1:
                    if tdata.dtype == object:
                        # Each trial has a cell array of spike times → need binning
                        # Skip trial-level binning for now, this is complex
                        continue
                    elif tdata.ndim == 2:
                        all_trial_data.append(tdata.astype(np.float32))
            except Exception:
                continue
        
        if all_trial_data:
            # Ensure consistent channel count
            n_chs = [d.shape[1] if d.ndim == 2 else d.shape[0] for d in all_trial_data]
            target_ch = max(set(n_chs), key=n_chs.count)  # mode
            filtered = [d for d in all_trial_data if d.shape[1] == target_ch]
            
            if filtered:
                concatenated = np.concatenate(filtered, axis=0)
                if concatenated.shape[0] < concatenated.shape[1]:
                    concatenated = concatenated.T
                duration = concatenated.shape[0] * bin_size_s
                return concatenated, {
                    "n_units": concatenated.shape[1],
                    "n_bins": concatenated.shape[0],
                    "duration_s": duration,
                    "bin_size_ms": bin_size_s * 1000,
                    "source_file": basename,
                }
    
    logger.debug(f"  No recognized spike data in {basename} (scipy). Keys: {mat_keys}")
    return None


def extract_subject_from_mat(filepath: str, dataset_id: str) -> str:
    """
    Extract subject/monkey name from .mat filename or path.
    
    Handles various naming conventions:
      O'Doherty: indy_20160407_02.mat → Indy, loco_20170210.mat → Loco
      Chowdhury: Han_20171122.mat → Han, Chips_20170913.mat → Chips
      Ma: sub-MonkeyName/session.mat → MonkeyName, or monkeyname_date.mat → Monkeyname
    """
    basename = os.path.basename(filepath).lower()
    basename_noext = os.path.splitext(basename)[0]
    
    if dataset_id == "odoherty_2017":
        # indy_20160407_02.mat → Indy, loco_20170210_03.mat → Loco
        first_part = basename_noext.split("_")[0]
        if first_part in ("indy", "loco"):
            return first_part.capitalize()
        return first_part.capitalize()
    
    elif dataset_id == "chowdhury_2020":
        # Files: C_20170907_TRT_TD.mat, H_20171101_TRT_TD.mat, L_20170921_TRT_TD.mat
        # C=Chips, H=Han, L=Lando (single letter prefix)
        first_part = basename_noext.split("_")[0].lower()
        letter_map = {"c": "Chips", "h": "Han", "l": "Lando"}
        known_monkeys = {"han", "chips", "lando"}
        
        if first_part in letter_map:
            return letter_map[first_part]
        if first_part in known_monkeys:
            return first_part.capitalize()
        # Check parent dir
        parent = os.path.basename(os.path.dirname(filepath)).lower()
        if parent in known_monkeys:
            return parent.capitalize()
        return first_part.capitalize()
    
    elif dataset_id == "ma_2023":
        # Files in: doi_.../Chewie_CO_2016/Chewie_20160927_001.mat
        # or: doi_.../Greyson_Key_2019/Greyson_20190812_Key_001.mat
        # Subject is the first part of filename before underscore+date
        first_part = basename_noext.split("_")[0]
        known_monkeys = {"chewie", "greyson", "mihili", "jaco", "jango", "lando", "thor", "spike"}
        if first_part.lower() in known_monkeys:
            return first_part.capitalize()
        # Check parent directories
        parent = os.path.basename(os.path.dirname(filepath))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        for level in [parent, grandparent]:
            level_lower = level.lower()
            for monkey in known_monkeys:
                if level_lower.startswith(monkey):
                    return monkey.capitalize()
        return first_part.capitalize()
    
    # Generic fallback
    first_part = basename_noext.split("_")[0]
    parent = os.path.basename(os.path.dirname(filepath))
    if parent and parent.lower() not in (dataset_id, ".", ""):
        return parent.capitalize()
    return first_part.capitalize()


# ==============================================================================
# Probe / inspect .mat files (for debugging before processing)
# ==============================================================================

def probe_mat_file(filepath: str, max_depth: int = 3) -> str:
    """
    Inspect a .mat file and return a formatted string describing its structure.
    Works with both HDF5 (v7.3) and classic .mat files.
    """
    lines = [f"\n{'='*70}", f"PROBING: {os.path.basename(filepath)}", f"{'='*70}"]
    
    # Try HDF5 first
    try:
        with h5py.File(filepath, "r") as f:
            lines.append(f"Format: HDF5 / MATLAB v7.3")
            lines.append(f"Top-level keys: {list(f.keys())}")
            lines.append("")
            
            def _walk_h5(obj, prefix="", depth=0):
                if depth > max_depth:
                    return
                for key in obj.keys():
                    item = obj[key]
                    indent = "  " * depth
                    if isinstance(item, h5py.Dataset):
                        dtype_str = str(item.dtype)
                        if item.dtype == h5py.ref_dtype:
                            dtype_str = "object_refs (cell array)"
                        lines.append(
                            f"{indent}{prefix}{key}: Dataset "
                            f"shape={item.shape} dtype={dtype_str}"
                        )
                        # Show a few values for small datasets
                        if item.size <= 10 and item.dtype != h5py.ref_dtype:
                            try:
                                vals = item[:].flatten()[:5]
                                lines.append(f"{indent}  values (first 5): {vals}")
                            except Exception:
                                pass
                        # For ref arrays, show what one reference looks like
                        elif item.dtype == h5py.ref_dtype and item.size > 0:
                            try:
                                ref = item.flat[0]
                                deref = f[ref]
                                lines.append(
                                    f"{indent}  → ref[0]: shape={deref.shape} "
                                    f"dtype={deref.dtype}"
                                )
                                if deref.size <= 5:
                                    lines.append(
                                        f"{indent}    values: {deref[:].flatten()[:5]}"
                                    )
                            except Exception:
                                pass
                    elif isinstance(item, h5py.Group):
                        lines.append(f"{indent}{prefix}{key}: Group ({len(item)} items)")
                        _walk_h5(item, prefix=f"{key}/", depth=depth + 1)
            
            _walk_h5(f)
            return "\n".join(lines)
    except Exception:
        pass
    
    # Try scipy
    if HAS_SCIPY:
        try:
            mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=True)
            lines.append(f"Format: Classic MATLAB (v5/v7)")
            mat_keys = [k for k in mat.keys() if not k.startswith("__")]
            lines.append(f"Variables: {mat_keys}")
            lines.append("")
            
            def _describe_var(val, indent=0):
                prefix = "  " * indent
                if isinstance(val, np.ndarray):
                    if val.dtype == object:
                        lines.append(f"{prefix}  object array, shape={val.shape}")
                        # Show first element
                        flat = val.flatten()
                        if len(flat) > 0 and flat[0] is not None:
                            first = flat[0]
                            if isinstance(first, np.ndarray):
                                lines.append(
                                    f"{prefix}  → [0]: ndarray shape={first.shape} "
                                    f"dtype={first.dtype}"
                                )
                            else:
                                lines.append(f"{prefix}  → [0]: {type(first).__name__}")
                    elif val.dtype.names:
                        # Struct
                        lines.append(f"{prefix}  struct, shape={val.shape}")
                        lines.append(f"{prefix}  fields: {list(val.dtype.names)}")
                        if indent < max_depth:
                            for fname in val.dtype.names[:10]:
                                lines.append(f"{prefix}  .{fname}:")
                                try:
                                    fval = val[fname].flatten()[0]
                                    if isinstance(fval, np.ndarray):
                                        lines.append(
                                            f"{prefix}    ndarray shape={fval.shape} "
                                            f"dtype={fval.dtype}"
                                        )
                                    else:
                                        lines.append(
                                            f"{prefix}    {type(fval).__name__}: {fval}"
                                        )
                                except Exception:
                                    pass
                    else:
                        lines.append(
                            f"{prefix}  ndarray shape={val.shape} dtype={val.dtype}"
                        )
                        if val.size <= 5:
                            lines.append(f"{prefix}  values: {val.flatten()[:5]}")
                else:
                    lines.append(f"{prefix}  {type(val).__name__}: {val}")
            
            for key in mat_keys:
                lines.append(f"  {key}:")
                _describe_var(mat[key], indent=1)
            
            return "\n".join(lines)
        except Exception as e:
            lines.append(f"Format: Unknown (failed to open: {e})")
            return "\n".join(lines)
    
    lines.append("Cannot open file (neither h5py nor scipy succeeded)")
    return "\n".join(lines)


def probe_dataset(dataset_id: str, data_root: str, max_files: int = 3):
    """Probe the first N files of a dataset to understand its structure."""
    dataset_dir = os.path.join(data_root, dataset_id)
    if not os.path.isdir(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return
    
    mat_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.mat"), recursive=True))
    nwb_files = sorted(glob.glob(os.path.join(dataset_dir, "**/*.nwb"), recursive=True))
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_id}")
    print(f"Location: {dataset_dir}")
    print(f"Files found: {len(mat_files)} .mat, {len(nwb_files)} .nwb")
    
    # Show directory structure
    print(f"\nDirectory structure (first 20 entries):")
    all_files = sorted(mat_files + nwb_files)
    for f in all_files[:20]:
        rel = os.path.relpath(f, dataset_dir)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  {rel}  ({size_mb:.1f} MB)")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more files")
    
    # Probe first N .mat files
    for fpath in mat_files[:max_files]:
        print(probe_mat_file(fpath))
    
    # Probe first N .nwb files
    for fpath in nwb_files[:max_files]:
        print(f"\n{'='*70}")
        print(f"PROBING NWB: {os.path.basename(fpath)}")
        try:
            with h5py.File(fpath, "r") as f:
                print(f"Top-level keys: {list(f.keys())}")
                if "units" in f:
                    units = f["units"]
                    print(f"  units: {list(units.keys())}")
                if "acquisition" in f:
                    acq = f["acquisition"]
                    print(f"  acquisition: {list(acq.keys())}")
        except Exception as e:
            print(f"  Error: {e}")


# ==============================================================================
# Main processing pipeline per dataset
# ==============================================================================

def process_dataset(
    dataset_cfg: DatasetConfig,
    pipeline_cfg: PipelineConfig,
    logger: logging.Logger,
) -> Dict[str, list]:
    """
    Process all files for a dataset (NWB or .mat).
    
    Returns:
        subject_data: {subject_id: [(binned_data, metadata), ...]}
    """
    dataset_root = os.path.join(pipeline_cfg.monkey_data_root, dataset_cfg.dataset_id)
    
    if not os.path.isdir(dataset_root):
        logger.error(f"Dataset directory not found: {dataset_root}")
        return {}
    
    # Find files based on data type
    is_mat = dataset_cfg.data_type == "mat_spikes"
    if is_mat:
        files = sorted(glob.glob(os.path.join(dataset_root, "**/*.mat"), recursive=True))
        logger.info(f"Found {len(files)} .mat files in {dataset_cfg.dataset_id}")
    else:
        files = sorted(glob.glob(os.path.join(dataset_root, "**/*.nwb"), recursive=True))
        logger.info(f"Found {len(files)} NWB files in {dataset_cfg.dataset_id}")
    
    if len(files) == 0:
        return {}
    
    # Group by subject
    subject_data: Dict[str, list] = {}
    bin_size_s = pipeline_cfg.bin_size_ms / 1000.0
    
    for i, fpath in enumerate(files):
        # Extract subject
        if is_mat:
            subject = extract_subject_from_mat(fpath, dataset_cfg.dataset_id)
        else:
            subject = extract_subject_from_nwb(fpath)
        
        logger.info(f"  [{i+1}/{len(files)}] Processing {os.path.basename(fpath)} ...")
        
        # Process based on data type
        if dataset_cfg.data_type == "spike_times":
            result = process_spike_times_nwb(fpath, bin_size_s, logger)
        elif dataset_cfg.data_type == "binned_tc":
            result = process_binned_tc_nwb(fpath, logger)
        elif dataset_cfg.data_type == "mat_spikes":
            result = process_mat_spikes(fpath, dataset_cfg.dataset_id, bin_size_s, logger)
        else:
            logger.warning(f"  Unknown data_type: {dataset_cfg.data_type}")
            continue
        
        if result is None:
            continue
        
        binned, meta = result
        
        # Skip very short sessions
        if meta["duration_s"] < pipeline_cfg.min_session_duration_s:
            logger.debug(f"  Skipping short session: {meta['duration_s']:.1f}s < {pipeline_cfg.min_session_duration_s}s")
            continue
        
        # Dead channel detection
        session_duration_min = meta["duration_s"] / 60.0
        scaled_min_spikes = max(3, int(pipeline_cfg.min_spikes_alive * min(1.0, session_duration_min / 10.0)))
        dead_mask = detect_dead_channels(binned, scaled_min_spikes)
        n_dead = int(np.sum(dead_mask))
        meta["n_dead_channels"] = n_dead
        meta["dead_channel_indices"] = np.where(dead_mask)[0].tolist()
        
        if n_dead > pipeline_cfg.max_dead_channels:
            logger.warning(
                f"  EXCLUDING {os.path.basename(fpath)}: {n_dead} dead channels "
                f"(> {pipeline_cfg.max_dead_channels} max)"
            )
            continue
        
        if n_dead > 0:
            logger.info(f"  {os.path.basename(fpath)}: {n_dead} dead channels → interpolating")
            binned = handle_dead_channels(binned, dead_mask, pipeline_cfg.interpolate_dead)
        
        # Store (use subject_name from meta if available, e.g. from mat processor)
        subj = meta.get("subject_name", subject)
        if subj not in subject_data:
            subject_data[subj] = []
        subject_data[subj].append((binned, meta))
    
    return subject_data


def zscore_and_save(
    subject_data: Dict[str, list],
    dataset_cfg: DatasetConfig,
    pipeline_cfg: PipelineConfig,
    logger: logging.Logger,
) -> dict:
    """
    Z-score across sessions per subject and save to HDF5.
    
    Returns:
        stats: summary statistics for logging
    """
    stats = {}
    
    for subject_id, sessions in subject_data.items():
        if len(sessions) == 0:
            continue
        
        # Pad all sessions to the max channel count within this subject.
        # In spike-sorted datasets, n_units varies per session. Rather than 
        # fragmenting into one "subject" per channel count (which breaks z-scoring
        # and creates 80+ tiny subjects), we pad shorter sessions with zeros.
        # After z-scoring, padded channels → ~0, so the model learns to ignore them.
        n_channels_list = [s[0].shape[1] for s in sessions]
        max_ch = max(n_channels_list)
        min_ch = min(n_channels_list)
        
        if max_ch != min_ch:
            logger.info(
                f"  Subject {subject_id}: channel counts vary {min_ch}–{max_ch}. "
                f"Padding all to {max_ch}."
            )
            padded_sessions = []
            for sess_data, sess_meta in sessions:
                n_ch_sess = sess_data.shape[1]
                if n_ch_sess < max_ch:
                    pad_width = max_ch - n_ch_sess
                    sess_data = np.pad(
                        sess_data, ((0, 0), (0, pad_width)),
                        mode="constant", constant_values=0
                    )
                    sess_meta["n_units_original"] = n_ch_sess
                    sess_meta["n_units_padded"] = max_ch
                padded_sessions.append((sess_data, sess_meta))
            sessions = padded_sessions
        
        n_ch = max_ch
        subj_key = f"{dataset_cfg.dataset_id}_{subject_id}"
        
        # Compute z-score statistics across all sessions
        all_data = [s[0] for s in sessions]
        
        if pipeline_cfg.zscore_across_sessions:
            global_mean, global_std = compute_zscore_stats(all_data)
        
        # Create output directory
        out_dir = os.path.join(pipeline_cfg.output_dir, subj_key)
        os.makedirs(out_dir, exist_ok=True)
        
        total_bins = 0
        total_duration = 0.0
        
        for sess_idx, (sess_data, sess_meta) in enumerate(sessions):
            # Apply z-scoring
            if pipeline_cfg.zscore_across_sessions:
                sess_data = apply_zscore(
                    sess_data, global_mean, global_std, pipeline_cfg.zscore_eps
                )
            
            # Save
            out_path = os.path.join(out_dir, f"session_{sess_idx:04d}.h5")
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset(
                    "neural_data", 
                    data=sess_data.astype(np.float32),
                    compression="gzip",
                    compression_opts=4,
                )
                # Metadata
                hf.attrs["subject_id"] = subj_key
                hf.attrs["dataset_id"] = dataset_cfg.dataset_id
                hf.attrs["dataset_name"] = dataset_cfg.name
                hf.attrs["brain_area"] = dataset_cfg.brain_area
                hf.attrs["n_electrodes"] = int(n_ch)
                hf.attrs["n_electrodes_original"] = int(sess_meta.get("n_units_original", n_ch))
                hf.attrs["n_bins"] = int(sess_data.shape[0])
                hf.attrs["duration_s"] = float(sess_meta["duration_s"])
                hf.attrs["bin_size_ms"] = float(sess_meta["bin_size_ms"])
                hf.attrs["source_file"] = sess_meta["source_file"]
                hf.attrs["n_dead_channels"] = int(sess_meta.get("n_dead_channels", 0))
                hf.attrs["zscore_applied"] = pipeline_cfg.zscore_across_sessions
                hf.attrs["session_index"] = sess_idx
            
            total_bins += sess_data.shape[0]
            total_duration += sess_meta["duration_s"]
        
        # Save z-score stats for reproducibility
        if pipeline_cfg.zscore_across_sessions:
            stats_path = os.path.join(out_dir, "zscore_stats.h5")
            with h5py.File(stats_path, "w") as hf:
                hf.create_dataset("mean", data=global_mean)
                hf.create_dataset("std", data=global_std)
                hf.attrs["n_sessions"] = len(sessions)
                hf.attrs["total_bins"] = int(total_bins)
                hf.attrs["max_channels"] = int(max_ch)
                hf.attrs["min_channels_original"] = int(min_ch)
        
        stats[subj_key] = {
            "n_sessions": len(sessions),
            "n_electrodes": n_ch,
            "total_hours": total_duration / 3600,
            "total_bins": total_bins,
        }
        
        logger.info(
            f"  {subj_key}: {len(sessions)} sessions, "
            f"{n_ch} channels ({min_ch}–{max_ch} original), "
            f"{total_duration/3600:.2f}h"
        )
    
    return stats


# ==============================================================================
# Summary manifest
# ==============================================================================

def write_manifest(
    all_stats: Dict[str, dict],
    pipeline_cfg: PipelineConfig,
    logger: logging.Logger,
):
    """Write a summary manifest of all processed data."""
    manifest_path = os.path.join(pipeline_cfg.output_dir, "manifest.txt")
    
    total_hours = 0
    total_sessions = 0
    
    lines = ["NHP Pretraining Data Manifest", "=" * 50, ""]
    
    for subj_key, stats in sorted(all_stats.items()):
        line = (
            f"{subj_key:40s} | "
            f"{stats['n_sessions']:4d} sessions | "
            f"{stats['n_electrodes']:4d} ch | "
            f"{stats['total_hours']:7.2f}h"
        )
        lines.append(line)
        total_hours += stats["total_hours"]
        total_sessions += stats["n_sessions"]
    
    lines.extend([
        "",
        "-" * 50,
        f"{'TOTAL':40s} | {total_sessions:4d} sessions | {'':8s} {total_hours:7.2f}h",
        f"",
        f"BIT paper used ~269h monkey data for comparison.",
    ])
    
    with open(manifest_path, "w") as f:
        f.write("\n".join(lines))
    
    logger.info(f"\nManifest written to {manifest_path}")
    logger.info(f"TOTAL: {total_sessions} sessions, {total_hours:.2f}h")


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Preprocess NHP data for SSL pretraining")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Dataset IDs to process (e.g., 000070 001201). Default: all registered."
    )
    parser.add_argument(
        "--monkey-data-root", type=str, default=None,
        help="Override monkey data root directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--no-zscore", action="store_true",
        help="Disable z-scoring (for debugging)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file (overrides command-line args)"
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Probe/inspect dataset files without processing. Shows structure of .mat/.nwb files."
    )
    parser.add_argument(
        "--probe-max-files", type=int, default=3,
        help="Max files to probe per dataset (default: 3)"
    )
    args = parser.parse_args()
    
    # Build pipeline config
    cfg = PipelineConfig()
    
    # Load YAML config if provided
    if args.config and HAS_YAML:
        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f)
        for k, v in yaml_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    
    # CLI overrides
    if args.monkey_data_root:
        cfg.monkey_data_root = args.monkey_data_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_zscore:
        cfg.zscore_across_sessions = False
    cfg.log_level = args.log_level
    
    # Setup
    logger = setup_logging(cfg.log_level)
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("NHP Preprocessing Pipeline for SSL Pretraining")
    logger.info("=" * 60)
    logger.info(f"Data root: {cfg.monkey_data_root}")
    logger.info(f"Output dir: {cfg.output_dir}")
    logger.info(f"Bin size: {cfg.bin_size_ms}ms")
    logger.info(f"Z-score: {cfg.zscore_across_sessions}")
    logger.info(f"Dead channel policy: max {cfg.max_dead_channels}, interpolate={cfg.interpolate_dead}")
    logger.info("")
    
    # Select datasets
    if args.datasets:
        dataset_ids = args.datasets
    else:
        dataset_ids = sorted(DATASET_REGISTRY.keys())
    
    # ---- Probe mode: inspect files without processing ----
    if args.probe:
        logger.info("PROBE MODE — inspecting file structures")
        for ds_id in dataset_ids:
            if ds_id in SKIP_DATASETS:
                continue
            probe_dataset(ds_id, cfg.monkey_data_root, max_files=args.probe_max_files)
        logger.info("\nProbe complete!")
        return
    
    # Process each dataset
    all_stats = {}
    
    for ds_id in dataset_ids:
        if ds_id in SKIP_DATASETS:
            logger.info(f"Skipping {ds_id} (in skip list)")
            continue
        
        if ds_id not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset ID: {ds_id}, skipping")
            continue
        
        ds_cfg = DATASET_REGISTRY[ds_id]
        ds_cfg.root_path = os.path.join(cfg.monkey_data_root, ds_id)
        
        logger.info(f"{'='*60}")
        logger.info(f"Processing {ds_id} — {ds_cfg.name}")
        logger.info(f"  Type: {ds_cfg.data_type}, Area: {ds_cfg.brain_area}, BIT: {ds_cfg.in_bit_paper}")
        logger.info(f"{'='*60}")
        
        # Process all files
        subject_data = process_dataset(ds_cfg, cfg, logger)
        
        if not subject_data:
            logger.warning(f"  No valid data from {ds_id}")
            continue
        
        # Z-score and save
        stats = zscore_and_save(subject_data, ds_cfg, cfg, logger)
        all_stats.update(stats)
    
    # Write manifest
    if all_stats:
        write_manifest(all_stats, cfg, logger)
    else:
        logger.warning("No data was processed!")
    
    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    main()
