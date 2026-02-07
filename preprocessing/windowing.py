"""
Intelligent windowing and segmentation for PPG signals.
Optimized for temporal model input (CNN, LSTM, CNN-LSTM).
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from scipy import signal as sp_signal


def segment_signal(sig: np.ndarray, fs: float, window_duration: float = 10.0, 
                   overlap: float = 0.5, labels: Optional[Dict] = None, 
                   min_length: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Segment signal into overlapping windows and synchronize labels if provided.
    
    Args:
        sig: Input signal (1D array)
        fs: Sampling frequency (Hz)
        window_duration: Window duration in seconds, default 10.0
        overlap: Overlap fraction (0-1), default 0.5 (50%)
        labels: Optional dict of label arrays (e.g., {'hr': hr_array})
        min_length: Minimum signal length to process (optional)
        
    Returns:
        Tuple of (windows, indices, segmented_labels)
    """
    window_length = int(window_duration * fs)
    stride = int(window_length * (1 - overlap))
    
    # Check minimum length
    if min_length and len(sig) < min_length:
        raise ValueError(f"Signal too short: {len(sig)} < {min_length}")
    
    # Calculate number of windows
    n_windows = (len(sig) - window_length) // stride + 1
    
    if n_windows <= 0:
        return np.array([]), np.array([]), None
    
    # Create windows
    windows = np.zeros((n_windows, window_length))
    indices = np.zeros(n_windows, dtype=int)
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_length
        
        if end_idx <= len(sig):
            windows[i] = sig[start_idx:end_idx]
            indices[i] = start_idx

    # Segment labels if provided
    segmented_labels = {}
    if labels is not None:
        duration_sec = len(sig) / fs
        for key, label_vals in labels.items():
            if not isinstance(label_vals, np.ndarray) and not isinstance(label_vals, list):
                continue
            
            label_vals = np.array(label_vals)
            if len(label_vals) == 0:
                continue
            
            # Map labels to windows based on time
            label_fs = len(label_vals) / duration_sec
            label_window_len = int(window_duration * label_fs)
            label_indices = (indices / fs * label_fs).astype(int)
            
            seg_vals = []
            for start_idx in label_indices:
                end_idx = start_idx + label_window_len
                chunk = label_vals[start_idx : min(end_idx, len(label_vals))]
                if len(chunk) > 0:
                    seg_vals.append(np.nanmean(chunk))
                else:
                    seg_vals.append(np.nan)
            segmented_labels[key] = np.array(seg_vals)
    
    return windows, indices, segmented_labels if labels else None


def create_sequences(windows: np.ndarray, sequence_length: int = 5, 
                    overlap: int = 1) -> np.ndarray:
    """
    Create sequences of windows for CNN-LSTM models.
    
    Args:
        windows: Array of windows, shape (n_windows, window_length)
        sequence_length: Number of windows per sequence
        overlap: Number of overlapping windows between sequences
        
    Returns:
        Array of shape (n_sequences, sequence_length, window_length)
        
    Notes:
        For CNN-LSTM architecture:
        - Each sequence contains multiple consecutive windows
        - CNN extracts features from each window
        - LSTM models temporal dependencies across windows
        
        Example: 5 windows per sequence captures ~50 seconds of data
    """
    n_windows = len(windows)
    stride = sequence_length - overlap
    
    n_sequences = (n_windows - sequence_length) // stride + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough windows for sequences: {n_windows} windows, need {sequence_length}")
    
    sequences = np.zeros((n_sequences, sequence_length, windows.shape[1]))
    
    for i in range(n_sequences):
        start_idx = i * stride
        end_idx = start_idx + sequence_length
        sequences[i] = windows[start_idx:end_idx]
    
    return sequences


def ensure_heartbeats(window: np.ndarray, fs: float, min_beats: int = 5) -> bool:
    """
    Verify that window contains minimum number of heartbeats.
    
    Args:
        window: Signal window
        fs: Sampling frequency (Hz)
        min_beats: Minimum number of heartbeats required
        
    Returns:
        True if window contains sufficient heartbeats
        
    Notes:
        Uses peak detection to count heartbeats.
        Ensures window has enough cardiac cycles for meaningful analysis.
    """
    # Detect peaks (systolic peaks)
    # Use adaptive threshold based on signal statistics
    threshold = np.mean(window) + 0.5 * np.std(window)
    
    # Minimum distance between peaks (physiological constraint)
    # At max HR of 180 BPM: 60/180 = 0.33 seconds
    min_distance = int(0.33 * fs)
    
    peaks, _ = sp_signal.find_peaks(window, height=threshold, distance=min_distance)
    
    return len(peaks) >= min_beats


def reject_low_quality_windows(windows: np.ndarray, fs: float, 
                                quality_threshold: float = 0.5,
                                min_beats: int = 5,
                                labels: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Reject low-quality windows based on signal quality indices.
    
    Args:
        windows: Array of windows, shape (n_windows, window_length)
        fs: Sampling frequency (Hz)
        quality_threshold: Minimum quality score (0-1)
        min_beats: Minimum number of heartbeats per window
        labels: Optional dict of segmented labels
        
    Returns:
        Tuple of (filtered_windows, quality_mask, filtered_labels)
    """
    from .quality import assess_window_quality
    
    n_windows = len(windows)
    quality_mask = np.ones(n_windows, dtype=bool)
    
    for i in range(n_windows):
        window = windows[i]
        
        # Check for NaN or Inf
        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            quality_mask[i] = False
            continue
        
        # Check signal variation
        if np.std(window) < 0.01:
            quality_mask[i] = False
            continue
        
        # Check for excessive clipping
        min_val, max_val = np.min(window), np.max(window)
        clipping_ratio = (np.sum(window == min_val) + np.sum(window == max_val)) / len(window)
        if clipping_ratio > 0.05:
            quality_mask[i] = False
            continue
        
        # Check number of heartbeats
        if not ensure_heartbeats(window, fs, min_beats):
            quality_mask[i] = False
            continue
        
        # Assess overall quality
        quality_score = assess_window_quality(window, fs)
        if quality_score < quality_threshold:
            quality_mask[i] = False
            continue
    
    filtered_windows = windows[quality_mask]
    
    filtered_labels = {}
    if labels is not None:
        for key, val in labels.items():
            filtered_labels[key] = val[quality_mask]
    
    rejection_rate = 1 - np.mean(quality_mask)
    print(f"Window quality filtering: {np.sum(quality_mask)}/{n_windows} windows accepted ({rejection_rate*100:.1f}% rejected)")
    
    return filtered_windows, quality_mask, filtered_labels if labels else None


def create_train_val_test_split(windows: np.ndarray, labels: Optional[Any] = None,
                                subject_ids: Optional[np.ndarray] = None,
                                train_ratio: float = 0.7, val_ratio: float = 0.15,
                                random_seed: int = 42) -> dict:
    """
    Create train/validation/test splits with no subject overlap.
    """
    np.random.seed(random_seed)
    
    if subject_ids is not None:
        unique_subjects = np.unique(subject_ids)
        n_subjects = len(unique_subjects)
        shuffled_subjects = np.random.permutation(unique_subjects)
        
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)
        
        train_subjects = shuffled_subjects[:n_train]
        val_subjects = shuffled_subjects[n_train:n_train+n_val]
        test_subjects = shuffled_subjects[n_train+n_val:]
        
        train_mask = np.isin(subject_ids, train_subjects)
        val_mask = np.isin(subject_ids, val_subjects)
        test_mask = np.isin(subject_ids, test_subjects)
    else:
        n_windows = len(windows)
        indices = np.random.permutation(n_windows)
        n_train = int(n_windows * train_ratio)
        n_val = int(n_windows * val_ratio)
        
        train_mask = np.zeros(n_windows, dtype=bool)
        val_mask = np.zeros(n_windows, dtype=bool)
        test_mask = np.zeros(n_windows, dtype=bool)
        
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train+n_val]] = True
        test_mask[indices[n_train+n_val:]] = True

    def filter_labels(l_obj, mask):
        if l_obj is None: return None
        if isinstance(l_obj, dict):
            return {k: v[mask] for k, v in l_obj.items()}
        return l_obj[mask]

    splits = {
        'train': {
            'windows': windows[train_mask],
            'labels': filter_labels(labels, train_mask),
            'subject_ids': subject_ids[train_mask] if subject_ids is not None else None
        },
        'val': {
            'windows': windows[val_mask],
            'labels': filter_labels(labels, val_mask),
            'subject_ids': subject_ids[val_mask] if subject_ids is not None else None
        },
        'test': {
            'windows': windows[test_mask],
            'labels': filter_labels(labels, test_mask),
            'subject_ids': subject_ids[test_mask] if subject_ids is not None else None
        }
    }
    
    print(f"Data split: Train={np.sum(train_mask)}, Val={np.sum(val_mask)}, Test={np.sum(test_mask)}")
    return splits


if __name__ == "__main__":
    # Test windowing functions
    print("Testing windowing functions...")
    
    # Generate test signal
    fs = 64.0
    duration = 60.0  # 60 seconds
    t = np.arange(0, duration, 1/fs)
    sig = np.sin(2 * np.pi * 1.0 * t) + 0.1 * np.random.randn(len(t))
    
    # Test segmentation
    windows, indices = segment_signal(sig, fs, window_duration=10.0, overlap=0.5)
    print(f"✓ Segmentation: {len(sig)} samples -> {len(windows)} windows of {windows.shape[1]} samples")
    
    # Test sequence creation
    sequences = create_sequences(windows, sequence_length=5, overlap=1)
    print(f"✓ Sequences: {len(windows)} windows -> {len(sequences)} sequences of {sequences.shape[1]} windows")
    
    # Test heartbeat detection
    has_beats = ensure_heartbeats(windows[0], fs, min_beats=5)
    print(f"✓ Heartbeat detection: Window has sufficient beats = {has_beats}")
    
    # Test quality filtering
    filtered_windows, quality_mask = reject_low_quality_windows(windows, fs)
    print(f"✓ Quality filtering: {len(windows)} -> {len(filtered_windows)} windows")
    
    # Test train/val/test split
    subject_ids = np.repeat(np.arange(5), len(windows)//5 + 1)[:len(windows)]
    splits = create_train_val_test_split(windows, subject_ids=subject_ids)
    print(f"✓ Data split: Train={len(splits['train']['windows'])}, Val={len(splits['val']['windows'])}, Test={len(splits['test']['windows'])}")
    
    print("\n✓ All windowing functions working correctly!")
