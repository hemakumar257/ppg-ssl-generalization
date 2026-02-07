"""
Dataset-aware signal loaders for PPG-DaLiA, WESAD, and BIDMC datasets.
Handles different file formats and provides unified interface.
"""

import numpy as np
import pandas as pd
import h5py
import pickle
from scipy import signal
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings


def resample_signal(sig: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """
    Resample signal to target sampling rate using anti-aliasing filter.
    
    Args:
        sig: Input signal array
        original_fs: Original sampling frequency (Hz)
        target_fs: Target sampling frequency (Hz)
        
    Returns:
        Resampled signal array
        
    Notes:
        Uses scipy.signal.resample_poly for high-quality resampling with anti-aliasing.
        This prevents aliasing artifacts that can increase MAE by 5-10%.
    """
    if original_fs == target_fs:
        return sig
    
    # Calculate resampling ratio
    gcd = np.gcd(int(original_fs), int(target_fs))
    up = int(target_fs) // gcd
    down = int(original_fs) // gcd
    
    # Use polyphase resampling with anti-aliasing
    resampled = signal.resample_poly(sig, up, down, axis=0)
    
    return resampled


def load_ppg_dalia(subject_id: int, base_path: str = "datasets/PPG+Dalia/PPG_FieldStudy") -> Tuple[np.ndarray, float, Dict]:
    """
    Load PPG signal from PPG-DaLiA dataset (Pickle format).
    Supports both Chest PPG (700Hz) and Wrist BVP (64Hz) with automatic detection.
    """
    base_path = Path(base_path)
    subject_dir = base_path / f"S{subject_id}"
    
    # Load Pickle file (contains everything needed)
    pkl_file = subject_dir / f"S{subject_id}.pkl"
    
    if not pkl_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Robust signal extraction
    signal_dict = data['signal']
    sensor_location = 'chest'
    
    if 'chest' in signal_dict and 'PPG' in signal_dict['chest']:
        ppg_signal = signal_dict['chest']['PPG'].flatten()
        sampling_rate = 700.0  # RespiBAN PPG sampling rate
        sensor_location = 'chest'
    elif 'wrist' in signal_dict and 'BVP' in signal_dict['wrist']:
        ppg_signal = signal_dict['wrist']['BVP'].flatten()
        sampling_rate = 64.0  # Empatica E4 BVP sampling rate
        sensor_location = 'wrist'
        print(f"  [INFO] Fallback to wrist BVP for S{subject_id}")
    else:
        raise ValueError(f"No PPG or BVP signal found for S{subject_id}")
    
    # Extract labels (activity annotations)
    labels = data.get('label', None)
    
    # Extract subject info
    subject_info = data.get('subject', None)
    
    metadata = {
        'subject_id': f"PPG-DaLiA_S{subject_id}",
        'dataset': 'PPG-DaLiA',
        'sensor_location': sensor_location,
        'original_fs': sampling_rate,
        'labels': labels,
        'subject_info': subject_info
    }
    
    return ppg_signal, sampling_rate, labels, metadata


def load_wesad(subject_id: int, base_path: str = "datasets/WESAD/WESAD") -> Tuple[np.ndarray, float, Dict]:
    """
    Load BVP (PPG) signal from WESAD dataset.
    
    Args:
        subject_id: Subject number (2-17, excluding 1 and 12)
        base_path: Path to WESAD directory
        
    Returns:
        Tuple of (signal, sampling_rate, metadata)
        - signal: BVP signal array (wrist sensor, treated as PPG)
        - sampling_rate: 64 Hz (Empatica E4 native)
        - metadata: Dict with subject_id, stress_labels, etc.
        
    Notes:
        WESAD uses Empatica E4 wristband with BVP (Blood Volume Pulse).
        BVP is equivalent to PPG and sampled at 64 Hz.
        Includes stress/affect labels for supervised learning.
    """
    base_path = Path(base_path)
    subject_dir = base_path / f"S{subject_id}"
    
    # Load pickle file
    pkl_file = subject_dir / f"S{subject_id}.pkl"
    
    if not pkl_file.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Extract BVP signal (wrist sensor)
    bvp_signal = data['signal']['wrist']['BVP'].flatten()
    sampling_rate = 64.0  # Empatica E4 BVP sampling rate
    
    # Extract labels (stress/affect annotations)
    labels = data.get('label', None)
    
    # Extract subject info
    subject_info = data.get('subject', None)
    
    metadata = {
        'subject_id': f"WESAD_S{subject_id}",
        'dataset': 'WESAD',
        'sensor_location': 'wrist',
        'original_fs': sampling_rate,
        'labels': labels,
        'subject_info': subject_info
    }
    
    return bvp_signal, sampling_rate, labels, metadata


def load_bidmc(subject_id: int, base_path: str = "Datasets/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc-ppg-and-respiration-dataset-1.0.0") -> Tuple[np.ndarray, float, Dict]:
    """
    Load PLETH (PPG) signal from BIDMC dataset.
    
    Args:
        subject_id: Subject number (1-53)
        base_path: Path to BIDMC dataset directory
        
    Returns:
        Tuple of (signal, sampling_rate, metadata)
        - signal: PLETH signal array (clinical PPG)
        - sampling_rate: 125 Hz
        - metadata: Dict with subject_id, breath_annotations, etc.
        
    Notes:
        BIDMC uses clinical pulse oximetry sampled at 125 Hz.
        Includes manual breath annotations for validation.
        8-minute recordings per subject.
    """
    base_path = Path(base_path)
    csv_dir = base_path / "bidmc_csv"
    
    # Format subject ID with leading zeros
    subject_str = f"{subject_id:02d}"
    
    # Load signals CSV
    signals_file = csv_dir / f"bidmc_{subject_str}_Signals.csv"
    
    if not signals_file.exists():
        raise FileNotFoundError(f"Signals file not found: {signals_file}")
    
    # Load signal data
    signals_df = pd.read_csv(signals_file)
    
    # Extract PLETH (PPG) signal
    if ' PLETH' in signals_df.columns:
        ppg_signal = signals_df[' PLETH'].values.flatten()
    elif 'PLETH' in signals_df.columns:
        ppg_signal = signals_df['PLETH'].values.flatten()
    else:
        raise ValueError(f"PLETH column not found in {signals_file}")
    
    sampling_rate = 125.0  # BIDMC sampling rate
    
    # Load breath annotations
    breaths_file = csv_dir / f"bidmc_{subject_str}_Breaths.csv"
    breath_annotations = None
    if breaths_file.exists():
        breaths_df = pd.read_csv(breaths_file)
        breath_annotations = breaths_df.to_dict('records')
    
    # Load numerics (HR, RR, SpO2)
    numerics_file = csv_dir / f"bidmc_{subject_str}_Numerics.csv"
    labels = None
    if numerics_file.exists():
        numerics_df = pd.read_csv(numerics_file)
        # Extract HR and RR as primary labels (resampled to signal FS later)
        # Numerics are usually at 1Hz, we need to handle this
        hr_col = next((c for c in numerics_df.columns if c.strip() in ['HR', 'PULSE']), None)
        rr_col = next((c for c in numerics_df.columns if c.strip() in ['RR', 'RESP']), None)
        
        labels = {}
        if hr_col: labels['hr'] = numerics_df[hr_col].values
        if rr_col: labels['rr'] = numerics_df[rr_col].values
        if 'Time [s]' in numerics_df.columns:
            labels['time'] = numerics_df['Time [s]'].values
    
    # Load fixed parameters (age, gender, location)
    fix_file = csv_dir / f"bidmc_{subject_str}_Fix.txt"
    fixed_params = {}
    if fix_file.exists():
        with open(fix_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    fixed_params[key.strip()] = value.strip()
    
    metadata = {
        'subject_id': f"BIDMC_{subject_str}",
        'dataset': 'BIDMC',
        'sensor_location': 'clinical',
        'original_fs': sampling_rate,
        'fixed_params': fixed_params
    }
    
    return ppg_signal, sampling_rate, labels, metadata


def load_all_subjects(dataset: str, base_path: Optional[str] = None) -> List[Tuple[np.ndarray, float, Dict]]:
    """
    Load all subjects from a specific dataset.
    
    Args:
        dataset: Dataset name ('ppg_dalia', 'wesad', or 'bidmc')
        base_path: Optional custom base path
        
    Returns:
        List of (signal, sampling_rate, labels, metadata) tuples for all subjects
    """
    dataset = dataset.lower()
    all_data = []
    
    if dataset == 'ppg_dalia':
        subject_ids = range(1, 16)  # S1-S15
        loader = load_ppg_dalia
    elif dataset == 'wesad':
        # WESAD subjects (excluding S1 and S12)
        subject_ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
        loader = load_wesad
    elif dataset == 'bidmc':
        subject_ids = range(1, 54)  # bidmc01-bidmc53
        loader = load_bidmc
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    for subject_id in subject_ids:
        try:
            # Explicitly pass None if base_path is empty to use loader defaults
            actual_path = base_path if base_path else None
            if actual_path:
                signal, fs, labels, metadata = loader(subject_id, actual_path)
            else:
                signal, fs, labels, metadata = loader(subject_id)
            all_data.append((signal, fs, labels, metadata))
            print(f"  [OK] Loaded {dataset.upper()} subject {subject_id}")
        except Exception as e:
            warnings.warn(f"Failed to load {dataset.upper()} subject {subject_id}: {e}")
    
    return all_data


if __name__ == "__main__":
    # Test loaders
    print("Testing PPG-DaLiA loader...")
    sig, fs, meta = load_ppg_dalia(1)
    print(f"  Signal shape: {sig.shape}, FS: {fs} Hz, Subject: {meta['subject_id']}")
    
    print("\nTesting WESAD loader...")
    sig, fs, meta = load_wesad(2)
    print(f"  Signal shape: {sig.shape}, FS: {fs} Hz, Subject: {meta['subject_id']}")
    
    print("\nTesting BIDMC loader...")
    sig, fs, meta = load_bidmc(1)
    print(f"  Signal shape: {sig.shape}, FS: {fs} Hz, Subject: {meta['subject_id']}")
    
    print("\n✓ All loaders working correctly!")
