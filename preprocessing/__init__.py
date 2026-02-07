"""
Preprocessing package for PPG signal processing.
Phase-2: Advanced, MAE-optimized preprocessing pipeline.
"""

__version__ = "1.0.0"
__author__ = "PPG Deep Learning Project"

from .loaders import load_ppg_dalia, load_wesad, load_bidmc, resample_signal
from .filters import bandpass_filter, remove_baseline_wander, motion_artifact_mitigation
from .normalization import per_window_zscore, robust_scaling
from .windowing import segment_signal, create_sequences
from .quality import compute_snr, assess_window_quality
from .pipeline import PreprocessingPipeline

__all__ = [
    'load_ppg_dalia',
    'load_wesad',
    'load_bidmc',
    'resample_signal',
    'bandpass_filter',
    'remove_baseline_wander',
    'motion_artifact_mitigation',
    'per_window_zscore',
    'robust_scaling',
    'segment_signal',
    'create_sequences',
    'compute_snr',
    'assess_window_quality',
    'PreprocessingPipeline'
]
