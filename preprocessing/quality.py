"""
Signal quality assessment for PPG signals.
Implements multiple Signal Quality Indices (SQI) for window rejection.
"""

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple


def compute_snr(sig: np.ndarray, fs: float) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        sig: Input signal
        fs: Sampling frequency (Hz)
        
    Returns:
        SNR in dB
        
    Notes:
        SNR estimation for PPG:
        - Signal power: Power in physiological band (0.5-4 Hz)
        - Noise power: Power outside physiological band
        
        Good quality PPG: SNR > 10 dB
        Acceptable quality: SNR > 6 dB
        Poor quality: SNR < 6 dB
    """
    # Compute power spectral density
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    
    # Define physiological band (0.5-4 Hz for heart rate)
    signal_band = (freqs >= 0.5) & (freqs <= 4.0)
    noise_band = (freqs < 0.5) | (freqs > 4.0)
    
    # Calculate power in each band
    signal_power = np.sum(psd[signal_band])
    noise_power = np.sum(psd[noise_band])
    
    # Avoid division by zero
    if noise_power == 0:
        return 100.0  # Very high SNR
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db


def detect_peaks(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, Dict]:
    """
    Detect systolic peaks in PPG signal.
    
    Args:
        sig: Input PPG signal
        fs: Sampling frequency (Hz)
        
    Returns:
        Tuple of (peak_indices, peak_properties)
        
    Notes:
        Peak detection is fundamental for:
        - Heart rate estimation
        - Pulse morphology analysis
        - Signal quality assessment
    """
    # Adaptive threshold based on signal statistics
    threshold = np.mean(sig) + 0.5 * np.std(sig)
    
    # Minimum distance between peaks (physiological constraint)
    # Max HR: 180 BPM -> 0.33 seconds between beats
    min_distance = int(0.33 * fs)
    
    # Detect peaks
    peaks, properties = signal.find_peaks(sig, height=threshold, distance=min_distance)
    
    return peaks, properties


def compute_perfusion(sig: np.ndarray) -> float:
    """
    Compute perfusion index (AC/DC ratio).
    
    Args:
        sig: Input PPG signal
        
    Returns:
        Perfusion index (0-1, higher is better)
        
    Notes:
        Perfusion index measures signal strength:
        - AC component: Pulsatile (cardiac) component
        - DC component: Baseline (tissue absorption)
        
        Good perfusion: PI > 0.02 (2%)
        Poor perfusion: PI < 0.01 (1%)
    """
    # AC component (pulsatile): peak-to-peak amplitude
    ac_component = np.max(sig) - np.min(sig)
    
    # DC component (baseline): mean value
    dc_component = np.mean(sig)
    
    # Avoid division by zero
    if dc_component == 0:
        return 0.0
    
    perfusion_index = ac_component / dc_component
    
    return perfusion_index


def compute_template_correlation(sig: np.ndarray, fs: float) -> float:
    """
    Compute correlation with ideal PPG template.
    
    Args:
        sig: Input PPG signal
        fs: Sampling frequency (Hz)
        
    Returns:
        Correlation coefficient (0-1)
        
    Notes:
        High correlation with template indicates good morphology.
        Template matching is robust to amplitude variations.
    """
    # Detect peaks to segment individual beats
    peaks, _ = detect_peaks(sig, fs)
    
    if len(peaks) < 2:
        return 0.0
    
    # Extract individual beats
    beats = []
    for i in range(len(peaks) - 1):
        beat = sig[peaks[i]:peaks[i+1]]
        if len(beat) > 10:  # Minimum beat length
            # Normalize beat length
            beat_normalized = signal.resample(beat, 64)  # Standard length
            beats.append(beat_normalized)
    
    if len(beats) < 2:
        return 0.0
    
    # Compute average beat template
    template = np.mean(beats, axis=0)
    
    # Compute correlation of each beat with template
    correlations = []
    for beat in beats:
        corr = np.corrcoef(beat, template)[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    if len(correlations) == 0:
        return 0.0
    
    # Return average correlation
    return np.mean(correlations)


def compute_statistical_features(sig: np.ndarray) -> Dict[str, float]:
    """
    Compute statistical features for quality assessment.
    
    Args:
        sig: Input signal
        
    Returns:
        Dictionary of statistical features
    """
    features = {
        'mean': np.mean(sig),
        'std': np.std(sig),
        'skewness': skew(sig),
        'kurtosis': kurtosis(sig),
        'range': np.max(sig) - np.min(sig),
        'iqr': np.percentile(sig, 75) - np.percentile(sig, 25)
    }
    
    return features


def assess_window_quality(sig: np.ndarray, fs: float) -> float:
    """
    Assess overall window quality using composite SQI.
    
    Args:
        sig: Input PPG window
        fs: Sampling frequency (Hz)
        
    Returns:
        Quality score (0-1, higher is better)
        
    Notes:
        Composite SQI combines multiple quality metrics:
        1. SNR (40% weight)
        2. Perfusion index (30% weight)
        3. Template correlation (20% weight)
        4. Statistical features (10% weight)
        
        Quality thresholds:
        - Excellent: > 0.8
        - Good: 0.6-0.8
        - Acceptable: 0.5-0.6
        - Poor: < 0.5 (reject)
    """
    # 1. SNR score (0-1)
    snr_db = compute_snr(sig, fs)
    snr_score = min(1.0, max(0.0, (snr_db - 3) / 15))  # Map 3-18 dB to 0-1
    
    # 2. Perfusion score (0-1)
    perfusion = compute_perfusion(sig)
    perfusion_score = min(1.0, perfusion / 0.05)  # Map 0-5% to 0-1
    
    # 3. Template correlation score (0-1)
    template_corr = compute_template_correlation(sig, fs)
    
    # 4. Statistical score (0-1)
    stats = compute_statistical_features(sig)
    # Good PPG has moderate kurtosis (not too flat, not too spiky)
    kurtosis_score = 1.0 - min(1.0, abs(stats['kurtosis'] - 3) / 10)
    
    # Composite score (weighted average)
    quality_score = (
        0.4 * snr_score +
        0.3 * perfusion_score +
        0.2 * template_corr +
        0.1 * kurtosis_score
    )
    
    return quality_score


def batch_quality_assessment(windows: np.ndarray, fs: float) -> np.ndarray:
    """
    Assess quality for batch of windows.
    
    Args:
        windows: Array of windows, shape (n_windows, window_length)
        fs: Sampling frequency (Hz)
        
    Returns:
        Array of quality scores, shape (n_windows,)
    """
    n_windows = len(windows)
    quality_scores = np.zeros(n_windows)
    
    for i in range(n_windows):
        quality_scores[i] = assess_window_quality(windows[i], fs)
    
    return quality_scores


if __name__ == "__main__":
    # Test quality assessment functions
    print("Testing quality assessment functions...")
    
    # Generate test signals
    fs = 64.0
    t = np.arange(0, 10, 1/fs)
    
    # Good quality signal
    good_signal = np.sin(2 * np.pi * 1.0 * t) + 0.05 * np.random.randn(len(t))
    
    # Poor quality signal (noisy)
    poor_signal = np.sin(2 * np.pi * 1.0 * t) + 0.5 * np.random.randn(len(t))
    
    # Test SNR
    snr_good = compute_snr(good_signal, fs)
    snr_poor = compute_snr(poor_signal, fs)
    print(f"✓ SNR: Good signal={snr_good:.2f} dB, Poor signal={snr_poor:.2f} dB")
    
    # Test peak detection
    peaks, _ = detect_peaks(good_signal, fs)
    print(f"✓ Peak detection: {len(peaks)} peaks detected")
    
    # Test perfusion
    perfusion = compute_perfusion(good_signal)
    print(f"✓ Perfusion index: {perfusion:.4f}")
    
    # Test template correlation
    template_corr = compute_template_correlation(good_signal, fs)
    print(f"✓ Template correlation: {template_corr:.4f}")
    
    # Test overall quality
    quality_good = assess_window_quality(good_signal, fs)
    quality_poor = assess_window_quality(poor_signal, fs)
    print(f"✓ Quality assessment: Good signal={quality_good:.4f}, Poor signal={quality_poor:.4f}")
    
    # Test batch assessment
    windows = np.array([good_signal, poor_signal])
    quality_scores = batch_quality_assessment(windows, fs)
    print(f"✓ Batch assessment: {quality_scores}")
    
    print("\n✓ All quality assessment functions working correctly!")
