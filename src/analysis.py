"""Spectral analysis engine with exact window normalization and sub-bin parabolic interpolation.

Adheres to DSP principles:
1. True Amplitude Recovery: Division by window coherent gain (0.5 * sum(w[n]))
2. Sub-Bin Frequency Accuracy: Parabolic interpolation on log-magnitudes
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 44100,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Loads audio from file, converts to mono float64, and resamples if target_sr is provided."""
    y, sr = librosa.load(str(file_path), sr=target_sr, mono=mono, dtype=np.float64)
    return y, sr


def compute_normalized_stft(
    y: np.ndarray,
    sr: int = 44100,
    n_fft: int = 4096,
    hop_length: int = 1024,
    window_name: str = "hann",
) -> Dict[str, Any]:
    """Computes STFT with exact coherent gain window normalization.

    Coherent gain factor:
        W_coherent = 0.5 * sum(w[n])
    Dividing raw STFT magnitudes by W_coherent yields true physical sine amplitudes.
    """
    w = signal.get_window(window_name, n_fft)
    coherent_gain = 0.5 * np.sum(w)

    # Compute complex STFT
    stft_matrix = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=w,
        center=True,
        pad_mode="reflect",
    )

    raw_mags = np.abs(stft_matrix)
    normalized_mags = raw_mags / coherent_gain
    phases = np.angle(stft_matrix)

    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    time_frames = librosa.frames_to_time(
        np.arange(stft_matrix.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return {
        "stft": stft_matrix,
        "raw_mags": raw_mags,
        "normalized_mags": normalized_mags,
        "phases": phases,
        "freq_bins": freq_bins,
        "time_frames": time_frames,
        "coherent_gain": coherent_gain,
        "sr": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
    }


def find_parabolic_peaks(
    raw_mags_frame: np.ndarray,
    coherent_gain: float,
    sr: int,
    n_fft: int,
    phase_frame: Optional[np.ndarray] = None,
    threshold: float = 0.005,
    min_bin_distance: int = 3,
    max_peaks: int = 40,
) -> List[Dict[str, float]]:
    """Detects spectral peaks and applies parabolic interpolation on log-magnitudes.

    Given a local peak bin k and neighbors k-1, k+1:
        alpha = ln(|X[k-1]|)
        beta  = ln(|X[k]|)
        gamma = ln(|X[k+1]|)
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        f_true = (k + p) * sr / n_fft
        true_log_mag = beta - 0.25 * (alpha - gamma) * p
        A = exp(true_log_mag) / coherent_gain
    """
    normalized_frame = raw_mags_frame / coherent_gain
    # Find local maxima bins
    min_height = threshold * coherent_gain
    peak_bins, _ = signal.find_peaks(
        raw_mags_frame,
        height=min_height,
        distance=min_bin_distance,
    )

    results: List[Dict[str, float]] = []
    n_bins = len(raw_mags_frame)
    eps = 1e-12

    for k in peak_bins:
        # Avoid boundary bins where neighbors are missing
        if k <= 0 or k >= n_bins - 1:
            continue

        m_prev = raw_mags_frame[k - 1]
        m_curr = raw_mags_frame[k]
        m_next = raw_mags_frame[k + 1]

        # Parabolic interpolation in log space
        alpha = np.log(max(m_prev, eps))
        beta = np.log(max(m_curr, eps))
        gamma = np.log(max(m_next, eps))

        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            p = 0.5 * (alpha - gamma) / denom
            # Clamp offset to valid neighboring interval [-0.5, 0.5]
            p = np.clip(p, -0.5, 0.5)
        else:
            p = 0.0

        f_true = float((k + p) * sr / n_fft)
        true_log_mag = beta - 0.25 * (alpha - gamma) * p
        raw_peak = float(np.exp(true_log_mag))
        amp_true = float(raw_peak / coherent_gain)

        # Estimate phase at peak
        if phase_frame is not None:
            # # ponytail: direct bin phase with fractional offset adjustment; suffices for additive reconstruction
            phi = float(phase_frame[k] - np.pi * p)
            # Wrap phase to [-pi, pi]
            phi = float((phi + np.pi) % (2.0 * np.pi) - np.pi)
        else:
            phi = 0.0

        results.append({
            "frequency": f_true,
            "amplitude": amp_true,
            "phase": phi,
            "bin": int(k),
            "raw_mag": raw_peak,
        })

    # Sort descending by amplitude and cap to max_peaks
    results.sort(key=lambda item: item["amplitude"], reverse=True)
    return results[:max_peaks]
