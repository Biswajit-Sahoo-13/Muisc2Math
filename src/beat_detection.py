"""Beat and onset detection via multi-band spectral flux.

Implements a self-contained analysis pipeline:
1. Multi-band spectral-flux onset novelty (rectified STFT changes per band).
2. Tempo estimation from the autocorrelation of the novelty curve.
3. Dynamic local-search beat tracker locked to the estimated period.

This is designed to be more robust than a single global argmax while remaining
deterministic and dependency-light (numpy/scipy/librosa only).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import librosa


def _frame_times(n_frames: int, sr: int, hop_length: int) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)


def compute_spectral_flux(
    y: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 4,
    pre_emphasis: float = 0.97,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a multi-band spectral-flux onset-strength envelope.

    Positive frame-to-frame magnitude changes are summed independently inside
    several frequency bands and then combined, so that a kick in the lows and a
    hi-hat in the highs can both produce clear novelty peaks.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    # Simple pre-emphasis to sharpen transient edges.
    if pre_emphasis != 0.0 and y.size > 1:
        y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    stft = np.abs(
        librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann", center=True)
    )
    if stft.shape[1] < 2:
        return np.zeros(stft.shape[1], dtype=np.float64), _frame_times(stft.shape[1], sr, hop_length)

    # Split linear STFT bins into contiguous bands.
    n_bins = stft.shape[0]
    edges = np.unique(np.round(np.linspace(0, n_bins, n_bands + 1)).astype(int))
    if len(edges) < 2:
        edges = np.array([0, n_bins])

    band_fluxes: List[np.ndarray] = []
    for i in range(len(edges) - 1):
        lo, hi = int(edges[i]), int(edges[i + 1])
        if hi - lo < 1:
            continue
        band = stft[lo:hi, :]
        diff = np.diff(band, axis=1)
        flux = np.sum(np.maximum(diff, 0.0), axis=0)
        # Normalise each band so no single band dominates.
        scale = np.max(flux) if np.max(flux) > 1e-12 else 1.0
        band_fluxes.append(flux / scale)

    if not band_fluxes:
        flux = np.zeros(stft.shape[1] - 1, dtype=np.float64)
    else:
        flux = np.mean(np.vstack(band_fluxes), axis=0)

    # Smooth with a short Hamming window.
    if flux.size >= 3:
        win = np.hamming(5)
        win /= np.sum(win)
        flux = np.convolve(flux, win, mode="same")

    # Pad to frame count.
    onset_env = np.zeros(stft.shape[1], dtype=np.float64)
    onset_env[1:] = flux
    if onset_env.size > 0:
        onset_env[0] = onset_env[1] if onset_env.size > 1 else 0.0

    times = _frame_times(onset_env.size, sr, hop_length)
    return onset_env, times


def estimate_tempo(
    onset_env: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    tempo_min: float = 40.0,
    tempo_max: float = 240.0,
) -> Tuple[float, int, float]:
    """Estimate tempo from autocorrelation of an onset envelope.

    Returns (tempo_bpm, best_lag_frames, confidence).
    """
    env = np.asarray(onset_env, dtype=np.float64).ravel()
    if env.size < 8:
        return 120.0, 0, 0.0

    envc = env - np.mean(env)
    ac = np.correlate(envc, envc, mode="full")
    ac = ac[env.size - 1:]
    if ac.size == 0 or not np.any(np.isfinite(ac)):
        return 120.0, 0, 0.0

    # Normalise autocorrelation.
    denom = ac[0] if ac[0] > 1e-12 else 1.0
    ac_norm = ac / denom

    min_lag = max(1, int(sr * 60.0 / (tempo_max * hop_length)))
    max_lag = min(len(ac_norm) - 1, int(sr * 60.0 / (tempo_min * hop_length)))
    if max_lag <= min_lag:
        return 120.0, 0, 0.0

    search_region = ac_norm[min_lag:max_lag + 1]
    rel_idx = int(np.argmax(search_region))
    lag = min_lag + rel_idx

    # Parabolic refinement around the peak.
    if 0 < lag < len(ac_norm) - 1:
        a, b, c = ac_norm[lag - 1], ac_norm[lag], ac_norm[lag + 1]
        denom_p = a - 2.0 * b + c
        if abs(denom_p) > 1e-12:
            p = 0.5 * (a - c) / denom_p
            p = float(np.clip(p, -0.8, 0.8))
            lag_refined = lag + p
        else:
            lag_refined = float(lag)
    else:
        lag_refined = float(lag)

    tempo = 60.0 * sr / (hop_length * float(lag_refined)) if lag_refined > 0 else 120.0
    tempo = float(np.clip(tempo, tempo_min, tempo_max))
    confidence = float(ac_norm[lag])
    return tempo, int(lag), confidence


def track_beats(
    onset_env: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    tempo: Optional[float] = None,
    search_ratio: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dynamic local-search beat tracker.

    Returns (beat_frames, beat_times). Tracks along the onset curve by jumping
    roughly one beat period at a time, allowing a small local search.
    """
    env = np.asarray(onset_env, dtype=np.float64).ravel()
    if env.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if tempo is None:
        tempo, _, _ = estimate_tempo(env, sr=sr, hop_length=hop_length)
    period = max(1, int(round(sr * 60.0 / (tempo * hop_length))))
    search = max(1, int(round(search_ratio * period)))

    # Start at the first detected onset peak within the first beat periods, so the
    # grid phase locks to an attack rather than to an arbitrary magnitude max.
    first_end = min(env.size, period * 2)
    onset_frames, _ = detect_onsets(env, sr=sr, hop_length=hop_length, tempo=tempo)
    candidates = onset_frames[onset_frames < first_end]
    if candidates.size > 0:
        start = int(candidates[0])
    else:
        start = int(np.argmax(env[:first_end])) if first_end > 0 else 0

    beat_frames: List[int] = [start]
    pos = start
    while pos + period < env.size:
        lo = max(0, pos + period - search)
        hi = min(env.size, pos + period + search + 1)
        if hi <= lo:
            break
        rel = int(np.argmax(env[lo:hi]))
        new_pos = lo + rel
        if new_pos <= pos:
            new_pos = min(env.size - 1, pos + period)
        beat_frames.append(new_pos)
        pos = new_pos

    frames = np.asarray(beat_frames, dtype=int)
    times = frames * (hop_length / float(sr))
    return frames, times


def detect_onsets(
    onset_env: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    tempo: Optional[float] = None,
    threshold_std: float = 1.0,
    min_distance_ratio: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect onset peak frames from an onset envelope."""
    env = np.asarray(onset_env, dtype=np.float64).ravel()
    if env.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

    if tempo is None:
        tempo, _, _ = estimate_tempo(env, sr=sr, hop_length=hop_length)
    period = max(1, int(round(sr * 60.0 / (tempo * hop_length))))
    min_distance = max(1, int(round(min_distance_ratio * period)))
    height = float(np.mean(env) + threshold_std * np.std(env))

    peaks, _ = signal.find_peaks(env, height=height, distance=min_distance)
    if peaks.size == 0:
        peaks, _ = signal.find_peaks(env, distance=min_distance)

    frames = np.asarray(peaks, dtype=int)
    times = frames * (hop_length / float(sr))
    return frames, times


def detect_beats(
    y: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_bands: int = 4,
    tempo_min: float = 40.0,
    tempo_max: float = 240.0,
) -> Dict[str, Any]:
    """High-level beat/onset detection from raw audio.

    Returns a dictionary containing tempo, beat frames/times, onset frames/times,
    the onset-strength envelope, and analysis parameters.
    """
    onset_env, onset_times = compute_spectral_flux(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands
    )
    tempo, best_lag, confidence = estimate_tempo(
        onset_env, sr=sr, hop_length=hop_length, tempo_min=tempo_min, tempo_max=tempo_max
    )
    beat_frames, beat_times = track_beats(
        onset_env, sr=sr, hop_length=hop_length, tempo=tempo
    )
    onset_frames, onset_time_list = detect_onsets(
        onset_env, sr=sr, hop_length=hop_length, tempo=tempo
    )

    return {
        "tempo": float(tempo),
        "confidence": float(confidence),
        "best_lag_frames": int(best_lag),
        "beat_period_frames": int(round(sr * 60.0 / (tempo * hop_length))) if tempo > 0 else 0,
        "beat_frames": beat_frames,
        "beat_times": beat_times,
        "onset_frames": onset_frames,
        "onset_times": onset_time_list,
        "onset_strength": onset_env,
        "onset_times_grid": onset_times,
        "sr": int(sr),
        "n_fft": int(n_fft),
        "hop_length": int(hop_length),
        "n_bands": int(n_bands),
    }


# Compatibility aliases.
BeatDetector = detect_beats
estimate_beats = detect_beats
detect_beat_times = lambda y, sr=44100, **kw: detect_beats(y, sr=sr, **kw)["beat_times"]  # noqa: E731


__all__ = [
    "compute_spectral_flux",
    "estimate_tempo",
    "track_beats",
    "detect_onsets",
    "detect_beats",
    "BeatDetector",
    "estimate_beats",
]
