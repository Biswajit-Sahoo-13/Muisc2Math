"""Residual analysis, objective DSP error metrics, and iterative sinusoidal correction engine.

Adheres to DSP principles:
1. Exact projection/least-squares for residual sinusoidal addition
2. Comprehensive perceptual and objective metrics: MSE, SNR, Spectral MSE, Log-Spectral Distance
3. Iterative refinement loop: r_{k+1}(t) = x(t) - x_hat_{k+1}(t)
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.signal as signal
import librosa


def align_signals(x: np.ndarray, x_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Ensures both signals have identical length by trimming or zero-padding."""
    min_len = min(len(x), len(x_hat))
    return x[:min_len], x_hat[:min_len]


def compute_error_metrics(
    x: np.ndarray,
    x_hat: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    eps: float = 1e-12,
    extras: bool = False,
) -> Dict[str, Any]:
    """Computes rigorous DSP error metrics: MSE, SNR (dB), Spectral Error, and Log-Spectral Distance (LSD).

    ``extras=True`` adds the evaluation-upgrade metrics —
    ``mrstft_db`` (multi-resolution log-spectral distance over several
    window sizes, which sees tonal-vs-textural errors a single window
    understates) and ``band_corr`` (waveform correlation per frequency
    band, which says *where* in the spectrum the error lives). The referee
    hot path keeps ``extras=False`` so its cost and behaviour stay put.
    """
    x, x_hat = align_signals(x, x_hat)
    residual = x - x_hat

    # 1. Mean Squared Error (MSE)
    mse = float(np.mean(residual ** 2))

    # 2. Signal-to-Noise Ratio (SNR) in dB
    power_signal = np.sum(x ** 2)
    power_noise = np.sum(residual ** 2)
    snr_db = float(10.0 * np.log10((power_signal + eps) / (power_noise + eps)))

    # 3. Spectral Error and Log-Spectral Distance (LSD)
    w = signal.get_window("hann", n_fft)
    coherent_gain = 0.5 * np.sum(w)
    s_orig = np.abs(librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window=w)) / coherent_gain
    s_recon = np.abs(librosa.stft(x_hat, n_fft=n_fft, hop_length=hop_length, window=w)) / coherent_gain

    spectral_mse = float(np.mean((s_orig - s_recon) ** 2))

    # Log-Spectral Distance (LSD) across frames, clamped to standard audible floor (-80 dB)
    floor_db = -80.0
    log_orig_db = np.maximum(20.0 * np.log10(s_orig + eps), floor_db)
    log_recon_db = np.maximum(20.0 * np.log10(s_recon + eps), floor_db)
    frame_lsd = np.sqrt(np.mean((log_orig_db - log_recon_db) ** 2, axis=0))
    mean_lsd_db = float(np.mean(frame_lsd))

    out: Dict[str, Any] = {
        "mse": mse,
        "snr_db": snr_db,
        "spectral_mse": spectral_mse,
        "log_spectral_distance_db": mean_lsd_db,
        "signal_energy": float(power_signal),
        "residual_energy": float(power_noise),
    }
    if extras:
        out["mrstft_db"] = _mrstft_db(x, x_hat, sr, eps=eps)
        out["band_corr"] = _band_corr(x, x_hat, sr)
    return out


# Bands for the per-band correlation metric. Chosen to match the error
# budget conventions used across the research docs (sub-bass, core, texture,
# air) so ledger numbers compare directly with docs/PHYSICS_ROADMAP.md.
_BAND_EDGES_HZ = (150.0, 2000.0, 8000.0)
_BAND_NAMES = ("low", "mid", "high", "top")


def _band_corr(
    x: np.ndarray, x_hat: np.ndarray, sr: int,
    edges: Tuple[float, ...] = _BAND_EDGES_HZ,
) -> Dict[str, float]:
    """Waveform correlation per frequency band (FFT band-split, no window).

    Both signals are split with brick-wall masks on a shared rfft grid and
    correlated within each band. A band with (near-)zero energy in either
    signal correlates 0.0 — there is nothing to agree about, which is the
    honest score for a band one side does not render at all.
    """
    n = min(len(x), len(x_hat))
    if n < 2:
        return {name: 0.0 for name in _BAND_NAMES}
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    edges_hz = (0.0,) + tuple(edges) + (float(freqs[-1]) + 1.0,)
    out: Dict[str, float] = {}
    for i, name in enumerate(_BAND_NAMES):
        mask = (freqs >= edges_hz[i]) & (freqs < edges_hz[i + 1])
        if not mask.any():
            out[name] = 0.0
            continue
        bx = np.fft.irfft(np.fft.rfft(x[:n]) * mask, n)
        br = np.fft.irfft(np.fft.rfft(x_hat[:n]) * mask, n)
        sa, sb = float(np.std(bx)), float(np.std(br))
        if sa <= 1e-12 or sb <= 1e-12:
            out[name] = 0.0
            continue
        out[name] = float(np.corrcoef(bx, br)[0, 1])
    return out


def _mrstft_db(
    x: np.ndarray,
    x_hat: np.ndarray,
    sr: int,
    sizes: Tuple[int, ...] = (512, 1024, 2048, 4096),
    floor_db: float = -80.0,
    eps: float = 1e-12,
) -> float:
    """Multi-resolution log-spectral distance (mean LSD over window sizes).

    Same convention as the single-window LSD (Hann, coherent-gain
    normalised, -80 dB floor) averaged over several resolutions, following
    the multi-resolution STFT loss family (Yamamoto et al. 2019): tonal
    detail shows at long windows, transients at short ones, and a single
    window can miss either.
    """
    n = min(len(x), len(x_hat))
    distances: List[float] = []
    for size in sizes:
        # A window longer than the signal would be mostly padding; clamp to
        # the largest power of two that still slides, floor at 256.
        eff = size
        while eff > 256 and eff > n:
            eff //= 2
        if eff > n:
            continue
        win = signal.get_window("hann", eff)
        gain = 0.5 * float(np.sum(win))
        so = np.abs(librosa.stft(x[:n], n_fft=eff, hop_length=eff // 4, window=win)) / gain
        sr_ = np.abs(librosa.stft(x_hat[:n], n_fft=eff, hop_length=eff // 4, window=win)) / gain
        lo = np.maximum(20.0 * np.log10(so + eps), floor_db)
        lr = np.maximum(20.0 * np.log10(sr_ + eps), floor_db)
        distances.append(float(np.mean(np.sqrt(np.mean((lo - lr) ** 2, axis=0)))))
    return float(np.mean(distances)) if distances else 0.0


def segmental_snr_db(
    x: np.ndarray,
    x_hat: np.ndarray,
    sr: int,
    frame_s: float = 0.04,
    hop_s: float = 0.02,
    floor_db: float = -60.0,
    cap_db: float = 40.0,
) -> float:
    """Frame-wise SNR averaged over active frames (segmental SNR, ITU-style).

    Frames whose RMS sits more than ``floor_db`` below the global peak are
    skipped (silence says nothing about fidelity); the mean is clamped to
    ``cap_db`` so a single near-perfect frame cannot dominate.
    """
    x, x_hat = align_signals(x, x_hat)
    nf = max(16, int(round(frame_s * sr)))
    nh = max(8, int(round(hop_s * sr)))
    if len(x) < nf or nf < 32:
        return 0.0
    peak = float(np.max(np.abs(x)))
    if peak <= 0:
        return 0.0
    thr = peak * (10.0 ** (floor_db / 20.0))
    vals: List[float] = []
    for s in range(0, len(x) - nf + 1, nh):
        ye = x[s:s + nf]
        r = x_hat[s:s + nf] - ye
        pe = float(np.sum(ye ** 2))
        pn = float(np.sum(r ** 2))
        if np.sqrt(pe / nf) < thr or pe < 1e-20:
            continue
        vals.append(min(10.0 * np.log10((pe + 1e-20) / (pn + 1e-20)), cap_db))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def onset_envelope(
    x: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
) -> np.ndarray:
    """Half-wave-rectified spectral-flux onset envelope (one value per frame gap)."""
    if len(x) < n_fft:
        return np.zeros(0)
    frames = 1 + (len(x) - n_fft) // hop
    w = np.hanning(n_fft)
    env = np.zeros(frames - 1)
    prev = None
    for f in range(frames):
        mag = np.abs(np.fft.rfft(x[f * hop:f * hop + n_fft] * w))
        if prev is not None:
            env[f - 1] = float(np.sum(np.maximum(mag - prev, 0.0)))
        prev = mag
    return env


def onset_corr(
    x: np.ndarray,
    x_hat: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
) -> float:
    """Correlation of the onset envelopes: does the render strike when the target strikes?"""
    x, x_hat = align_signals(x, x_hat)
    ex = onset_envelope(x, sr, n_fft, hop)
    er = onset_envelope(x_hat, sr, n_fft, hop)
    if ex.size < 3 or er.size != ex.size:
        return 0.0
    if float(np.std(ex)) <= 1e-12 or float(np.std(er)) <= 1e-12:
        return 0.0
    c = np.corrcoef(ex, er)[0, 1]
    return 0.0 if not np.isfinite(c) else float(c)


def estimate_sinusoid_by_projection(
    signal_arr: np.ndarray,
    sr: int,
    freq: float,
) -> Tuple[float, float]:
    """Estimates optimal amplitude and phase for a known frequency using least-squares projection.

    Models: signal ~ c_cos * cos(2*pi*f*t) + c_sin * sin(2*pi*f*t)
    Equivalent to: A * sin(2*pi*f*t + phi)
        where A = sqrt(c_cos^2 + c_sin^2), phi = atan2(c_cos, c_sin)
    This is guaranteed to minimize residual MSE for that single frequency.
    """
    n = len(signal_arr)
    t = np.arange(n, dtype=np.float64) / sr
    omega_t = 2.0 * np.pi * freq * t

    s_sin = np.sin(omega_t)
    s_cos = np.cos(omega_t)

    # Solve least squares [sin, cos] @ [c_sin, c_cos] = signal
    mat = np.column_stack((s_sin, s_cos))
    coefs, _, _, _ = np.linalg.lstsq(mat, signal_arr, rcond=None)
    c_sin, c_cos = coefs[0], coefs[1]

    amplitude = float(np.sqrt(c_sin ** 2 + c_cos ** 2))
    phase = float(np.arctan2(c_cos, c_sin))

    return amplitude, phase


def find_dominant_residual_frequencies(
    residual: np.ndarray,
    sr: int,
    top_k: int = 5,
    min_freq: float = 20.0,
    max_freq: float = 12000.0,
    min_distance_hz: float = 15.0,
) -> List[float]:
    """Extracts top dominant frequency peaks from the residual signal."""
    n = len(residual)
    w = signal.get_window("hann", n)
    spec = np.abs(np.fft.rfft(residual * w))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    mask = (freqs >= min_freq) & (freqs <= max_freq)
    valid_freqs = freqs[mask]
    valid_spec = spec[mask]

    min_bin_dist = max(1, int(np.round(min_distance_hz / (sr / n))))
    peaks, _ = signal.find_peaks(valid_spec, distance=min_bin_dist)

    if len(peaks) == 0:
        return []

    sorted_peak_indices = peaks[np.argsort(valid_spec[peaks])[::-1]]
    dominant_f = [float(valid_freqs[idx]) for idx in sorted_peak_indices[:top_k]]
    return dominant_f


def iterative_residual_refinement(
    original_audio: np.ndarray,
    sr: int,
    initial_model: Optional[Dict[str, Any]] = None,
    max_iterations: int = 20,
    target_snr_db: float = 30.0,
    min_improvement_db: float = 0.2,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, float]]:
    """Performs iterative residual refinement matching.

    Loop:
        1. Synthesize current model x_hat(t)
        2. Compute residual r(t) = x(t) - x_hat(t)
        3. Check stop condition (SNR >= target_snr_db or max iterations)
        4. Detect strongest missing frequency in residual
        5. Estimate amplitude & phase via projection
        6. Add component to model and repeat
    """
    from src.synthesis import render_model

    duration = float(len(original_audio) / sr)

    if initial_model is None:
        model: Dict[str, Any] = {
            "model_name": "iterative_refined_model",
            "sample_rate": sr,
            "duration": duration,
            "components": [],
        }
    else:
        # Deep copy
        model = json.loads(json.dumps(initial_model))
        model["model_name"] = "iterative_refined_model"

    x_hat, _ = render_model(model, duration=duration, sr=sr)
    x_hat = x_hat[: len(original_audio)]
    current_metrics = compute_error_metrics(original_audio, x_hat, sr=sr)
    prev_snr = current_metrics["snr_db"]

    for iteration in range(max_iterations):
        if current_metrics["snr_db"] >= target_snr_db:
            if verbose:
                print(f"[Refine] Target SNR reached: {current_metrics['snr_db']:.2f} dB")
            break

        residual = original_audio - x_hat
        dom_freqs = find_dominant_residual_frequencies(residual, sr, top_k=1)
        if not dom_freqs:
            if verbose:
                print("[Refine] No more spectral peaks found in residual.")
            break

        cand_freq = dom_freqs[0]
        cand_amp, cand_phase = estimate_sinusoid_by_projection(residual, sr, cand_freq)

        if cand_amp < 1e-4:
            if verbose:
                print(f"[Refine] Peak amplitude below threshold: {cand_amp:.6f}")
            break

        # Create new component
        new_comp = {
            "id": f"residual_partial_{len(model['components'])}",
            "frequency": round(cand_freq, 3),
            "amplitude": round(cand_amp, 5),
            "phase": round(cand_phase, 5),
            "decay_rate": 0.0,
            "envelope": {
                "type": "constant",
                "a0": round(cand_amp, 5),
                "decay_rate": 0.0,
            },
        }

        # Candidate test: ensure model improves
        test_components = list(model["components"]) + [new_comp]
        test_model = dict(model, components=test_components)
        new_x_hat, _ = render_model(test_model, duration=duration, sr=sr)
        new_x_hat = new_x_hat[: len(original_audio)]
        new_metrics = compute_error_metrics(original_audio, new_x_hat, sr=sr)

        snr_gain = new_metrics["snr_db"] - prev_snr
        if snr_gain < min_improvement_db and iteration > 0:
            if verbose:
                print(f"[Refine] Marginal SNR improvement ({snr_gain:.3f} dB < {min_improvement_db} dB). Terminating.")
            break

        # Accept component
        model["components"].append(new_comp)
        x_hat = new_x_hat
        current_metrics = new_metrics
        prev_snr = current_metrics["snr_db"]

        if verbose:
            print(
                f"[Refine Iter {iteration + 1}] Added f={cand_freq:.2f} Hz, "
                f"A={cand_amp:.4f} -> SNR={current_metrics['snr_db']:.2f} dB (+{snr_gain:.2f} dB)"
            )

    model["final_metrics"] = current_metrics
    return model, x_hat, current_metrics
