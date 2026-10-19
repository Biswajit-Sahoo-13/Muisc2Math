"""Closed-loop error correction and before/after comparison for recreations.

A tone-based recreation captures notes and timbre but not the original phase,
level or timing, so its waveform correlation against the source starts near
zero even when the spectral content matches. This module closes that gap:

1. **Lag alignment** - cross-correlates estimate against reference to find the
   best global time offset before anything else.
2. **Residual correction layers** - repeatedly discovers decaying partials on
   the residual with the library-learning engine and adds them, accepting a
   layer only when it measurably improves the objective.
3. **Level correction** - a least-squares global gain refit.

Every correction is itself a plain math equation string, so the corrected audio
is still fully described by equations rather than by raw samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa

from src.formula_bank import fit_equation_library
from src.residual import compute_error_metrics
from src.safe_eval import safe_eval

_DARK = "#07090e"
_FG = "#c5c6c7"
_CORR_COLOR = "#66fcf1"
_ORIG_COLOR = "#ff7597"


def evaluate_equation(equation: str, n_samples: int, sr: int) -> np.ndarray:
    """Render a discovered equation to a waveform via the safe evaluator."""
    if not equation or n_samples <= 0:
        return np.zeros(max(0, n_samples), dtype=np.float64)
    t = np.arange(n_samples, dtype=np.float64) / sr
    y = safe_eval(equation, {"t": t, "u": t, "s": t, "rnd": np.zeros_like(t), "beat": 0.5})
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.size < n_samples:
        y = np.pad(y, (0, n_samples - y.size))
    return y[:n_samples]


def rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64).ravel()
    if arr.size < 1:
        return 0.0
    return float(np.sqrt(np.mean(arr ** 2)))


def match_rms_level(estimate: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, float]:
    """Scale ``estimate`` so its RMS matches the reference's RMS.

    A peak-normalised recreation is typically far louder than the source, which
    makes the residual meaningless (it is dominated by the recreation itself)
    and tempts the correction stage into simply cancelling everything. Matching
    level first makes the residual an honest measure of what is missing.
    """
    r_ref = rms(reference)
    r_est = rms(estimate)
    if r_ref < 1e-12 or r_est < 1e-12:
        return estimate, 1.0
    gain = float(np.clip(r_ref / r_est, 1e-4, 1e4))
    return estimate * gain, gain


def energy_guard(
    candidate: np.ndarray,
    reference: np.ndarray,
    min_ratio: float = 0.35,
    max_ratio: float = 3.0,
) -> bool:
    """Reject candidates whose energy drifted too far from the reference.

    Without this guard the optimisation is trivial to cheat: emitting silence
    yields 0 dB SNR, so a correction stage that only maximises SNR will happily
    cancel the whole signal.
    """
    r_ref = rms(reference)
    if r_ref < 1e-12:
        return True
    ratio = rms(candidate) / r_ref
    return bool(min_ratio <= ratio <= max_ratio)


def waveform_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation between two signals."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    a = np.asarray(x[:n], dtype=np.float64)
    b = np.asarray(y[:n], dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def find_best_lag(
    reference: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    max_lag_s: float = 0.06,
) -> Tuple[int, float]:
    """Find the lag (in samples) that best aligns ``estimate`` to ``reference``.

    Returns ``(lag, correlation)`` where a positive lag means the estimate
    should be delayed by that many samples to match the reference.
    """
    n = min(len(reference), len(estimate))
    if n < 64:
        return 0, waveform_correlation(reference, estimate)

    max_lag = int(max(0.0, max_lag_s) * sr)
    if max_lag < 1:
        return 0, waveform_correlation(reference, estimate)

    a = np.asarray(reference[:n], dtype=np.float64)
    a = a - a.mean()
    if np.std(a) < 1e-12:
        return 0, 0.0

    size = 1 << int(np.ceil(np.log2(2 * n)))
    fa = np.fft.rfft(a, size)
    fb = np.fft.rfft(np.asarray(estimate[:n], dtype=np.float64), size)
    corr = np.fft.irfft(fa * np.conj(fb), size)

    positive = corr[: max_lag + 1]
    negative = corr[-max_lag:]
    best_pos = int(np.argmax(positive)) if positive.size else 0
    best_neg = int(np.argmax(negative)) if negative.size else 0
    best_neg_lag = -(negative.size - best_neg)

    if positive.size and negative.size:
        peak_pos = float(positive[best_pos])
        peak_neg = float(negative[best_neg])
        best_lag = best_pos if peak_pos >= peak_neg else best_neg_lag
    elif positive.size:
        best_lag = best_pos
    else:
        best_lag = best_neg_lag
    return int(best_lag), waveform_correlation(reference, estimate)


def apply_lag(y: np.ndarray, lag: int, pad: bool = False) -> np.ndarray:
    """Delay ``y`` by ``lag`` samples (positive lag = shift later, negative = shift earlier)."""
    if lag == 0 or len(y) == 0:
        return y
    n = len(y)
    if lag > 0:
        if lag >= n:
            return np.zeros(n, dtype=np.float64)
        return np.concatenate([np.zeros(lag, dtype=np.float64), y[:-lag]])
    abs_lag = -lag
    if abs_lag >= n:
        return np.zeros(n, dtype=np.float64) if pad else np.zeros(0, dtype=np.float64)
    if pad:
        return np.concatenate([y[abs_lag:], np.zeros(abs_lag, dtype=np.float64)])
    return y[abs_lag:]


def least_squares_gain(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scalar gain that best matches ``estimate`` to ``reference``."""
    n = min(len(reference), len(estimate))
    if n < 2:
        return 1.0
    a = np.asarray(reference[:n], dtype=np.float64)
    b = np.asarray(estimate[:n], dtype=np.float64)
    denom = float(np.dot(b, b))
    if denom < 1e-20:
        return 1.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 2.0))


def _summarize(reference: np.ndarray, estimate: np.ndarray, sr: int) -> Dict[str, Any]:
    n = min(len(reference), len(estimate))
    metrics = compute_error_metrics(reference[:n], estimate[:n], sr=sr)
    return {
        "snr_db": round(float(metrics["snr_db"]), 3),
        "spectral_mse": round(float(metrics["spectral_mse"]), 8),
        "log_spectral_distance_db": round(float(metrics["log_spectral_distance_db"]), 3),
        "correlation": round(waveform_correlation(reference, estimate), 4),
    }


def refine_segmented(
    original: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    segments: List[Tuple[int, int]],
    rounds: int = 2,
    max_terms_per_segment: int = 2,
    min_gain_db: float = 0.05,
) -> Dict[str, Any]:
    """Beat-segmented residual correction.

    A single stationary decaying partial cannot describe seconds of music, so
    correction is fitted inside short beat-length windows where the
    stationary-partial assumption actually holds. Each accepted window
    contributes its own equation, recorded with its start time.
    """
    n = min(len(original), len(estimate))
    work = np.asarray(estimate[:n], dtype=np.float64).copy()
    before = _summarize(original, work, sr)
    layers: List[Dict[str, Any]] = []

    for round_idx in range(max(0, int(rounds))):
        residual = original[:n] - work
        correction = np.zeros(n, dtype=np.float64)
        round_layers: List[Dict[str, Any]] = []

        for start_idx, end_idx in segments:
            start_idx = max(0, int(start_idx))
            end_idx = min(n, int(end_idx))
            if end_idx - start_idx < 256:
                continue
            seg_res = residual[start_idx:end_idx]
            if float(np.max(np.abs(seg_res))) < 1e-6:
                continue
            lib = fit_equation_library(
                seg_res, sr, f0=220.0, max_terms=int(max_terms_per_segment), n_candidates=3
            )
            equation = lib.get("equation", "")
            if not equation:
                continue
            seg_corr = evaluate_equation(equation, end_idx - start_idx, sr)
            if float(np.max(np.abs(seg_corr))) < 1e-9:
                continue
            correction[start_idx:end_idx] += seg_corr
            round_layers.append({
                "start_time": round(start_idx / sr, 4),
                "equation": equation,
                "terms": lib["term_count"],
            })

        if not round_layers:
            break

        candidate = work + correction
        after = _summarize(original, candidate, sr)
        gain_db = after["snr_db"] - before["snr_db"]
        if gain_db < float(min_gain_db):
            break

        work = candidate
        layers.extend(round_layers)
        before = after

    return {
        "corrected_audio": work,
        "metrics_before": None,
        "layers": layers,
        "segment_count": len(layers),
        "metrics_after": before,
    }


def correct_reconstruction(
    original: np.ndarray,
    estimate: np.ndarray,
    sr: int,
    rounds: int = 3,
    terms_per_round: int = 4,
    min_gain_db: float = 0.2,
    align_lag: bool = True,
    max_lag_s: float = 0.06,
    fit_level: bool = True,
    segments: Optional[List[Tuple[int, int]]] = None,
    segmented_rounds: int = 2,
) -> Dict[str, Any]:
    """Run closed-loop error correction on a recreation.

    Each round discovers decaying partials on the current residual and adds
    them, but only if the objective actually improves. The discovered terms are
    returned as equation strings so the correction stays auditable.

    When ``segments`` (beat-window index pairs) is supplied, a segmented pass
    runs first, because short windows are the only place where the
    stationary-partial model is actually valid for real music.
    """
    original = np.asarray(original, dtype=np.float64).ravel()
    n = min(len(original), len(estimate))
    work = np.asarray(estimate[:n], dtype=np.float64).copy()

    report: Dict[str, Any] = {
        "lag_samples": 0,
        "lag_ms": 0.0,
        "gain": 1.0,
        "initial_level_gain": 1.0,
        "rounds": [],
        "equations": [],
    }

    if n <= 0:
        empty_metrics = {
            "snr_db": 0.0,
            "spectral_mse": 0.0,
            "log_spectral_distance_db": 0.0,
            "correlation": 0.0,
        }
        report["metrics_before"] = empty_metrics
        report["metrics_after"] = dict(empty_metrics)
        report["corrected_audio"] = np.zeros(0, dtype=np.float64)
        report["residual_audio"] = np.zeros(0, dtype=np.float64)
        report["total_snr_gain_db"] = 0.0
        report["total_correlation_gain"] = 0.0
        return report

    work, level_gain = match_rms_level(work, original[:n])
    report["initial_level_gain"] = round(float(level_gain), 6)

    if align_lag:
        lag, _ = find_best_lag(original, work, sr, max_lag_s=max_lag_s)
        if lag:
            padded = np.zeros(n, dtype=np.float64)
            shifted = apply_lag(work, lag)
            m = min(n, len(shifted))
            padded[:m] = shifted[:m]
            work = padded
        report["lag_samples"] = int(lag)
        report["lag_ms"] = round(1000.0 * lag / sr, 3)

    before = _summarize(original, work, sr)
    report["metrics_before"] = before

    if segments:
        segmented = refine_segmented(
            original,
            work,
            sr=sr,
            segments=segments,
            rounds=int(segmented_rounds),
        )
        candidate = segmented["corrected_audio"]
        after = _summarize(original, candidate, sr)
        if after["snr_db"] > before["snr_db"] and energy_guard(candidate, original[:n]):
            work = candidate
            report["segmented"] = {
                "accepted": True,
                "segments": segmented["segment_count"],
                "snr_gain_db": round(after["snr_db"] - before["snr_db"], 3),
                "correlation_gain": round(after["correlation"] - before["correlation"], 4),
            }
            report["segment_equations"] = segmented["layers"][:64]
            before = after
        else:
            report["segmented"] = {"accepted": False}

    for round_idx in range(max(0, int(rounds))):
        residual = original - work
        if float(np.max(np.abs(residual))) < 1e-6:
            break

        lib = fit_equation_library(
            residual, sr, f0=220.0, max_terms=int(terms_per_round), n_candidates=5
        )
        equation = lib.get("equation", "")
        if not equation:
            break

        correction = evaluate_equation(equation, n, sr)
        if float(np.max(np.abs(correction))) < 1e-9:
            break

        candidate = work + correction
        candidate_metrics = _summarize(original, candidate, sr)
        gain_db = candidate_metrics["snr_db"] - before["snr_db"]
        corr_gain = candidate_metrics["correlation"] - before["correlation"]

        if gain_db < float(min_gain_db):
            report["rounds"].append({
                "round": round_idx + 1,
                "accepted": False,
                "reason": f"snr gain {gain_db:.3f} dB below threshold",
                "terms": lib["term_count"],
            })
            break

        if not energy_guard(candidate, original[:n]):
            report["rounds"].append({
                "round": round_idx + 1,
                "accepted": False,
                "reason": "energy guard: correction would cancel too much signal",
                "terms": lib["term_count"],
            })
            break

        work = candidate
        report["equations"].append(equation)
        report["rounds"].append({
            "round": round_idx + 1,
            "accepted": True,
            "terms": lib["term_count"],
            "snr_db": candidate_metrics["snr_db"],
            "correlation": candidate_metrics["correlation"],
            "snr_gain_db": round(gain_db, 3),
            "correlation_gain": round(corr_gain, 4),
        })
        before = candidate_metrics

    if fit_level:
        gain = least_squares_gain(original, work)
        if abs(gain - 1.0) > 1e-3:
            scaled = work * gain
            scaled_metrics = _summarize(original, scaled, sr)
            if (
                scaled_metrics["snr_db"] >= before["snr_db"]
                and energy_guard(scaled, original[:n])
            ):
                work = scaled
                before = scaled_metrics
                report["gain"] = round(gain, 5)

    report["metrics_after"] = before
    report["total_snr_gain_db"] = round(
        float(report["metrics_after"]["snr_db"] - report["metrics_before"]["snr_db"]), 3
    )
    report["total_correlation_gain"] = round(
        float(
            report["metrics_after"]["correlation"] - report["metrics_before"]["correlation"]
        ),
        4,
    )
    report["corrected_audio"] = work
    report["residual_audio"] = original[: len(work)] - work
    return report


def plot_ab_comparison(
    original: np.ndarray,
    estimate: np.ndarray,
    corrected: np.ndarray,
    sr: int,
    out_path: Union[str, Path],
    title: str = "Original vs Recreation",
    view_s: float = 6.0,
    metrics_before: Optional[Dict[str, Any]] = None,
    metrics_after: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a 3-panel before/after comparison figure."""
    n = min(len(original), len(corrected), len(estimate))
    view = min(n, int(view_s * sr))
    t = np.arange(view) / sr
    orig = original[:view]
    corr = corrected[:view]
    res = orig - corr

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    fig.patch.set_facecolor(_DARK)

    axes[0].set_facecolor(_DARK)
    axes[0].plot(t, orig, lw=0.5, color=_ORIG_COLOR, label="Original")
    axes[0].plot(t, estimate[:view], lw=0.5, color="#f39c12", alpha=0.75, label="Recreation (before correction)")
    axes[0].plot(t, corr, lw=0.5, color=_CORR_COLOR, alpha=0.85, label="After error correction")
    axes[0].set_title(title, color=_CORR_COLOR, fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude", color=_FG)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.25, color="#333")

    axes[1].set_facecolor(_DARK)
    axes[1].plot(t, res, lw=0.5, color="#e74c3c")
    axes[1].set_title("Remaining error (original - corrected)", color="#e74c3c", fontsize=10)
    axes[1].set_ylabel("Error", color=_FG)
    axes[1].grid(True, alpha=0.25, color="#333")

    axes[2].set_facecolor(_DARK)
    orig_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(orig, n_fft=2048, hop_length=256)), ref=np.max
    )
    corr_db = librosa.amplitude_to_db(
        np.abs(librosa.stft(corr, n_fft=2048, hop_length=256)), ref=np.max
    )
    im = axes[2].imshow(
        orig_db - corr_db, aspect="auto", origin="lower", cmap="coolwarm",
        vmin=-60, vmax=60,
    )
    axes[2].set_title(
        "Spectrogram difference (dB):  white = corrected matches original",
        color=_FG, fontsize=10,
    )
    axes[2].set_xlabel("Frame", color=_FG)
    axes[2].set_ylabel("Frequency bin", color=_FG)
    fig.colorbar(im, ax=axes[2], format="%+2.0f dB")

    summary = ""
    if metrics_before and metrics_after:
        summary = (
            f"   |   SNR {metrics_before['snr_db']:.2f} -> {metrics_after['snr_db']:.2f} dB"
            f"   |   correlation {metrics_before['correlation']:.3f} -> "
            f"{metrics_after['correlation']:.3f}"
        )
    axes[0].set_xlabel(f"Time (seconds){summary}", color=_FG)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=130)
    plt.close(fig)


def bark_scale(f: np.ndarray) -> np.ndarray:
    """Convert frequency in Hz to Bark critical-band rate."""
    f_arr = np.maximum(np.asarray(f, dtype=np.float64), 1.0)
    return 13.0 * np.arctan(0.00076 * f_arr) + 3.5 * np.arctan((f_arr / 7500.0) ** 2)


def ath_linear(f: np.ndarray) -> np.ndarray:
    """Absolute threshold of hearing (quiet threshold) in linear amplitude."""
    f_khz = np.maximum(np.asarray(f, dtype=np.float64), 20.0) / 1000.0
    t_q = (
        3.64 * (f_khz ** -0.8)
        - 6.5 * np.exp(-0.6 * ((f_khz - 3.3) ** 2))
        + 1e-3 * (f_khz ** 4)
    )
    # Convert dB SPL to normalized linear threshold scale
    return 1e-3 * (10.0 ** ((t_q - 40.0) / 20.0))


def psychoacoustic_masking_threshold(
    signal: np.ndarray,
    sr: int,
    n_fft: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate psychoacoustic spectral masking threshold for a signal.

    Implements a spreading function across the Bark critical band domain.
    Strong spectral components mask neighboring frequencies, with an upward
    masking slope (-24 dB/Bark) and downward masking slope (+15 dB/Bark).
    The total threshold is bounded below by the absolute threshold of hearing (ATH).
    """
    n = len(signal)
    if n == 0:
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
        return freqs, np.zeros_like(freqs)

    w_len = min(n, n_fft)
    w = np.hanning(w_len)
    seg = signal[:w_len] * w
    spec_mag = np.abs(np.fft.rfft(seg, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    ath = ath_linear(freqs)

    # If silent, return ATH
    if np.max(spec_mag) < 1e-9:
        return freqs, ath

    # Compute spreading across Bark domain
    z = bark_scale(freqs)
    dz = z[:, None] - z[None, :]
    # Upward masking (dz >= 0): slope -24 dB/Bark -> 10^(-1.2 * dz)
    # Downward masking (dz < 0): slope +15 dB/Bark -> 10^(0.75 * dz)
    spread = np.where(dz >= 0, 10.0 ** (-1.2 * dz), 10.0 ** (0.75 * dz))

    # Tonal masker factor: -16 dB relative to peak component
    tonal_factor = 0.16
    tonal_mask = tonal_factor * np.max(spec_mag[None, :] * spread, axis=1)

    total_threshold = np.maximum(tonal_mask, ath)
    return freqs, total_threshold


def masking_weighted_residual_error(
    residual: np.ndarray,
    reference: np.ndarray,
    sr: int,
    n_fft: int = 1024,
) -> float:
    """Compute psychoacoustically-weighted residual error.

    Components of the residual that lie below the spectral masking threshold
    are inaudible to human perception and are heavily discounted (penalized
    less). Audibly unmasked components above the threshold are penalized in full.
    """
    n = min(len(residual), len(reference))
    if n < 8:
        return float(np.sum(residual ** 2))

    _, mask = psychoacoustic_masking_threshold(reference, sr, n_fft=n_fft)

    w_len = min(len(residual), n_fft)
    w = np.hanning(w_len)
    r_seg = residual[:w_len] * w
    r_mag = np.abs(np.fft.rfft(r_seg, n=n_fft))

    # Inaudible components (below mask) get discounted by factor 0.05
    # Audible components get full weight
    weights = np.where(r_mag <= mask, 0.05, 1.0)
    weighted_err = float(np.sum(weights * (r_mag ** 2)))
    return weighted_err


__all__ = [
    "apply_lag",
    "bark_scale",
    "correct_reconstruction",
    "evaluate_equation",
    "find_best_lag",
    "least_squares_gain",
    "masking_weighted_residual_error",
    "plot_ab_comparison",
    "psychoacoustic_masking_threshold",
    "waveform_correlation",
]