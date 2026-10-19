"""RES synthesis and evaluation primitives.

Shared mathematical kernels for partial rendering, trajectory interpolation,
support masking, and referee metrics, used across both the coordinate lifter
(res_loop) and the hypothesis tournament bank (res_bank).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.signal
from scipy.signal import hilbert

from src.sinusoidal_model import _smooth_envelope
from src.residual import compute_error_metrics, segmental_snr_db, onset_corr


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient between two 1D vectors."""
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a_sl = np.asarray(a[:n], dtype=np.float64)
    b_sl = np.asarray(b[:n], dtype=np.float64)
    sa, sb = float(np.std(a_sl)), float(np.std(b_sl))
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.corrcoef(a_sl, b_sl)[0, 1])


def _glide_traj(
    tt: np.ndarray,
    ff: np.ndarray,
    aa: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Amplitude + integrated-phase trajectories on an arbitrary global grid.

    Exact replica of sinusoidal_model._trajectories (9-frame median-smoothed
    frequency for the integral, 5-frame-smoothed amplitude), evaluated on
    ``t`` so section-offset renders stay on the global time axis.
    """
    if tt.size == 1:
        return (
            np.full(t.size, float(aa[0])),
            np.zeros(t.size),
        )
    f_smooth = _smooth_envelope(ff, 9)
    a_frames = _smooth_envelope(aa)
    integral_frames = np.zeros(tt.size, dtype=np.float64)
    integral_frames[1:] = np.cumsum(
        2.0 * np.pi * 0.5 * (f_smooth[:-1] + f_smooth[1:]) * np.diff(tt)
    )
    a_env = np.interp(t, tt, a_frames)
    integral = np.interp(t, tt, integral_frames)
    return a_env, integral


_WHISTLE_FREQ_HZ = 1000.0
_MIN_DECAY_BASE = 2.5
_MIN_DECAY_SLOPE = 0.002
_MAX_GLIDE_HZ = 1800.0
_RELEASE_FADE_S = 0.03
_ZERO_DECAY_EPS = 1e-6
_MIN_RING_S = 0.10


def minimum_decay_for(freq: float) -> float:
    """Frequency-dependent damping floor that keeps highs from ringing.

    A zero-decay partial above ~1 kHz holds a pure sine for its whole window,
    which is audibly a whistle rather than a musical tone. Requiring decay to
    grow with frequency models the real air/surface losses instead.
    """
    if freq <= _WHISTLE_FREQ_HZ:
        return 0.0
    return _MIN_DECAY_BASE + _MIN_DECAY_SLOPE * (freq - _WHISTLE_FREQ_HZ)


def sanitize_partial(p: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce the no-birdie rules on a partial without mutating the input.

    Only one rule survived measurement, and it is deliberately narrow:

    - A *zero-decay* partial above ~1 kHz that also *sustains* for longer than
      ``_MIN_RING_S`` is a whistle: it holds a pure sine across the whole window.
      Those get a frequency-dependent damping floor and a ``damped`` model,
      because ``constant`` cannot express decay.

    Three rules that looked reasonable were removed because measurement showed
    they cost more than they bought:

    - Raising an already-positive decay is destructive, not corrective. On a
      synthetic chord it forced a real 1720 Hz harmonic from its fitted decay of
      1.38 up to the 3.94 floor, costing 0.0066 correlation.
    - A span cap of ``3.0 / decay`` was provably inert: zero-decay partials
      above 1 kHz are already only ~23 ms long, so the bound never bound
      (measured delta exactly 0.0000).
    - Demoting glide tracks above 1.8 kHz cost the drums stem -0.0743
      correlation on its own, because swept high-frequency content is what
      hats and cymbals actually are. The birdie it targeted was never
      isolated; the loss was.

    The minimum-span condition matters for the same reason: 403 of 1104 drums
    partials are zero-decay above 1 kHz, but almost all are ~23 ms transients
    that cannot ring at all. Flooring them only attenuated genuine transients.
    """
    q = dict(p)
    freq = float(q.get("frequency") or 0.0)
    model = q.get("model")
    decay = float(q.get("envelope_decay") or 0.0)
    floor = minimum_decay_for(freq)

    if floor > 0.0 and decay <= _ZERO_DECAY_EPS:
        st, et = q.get("start_time"), q.get("end_time")
        span = (float(et) - float(st)) if (st is not None and et is not None) else 0.0
        if span > _MIN_RING_S:
            q["envelope_decay"] = floor
            if model == "constant":
                q["model"] = "damped"
    return q


def _support_mask(
    t: np.ndarray,
    start_time: float,
    end_time: float,
    decay: float,
    amp: float,
    stft_half_window_s: float,
) -> np.ndarray:
    """Support window matching the SMS renderer (see sinusoidal_model).

    Content one STFT half-window before the first frame is what the detecting
    window measured (backed by a rising ramp), the partial lives at full
    strength through the projection window, and past the last frame a fitted
    decay holds a little before a 0.25 s fade. Without a half-window an
    approximate 20 ms attack is used.
    """
    if stft_half_window_s > 0.0:
        pre_start = max(0.0, start_time - stft_half_window_s)
        if (start_time - stft_half_window_s) < 0.0:
            attack_term = np.ones_like(t)
        else:
            attack_term = np.clip(
                (t - pre_start) / max(start_time - pre_start, 1e-9), 0.0, 1.0
            )
    else:
        attack_len = min(0.02, 0.5 * max(start_time, 1e-6))
        attack_term = (
            np.clip((t - start_time) / attack_len, 0.0, 1.0)
            if attack_len > 1e-9
            else np.ones_like(t)
        )
    if decay > 0.0 and amp > 0.0:
        hold_s = min(1.5, float(np.log(100.0)) / decay)
    else:
        hold_s = 0.25
    support_end = end_time + hold_s
    release_term = np.clip((t - support_end) / _RELEASE_FADE_S, 0.0, 1.0)
    return attack_term * (1.0 - release_term)


def _support_window(
    p: Dict[str, Any],
    n: int,
    sr: int,
    start_sample: int,
    stft_half_window_s: float,
) -> Optional[Tuple[int, int]]:
    """Exact ``[i0, i1)`` where ``_support_mask`` can be non-zero.

    Mirrors the mask's own thresholds: the attack ramp before ``start_time``
    (one STFT half-window when available, otherwise a short rise) and the decay
    hold plus release fade after ``end_time`` -- the hold alone can reach 1.5 s,
    so slicing at ``end_time`` would truncate audible signal. Two guard samples
    on each side keep the window a strict superset of the live region.
    Returns ``None`` when no support window applies, so the caller renders at
    full length.
    """
    st = p.get("start_time", None)
    et = p.get("end_time", None)
    amp = float(p.get("amplitude", 0.0))
    if st is None or et is None or amp <= 0.0:
        return None
    st = float(st)
    et = float(et)
    if stft_half_window_s > 0.0:
        pre = st - stft_half_window_s
        t_lo = max(0.0, pre) if pre >= 0.0 else 0.0
    else:
        t_lo = st
    decay = float(p.get("envelope_decay", 0.0))
    hold_s = min(1.5, float(np.log(100.0)) / decay) if decay > 0.0 else 0.25
    t_hi = et + hold_s + _RELEASE_FADE_S
    # t is global seconds, t[i] = (i + start_sample) / sr  =>  i = t*sr - start_sample
    i0 = max(0, int(np.floor(t_lo * sr - start_sample)) - 1)
    i1 = min(n, int(np.ceil(t_hi * sr - start_sample)) + 2)
    if i1 <= i0:
        return (0, 0)
    return (i0, i1)


def _render_partial(
    p: Dict[str, Any],
    n: int,
    sr: int,
    start_sample: int = 0,
    stft_half_window_s: float = 0.0,
) -> np.ndarray:
    """Render one equation partial over n samples.

    ``start_sample`` places the render on the global time axis; partial
    parameters (envelope_t0, trajectory times) are always global, so a
    section render must pass its slice's first sample. When the partial
    carries its measured support (start_time/end_time), the render is windowed
    the same way the SMS engine windows its tracks. For window-fitted partials
    the stored phase is referenced to the section's own time origin, which is
    exactly the ``start_sample`` used during fitting - no extra offset
    argument needed.
    """
    t = (np.arange(n, dtype=np.float64) + start_sample) / sr
    t_fit = (np.arange(n, dtype=np.float64)) / sr
    p = sanitize_partial(p)
    # Deterministic models (sinusoid, glide) are pointwise in t, so they can be
    # evaluated only across the support window and written into a zero buffer.
    # The stochastic models below cannot: their RNG stream position, IIR state
    # and RMS normalisation all depend on the full length n, so they keep
    # evaluating over the whole buffer.
    win = _support_window(p, n, sr, start_sample, stft_half_window_s)
    if win is not None:
        i0, i1 = win
        if i1 <= i0:
            return np.zeros(n)
        t = (np.arange(i0, i1, dtype=np.float64) + start_sample) / sr
        t_fit = np.arange(i0, i1, dtype=np.float64) / sr
    if p.get("model") == "glide" and p.get("trajectory"):
        traj = p["trajectory"]
        tt = np.asarray(traj.get("t", []), dtype=np.float64)
        ff = np.asarray(traj.get("f_hz", []), dtype=np.float64)
        aa = np.asarray(traj.get("a", []), dtype=np.float64)
        if tt.size == 0 or ff.size != tt.size or aa.size != tt.size:
            return np.zeros(n)
        a_env, integral = _glide_traj(tt, ff, aa, t)
        freq_ref = float(p.get("frequency", 0.0))
        theta = integral + 2.0 * np.pi * (freq_ref - float(np.median(ff))) * t_fit
        gain = float(p.get("projection_gain", 1.0))
        ph = float(p.get("phase", 0.0))
        decay = float(p.get("envelope_decay", 0.0))
        amp = float(p.get("amplitude", 0.0))
        sig = gain * a_env * np.cos(theta - ph)
        st = p.get("start_time", None)
        et = p.get("end_time", None)
        if st is not None and et is not None and amp > 0.0:
            sig = sig * _support_mask(
                t, float(st), float(et), decay, amp, stft_half_window_s
            )
        if win is not None:
            out = np.zeros(n)
            out[i0:i1] = sig
            return out
        return sig

    if p.get("model") == "filtered_noise":
        t_full = (np.arange(n, dtype=np.float64) + start_sample) / sr
        amp = float(p.get("amplitude", 0.0))
        if amp <= 0.0 or n < 8:
            return np.zeros(n)
        seed = int(p.get("seed", 42))
        rng = np.random.default_rng(seed)
        white = rng.standard_normal(n)
        band_centers = np.asarray(p.get("band_centers", []), dtype=np.float64)
        band_mags = np.asarray(p.get("band_mags", []), dtype=np.float64)
        if band_centers.size >= 2 and band_mags.size == band_centers.size:
            w_fft = np.fft.rfft(white)
            freq_bins = np.fft.rfftfreq(n, 1.0 / sr)
            min_f = max(10.0, float(band_centers[0]))
            log_bc = np.log10(np.maximum(band_centers, min_f))
            log_bins = np.log10(np.maximum(freq_bins, min_f))
            env_bins = np.interp(log_bins, log_bc, band_mags, left=band_mags[0], right=band_mags[-1])
            sig = np.fft.irfft(w_fft * env_bins, n=n)
        else:
            sig = white
        sig_rms = float(np.sqrt(np.mean(sig ** 2)))
        if sig_rms > 1e-12:
            sig = sig * (amp / sig_rms)
        else:
            sig = sig * amp
        st = p.get("start_time", None)
        et = p.get("end_time", None)
        decay = float(p.get("envelope_decay", 0.0))
        if st is not None and et is not None:
            sig = sig * _support_mask(t_full, float(st), float(et), decay, amp, stft_half_window_s)
        return sig

    if p.get("model") == "lpc_noise":
        t_full = (np.arange(n, dtype=np.float64) + start_sample) / sr
        amp = float(p.get("amplitude", 0.0))
        if amp <= 0.0 or n < 8:
            return np.zeros(n)
        seed = int(p.get("seed", 42))
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(n)
        coeffs = p.get("lpc_coeffs", [])
        if coeffs:
            a_den = np.concatenate([[1.0], -np.array(coeffs, dtype=np.float64)])
            sig = scipy.signal.lfilter([1.0], a_den, noise)
            sig_rms = float(np.sqrt(np.mean(sig ** 2)))
            if sig_rms > 1e-12:
                sig = sig * (amp / sig_rms)
            else:
                sig = sig * amp
        else:
            sig = amp * noise
        st = p.get("start_time", None)
        et = p.get("end_time", None)
        decay = float(p.get("envelope_decay", 0.0))
        if st is not None and et is not None:
            sig = sig * _support_mask(t_full, float(st), float(et), decay, amp, stft_half_window_s)
        return sig

    amp = float(p.get("amplitude", 0.0))
    decay = float(p.get("envelope_decay", 0.0))
    t0 = float(p.get("envelope_t0", 0.0))
    f0 = float(p.get("frequency", 0.0))
    ph = float(p.get("phase", 0.0))
    env = amp * np.exp(-decay * np.maximum(t - t0, 0.0))
    sig = env * np.cos(2.0 * np.pi * f0 * t_fit - ph)
    st = p.get("start_time", None)
    et = p.get("end_time", None)
    if st is not None and et is not None and amp > 0.0:
        sig = sig * _support_mask(t, float(st), float(et), decay, amp, stft_half_window_s)
    if win is not None:
        out = np.zeros(n)
        out[i0:i1] = sig
        return out
    return sig


def _score(y: np.ndarray, recon: np.ndarray, sr: int) -> Dict[str, float]:
    """Score reconstruction against target with all referee metrics."""
    m = compute_error_metrics(y, recon, sr=sr)
    return {
        "correlation": round(_corr(y, recon), 6),
        "snr_db": round(float(m["snr_db"]), 3),
        "log_spectral_distance_db": round(float(m["log_spectral_distance_db"]), 3),
        "seg_snr_db": round(segmental_snr_db(y, recon, sr), 3),
        "onset_corr": round(onset_corr(y, recon, sr), 6),
    }


def align_instantaneous_phase(
    target: np.ndarray,
    partial: Dict[str, Any],
    sr: int,
    start_sample: int = 0,
    stft_half_window_s: float = 0.0,
) -> float:
    """Align the partial's phase to the target using the analytic signal.

    Computes the analytic signals via Hilbert transform, extracts the
    weighted circular mean phase offset across the partial's support,
    and updates the partial's stored phase in place.
    Returns the phase correction delta applied.
    """
    n = len(target)
    if n < 8:
        return 0.0
    sig = _render_partial(
        partial, n, sr, start_sample=start_sample, stft_half_window_s=stft_half_window_s
    )
    if np.sum(sig ** 2) <= 1e-12 or np.sum(target ** 2) <= 1e-12:
        return 0.0
    z_target = hilbert(target)
    z_sig = hilbert(sig)
    cross = z_target * np.conj(z_sig)
    weights = np.abs(z_sig) ** 2
    c = float(np.sum(weights * np.real(cross)))
    s = float(np.sum(weights * np.imag(cross)))
    if np.hypot(c, s) <= 1e-12:
        return 0.0
    d_phi = -float(np.arctan2(s, c))
    cur_phi = float(partial.get("phase", 0.0))
    new_phi = float((cur_phi + d_phi) % (2.0 * np.pi))
    partial["phase"] = round(new_phi, 6)
    return d_phi


__all__ = [
    "_corr",
    "_glide_traj",
    "_support_mask",
    "_render_partial",
    "_score",
    "align_instantaneous_phase",
]
