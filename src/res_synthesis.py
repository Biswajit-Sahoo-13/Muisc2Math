"""RES synthesis and evaluation primitives.

Shared mathematical kernels for partial rendering, trajectory interpolation,
support masking, and referee metrics, used across both the coordinate lifter
(res_loop) and the hypothesis tournament bank (res_bank).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.signal
from scipy.signal import hilbert

from src.sinusoidal_model import _trajectories
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

    Delegates to sinusoidal_model._trajectories (3-frame median-smoothed
    frequency for the integral, 5-frame-smoothed amplitude, and the same
    analytical intra-frame phase correction) evaluated on ``t`` so
    section-offset renders stay on the global time axis. One implementation,
    so analyser and renderer cannot drift;
    tests/test_res_trajectory_defects.py asserts they agree to 1e-11 rad
    across arbitrary section offsets.
    """
    a_env, integral, _f_traj = _trajectories(tt, ff, aa, grid=t)
    return a_env, integral


_WHISTLE_FREQ_HZ = 1000.0
# Bands/trajectories below this relative level are gated to silence: a
# mastered recording drops to near-zero between events, and an ungated
# noise floor hisses there no matter what LSD says. High bands gate
# harder - gaps above 2 kHz are musically silent, while low-band level
# 0.2 can be a genuine quiet verse.
_NOISE_GATE_S = 0.10
_NOISE_GATE_S_HIGH = 0.30
_NOISE_GATE_HZ = 2000.0


def _noise_gate_for(center_hz: float) -> float:
    return (_NOISE_GATE_S_HIGH if center_hz >= _NOISE_GATE_HZ
            else _NOISE_GATE_S)
_MIN_DECAY_BASE = 2.5
_MIN_DECAY_SLOPE = 0.002
_MAX_GLIDE_HZ = 1800.0
# The release fade is deliberately SHORTER than the analyser's 0.25 s fade.
# The analyser (sinusoidal_model's inline support mask) holds a partial at a
# clamped amplitude for hold_s past its last frame and then fades over 0.25 s;
# the renderer deliberately refuses to ring that long, so a partial stops at
# its support instead of trailing off (tests/test_res_whistle_suppression.py
# asserts the fade stays <= 0.05 s and that the mask is exactly zero past
# support_end + fade).
#
# The consequence is that the renderer is not a bit-exact replica of the
# analyser: measured, the replica is 0.9906 on a full mix and 0.9962 on the
# bass stem, and matching the analyser's 0.25 s fade would take it to
# 0.9992 / 1.0000 - for a fidelity cost of about 0.001 correlation. The
# 0.99 is the intended state, not drift.
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


def _stochastic_onesided_enabled() -> bool:
    """Whether a stochastic channel honours one-sidedness and decay.

    Off by default so the shipped numbers do not move until the exciter
    has been measured. The switch is read per call rather than at import
    so one process can compare both behaviours.
    """
    import os

    return os.environ.get("M2M_EXCITER", "0") == "1"


def _support_mask(
    t: np.ndarray,
    start_time: float,
    end_time: float,
    decay: float,
    amp: float,
    stft_half_window_s: float,
    release_fade_s: float = _RELEASE_FADE_S,
) -> np.ndarray:
    """Support window shared by the SMS analyser and this renderer.

    Content one STFT half-window before the first frame is what the detecting
    window measured (backed by a rising ramp), the partial lives at full
    strength through the projection window, and past the last frame a fitted
    decay holds a little before a release fade. Without a half-window an
    approximate 20 ms attack is used.

    ``release_fade_s`` is the ONE deliberate analyser/renderer difference:
    the renderer passes the short default (``_RELEASE_FADE_S``) so a partial
    stops at its support instead of trailing off, while the analyser
    (``sinusoidal_model.refine_track``) passes 0.25 s. Both values are
    measured trade-offs - see the comment at ``_RELEASE_FADE_S`` and
    tests/test_res_whistle_suppression.py.
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
    release_term = np.clip((t - support_end) / release_fade_s, 0.0, 1.0)
    return attack_term * (1.0 - release_term)


def shape_noise_spectrum(
    seed: int,
    n: int,
    sr: int,
    band_centers: Sequence[float],
    band_mags: Sequence[float],
    zero_below_band: bool = False,
) -> np.ndarray:
    """Seeded white noise shaped by a log-spaced band envelope (spectrum).

    The shared kernel behind every ``filtered_noise`` proposal: draw the
    seeded white stream, interpolate the band magnitudes onto the rfft bins
    in log-frequency, and return the shaped complex spectrum. Callers do
    their own ``irfft`` - the class-C++ channel normalises the result, the
    sibilance burst keeps it raw, and the renderer may split it per band.

    ``zero_below_band`` silences everything below the first band centre:
    log-interpolation would otherwise hold the first magnitude flat across
    the low end and fill a highs-only burst with bass energy.
    """
    centers = np.asarray(band_centers, dtype=np.float64)
    mags = np.asarray(band_mags, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal(n)
    freq_bins = np.fft.rfftfreq(n, 1.0 / sr)
    min_f = max(10.0, float(centers[0]))
    log_bc = np.log10(np.maximum(centers, min_f))
    log_bins = np.log10(np.maximum(freq_bins, min_f))
    env_bins = np.interp(
        log_bins, log_bc, mags, left=mags[0], right=mags[-1]
    )
    if zero_below_band:
        env_bins = np.where(freq_bins >= centers[0], env_bins, 0.0)
    return np.fft.rfft(white) * env_bins


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


def fit_noise_floor_partial(
      residual: np.ndarray,
      sr: int,
      start_time: float = 0.0,
      end_time: float = 0.0,
      seed: int = 20240617,
      n_bands: int = 12,
      f_lo: float = 200.0,
) -> Optional[Dict[str, Any]]:
    """Model a residual's broadband noise floor as one ``filtered_noise`` partial.

    A bank of sinusoids cannot represent breath, consonants or cymbals. The
    result is a sparse set of exponentials on a dead floor: the envelope
    collapses between syllables (p95/p5 dynamic range measured at 160 dB for a
    real voice against 40 dB for the source) and everything above 4 kHz is
    missing (0.47% of the energy against 3.10% in the source). That is what
    makes a reconstruction sound like a robot rather than a voice.

    One seeded band-limited noise partial restores the floor. The seed is fixed
    so the ledger stays reproducible and the export stays deterministic.
    """
    x = np.asarray(residual, dtype=np.float64).ravel()
    if x.size < 512:
        return None
    n = x.size
    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(x * w))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    if freqs.size < 8:
        return None

    edges = np.geomspace(max(f_lo, 20.0), sr * 0.5, n_bands + 1)
    centers: List[float] = []
    mags: List[float] = []
    for i in range(n_bands):
        lo, hi = float(edges[i]), float(edges[i + 1])
        sel = (freqs >= lo) & (freqs < hi)
        if not np.any(sel):
            continue
        centers.append(float(np.sqrt(lo * hi)))
        # median, not mean: robust to the tonal peaks still in the residual
        mags.append(float(np.median(mag[sel])))
    if len(centers) < 3:
        return None

    m = np.asarray(mags, dtype=np.float64)
    peak = float(np.max(m))
    if peak <= 0.0:
        return None
    m = m / peak
    # keep only the floor, so we add noise rather than re-rendering the residual
    m = np.maximum(m, float(np.percentile(m, 25)))

    rms = float(np.sqrt(np.mean(x ** 2)))
    if rms <= 1e-6:
        return None
    partial = {
        "model": "filtered_noise",
        "amplitude": round(rms, 6),
        "seed": int(seed),
        "band_centers": [round(c, 2) for c in centers],
        "band_mags": [round(float(v), 6) for v in m],
        "envelope_decay": 0.0,
        "envelope_t0": float(start_time),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "phase": 0.0,
        "frequency": 0.0,
    }
    # Measure the residual's own level envelope over a few segments. Real
    # residual noise is transient-modulated (fills, breaths, cymbal decay);
    # a static floor renders its time-average - correct spectrum, no timing.
    # Only attach a trajectory when the level is genuinely uneven, so a
    # stationary floor stays byte-identical to the shipped behaviour.
    dur = float(end_time) - float(start_time)
    if dur > 0.0 and x.size >= 2048:
        # ~50 ms envelope resolution where the section allows it: 16 fixed
        # segments cannot time a hi-hat pattern (150 ms per step).
        n_seg = max(16, min(48, int(dur / 0.05)))
        seg = x.size // n_seg
        if seg >= 256:
            t_centers = [
                float(start_time) + (i + 0.5) * dur / n_seg
                for i in range(n_seg)
            ]
            seg_rms = np.array([
                float(np.sqrt(np.mean(x[i * seg:(i + 1) * seg] ** 2)))
                for i in range(n_seg)
            ])
            if rms > 1e-12 and float(seg_rms.max()) > 1e-12:
                shape = seg_rms / rms
                lo, hi = float(shape.min()), float(shape.max())
                if hi > 1e-9 and lo / hi < 0.5:
                    # 3-frame median: single-segment spikes are estimation
                    # noise, not envelopes (same convention as the glide
                    # frequency smoothing).
                    pad = np.concatenate(([shape[0]], shape, [shape[-1]]))
                    shape_s = np.array([
                        float(np.median(pad[max(0, i - 1):i + 2]))
                        for i in range(1, n_seg + 1)
                    ])
                    partial["trajectory"] = {
                        "t": [round(v, 6) for v in t_centers],
                        "a": [round(float(v), 6) for v in shape_s],
                    }
                # Per-band envelopes - the NoiseBandNet idea in closed form.
                # A cymbal's band switches on when the stick lands while the
                # bass band never moves; one global envelope cannot express
                # both. Measure each band's own short-time level; attach
                # only when some band is genuinely uneven (a flat floor
                # stays byte-identical to the shipped behaviour).
                if len(centers) >= 3 and x.size >= 4 * seg:
                    edges = np.geomspace(max(f_lo, 20.0), sr * 0.5,
                                         len(centers) + 1)
                    X = np.fft.rfft(x)
                    rows = []
                    uneven = False
                    for i in range(len(centers)):
                        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
                        if not mask.any():
                            rows.append(np.ones(n_seg))
                            continue
                        xb = np.fft.irfft(X * mask, n=x.size)
                        rb = np.sqrt(np.mean(
                            xb[: n_seg * seg].reshape(n_seg, seg) ** 2,
                            axis=1))
                        # Normalise to the band's PEAK segment, not its mean:
                        # gaps must reach ~0 so the render-side gate can
                        # silence them (measured: mean-normalised envelopes
                        # never dropped below the gate, and the render kept
                        # hissing 3.4x the source level in quiet gaps).
                        peak_b = float(rb.max())
                        shape_b = (rb / peak_b if peak_b > 1e-12
                                   else np.ones(n_seg))
                        pad = np.concatenate(
                            ([shape_b[0]], shape_b, [shape_b[-1]]))
                        shape_b = np.array([
                            float(np.median(pad[max(0, j - 1):j + 2]))
                            for j in range(1, n_seg + 1)
                        ])
                        blo, bhi = float(shape_b.min()), float(shape_b.max())
                        if bhi > 1e-9 and blo / bhi < 0.4:
                            uneven = True
                        rows.append(shape_b)
                    if uneven:
                        partial["band_edges"] = [
                            round(float(v), 2) for v in edges
                        ]
                        partial["band_traj"] = {
                            "t": [round(v, 6) for v in t_centers],
                            "a": [[round(float(v), 6) for v in row]
                                  for row in rows],
                        }
                        # The per-band envelope supersedes the global one -
                        # applying both would modulate twice.
                        partial.pop("trajectory", None)
    return partial


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
        # The phase reference term must use the SAME anchor frequency the SMS
        # engine projected against. Re-estimating it here (median of the
        # smoothed trajectory) silently changes the term, and the phase error
        # then grows with t instead of staying constant.
        ref_hz = p.get("phase_ref_hz")
        anchor = (
            float(ref_hz) if ref_hz is not None else float(np.median(ff))
        )
        theta = integral + 2.0 * np.pi * (freq_ref - anchor) * t_fit
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
        # The analyser renders each partial inside its own analysis window and
        # pastes the result at the window's start, so the partial's ringing
        # tail is clipped at the window's far edge. Without this clip the tail
        # leaks into the next section, and the section-based render no longer
        # reproduces what the analyser produced.
        wend = p.get("window_end")
        if wend is not None:
            sig = np.where(t < float(wend), sig, 0.0)
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
        band_centers = np.asarray(p.get("band_centers", []), dtype=np.float64)
        band_mags = np.asarray(p.get("band_mags", []), dtype=np.float64)
        if band_centers.size >= 2 and band_mags.size == band_centers.size:
            shaped = shape_noise_spectrum(seed, n, sr, band_centers, band_mags)
            freq_bins = np.fft.rfftfreq(n, 1.0 / sr)
            # Per-band envelopes: split the shaped spectrum by the fitted
            # band edges, ring each band down/up on its own measured
            # envelope, and sum. This is what turns "stationary hiss" into
            # a cymbal that strikes when the stick lands.
            btraj = p.get("band_traj")
            bedges = p.get("band_edges")
            used_bands = False
            if btraj and bedges:
                tt = np.asarray(btraj.get("t", []), dtype=np.float64)
                aa = np.asarray(btraj.get("a", []), dtype=np.float64)
                be = np.asarray(bedges, dtype=np.float64)
                if (tt.size >= 2 and aa.ndim == 2
                        and aa.shape[0] == be.size - 1
                        and aa.shape[1] == tt.size):
                    sig = np.zeros(n)
                    for i in range(aa.shape[0]):
                        mask = (freq_bins >= be[i]) & (freq_bins < be[i + 1])
                        if not mask.any():
                            continue
                        sig_b = np.fft.irfft(shaped * mask, n=n)
                        env_b = np.interp(
                            t_full, tt, aa[i], left=aa[i][0], right=aa[i][-1]
                        )
                        center = float(np.sqrt(be[i] * be[i + 1]))
                        gate = _noise_gate_for(center)
                        env_b = np.where(env_b < gate, 0.0, env_b)
                        sig += sig_b * np.maximum(env_b, 0.0)
                    used_bands = True
            if not used_bands:
                sig = np.fft.irfft(shaped, n=n)
        else:
            rng = np.random.default_rng(seed)
            sig = rng.standard_normal(n)
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
        # A one-sided model must not sound before its onset. The support mask's
        # attack ramp reaches half a STFT window back, so without this a noise
        # burst carries its energy before the hit it belongs to - the same
        # defect the bank already avoids on the damped modes.
        if p.get("unit_step") and _stochastic_onesided_enabled():
            t0 = float(p.get("envelope_t0", st if st is not None else 0.0))
            sig = np.where(t_full >= t0, sig, 0.0)
        # An optional decay turns a stationary noise channel into a burst that
        # rings down, which is what a struck body does. Zero means stationary,
        # which is what the noise floor wants.
        if decay > 0.0 and _stochastic_onesided_enabled():
            t0 = float(p.get("envelope_t0", st if st is not None else 0.0))
            sig = sig * np.exp(-decay * np.maximum(t_full - t0, 0.0))
        # A measured amplitude trajectory redistributes the level in time -
        # the difference between stationary hiss and an actual fill or
        # breath. Times are global (same convention as the glide
        # trajectories); the global RMS stays the partial's amplitude.
        # A noise gate keeps the gaps silent: real recordings drop to
        # near-zero between events, and an ungated floor hisses there
        # (measured 3.4x the source's gap level) no matter what the
        # spectral metrics say.
        traj = p.get("trajectory")
        if traj:
            tt = np.asarray(traj.get("t", []), dtype=np.float64)
            aa = np.asarray(traj.get("a", []), dtype=np.float64)
            if tt.size >= 2 and aa.size == tt.size:
                env = np.interp(t_full, tt, aa, left=aa[0], right=aa[-1])
                env = np.where(env < _NOISE_GATE_S, 0.0, env)
                sig = sig * np.maximum(env, 0.0)
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
        if p.get("unit_step") and _stochastic_onesided_enabled():
            t0 = float(p.get("envelope_t0", st if st is not None else 0.0))
            sig = np.where(t_full >= t0, sig, 0.0)
        if decay > 0.0 and _stochastic_onesided_enabled():
            t0 = float(p.get("envelope_t0", st if st is not None else 0.0))
            sig = sig * np.exp(-decay * np.maximum(t_full - t0, 0.0))
        return sig

    amp = float(p.get("amplitude", 0.0))
    decay = float(p.get("envelope_decay", 0.0))
    t0 = float(p.get("envelope_t0", 0.0))
    f0 = float(p.get("frequency", 0.0))
    ph = float(p.get("phase", 0.0))
    env = amp * np.exp(-decay * np.maximum(t - t0, 0.0))
    if p.get("unit_step"):
        # One-sided model: no energy before the envelope's onset. The bank
        # sets this on its hypotheses, whose support mask otherwise lets a
        # half-window attack ramp carry a fast mode's energy before the hit.
        env = np.where(t >= t0, env, 0.0)
    sig = env * np.cos(2.0 * np.pi * f0 * t_fit - ph)
    st = p.get("start_time", None)
    et = p.get("end_time", None)
    if st is not None and et is not None and amp > 0.0:
        sig = sig * _support_mask(t, float(st), float(et), decay, amp, stft_half_window_s)
    # Clipped to the analysis window the partial was fitted in - see the
    # glide branch above for why.
    wend = p.get("window_end")
    if wend is not None:
        sig = np.where(t < float(wend), sig, 0.0)
    if win is not None:
        out = np.zeros(n)
        out[i0:i1] = sig
        return out
    return sig


def _score(y: np.ndarray, recon: np.ndarray, sr: int) -> Dict[str, float]:
    """Score reconstruction against target with all referee metrics."""
    m = compute_error_metrics(y, recon, sr=sr, extras=True)
    return {
        "correlation": round(_corr(y, recon), 6),
        "snr_db": round(float(m["snr_db"]), 3),
        "log_spectral_distance_db": round(float(m["log_spectral_distance_db"]), 3),
        "seg_snr_db": round(segmental_snr_db(y, recon, sr), 3),
        "onset_corr": round(onset_corr(y, recon, sr), 6),
        "mrstft_db": round(float(m["mrstft_db"]), 3),
        "band_corr": {k: round(float(v), 4) for k, v in m["band_corr"].items()},
    }


def apply_energy_parity(
    y: np.ndarray,
    recon: np.ndarray,
    spans: Sequence[Tuple[int, int]],
    max_gain: float = 50.0,
    mode: str = "ls",
) -> Tuple[np.ndarray, List[float]]:
    """Refit each span's render level against the source.

    Correlation - the referee's own metric - is scale-invariant, so nothing
    in the loop enforces loudness: measured, a drums stem shipped its best
    render at 2.1% of the source energy. Two modes:

    - ``mode="ls"`` (default): the least-squares gain per span, the one
      that minimises that span's residual energy and therefore maximises
      SNR. It provably lands the render at ``(correlation) x`` source RMS -
      SNR-optimal, but a low-corr render ends up sounding too quiet.
    - ``mode="rms"``: scale the span to the source's RMS - loudness parity,
      the "sounds like the original" choice, at a small SNR cost when the
      correlation is low.

    Guards: a (near-)silent render span is left alone, the gain never goes
    negative (never invert a render) and never exceeds ``max_gain``.
    """
    if mode not in ("ls", "rms"):
        raise ValueError(f"unknown parity mode: {mode!r}")
    out = np.array(recon, dtype=np.float64, copy=True)
    gains: List[float] = []
    for start, end in spans:
        start, end = int(start), int(end)
        if not (0 <= start < end <= out.size):
            gains.append(1.0)
            continue
        seg = out[start:end]
        e_r = float(np.dot(seg, seg))
        if e_r <= 1e-15:
            gains.append(1.0)
            continue
        if mode == "rms":
            rms_y = float(np.sqrt(np.mean(y[start:end] ** 2)))
            rms_r = float(np.sqrt(e_r / (end - start)))
            g = rms_y / rms_r if rms_y > 1e-12 else 1.0
        else:
            g = float(np.dot(y[start:end], seg) / e_r)
        if not np.isfinite(g):
            g = 1.0
        g = min(max(g, 0.0), float(max_gain))
        out[start:end] = seg * g
        gains.append(g)
    return out, gains


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
