"""Class E: the exciter - the broadband attack of a struck body.

The percussive class models a hit as a set of quickly-dying modes. That
is the *body*: the shell or the head ringing after the strike. It does
not model the strike itself, and for a snare the strike is most of the
energy - a noise burst exciting a set of resonances.

This is the standard exciter-resonator formulation: an excitation filtered
by a bank of resonances. The difference spectrum (the attack minus the
pre-hit window) exposes what actually began at the hit; the narrow peaks
in it are the modes, and the median level of each band is the broadband
floor the exciter has to supply. Taking the median is what makes this
robust - it ignores the modes and keeps the shape of the noise.

The exciter is proposed as a short, seeded noise burst shaped by that
broadband envelope, gated to the hit and ringing down. It reuses the
existing ``filtered_noise`` render path, so there is no new render
machinery and the equation sheet keeps working: a noise channel with a
spectral envelope, an onset, and a decay.

The amplitude is measured by projecting the candidate against what the
modes did not explain, through the real render path, so the proposal is
scored on exactly the signal the renderer will produce.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.res_synthesis import _render_partial


def broadband_envelope(
    mag: np.ndarray,
    bins: np.ndarray,
    band_lo: float = 45.0,
    band_hi: float = 9000.0,
    n_bands: int = 8,
) -> Tuple[List[float], List[float]]:
    """The broadband level of a magnitude spectrum, in log-spaced bands.

    The median of each band is used rather than its mean, so a narrow
    modal peak does not pass for broadband energy. That is the whole
    point: the modes are already modelled, and what is left is the
    strike.
    """
    centers: List[float] = []
    mags: List[float] = []
    if mag.size == 0 or bins.size != mag.size:
        return centers, mags
    lo = float(max(band_lo, float(bins[0])))
    hi = float(min(band_hi, float(bins[-1])))
    if not (hi > lo):
        return centers, mags
    edges = np.logspace(np.log10(lo), np.log10(hi), int(n_bands) + 1)
    for i in range(int(n_bands)):
        a, b = float(edges[i]), float(edges[i + 1])
        sel = (bins >= a) & (bins < b)
        if not np.any(sel):
            continue
        centers.append(float(np.sqrt(a * b)))
        mags.append(float(np.median(mag[sel])))
    return centers, mags


def spectral_log_distance(
    reference: np.ndarray, estimate: np.ndarray
) -> float:
    """Root-mean-square log-magnitude distance between two signals.

    This is the metric that can see a broadband component. A noise burst
    cannot be sample-matched - two realisations of noise are uncorrelated,
    so a waveform error never rewards adding one - but its *spectrum* can
    be matched, and a spectral error does.
    """
    n = int(min(reference.size, estimate.size))
    if n < 32:
        return float("inf")
    w = np.hanning(n)
    ref = np.abs(np.fft.rfft(reference[:n] * w))
    est = np.abs(np.fft.rfft(estimate[:n] * w))
    floor = max(float(ref.max()), 1e-9) * 1e-4
    ref = np.maximum(ref, floor)
    est = np.maximum(est, floor)
    return float(np.sqrt(np.mean((np.log10(est) - np.log10(ref)) ** 2)))


def resonant_burst_hypothesis(
    target: np.ndarray,
    sr: int,
    start_sample: int,
    stft_half_window_s: float,
    hit_sample: int,
    band_centers: List[float],
    band_mags: List[float],
    burst_s: float = 0.012,
    seed: Optional[int] = None,
    min_energy_share: float = 0.01,
) -> Optional[Dict[str, Any]]:
    """Propose the broadband strike that excites a hit's modes.

    ``target`` is what the modes have not already explained.

    The amplitude is set by an **energy match** over the burst window,
    not by a least-squares projection. A projection would be the obvious
    choice and it is wrong here: two realisations of noise are
    uncorrelated, so the projection of any candidate burst onto the
    target is near zero and the proposal would be rejected even when the
    burst is exactly what the hit needs. Noise is predictable in
    *spectrum*, not in samples.

    The proposal is only made when the unexplained energy in the burst
    window is a real share of the section, and the caller is expected to
    judge it on a spectral criterion.
    """
    n = int(target.size)
    if n < 16 or len(band_centers) < 2 or len(band_mags) < 2:
        return None
    hit = int(hit_sample) - int(start_sample)
    if hit < 0 or hit >= n:
        return None
    hit_t = hit_sample / float(sr)
    span = int(min(burst_s * sr, max(n - hit, 0)))
    if span < 16:
        return None
    if seed is None:
        seed = (int(start_sample) * 131 + int(hit_sample)) % (2 ** 31 - 1)
    decay = 1.0 / (2.0 * burst_s)
    partial: Dict[str, Any] = {
        "amplitude": 1.0,
        "envelope_decay": round(decay, 6),
        "envelope_t0": round(hit_t, 6),
        "model": "filtered_noise",
        "start_time": round(hit_t, 6),
        "end_time": round(hit_t + 4.0 * burst_s, 6),
        "band_centers": [round(float(c), 4) for c in band_centers],
        "band_mags": [round(float(m), 9) for m in band_mags],
        "seed": int(seed),
        # One-sided: the strike must not sound before the hit. The support
        # mask's attack ramp reaches half a STFT window back, so without
        # this the burst carries its energy before the hit it belongs to.
        "unit_step": True,
        "note_group": round(hit_t, 3),
    }
    sig = _render_partial(
        partial, n, sr, start_sample=int(start_sample),
        stft_half_window_s=stft_half_window_s,
    )
    w = slice(hit, hit + span)
    s = sig[w]
    t = target[w]
    e_burst = float(np.dot(s, s))
    e_target = float(np.dot(t, t))
    if e_burst <= 1e-18 or e_target <= 1e-18:
        return None
    section_energy = float(np.dot(target, target))
    if section_energy <= 1e-18:
        return None
    if e_target / section_energy < min_energy_share:
        # There is no broadband energy here worth a partial.
        return None
    # A strike is an *onset*. If the window before the hit is as loud as
    # the window after it, the energy here was already sounding and is
    # not a strike at all - it belongs to whatever sustained it.
    pre_len = int(min(span, max(hit, 0)))
    if pre_len >= 16:
        pre_rms = float(np.sqrt(np.mean(target[hit - pre_len:hit] ** 2)))
        post_rms = float(np.sqrt(np.mean(t ** 2)))
        if pre_rms > 0.5 * post_rms:
            return None
    # Energy match: the burst carries the energy the modes left behind.
    amp = float(np.sqrt(e_target / e_burst))
    if not np.isfinite(amp) or amp <= 0.0:
        return None
    partial["amplitude"] = round(amp, 6)
    return partial
