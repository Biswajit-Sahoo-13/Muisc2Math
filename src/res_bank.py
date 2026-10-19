"""RES hypothesis bank (v2): multiple equation classes compete per section.

Class A, ``damped_cosine``, is the verified v1 state: independent SMS partials
refitted by the greedy coordinate lifter (matching pursuit over continuous
damped-cosine atoms, Mallat & Zhang 1993).

Class B, ``harmonic_string``, nominates *whole notes*: partials locked to the
inharmonicity relation f_m = m*F0*sqrt(1 + B*m^2) (Valette & Cuesta string
stiffness; see arXiv 2409.06513 for the modern neural wrapper), with
per-partial decay optionally consolidated into the linear Valette-Cuesta
damping law sigma_m = a + b*f_m. Note candidates come from the section
*residual* via a Klapuri-style harmonic-sum salience scan (ISMIR 2006), so the
class only sees fundamentals the current state already fails to explain.

The tournament is additive and monotone: a candidate is accepted only when
appending its partials strictly decreases the section residual energy - the
same optimality criterion the class A lifter enforces per atom. Every
candidate, accepted or not, is recorded under ``section["hypothesis_bank"]``
for the audit trail; the ledger stays the honest transcript.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import scipy.signal
from scipy.optimize import minimize_scalar

from src.res_synthesis import _corr, _render_partial, _support_mask

_E_REL_MARGIN = 1e-6  # relative residual-energy improvement to accept
_NOISE_K = 4.0  # overfit guard: expected drop from p random params ~ p*sigma^2


def acceptance_margin(
    e_y: float, cur_err: float, param_count: int, n: int
) -> float:
    """Accept only when the gain clearly beats noise-chasing.

    Fitting p free parameters to a white residual reduces its sum of squares
    by ~p * sigma^2 on average, so a candidate must deliver strictly more than
    ``k * p/n`` of the current residual (plus a small relative floor) before
    the tournament accepts it. This keeps the ledger free of pseudo-notes.
    """
    base = _E_REL_MARGIN * e_y + 1e-12
    overfit = _NOISE_K * max(param_count, 0) / float(n) * cur_err
    return base + overfit


def spectral_flatness(x: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Spectral Flatness Measure (SFM / Wiener Entropy).

    Near 0.0 for pure tones / sinusoids; near 1.0 for white noise / unvoiced speech.
    """
    if len(x) < 8:
        return 0.0
    w = np.hanning(len(x))
    p = np.abs(np.fft.rfft(x * w)) ** 2 + eps
    geo = np.exp(np.mean(np.log(p)))
    arith = np.mean(p)
    return float(geo / max(arith, eps))


def band_spectral_flatness(
    x: np.ndarray,
    sr: int,
    f_lo: float = 3000.0,
    f_hi: float | None = None,
    eps: float = 1e-12,
) -> float:
    """Spectral flatness measured over one frequency band only.

    Whole-spectrum SFM is the wrong detector for sibilance. A vocal "s" is
    band-limited noise: its spectrum is near-zero below ~3 kHz, which drives
    the geometric mean to zero and pins whole-spectrum SFM at ~0 regardless of
    how noisy the fricative actually is. Measuring flatness inside the band
    where sibilance and cymbals live asks the right question - is *this band*
    noise-like - and separates unvoiced fricatives from tonal content.
    """
    if len(x) < 8:
        return 0.0
    hi = f_hi if f_hi is not None else (sr / 2.0)
    w = np.hanning(len(x))
    p = np.abs(np.fft.rfft(x * w)) ** 2 + eps
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    sel = (freqs >= f_lo) & (freqs <= hi)
    band = p[sel]
    if band.size < 4:
        return 0.0
    # An empty band is perfectly "flat" (a constant eps floor) and would invert
    # the gate: band-limited tonal content has no high-band energy at all, so
    # without this guard every pure tone reads as maximum flatness.
    band_e = float(np.sum(band) - band.size * eps)
    total_e = float(np.sum(p) - p.size * eps)
    if band_e <= 1e-18 or band_e <= 1e-4 * max(total_e, 1e-18):
        return 0.0
    geo = np.exp(np.mean(np.log(band)))
    arith = np.mean(band)
    return float(geo / max(arith, eps))


def zero_crossing_rate(x: np.ndarray) -> float:
    """Compute Zero-Crossing Rate (ZCR) of a signal."""
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(x)))))


def section_lsd(
    target: np.ndarray,
    recon: np.ndarray,
    n_fft: int = 512,
    eps: float = 1e-12,
) -> float:
    """Compute Log-Spectral Distance (LSD in dB) between target and reconstruction."""
    n = len(target)
    if n < 32:
        return 0.0
    win_len = min(n_fft, n)
    w = np.hanning(win_len)
    if n <= win_len:
        s_t = np.abs(np.fft.rfft(target[:win_len] * w))
        s_r = np.abs(np.fft.rfft(recon[:win_len] * w))
        log_t = np.maximum(20.0 * np.log10(s_t + eps), -80.0)
        log_r = np.maximum(20.0 * np.log10(s_r + eps), -80.0)
        return float(np.sqrt(np.mean((log_t - log_r) ** 2)))

    hop = win_len // 2
    n_frames = max(1, (n - win_len) // hop + 1)
    frame_lsds = []
    for f in range(n_frames):
        start = f * hop
        end = start + win_len
        s_t = np.abs(np.fft.rfft(target[start:end] * w))
        s_r = np.abs(np.fft.rfft(recon[start:end] * w))
        log_t = np.maximum(20.0 * np.log10(s_t + eps), -80.0)
        log_r = np.maximum(20.0 * np.log10(s_r + eps), -80.0)
        frame_lsds.append(np.sqrt(np.mean((log_t - log_r) ** 2)))
    return float(np.mean(frame_lsds))


# ----------------------------------------------------------------------
# Evidence channel: harmonic-sum salience (Klapuri 2006, simplified)
# ----------------------------------------------------------------------

def _hann_mag(x: np.ndarray) -> np.ndarray:
    w = np.hanning(x.size)
    return np.abs(np.fft.rfft(x * w))


def _f_grid() -> np.ndarray:
    return np.geomspace(55.0, 500.0, 221)


def f0_candidates(
    target: np.ndarray,
    sr: int,
    k: int = 2,
    M: int = 8,
) -> List[float]:
    """Top-k fundamental candidates for a residual via harmonic summation.

    salience(f) = sum_m |X_(m*f)| for m = 1..M; each coarse top bin is refined
    by a +-2 Hz bounded search. ``k`` caps how many notes get nominated.

    Polyphony upgrade (Klapuri's iterative estimation-cancellation): after the
    best f0 is found, its harmonic comb is notched out of the spectrum and the
    scan runs again, so a second instrument no longer disappears behind the
    first one's shared bins.
    """
    n = len(target)
    if n < 64:
        return []
    mag = _hann_mag(target)
    bins = np.fft.rfftfreq(n, 1.0 / sr)
    nyq_lo = 0.92 * sr / 2.0

    def salience(f: float, spectra: np.ndarray) -> float:
        """Harmonic-sum salience with 1/m weights (Klapuri 2006).

        Weighting demands the strong bins sit at LOW harmonic indices of the
        candidate, which is what separates a true fundamental from a
        subharmonic that merely samples its neighbors' strong bins.
        """
        m_max = min(M, max(1, int(np.floor(nyq_lo / max(f, 1e-6)))))
        freqs = f * np.arange(1, m_max + 1)
        keep = freqs <= nyq_lo
        if not np.any(keep):
            return 0.0
        freqs = freqs[keep]
        idx = np.searchsorted(bins, freqs)
        wts = 1.0 / np.arange(1, m_max + 1)[keep]
        return float(np.sum(wts * spectra[np.clip(idx, 0, spectra.size - 1)]))

    def refined_top(spectra: np.ndarray, take: int = 4) -> List[Tuple[float, float]]:
        grid = _f_grid()
        svals = np.array([salience(f, spectra) for f in grid])
        if not np.any(svals > 0):
            return []
        order = np.argsort(svals)[::-1]
        out: List[Tuple[float, float]] = []
        seen: List[float] = []
        for g in order[:8]:
            f0 = float(grid[g])
            if svals[g] <= 0:
                continue
            lo, hi = max(40.0, f0 - 2.0), f0 + 2.0
            try:
                res = minimize_scalar(
                    lambda fv: -salience(fv, spectra),
                    bounds=(lo, hi), method="bounded",
                )
                if res.success:
                    cand = float(res.x)
                    if salience(cand, spectra) >= salience(f0, spectra):
                        f0 = cand
            except Exception:
                pass
            if any(abs(f0 - s) < 0.015 * max(f0, s) for s in seen):
                continue
            seen.append(f0)
            out.append((f0, salience(f0, spectra)))
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:take]

    first = refined_top(mag, take=4)
    if not first:
        return []
    merged = list(first)

    def even_fraction(f: float, spectra: np.ndarray) -> float:
        """Fraction of a candidate's salience carried by its even harmonics.

        A subharmonic q*f0 has no odd partials of its own: all its weight
        sits on bins that are multiples of 2f. Klapuri's voicing check - a
        near-total even fraction means f is really a subharmonic of 2f.
        """
        tot = salience(f, spectra)
        if tot <= 0:
            return 0.0
        m_max = min(M // 2, max(1, int(np.floor(nyq_lo / max(f, 1e-6)) // 2)))
        evn = 0.0
        for m in range(1, m_max + 1):
            bin_i = int(np.searchsorted(bins, f * 2.0 * m))
            if bin_i < spectra.size and bins[bin_i] <= nyq_lo:
                evn += float(spectra[bin_i]) / (2.0 * m)
        return evn / tot

    def veto_subharmonics(cands: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        voiced = [(f0, s, even_fraction(f0, mag)) for f0, s in cands]
        passing = [
            (f0, s) for f0, s, frac in voiced
            if frac < 0.70 or salience(2.0 * f0, mag) < 0.75 * s
        ]
        return passing if passing else cands

    merged = veto_subharmonics(merged)
    if len(merged) >= 2 and k > 1:
        # Cancel pass: notch the best note's comb out of the spectrum.
        f0_best = merged[0][0]
        mag2 = mag.copy()
        for m in range(1, M + 1):
            fc = f0_best * m
            if fc >= nyq_lo:
                break
            bi = int(np.searchsorted(bins, fc))
            lo, hi = max(0, bi - 3), min(mag2.size, bi + 4)
            depth = float(np.max(mag2[lo:hi])) if hi > lo else 0.0
            if depth > 0:
                mag2[lo:hi] -= depth  # zero the local peak, keep the floor
        freqs_all = np.fft.rfftfreq(n, 1.0 / sr)
        mask = freqs_all <= nyq_lo
        second = refined_top(mag2 * mask, take=3)
        for f0, s in second:
            if any(abs(f0 - s0) < 0.015 * max(f0, s0) for s0, _ in merged):
                continue
            merged.append((f0, s))
        merged = veto_subharmonics(merged)
    merged.sort(key=lambda p: p[1], reverse=True)
    if not merged:
        return []
    s_best = merged[0][1]
    return [f0 for f0, s in merged[: max(1, k)] if s >= 0.2 * s_best]


# ----------------------------------------------------------------------
# Class B: inharmonic string bank (Valette & Cuesta via arXiv 2409.06513)
# ----------------------------------------------------------------------

def freq_m(f0: float, B: float, m: int) -> float:
    """Inharmonic partial frequency f_m = m*F0*sqrt(1 + B*m^2)."""
    return m * f0 * float(np.sqrt(1.0 + B * m * m))


def _masked_proj_basis(
    t: np.ndarray,
    t_fit: np.ndarray,
    t0: float,
    f: float,
    lam: float,
    sec_start: float,
    sec_end: float,
    stft_half_window_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Basis matching _render_partial exactly: masked damped (cos, sin)."""
    env = np.exp(-lam * np.maximum(t - t0, 0.0))
    ang = 2.0 * np.pi * f * t_fit
    mask = _support_mask(t, sec_start, sec_end, lam, 1.0, stft_half_window_s)
    return env * mask * np.cos(ang), env * mask * np.sin(ang)


def _variant_partial(
    amp: float,
    lam: float,
    t0: float,
    f: float,
    ph: float,
    f0: float,
    m: int,
    sec_start: float,
    sec_end: float,
) -> Dict[str, Any]:
    return {
        "amplitude": round(amp, 6),
        "envelope_decay": round(lam, 6),
        "envelope_t0": round(t0, 6),
        "frequency": round(f, 6),
        "phase": round(ph, 6),
        "model": "damped",
        "start_time": round(sec_start, 6),
        "end_time": round(sec_end, 6),
        "m": int(m),
        "note_group": round(f0, 3),
    }


def _variant_render(
    partials: List[Dict[str, Any]],
    t: np.ndarray,
    t_fit: np.ndarray,
    sec_start: float,
    sec_end: float,
    stft_half_window_s: float,
) -> np.ndarray:
    out = np.zeros(t.size)
    for p in partials:
        a = float(p["amplitude"])
        lam = float(p["envelope_decay"])
        t0 = float(p["envelope_t0"])
        f = float(p["frequency"])
        ph = float(p["phase"])
        env = a * np.exp(-lam * np.maximum(t - t0, 0.0))
        ang = 2.0 * np.pi * f * t_fit - ph
        mask = _support_mask(
            t, sec_start, sec_end, lam, a, stft_half_window_s
        )
        out += env * np.cos(ang) * mask
    return out


class _Variant:
    """One concrete class-B hypothesis candidate."""

    def __init__(
        self,
        partials: List[Dict[str, Any]],
        err: float,
        param_count: int,
        B: float,
        f0: float,
        t0: float,
        vc_lock: bool = False,
        sigma_a: float = 0.0,
        sigma_b: float = 0.0,
    ) -> None:
        self.partials = partials
        self.err = err
        self.param_count = param_count
        self.B = B
        self.f0 = f0
        self.t0 = t0
        self.vc_lock = vc_lock
        self.sigma_a = sigma_a
        self.sigma_b = sigma_b

    def as_record(self, m_count: int, accepted: bool, err_before: float) -> Dict[str, Any]:
        return {
            "class": "harmonic_string",
            "f0": round(self.f0, 4),
            "B": round(self.B, 8),
            "m_count": int(m_count),
            "param_count": int(self.param_count),
            "vc_lock": bool(self.vc_lock),
            "sigma_a": round(self.sigma_a, 6) if self.vc_lock else None,
            "sigma_b": round(self.sigma_b, 6) if self.vc_lock else None,
            "t0": round(self.t0, 6),
            "residual_before": round(err_before, 3),
            "residual_after": round(self.err, 3),
            "accepted": bool(accepted),
        }


def _fit_candidate(
    target: np.ndarray,
    t: np.ndarray,
    t_fit: np.ndarray,
    f0: float,
    B: float,
    t0: float,
    m_cap: int,
    sec_start: float,
    sec_end: float,
    stft_half_window_s: float,
    decay_grid: Sequence[float],
    sr: int,
) -> _Variant:
    """Fit one (f0, B, t0) note: per-harmonic masked projection per decay.

    For each harmonic m the (amplitude, phase) is projected onto the masked
    damped basis at every candidate decay; the best decay survives. Harmonics
    above the Nyquist margin and near-silent amplitudes are dropped, then the
    joint render error of the surviving set is measured with the renderer's
    own math, so candidate scores are directly comparable with the section's
    actual residual.
    """
    partials: List[Dict[str, Any]] = []
    nyq = 0.98 * sr / 2.0
    for m in range(1, m_cap + 1):
        f = freq_m(f0, B, m)
        if f >= nyq:
            break
        best: Optional[Tuple[float, float, float, float]] = None
        for lam in decay_grid:
            c, s = _masked_proj_basis(
                t, t_fit, t0, f, float(lam), sec_start, sec_end,
                stft_half_window_s,
            )
            base = float(np.sum(np.square(c)) + np.sum(np.square(s)))
            if base <= 1e-18:
                continue
            cc = float(np.dot(c, target)) / base
            ss = float(np.dot(s, target)) / base
            err = float(np.sum((target - (cc * c + ss * s)) ** 2))
            if best is None or err < best[0]:
                best = (err, float(lam), cc, ss)
        if best is None:
            continue
        _, lam_best, cc, ss = best
        a = float(np.hypot(cc, ss))
        ph = float(np.arctan2(ss, cc))
        if a > 0.0:
            partials.append(
                _variant_partial(a, lam_best, t0, f, ph, f0, m, sec_start, sec_end)
            )
    if not partials:
        return _Variant([], float(np.sum(target ** 2)), 0, B, f0, t0)
    a_max = max(float(p["amplitude"]) for p in partials)
    keep = [
        p for p in partials
        if float(p["amplitude"]) >= max(1e-4, 0.02 * a_max)
    ]
    if not keep:
        return _Variant([], float(np.sum(target ** 2)), 0, B, f0, t0)
    param_count = 2 + 3 * len(keep)  # f0, B + per-harmonic (a, ph, lam)
    sig = _variant_render(keep, t, t_fit, sec_start, sec_end, stft_half_window_s)
    err = float(np.sum((target - sig) ** 2))
    return _Variant(keep, err, param_count, B, f0, t0)


def _vc_lock(
    variant: _Variant,
    target: np.ndarray,
    t: np.ndarray,
    t_fit: np.ndarray,
    sec_start: float,
    sec_end: float,
    stft_half_window_s: float,
    e_rel_tol: float,
) -> _Variant:
    """Valeta-Cuesta lock: sigma_m = a + b*f_m, refit (a, ph) at locked decay.

    Physically the per-partial damping of a string note rises with partial
    frequency; when the decays follow that linear law (tight fit, nonnegative
    slope) the locked variant costs 2K+4 parameters instead of 3K+2. The lock
    is adopted when its error is no more than ``e_rel_tol`` worse than the
    free fit's - parsimony at (near) equal measured error.
    """
    k = len(variant.partials)
    if k < 3:
        return variant
    lam_s = np.array(
        [float(p["envelope_decay"]) for p in variant.partials], dtype=np.float64
    )
    f_s = np.array(
        [float(p["frequency"]) for p in variant.partials], dtype=np.float64
    )
    A = np.column_stack((np.ones_like(f_s), f_s))
    coef, *_ = np.linalg.lstsq(A, lam_s, rcond=None)
    a0, b0 = float(coef[0]), float(coef[1])
    denom = float(np.sum((lam_s - float(np.mean(lam_s))) ** 2))
    if not np.all(np.isfinite(np.array([a0, b0]))) or b0 < 0.0 or denom <= 1e-18:
        return variant
    fit_res = float(np.sum((lam_s - np.maximum(A @ np.array([a0, b0]), 0.0)) ** 2))
    if fit_res > 0.50 * denom:
        return variant
    locked: List[Dict[str, Any]] = []
    fitted: List[Tuple[float, Dict[str, Any]]] = []
    for p in variant.partials:
        f = float(p["frequency"])
        lam_lock = max(a0 + b0 * f, 0.0)
        c, s = _masked_proj_basis(
            t, t_fit, float(p["envelope_t0"]), f, lam_lock,
            sec_start, sec_end, stft_half_window_s,
        )
        base = float(np.sum(np.square(c)) + np.sum(np.square(s)))
        if base <= 1e-18:
            return variant
        cc = float(np.dot(c, target)) / base
        ss = float(np.dot(s, target)) / base
        a = float(np.hypot(cc, ss))
        ph = float(np.arctan2(ss, cc))
        if a > 0.0:
            fitted.append((
                a,
                _variant_partial(
                    a, lam_lock, float(p["envelope_t0"]), f, ph,
                    variant.f0, int(p.get("m", 1)), sec_start, sec_end,
                ),
            ))
    if len(fitted) < max(3, k - 1):
        return variant
    a_max = max(a for a, _ in fitted)
    fitted = [(a, q) for a, q in fitted if a >= max(1e-4, 0.02 * a_max)]
    if len(fitted) < max(3, k - 1) or not fitted:
        return variant
    locked = [q for _, q in fitted]
    sig = _variant_render(locked, t, t_fit, sec_start, sec_end, stft_half_window_s)
    err = float(np.sum((target - sig) ** 2))
    if err <= variant.err + e_rel_tol * max(variant.err, 1.0):
        return _Variant(
            locked, err, 2 + 2 * len(locked) + 2, variant.B, variant.f0,
            variant.t0, vc_lock=True, sigma_a=a0, sigma_b=b0,
        )
    return variant


def harmonic_string_hypothesis(
    target: np.ndarray,
    sr: int,
    start_sample: int,
    stft_half_window_s: float,
    f0: float,
    lam_hint: float = 1.0,
) -> _Variant:
    """Best class-B note candidate for ``target`` at fundamental ``f0``.

    Searches the small grids (B, note-onset t0, shared-decay scalings),
    consolidates per harmonic, then tries the Valette-Cuesta damping lock.
    ``target`` is the section residual the note must explain.
    """
    n = target.size
    t = (np.arange(n, dtype=np.float64) + start_sample) / sr
    t_fit = np.arange(n, dtype=np.float64) / sr
    sec_start = start_sample / sr
    sec_end = (start_sample + n) / sr
    nyq = 0.95 * sr / 2.0
    m_cap = min(14, max(1, int(nyq / max(f0, 1e-6))))
    lam0 = float(lam_hint) if lam_hint > 0.0 else 1.0
    decay_grid = tuple(sorted({0.0, 0.5 * lam0, lam0, 2.0 * lam0}))
    t_mid = sec_start + 0.5 * (sec_end - sec_start)
    best: Optional[_Variant] = None
    for t0 in (sec_start, t_mid):
        for b in (0.0, 1e-4, 1e-3):
            vari = _fit_candidate(
                target, t, t_fit, f0, b, t0, m_cap, sec_start, sec_end,
                stft_half_window_s, decay_grid, sr,
            )
            if best is None or vari.err < best.err:
                best = vari
    if best is None or not best.partials:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, f0, sec_start)
    locked = _vc_lock(
        best, target, t, t_fit, sec_start, sec_end, stft_half_window_s, 1e-3
    )
    return locked if locked.vc_lock else best


# ----------------------------------------------------------------------
# Class C: percussive bundle (Serra & Smith 1990 sines+transients insight)
# ----------------------------------------------------------------------

def detect_hits(
    resid: np.ndarray,
    sr: int,
    start_sample: int,
    k: int = 2,
    n_fft: int = 1024,
    hop: int = 256,
) -> List[int]:
    """Transients in a residual: spectral-flux onset peaks (Bello et al. 2005).

    Returns at most ``k`` hit positions (absolute sample offsets), at least
    60 ms apart, ordered weakest-first is *not* guaranteed - strongest first.
    """
    n = len(resid)
    if n < n_fft:
        return []
    frames = 1 + (n - n_fft) // hop
    if frames < 4:
        return []
    w = np.hanning(n_fft)
    env = np.zeros(frames - 1)
    prev = None
    for f in range(frames):
        seg = resid[f * hop:f * hop + n_fft]
        mag = np.abs(np.fft.rfft(seg * w))
        if prev is not None:
            env[f - 1] = float(np.sum(np.maximum(mag - prev, 0.0)))
        prev = mag
    if float(np.max(env)) <= 0:
        return []
    peak_idx = [
        i for i in range(1, env.size - 1)
        if env[i] > env[i - 1] and env[i] >= env[i + 1]
    ]
    if not peak_idx:
        return []
    peak_idx.sort(key=lambda i: -float(env[i]))
    # Adaptive threshold: hits must stand out from the flux floor.
    med = float(np.median(env))
    q90 = float(np.quantile(env, 0.9)) if env.size else med
    if q90 <= 0:
        return []
    thr = min(max(3.0 * med, 0.35 * q90), q90)
    step, wlen = 64, 128

    def onset_fine(sample: int) -> int:
        """Center the onset on the energy rise the flux peak came from.

        Frame-hop granularity puts the estimate a hop-window early; the
        energy-rise scan on a 64-sample step pulls it within a few ms.
        """
        lo = max(0, sample - n_fft // 4)
        hi = min(n - wlen, sample + n_fft // 2)
        if hi - lo < 4 * step:
            return sample
        e = np.array([
            float(np.sum(resid[s:s + wlen] ** 2))
            for s in range(lo, hi, step)
        ])
        if float(e.max()) <= 0.0:
            return sample
        rise = e[1:] - e[:-1]
        j = int(np.argmax(rise))
        return lo + step * (j + 1) + wlen // 2

    chosen: List[int] = []
    for i in peak_idx:
        if env[i] < thr:
            break
        sample = onset_fine((i + 1) * hop)
        if any(abs(sample - s) < 0.06 * sr for s in chosen):
            continue
        if len(chosen) >= k:
            break
        chosen.append(sample)
    return [start_sample + s for s in chosen]


def percussive_bundle(
    target: np.ndarray,
    sr: int,
    start_sample: int,
    stft_half_window_s: float,
    hit_sample: int,
    window_s: float = 0.18,
    band_lo: float = 45.0,
    max_modes: int = 6,
) -> _Variant:
    """One transient hit as a bundle of fast-decay damped modes.

    Physically a percussion hit IS a set of quickly-dying modes, so the class
    reuses the damped-cosine atom with short support [hit, hit+window] and a
    fast-decay grid - zero new render machinery, existing equation form.
    ``target`` is the residual the hit must explain.
    """
    n = target.size
    hit = int(hit_sample) - int(start_sample)
    hit_t = hit_sample / sr
    if hit < 0 or hit >= n:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, 0.0, hit_t)
    span = int(min(window_s * sr, max(n - hit, 0)))
    if span < 128:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, 0.0, hit_t)
    w = target[hit:hit + span]
    t_attack = max(int(0.06 * sr), 512)
    span_atk = min(t_attack, span)
    # Difference spectrum: sustained partials appear in both the attack and
    # the pre-hit window, so subtracting them exposes what actually *began*
    # at the hit (Serra & Smith's transient argument, in one line of DSP).
    atk = target[hit:hit + span_atk]
    pre_lo = max(0, hit - span_atk)
    pre = target[pre_lo:hit]
    L = atk.size
    atk_p = np.zeros(L)
    atk_p[:atk.size] = atk
    pre_p = np.zeros(L)
    if pre.size:
        pre_p[-min(pre.size, L):] = pre[-min(pre.size, L):]
    mw = np.maximum(
        _hann_mag(atk_p) - _hann_mag(pre_p), 0.0
    )
    bins = np.fft.rfftfreq(L, 1.0 / sr)
    lo_i = int(np.searchsorted(bins, band_lo))
    hi_i = int(np.searchsorted(bins, 0.98 * sr / 2.0))
    peaks: List[Tuple[int, float]] = []
    floor = float(np.median(mw[lo_i:hi_i]) + 3.0 * np.std(mw[lo_i:hi_i]))
    for i in range(max(lo_i + 1, 1), min(hi_i, mw.size - 1)):
        if mw[i] > mw[i - 1] and mw[i] >= mw[i + 1] and mw[i] > floor:
            peaks.append((i, float(mw[i])))
    if not peaks:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, 0.0, hit)
    peaks.sort(key=lambda p: -p[1])
    band_freqs: List[float] = []
    for i, _ in peaks:
        f = float(bins[i])
        if any(abs(f - g) < 30.0 for g in band_freqs):
            continue
        band_freqs.append(f)
        if len(band_freqs) >= max_modes:
            break
    if not band_freqs:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, 0.0, hit)

    decays = (30.0, 80.0, 200.0)
    hit_t = hit_sample / sr
    # Section coordinates: global t for mask/envelope, section-local t_fit
    # for the phase anchor - exactly the convention _render_partial uses.
    t_g = (np.arange(n, dtype=np.float64) + start_sample) / sr
    t_fit = np.arange(n, dtype=np.float64) / sr
    sup0, sup1 = hit, hit + span
    end_s = hit_t + span / sr
    w_full = np.zeros(n)
    w_full[sup0:sup1] = target[sup0:sup1]
    w_cur = w_full.copy()
    partials: List[Dict[str, Any]] = []
    for f in band_freqs:
        best: Optional[Tuple[float, float, float, float]] = None
        for lam in decays:
            c, s = _masked_proj_basis(
                t_g, t_fit, hit_t, f, float(lam), hit_t, end_s,
                stft_half_window_s,
            )
            base = float(np.sum(np.square(c)) + np.sum(np.square(s)))
            if base <= 1e-18:
                continue
            cc = float(np.dot(c, w_cur)) / base
            ss = float(np.dot(s, w_cur)) / base
            fit = cc * c + ss * s
            err = float(
                np.sum((w_cur[sup0:sup1] - fit[sup0:sup1]) ** 2)
            )
            if best is None or err < best[0]:
                best = (err, float(lam), cc, ss)
        if best is None:
            continue
        _, lam_best, cc, ss = best
        # Sequential deflation: each mode is fitted against what previous
        # modes left, so overlapping modes cannot double-account energy.
        c_best, s_best = _masked_proj_basis(
            t_g, t_fit, hit_t, f, lam_best, hit_t, end_s, stft_half_window_s
        )
        w_cur -= cc * c_best + ss * s_best
        a = float(np.hypot(cc, ss))
        ph = float(np.arctan2(ss, cc))
        if a <= 0.0:
            continue
        partials.append({
            "amplitude": round(a, 6),
            "envelope_decay": round(lam_best, 6),
            "envelope_t0": round(hit_t, 6),
            "frequency": round(f, 6),
            "phase": round(ph, 6),
            "model": "damped",
            "start_time": round(hit_t, 6),
            "end_time": round(end_s, 6),
            "m": None,
            "note_group": round(hit_t, 3),
        })
    if not partials:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, 0.0, hit_t)
    a_max = max(float(p["amplitude"]) for p in partials)
    keep = [p for p in partials if float(p["amplitude"]) >= max(1e-4, 0.05 * a_max)]
    if not keep:
        return _Variant([], float(np.sum(target ** 2)), 0, 0.0, 0.0, hit_t)
    # Measure against the FULL section grid through the real per-partial
    # support path, so the score is directly comparable with the state render
    # the acceptance gate will re-measure anyway.
    t_full = (np.arange(n, dtype=np.float64) + start_sample) / sr
    t_fit_full = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n)
    for p in keep:
        sig += _render_partial(
            p, n, sr, start_sample=start_sample,
            stft_half_window_s=stft_half_window_s,
        )
    err = float(np.sum((target - sig) ** 2))
    params = 1 + 3 * len(keep)
    return _Variant(keep, err, params, 0.0, 0.0, hit_t)


def filtered_noise_hypothesis(
    residual: np.ndarray,
    sr: int,
    start_sample: int,
    stft_half_window_s: float = 0.0,
    seed: int = 42,
    n_bands: int = 16,
) -> _Variant:
    """Class C++: stochastic noise channel shaped by residual spectral envelope."""
    n = residual.size
    t0 = float(start_sample) / sr
    e_res = float(np.sum(residual ** 2))
    if n < 128 or e_res < 1e-12:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    # 1. Compute spectral envelope across log-spaced bands
    nyq = 0.95 * sr / 2.0
    band_centers = np.geomspace(50.0, nyq, n_bands)
    r_fft = np.fft.rfft(residual)
    freq_bins = np.fft.rfftfreq(n, 1.0 / sr)
    mag = np.abs(r_fft)

    band_mags = np.zeros(n_bands)
    for b_idx, bc in enumerate(band_centers):
        lo = bc / 1.414
        hi = bc * 1.414
        in_band = (freq_bins >= lo) & (freq_bins <= hi)
        if np.any(in_band):
            band_mags[b_idx] = float(np.mean(mag[in_band]))
        else:
            band_mags[b_idx] = 1e-6

    smooth_mags = np.convolve(band_mags, np.ones(3) / 3.0, mode="same")
    smooth_mags = np.maximum(smooth_mags, 1e-6)

    # 2. Shape seeded white noise
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal(n)
    w_fft = np.fft.rfft(white)

    min_f = max(10.0, float(band_centers[0]))
    log_bc = np.log10(np.maximum(band_centers, min_f))
    log_bins = np.log10(np.maximum(freq_bins, min_f))
    env_bins = np.interp(log_bins, log_bc, smooth_mags, left=smooth_mags[0], right=smooth_mags[-1])

    shaped = np.fft.irfft(w_fft * env_bins, n=n)
    shaped_rms = float(np.sqrt(np.mean(shaped ** 2)))
    if shaped_rms > 1e-12:
        normed_shaped = shaped / shaped_rms
    else:
        normed_shaped = shaped

    # 3. Least squares fit gain to residual
    denom = float(np.dot(normed_shaped, normed_shaped))
    if denom > 1e-12:
        gain = float(np.clip(np.dot(residual, normed_shaped) / denom, 0.0, 5.0))
    else:
        gain = 0.0

    cand_signal = gain * normed_shaped
    cand_err = float(np.sum((residual - cand_signal) ** 2))

    if gain <= 1e-6 or cand_err >= e_res:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    p = {
        "model": "filtered_noise",
        "seed": int(seed),
        "amplitude": round(gain, 6),
        "band_centers": [round(float(c), 1) for c in band_centers],
        "band_mags": [round(float(m), 6) for m in smooth_mags],
        "envelope_decay": 0.0,
        "envelope_t0": round(t0, 6),
        "frequency": 0.0,
        "phase": 0.0,
        "start_time": round(t0, 6),
        "end_time": round(t0 + float(n) / sr, 6),
    }
    return _Variant([p], cand_err, 2 + n_bands, 0.0, 0.0, t0)


def hf_log_spectral_distance(
    y: np.ndarray, r: np.ndarray, sr: int, hf_hz: float = 4000.0,
    n_fft: int = 2048, hop_length: int = 512,
) -> float:
    """Log-spectral distance restricted to bins above ``hf_hz``.

    The time-domain high-frequency residual energy cannot serve as the spectral
    lane's acceptance test: a burst carries random phase, so adding it always
    *increases* measured high-frequency residual energy even while it improves
    the spectrum. Measured on a sibilant fixture, that energy rose from 9.7e6 to
    2.4e8 as the burst level rose, while log-spectral distance fell by 12.9 dB.
    Spectral fidelity has to be judged in the magnitude domain.
    """
    from src.residual import compute_error_metrics

    if len(y) < n_fft:
        return float("inf")
    full = compute_error_metrics(y, r, sr=sr, n_fft=n_fft,
                                 hop_length=hop_length)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    sel = freqs >= hf_hz
    if sel.sum() < 2:
        return float("inf")
    win = np.hanning(n_fft)
    total, count = 0.0, 0
    for s in range(0, len(y) - n_fft + 1, hop_length):
        a = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(y[s:s + n_fft] * win)), 1e-10))
        b = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(r[s:s + n_fft] * win)), 1e-10))
        total += float(np.mean(np.abs(a[sel] - b[sel])))
        count += 1
    return total / max(1, count)


def _hf_residual_energy(x: np.ndarray, sr: int, hf_hz: float = 4000.0) -> float:
    """Energy of ``x`` above ``hf_hz``, used by the spectral acceptance lane."""
    if len(x) < 16:
        return 0.0
    mag = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(np.sum(mag[freqs >= hf_hz]))


def sibilance_burst_hypothesis(
    residual: np.ndarray,
    sr: int,
    start_sample: int,
    stft_half_window_s: float = 0.0,
    seed: int = 20240,
    hf_hz: float = 3000.0,
    hf_ratio_min: float = 0.03,
    n_bands: int = 10,
    corr_floor: float = 0.60,
) -> _Variant:
    """Class D: sibilance / cymbal noise burst, gated to genuinely noisy highs.

    Pure sinusoidal modelling has no stochastic component, which is what makes a
    reconstruction read as low-bitrate: sibilance is noise, so it cannot be
    expressed by any sum of sines. This proposes a seeded noise burst shaped by
    the residual's own high-frequency envelope.

    The gate is deliberately just the high-frequency energy ratio. Spectral
    flatness was tried first and rejected on measurement: a fricative is
    band-limited, so whole-spectrum SFM pins it near 0 (its lower bins are
    empty), and band-restricted SFM is no better - measured 0.04 for a realistic
    4-8 kHz fricative versus 0.00 for a pure tone, versus 0.56 only for full-band
    white noise. Spectral flatness detects broadband hiss, not sibilance.
    Crest factor and energy participation ratio were also measured and overlap
    between tonal and noise-like cases. The high-frequency energy ratio does
    separate cleanly (tonal 0.00, fricative 1.00), so it proposes the burst and
    the spectral acceptance lane then verifies on the real mixed render whether
    the burst actually lowers log-spectral distance and high-frequency residual.
    A burst that does not help tonal content is reverted rather than gated out
    by a proxy statistic.
    """
    n = residual.size
    t0 = float(start_sample) / sr
    e_res = float(np.sum(residual ** 2))
    if n < 128 or e_res < 1e-12:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    sfm = band_spectral_flatness(residual, sr, hf_hz)
    r_mag = np.abs(np.fft.rfft(residual))
    freq_bins = np.fft.rfftfreq(n, 1.0 / sr)
    total = float(np.sum(r_mag ** 2))
    hf_ratio = float(np.sum(r_mag[freq_bins >= hf_hz] ** 2) / (total + 1e-12))
    if hf_ratio <= hf_ratio_min:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    nyq = 0.98 * sr / 2.0
    lo = max(hf_hz * 1.05, 100.0)
    if lo >= nyq:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)
    band_centers = np.geomspace(lo, nyq, n_bands)
    band_mags = np.zeros(n_bands)
    for b_idx, bc in enumerate(band_centers):
        in_band = (freq_bins >= bc / 1.414) & (freq_bins <= bc * 1.414)
        band_mags[b_idx] = float(np.mean(r_mag[in_band])) if np.any(in_band) else 1e-6
    smooth = np.maximum(np.convolve(band_mags, np.ones(3) / 3.0, mode="same"), 1e-6)

    rng = np.random.default_rng(int(seed))
    w_fft = np.fft.rfft(rng.standard_normal(n))
    min_f = max(10.0, float(band_centers[0]))
    env = np.interp(
        np.log10(np.maximum(freq_bins, min_f)),
        np.log10(np.maximum(band_centers, min_f)),
        smooth,
        left=smooth[0],
        right=smooth[-1],
    )
    # The interpolation holds the first band's magnitude flat across everything
    # below it, which would fill a "sibilance" burst with low-frequency energy
    # and make it fail its own fit. Below the band the burst must be silent.
    env = np.where(freq_bins >= band_centers[0], env, 0.0)
    shaped = np.fft.irfft(w_fft * env, n=n)
    rms = float(np.sqrt(np.mean(shaped ** 2)))
    if rms <= 1e-12:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)
    unit = shaped / rms

    # No least-squares gain fit here. The burst carries seeded random phase, so
    # projecting it onto the residual is a correlation between two unrelated
    # noise realisations - near zero and often negative - which would clip the
    # gain to zero and reject every candidate. Amplitude 1.0 puts the burst at
    # the level the envelope implies; the acceptance lane then verifies on the
    # real render that this level genuinely lowers LSD and high-frequency
    # residual, and reverts it when it does not.
    gain = 1.0
    cand = gain * unit
    cand_err = float(np.sum((residual - cand) ** 2))

    p = {
        "model": "filtered_noise",
        "seed": int(seed),
        "amplitude": round(gain, 6),
        "band_centers": [round(float(c), 1) for c in band_centers],
        "band_mags": [round(float(m), 6) for m in smooth],
        "envelope_decay": 0.0,
        "envelope_t0": round(t0, 6),
        "frequency": 0.0,
        "phase": 0.0,
        "start_time": round(t0, 6),
        "end_time": round(t0 + float(n) / sr, 6),
        "sibilance": True,
    }
    return _Variant([p], cand_err, 2 + n_bands, sfm, hf_ratio, t0)


def lpc_noise_hypothesis(
    residual: np.ndarray,
    sr: int,
    start_sample: int,
    stft_half_window_s: float = 0.0,
    order: int = 12,
    seed: int = 42,
    bandwidth_expansion: float = 0.98,
) -> _Variant:
    """Class E: LPC Formant Noise (vocal tract resonance & rap vocal texture)."""
    n = residual.size
    t0 = float(start_sample) / sr
    e_res = float(np.sum(residual ** 2))
    if n < 32 or e_res < 1e-12:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    p_order = min(order, n // 4)
    r = np.correlate(residual, residual, mode="full")
    r_pos = r[n - 1 : n - 1 + p_order + 1]
    if r_pos[0] <= 1e-12:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    try:
        a = scipy.linalg.solve_toeplitz(r_pos[:p_order], r_pos[1:])
    except Exception:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    gamma = float(bandwidth_expansion)
    damp = gamma ** np.arange(1, len(a) + 1)
    a_damped = a * damp

    a_den = np.concatenate([[1.0], -a_damped])

    best_candidate = None
    candidate_seeds = [int(seed)] + [int(seed) + offset for offset in range(1, 8)]
    for s in candidate_seeds:
        rng = np.random.default_rng(s)
        noise = rng.standard_normal(n)
        synth_unscaled = scipy.signal.lfilter([1.0], a_den, noise)
        sig_rms = float(np.sqrt(np.mean(synth_unscaled ** 2)))
        if sig_rms <= 1e-12:
            continue
        normed = synth_unscaled / sig_rms
        denom = float(np.sum(normed ** 2))
        dot = float(np.dot(residual, normed))
        if dot <= 1e-6:
            continue
        gain = float(dot / denom)
        synth = gain * normed
        cand_err = float(np.sum((residual - synth) ** 2))
        if cand_err < e_res and gain > 1e-6:
            best_candidate = (s, gain, cand_err)
            break

    if best_candidate is None:
        return _Variant([], e_res, 0, 0.0, 0.0, t0)

    best_seed, gain, cand_err = best_candidate
    p = {
        "model": "lpc_noise",
        "amplitude": round(float(gain), 6),
        "lpc_order": int(p_order),
        "lpc_coeffs": [round(float(c), 6) for c in a_damped],
        "seed": int(best_seed),
        "envelope_decay": 0.0,
        "envelope_t0": round(t0, 6),
        "frequency": 0.0,
        "phase": 0.0,
        "start_time": round(t0, 6),
        "end_time": round(t0 + float(n) / sr, 6),
    }
    return _Variant([p], cand_err, 2 + p_order, 0.0, 0.0, t0)


def _dedupe(hyp_partials: List[Dict[str, Any]],
            active_freqs: Sequence[float]) -> List[Dict[str, Any]]:
    """Drop hypothesis partials within a few Hz of an active state partial.

    Neighbors with active support already describe their band; duplicates are
    the near-collinear columns that made earlier coordinate updates unstable.
    """
    return [
        p for p in hyp_partials
        if min(
            (abs(float(p["frequency"]) - fa) for fa in active_freqs),
            default=1e9,
        ) > 1.5
    ]



def _spectral_corr_floor(
    corr_floor: float,
    corr_floor_mode: str,
    cur_corr: float,
    corr_max_drop: float,
) -> float:
    """Resolve the effective correlation floor for the spectral lane.

    'absolute' is the shipped behaviour: a fixed bar every acceptance must
    clear, which deliberately keeps the lane inert on stems that already sit
    below it.

    'relative' is a pure degradation budget, ``cur_corr - corr_max_drop``.

    Note the form that is NOT correct here: ``max(corr_floor, cur_corr - drop)``
    reduces to the absolute floor for every ``cur_corr <= corr_floor + drop``,
    so it never enables the lane on a low-correlation stem while looking like
    it does. Relative mode must therefore use ``cur_corr - drop`` on its own.
    """
    if corr_floor_mode == "relative":
        return float(cur_corr) - float(corr_max_drop)
    return float(corr_floor)


def _accept_spectral(
    y_sec: np.ndarray,
    partials: List[Dict[str, Any]],
    fresh: List[Dict[str, Any]],
    cur_err: float,
    cur_corr: float,
    render_state: Any,
sr: int,
    gain_ladder: Optional[Sequence[float]],
    corr_floor: float = 0.60,
    corr_floor_mode: str = "absolute",
    corr_max_drop: float = 0.05,
) -> Tuple[bool, float, float]:
    """Spectral acceptance lane for stochastic hypotheses.

    Two time-domain gates make every noise class unacceptable by construction,
    so both are replaced here:
      * the energy-margin precondition, because adding noise raises squared
        time-domain error by definition even while it lowers spectral error;
      * the correlation gate, because waveform correlation demands sample-exact
        alignment, which random-phase noise can never achieve against real
        noise.

Acceptance is instead judged primarily in the magnitude domain: overall
    log-spectral distance and high-band log-spectral distance must both strictly
    improve on the actual mixed render. Burst level is self-tuned over a gain
    ladder, because the level that helps is content-dependent - measured LSD was
    best at a quarter of unit amplitude while double amplitude was worse than no
    burst at all.

    ``corr_floor`` is the safety limit on the price paid. Waveform correlation is
    scale- and phase-sensitive, so a stochastic burst always costs some of it;
    the floor rejects any candidate whose acceptance would push the stem below
    the limit, regardless of how much LSD it bought. The original partial list
    is restored on every failure.
    """
    from src.residual import compute_error_metrics

    if not fresh:
        return False, cur_err, cur_corr
    keep = list(partials)
    render_prev = render_state()
    lsd_before = compute_error_metrics(
        y_sec, render_prev, sr=sr)["log_spectral_distance_db"]
    hf_before = hf_log_spectral_distance(y_sec, render_prev, sr)
    floor_now = _spectral_corr_floor(corr_floor, corr_floor_mode, cur_corr,
                                    corr_max_drop)

    best = None
    for g in (gain_ladder or (0.15, 0.25, 0.4, 0.6, 1.0)):
        partials[:] = keep
        trial = [dict(p) for p in fresh]
        for p in trial:
            p["amplitude"] = g
        partials.extend(trial)
        cand = render_state()
        lsd_c = compute_error_metrics(y_sec, cand, sr=sr)["log_spectral_distance_db"]
        hf_c = hf_log_spectral_distance(y_sec, cand, sr)
        if lsd_c < lsd_before - 1e-9 and hf_c < hf_before - 1e-9:
            c_c = _corr(y_sec, cand)
            if c_c < floor_now:
                continue
            score = lsd_c + hf_c
            if best is None or score < best[0]:
                best = (score, lsd_c, g,
                        float(np.sum((y_sec - cand) ** 2)), c_c)

    if best is None:
        partials[:] = keep
        return False, cur_err, cur_corr

    _score, _lsd, g, err_after, corr_after = best
    partials[:] = keep
    trial = [dict(p) for p in fresh]
    for p in trial:
        p["amplitude"] = g
    partials.extend(trial)
    return True, err_after, corr_after


def _accept_candidate(
    y_sec: np.ndarray,
    partials: List[Dict[str, Any]],
    fresh: List[Dict[str, Any]],
    param_count: int,
    vari_err: float,
    e_y: float,
    cur_err: float,
    cur_corr: float,
    render_state: Any,
    sr: Optional[int] = None,
spectral: bool = False,
    gain_ladder: Optional[Sequence[float]] = None,
    corr_floor: Optional[float] = None,
    corr_floor_mode: str = "absolute",
    corr_max_drop: float = 0.05,
) -> Tuple[bool, float, float]:
    """Dual gate + parity re-measure shared by every hypothesis class.

    Energy margin first (statistically beats noise-chasing), then the corr
    gate (the referee's own metric strictly improves), both judged on the
    actual mixed render the acceptance would produce. Restores the previous
    state when a candidate fails.

    ``spectral=True`` routes to :func:`_accept_spectral`, which replaces both
    time-domain gates with magnitude-domain spectral tests.
    """
    if spectral and sr:
        return _accept_spectral(
            y_sec, partials, fresh, cur_err, cur_corr,
            render_state, sr, gain_ladder,
            0.60 if corr_floor is None else float(corr_floor),
            corr_floor_mode, corr_max_drop,
        )

    if not fresh or not (
        vari_err < cur_err - acceptance_margin(e_y, cur_err, param_count,
                                               len(y_sec))
    ):
        return False, cur_err, cur_corr
    keep = list(partials)
    render_prev = render_state()
    partials.extend(fresh)
    render = render_state()
    resid_after = y_sec - render
    err_after = float(np.sum(resid_after ** 2))
    corr_after = _corr(y_sec, y_sec - resid_after)
    if err_after < cur_err and corr_after > cur_corr + 1e-5:
        return True, err_after, corr_after
    partials[:] = keep
    return False, cur_err, cur_corr


def tournament_section(
    y_sec: np.ndarray,
    partials: List[Dict[str, Any]],
    sr: int,
    start_sample: int = 0,
    stft_half_window_s: float = 0.0,
    k: int = 2,
    k_hits: int = 2,
    enable_noise: bool = False,
    enable_spectral_lane: bool = False,
    gain_ladder: Optional[Sequence[float]] = None,
    corr_floor: float = 0.60,
    corr_floor_mode: str = "absolute",
    corr_max_drop: float = 0.05,
) -> List[Dict[str, Any]]:
    """Run the hypothesis-bank tournament against the current state.

    Class B (harmonic_string whole notes) first, then Class C (percussive
    bundles) on whatever music the notes could not explain. Appends accepted
    hypothesis partials in place and returns the candidate records (accepted
    first, most error-reducing first). Glide partials are left untouched; the
    class A lifter owns them.
    """
    n = len(y_sec)
    records: List[Dict[str, Any]] = []
    if n < 256 or not partials:
        return records

    def state() -> np.ndarray:
        r = np.zeros(n)
        for q in partials:
            r += _render_partial(
                q, n, sr, start_sample=start_sample,
                stft_half_window_s=stft_half_window_s,
            )
        return r

    e_y = float(np.sum(y_sec ** 2))
    resid = y_sec - state()
    cur_err = float(np.sum(resid ** 2))
    cur_corr = _corr(y_sec, y_sec - resid)
    if cur_err <= _E_REL_MARGIN * e_y + 1e-12:
        return records

    lam_hints = [
        float(q.get("envelope_decay", 0.0))
        for q in partials
        if q.get("model") != "glide" and float(q.get("amplitude", 0.0)) > 0.0
    ]
    lam0_hint = float(np.median(lam_hints)) if lam_hints else 1.0

    active_freqs = [
        float(q.get("frequency", 0.0))
        for q in partials
        if float(q.get("amplitude", 0.0)) > 0.0
    ]

    def record_empty(
        cls: str, f0: float, before: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rec: Dict[str, Any] = {
            "class": cls, "f0": round(f0, 4),
            "m_count": 0, "param_count": 0, "vc_lock": False,
            "sigma_a": None, "sigma_b": None, "t0": None,
            "residual_before": round(before, 3),
            "residual_after": round(cur_err, 3), "accepted": False,
        }
        if extra:
            rec.update(extra)
        return rec

    # ---- Class B: harmonic_string whole notes ----
    f0s = f0_candidates(resid, sr, k=k)
    for f0 in f0s:
        before = cur_err
        vari = harmonic_string_hypothesis(
            resid, sr, start_sample, stft_half_window_s, f0, lam_hint=lam0_hint
        )
        fresh = _dedupe(vari.partials, active_freqs)
        if not fresh:
            records.append(record_empty("harmonic_string", f0, before,
                extra={"m_count": len(vari.partials),
                       "param_count": vari.param_count}))
            continue
        if len(fresh) < len(vari.partials):
            # Margin integrity: dropped (duplicate) partials contribute no
            # fitted parameters to the state, so do not charge for them.
            vari.param_count = 2 + 3 * len(fresh)
        ok, err_after, corr_after = _accept_candidate(
            y_sec, partials, fresh, vari.param_count, vari.err,
            e_y, cur_err, cur_corr, state, sr=sr,
        )
        if ok:
            cur_err, cur_corr = err_after, corr_after
        records.append(vari.as_record(len(fresh), ok, before))

    # ---- Class C: percussive bundles on what notes could not explain ----
    resid = y_sec - state()
    cur_err = float(np.sum(resid ** 2))
    cur_corr = _corr(y_sec, y_sec - resid)
    hits = detect_hits(resid, sr, start_sample, k=k_hits)
    for hit_sample in hits:
        before = cur_err
        vari = percussive_bundle(
            resid, sr, start_sample, stft_half_window_s, hit_sample
        )
        fresh = _dedupe(vari.partials, active_freqs)
        if not fresh:
            records.append(record_empty(
                "percussive_bundle", float(hit_sample) / sr, before,
                extra={"t0": round(float(hit_sample) / sr, 6),
                       "m_count": len(vari.partials),
                       "param_count": vari.param_count},
            ))
            continue
        ok, err_after, corr_after = _accept_candidate(
            y_sec, partials, fresh, vari.param_count, vari.err,
            e_y, cur_err, cur_corr, state, sr=sr,
        )
        if ok:
            cur_err, cur_corr = err_after, corr_after
        rec: Dict[str, Any] = {
            "class": "percussive_bundle", "f0": None,
            "t0": round(float(hit_sample) / sr, 6),
            "m_count": len(fresh), "param_count": vari.param_count,
            "residual_before": round(before, 3),
            "residual_after": round(cur_err, 3), "accepted": ok,
        }
        records.append(rec)

    # ---- Optional Class C++: filtered noise (only when enable_noise is explicitly requested) ----
    if enable_noise:
        resid = y_sec - state()
        cur_err = float(np.sum(resid ** 2))
        cur_corr = _corr(y_sec, y_sec - resid)
        vari_noise = filtered_noise_hypothesis(resid, sr, start_sample, stft_half_window_s)
        if vari_noise.partials:
            before = cur_err
            fresh = vari_noise.partials
            ok, err_after, corr_after = _accept_candidate(
                y_sec, partials, fresh, vari_noise.param_count, vari_noise.err,
                e_y, cur_err, cur_corr, state, sr=sr,
            )
            if ok:
                cur_err, cur_corr = err_after, corr_after
            records.append({
                "class": "filtered_noise", "f0": None,
                "t0": round(float(start_sample) / sr, 6),
                "m_count": len(fresh), "param_count": vari_noise.param_count,
                "residual_before": round(before, 3),
                "residual_after": round(cur_err, 3), "accepted": ok,
            })

    # ---- Class D: sibilance / cymbal bursts on the spectral lane (opt-in) ----
    if enable_spectral_lane:
        resid = y_sec - state()
        cur_err = float(np.sum(resid ** 2))
        cur_corr = _corr(y_sec, y_sec - resid)
        vari_sib = sibilance_burst_hypothesis(
            resid, sr, start_sample, stft_half_window_s
        )
        if vari_sib.partials:
            before = cur_err
            fresh_sib = vari_sib.partials
            ok, err_after, corr_after = _accept_candidate(
                y_sec, partials, fresh_sib, vari_sib.param_count, vari_sib.err,
                e_y, cur_err, cur_corr, state, sr=sr, spectral=True,
                gain_ladder=gain_ladder, corr_floor=corr_floor,
                corr_floor_mode=corr_floor_mode, corr_max_drop=corr_max_drop,
            )
            if ok:
                cur_err, cur_corr = err_after, corr_after
            records.append({
                "class": "sibilance_burst", "f0": None,
                "t0": round(float(start_sample) / sr, 6),
                "m_count": len(fresh_sib), "param_count": vari_sib.param_count,
                "hf_band_flatness": round(float(vari_sib.B), 6),
                "hf_ratio": round(float(vari_sib.f0), 6),
                "residual_before": round(before, 3),
                "residual_after": round(cur_err, 3), "accepted": ok,
            })

    records.sort(key=lambda r: (not r["accepted"], r["residual_after"]))
    return records
