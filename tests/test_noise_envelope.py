"""P1.1 first increment: time-varying noise envelopes.

The measured failure: every stochastic model carried a STATIC whole-section
envelope, so a drum fill or breath burst was rendered as stationary hiss at
its time-average level - correct spectrum, near-zero envelope match (the
error budget measured >=98% of 2-11 kHz unexplained on drums, and the
weighted dB error concentrates exactly there). This pins the first
increment of the fix: the filtered_noise render honours a piecewise-linear
amplitude trajectory, and the noise-floor fitter measures one from the
residual instead of assuming stationarity.

Seams under test (public interfaces):
- fit_noise_floor_partial: a transient residual yields a partial carrying
  trajectory{t, a} with the level shape of the residual; a flat residual
  stays static (backward compatible).
- _render_partial (filtered_noise): the render's level follows the
  trajectory in time.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.res_synthesis import _render_partial, fit_noise_floor_partial

SR = 22050


def _burst_residual(n: int) -> np.ndarray:
    """Noise gated to the second half (smooth edges) - the fill/breath shape."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(n)
    gate = np.zeros(n)
    gate[n // 2 :] = 1.0
    # 5 ms raised-cosine edges so the gate itself is band-limited-ish
    edge = int(0.005 * SR)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(edge) / edge)
    gate[n // 2 : n // 2 + edge] = ramp
    gate[n - edge :] = ramp[::-1]
    return 0.5 * x * gate


def test_fit_measures_an_envelope_for_a_transient_residual() -> None:
    n = SR
    x = _burst_residual(n)
    p = fit_noise_floor_partial(x, SR, start_time=0.0, end_time=1.0, seed=3)
    assert p is not None
    # A full-band burst produces per-band envelopes (which supersede the
    # global one) or, for a pre-band-trajectory engine, a global envelope.
    # Either way the level shape must be present and must track the burst.
    traj = p.get("trajectory")
    if traj:
        tt = np.asarray(traj["t"], dtype=np.float64)
        aa = np.asarray(traj["a"], dtype=np.float64)
    else:
        bt = p.get("band_traj")
        assert bt, "a transient residual must produce an envelope model"
        tt = np.asarray(bt["t"], dtype=np.float64)
        aa = np.mean(np.asarray(bt["a"], dtype=np.float64), axis=0)
    assert tt.size == aa.size and tt.size >= 4
    late = float(np.mean(aa[tt >= 0.6]))
    early = float(np.mean(aa[tt <= 0.4])) + 1e-9
    assert late / early > 3.0, "the envelope must track the burst timing"


def test_fit_stays_static_for_a_flat_residual() -> None:
    rng = np.random.default_rng(5)
    x = 0.3 * rng.standard_normal(SR)
    p = fit_noise_floor_partial(x, SR, start_time=0.0, end_time=1.0, seed=3)
    assert p is not None
    assert not p.get("trajectory"), (
        "a stationary residual must not grow an envelope (backward compat)"
    )


def test_render_follows_the_noise_envelope() -> None:
    """Same noise partial, two opposite trajectories: the level order in
    time must follow the trajectory, not the RNG draw."""
    base = {
        "model": "filtered_noise",
        "amplitude": 0.4,
        "seed": 1234,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "phase": 0.0,
        "frequency": 0.0,
    }
    n = SR
    tt = [0.0, 0.25, 0.75, 1.0]
    down = dict(base, trajectory={"t": tt, "a": [1.0, 1.0, 0.05, 0.05]})
    up = dict(base, trajectory={"t": tt, "a": [0.05, 0.05, 1.0, 1.0]})
    sig_down = _render_partial(down, n, SR)
    sig_up = _render_partial(up, n, SR)

    def half_rms(sig):
        e = int(0.45 * SR)
        return (
            float(np.sqrt(np.mean(sig[:e] ** 2))),
            float(np.sqrt(np.mean(sig[-e:] ** 2))),
        )

    d1, d2 = half_rms(sig_down)
    u1, u2 = half_rms(sig_up)
    assert d1 > 5.0 * d2, "decaying envelope must decay the render"
    assert u2 > 5.0 * u1, "rising envelope must rise in the render"


def test_render_without_trajectory_is_unchanged() -> None:
    """Backward compat: no trajectory key -> the shipped render exactly."""
    base = {
        "model": "filtered_noise",
        "amplitude": 0.4,
        "seed": 1234,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "phase": 0.0,
        "frequency": 0.0,
    }
    n = SR
    sig = _render_partial(base, n, SR)
    # The shipped render includes the support mask's edge ramps (mask mean
    # 0.99995), so parity holds to that tolerance, not to machine epsilon.
    assert float(np.sqrt(np.mean(sig ** 2))) == pytest.approx(0.4, rel=1e-3)


def test_enveloped_render_still_meets_its_global_rms() -> None:
    """The envelope redistributes level in time; the global RMS is the
    partial's amplitude (within the smoothing tolerance)."""
    base = {
        "model": "filtered_noise",
        "amplitude": 0.4,
        "seed": 1234,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "phase": 0.0,
        "frequency": 0.0,
    }
    tt = np.linspace(0.0, 1.0, 9)
    rng = np.random.default_rng(2)
    aa = rng.uniform(0.2, 1.0, size=tt.size)
    p = dict(base, trajectory={"t": list(tt), "a": [float(v) for v in aa]})
    sig = _render_partial(p, SR, SR)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    mean_a = float(np.mean(aa ** 2)) ** 0.5
    assert rms == pytest.approx(0.4 * mean_a, rel=0.25)


# --- per-band envelopes: NoiseBandNet in closed form ----------------------

def _band_noise(n: int, lo: float, hi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    return np.fft.irfft(X * ((f >= lo) & (f < hi)), n=n)


def test_fit_attaches_per_band_envelopes_for_band_transient() -> None:
    """A residual whose HIGH band bursts on late (cymbal pattern) while the
    LOW band stays steady must yield per-band envelopes: the high row is
    uneven, the low row flat, and band_edges describes the split."""
    n = SR
    gate = np.zeros(n)
    gate[n // 2:] = 1.0
    x = 0.5 * _band_noise(n, 40.0, 500.0, 1) + 0.8 * _band_noise(n, 2000.0, 9000.0, 2) * gate
    p = fit_noise_floor_partial(x, SR, start_time=0.0, end_time=1.0, seed=3)
    assert p is not None
    bt = p.get("band_traj")
    assert bt, "a band-transient residual must produce per-band envelopes"
    edges = p.get("band_edges")
    assert edges and len(edges) >= 4
    aa = np.asarray(bt["a"], dtype=np.float64)
    tt = np.asarray(bt["t"], dtype=np.float64)
    assert aa.ndim == 2 and aa.shape[1] == tt.size
    # find the band containing 5 kHz and check it tracks the late burst
    hi_row = None
    for i in range(aa.shape[0]):
        if edges[i] <= 5000.0 < edges[i + 1]:
            hi_row = aa[i]
    assert hi_row is not None
    late = float(np.mean(hi_row[tt >= 0.6]))
    early = float(np.mean(hi_row[tt <= 0.4])) + 1e-9
    assert late / early > 3.0, "the 5 kHz band envelope must track the burst"


def test_render_applies_per_band_envelopes_independently() -> None:
    """A burst envelope on the high band only must raise the render's
    late/early energy ratio in the high band while leaving the low band's
    ratio untouched."""
    base = {
        "model": "filtered_noise",
        "amplitude": 0.4,
        "seed": 1234,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "phase": 0.0,
        "frequency": 0.0,
        "band_centers": [141.0, 1000.0, 5000.0],
        "band_mags": [1.0, 1.0, 1.0],
    }
    n = SR
    edges = [max(10.0, base["band_centers"][0] / 2), 500.0, 2500.0, 10000.0]
    tt = [0.125, 0.375, 0.625, 0.875]
    flat = dict(base, band_edges=edges,
                band_traj={"t": tt, "a": [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]})
    burst = dict(base, band_edges=edges,
                 band_traj={"t": tt, "a": [[1, 1, 1, 1], [1, 1, 1, 1], [0.05, 0.05, 1.0, 1.0]]})

    def band_ratio(sig, lo, hi):
        X = np.fft.rfft(sig)
        f = np.fft.rfftfreq(n, 1.0 / SR)
        m = (f >= lo) & (f < hi)
        sb = np.fft.irfft(X * m, n=n)
        e = int(0.45 * n)
        return (float(np.sum(sb[e:] ** 2)) + 1e-12) / (float(np.sum(sb[:e] ** 2)) + 1e-12)

    flat_sig = _render_partial(flat, n, SR)
    burst_sig = _render_partial(burst, n, SR)
    hi_flat = band_ratio(flat_sig, 4000.0, 10000.0)
    hi_burst = band_ratio(burst_sig, 4000.0, 10000.0)
    lo_flat = band_ratio(flat_sig, 50.0, 400.0)
    lo_burst = band_ratio(burst_sig, 50.0, 400.0)
    assert hi_burst > 4.0 * hi_flat, "the high band must burst late"
    assert abs(lo_burst / lo_flat - 1.0) < 0.5, "the low band must not move"


def test_noise_floor_lane_allows_a_spectral_trade(tmp_path=None) -> None:
    """With the spectral lane on, the noise floor may trade a little
    waveform corr for a large spectral gain (stochastic content is judged
    spectrally). Without the lane the shipped conservative gate holds."""
    from src.res_loop import _apply_noise_floor

    n = SR
    t = np.arange(n) / SR
    rng = np.random.default_rng(8)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * rng.standard_normal(n)
    recon = 0.55 * np.sin(2 * np.pi * 440.0 * t)  # corr ~ 0.70 vs y
    residual = y - recon
    noise = fit_noise_floor_partial(
        residual, SR, start_time=0.0, end_time=1.0, seed=5
    )
    assert noise is not None
    rejected = _apply_noise_floor(
        y, recon, noise, SR, w_start=0, st_hw=0.0, lane=False
    )
    accepted = _apply_noise_floor(
        y, recon, noise, SR, w_start=0, st_hw=0.0, lane=True
    )
    assert accepted is not None, "the lane must let the floor trade corr for LSD"
    assert accepted[1]["log_spectral_distance_db"] < rejected_base_lsd(recon, y, SR)
    if rejected is not None:
        assert accepted[1]["log_spectral_distance_db"] <= rejected[1]["log_spectral_distance_db"] + 1e-9


def rejected_base_lsd(recon, y, sr):
    from src.res_synthesis import _score

    return _score(y, recon, sr)["log_spectral_distance_db"]


# --- noise-layer matching pursuit (the chase) -----------------------------

def test_chase_stacks_noise_layers_and_closes_the_spectral_gap() -> None:
    """A dense mastered mix leaves band-texture in the residual that one
    floor cannot shape. The chase stacks band-envelope noise layers, each
    fitted from the REMAINING residual - matching pursuit in the
    band-envelope domain. LSD is phase-blind, so this can go where
    waveform corr cannot: each layer must reduce the spectral distance."""
    from src.res_loop import _chase_noise_layers
    from src.res_synthesis import _score

    n = SR
    t = np.arange(n) / SR
    gate = np.zeros(n)
    gate[n // 2:] = 1.0
    y = (
        0.5 * np.sin(2 * np.pi * 440.0 * t)
        + 0.45 * _band_noise(n, 40.0, 800.0, 21)
        + 0.4 * _band_noise(n, 3000.0, 10000.0, 22) * gate
    )
    recon = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    base_lsd = _score(y, recon, SR)["log_spectral_distance_db"]

    added, final_recon = _chase_noise_layers(
        y, recon, SR, w_start=0, st_hw=0.0, lane=True, rounds=2,
        start_time=0.0, end_time=1.0,
    )
    assert len(added) >= 2, "the chase must stack layers while they help"
    final_lsd = _score(y, final_recon, SR)["log_spectral_distance_db"]
    assert final_lsd <= base_lsd - 3.0, (
        f"chase must close the spectral gap: {base_lsd:.2f} -> {final_lsd:.2f}"
    )


def test_chase_without_lane_adds_nothing() -> None:
    """The chase is opt-in (published numbers must not move)."""
    from src.res_loop import _chase_noise_layers

    n = SR
    t = np.arange(n) / SR
    rng = np.random.default_rng(31)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.5 * rng.standard_normal(n)
    recon = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    added, recon2 = _chase_noise_layers(
        y, recon, SR, w_start=0, st_hw=0.0, lane=False, rounds=2,
        start_time=0.0, end_time=1.0,
    )
    assert added == []
    assert np.array_equal(recon2, recon)


# --- the ear checks: no clicks at boundaries, no hiss in gaps -------------

def test_band_envelope_gaps_stay_silent() -> None:
    """A band whose envelope switches off must render SILENCE in the gap -
    not a hiss floor 3x the source's. Measured on a real render, the gaps
    hissed at 3.4x the source level, which is what 'sounds the same'
    means to an ear regardless of what LSD says."""
    base = {
        "model": "filtered_noise",
        "amplitude": 0.4,
        "seed": 1234,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "phase": 0.0,
        "frequency": 0.0,
        "band_centers": [141.0, 1000.0, 5000.0],
        "band_mags": [1.0, 1.0, 1.0],
        "band_edges": [10.0, 500.0, 2500.0, 10000.0],
        "band_traj": {"t": [0.125, 0.375, 0.625, 0.875],
                      "a": [[1, 1, 0.04, 0.04],
                            [1, 1, 0.04, 0.04],
                            [1, 1, 0.04, 0.04]]},
    }
    n = SR
    sig = _render_partial(base, n, SR)
    # The interp transitions between segment centers span ~125 ms; the
    # flat gated span [0.7, 1.0] s is where silence is required.
    e = int(0.7 * n)
    late = float(np.sqrt(np.mean(sig[e:] ** 2)))
    early = float(np.sqrt(np.mean(sig[:e] ** 2)))
    assert late < 0.05 * early, (
        f"a gated band must be silent, got late/early rms {late/early:.3f}"
    )


def test_render_ledger_ola_kills_boundary_clicks() -> None:
    """Sections rephase independently, so the raw paste clicks (measured:
    a 13.8x sample jump at a real section boundary). OLA plus the seam
    repair must shrink that discontinuity to inaudibility."""
    from src.res_loop import render_ledger, _repair_seams

    sr = SR
    n = sr
    half = n // 2
    mk = lambda f, ph: {"frequency": f, "amplitude": 0.4, "phase": ph,
                        "envelope_decay": 0.0, "envelope_t0": 0.0,
                        "start_time": 0.0, "end_time": 1.0,
                        "model": "damped_cosine"}
    ledger = {
        "schema": "res/1.1", "sample_rate": sr, "n_samples": n,
        "sections": [
            {"start": 0.0, "end": 0.5, "start_sample": 0,
             "partials": [mk(220.0, 0.0)]},
            {"start": 0.5, "end": 1.0, "start_sample": half,
             "partials": [mk(220.0, 1.7)]},  # deliberately out of phase
        ],
    }
    raw = render_ledger(ledger, n, ola=False)
    smooth = render_ledger(ledger, n, ola=True)
    _repair_seams(smooth, [half], sr=sr)
    d_raw = np.abs(np.diff(raw))
    d_smooth = np.abs(np.diff(smooth))
    seam = slice(half - 400, half + 400)
    typ = float(np.median(d_smooth))
    jump_raw = float(d_raw[seam].max()) / typ
    jump_smooth = float(d_smooth[seam].max()) / typ
    assert jump_smooth < 3.0, (
        f"OLA + seam repair must de-click: raw {jump_raw:.1f}x -> "
        f"{jump_smooth:.1f}x"
    )
