"""Four measured defects in the frequency/trajectory path, pinned as tests.

Each defect below was reproduced against the current tree; the numbers in the
docstrings are the measured ones, not projections.

  D9  the 9-frame (104.49 ms) frequency median filter is 1.25x the half
      period of a 6 Hz vibrato, so it removes 36.3% of the excursion
      (20.00 Hz -> 12.74 Hz measured) and leaves the upper harmonics
      untrackable.
  D10 the upper vibrato harmonics shatter into stationary-phase caustics.
      The analyser leaves 161 partials for an 8-harmonic vibrato tone, of
      which 143 (89%) sit within 25 Hz of no harmonic at all, and the
      harmonics above 4th are absent entirely.
  D11 outlier rejection in _prepare_frames is referenced to the *static*
      track median, so a wide linear sweep loses its endpoints: the
      200 -> 600 Hz sweep keeps only [0.348, 2.055] s, i.e. 28.8% of the
      frames and 200-600 Hz truncated to 255.7-528.8 Hz.
  D12 the bounded Brent frequency search can leave the projection main
      lobe. With a 2.5 s sustained partial the main lobe is only
      |df| < 1/T = 0.4000 Hz wide and the first sidelobe minimum sits at
      1.4303/T = 0.5721 Hz, while scipy's first probes land at
      f_guess +- 0.4721 Hz - already outside the main lobe. Measured:
      refine_track(220.098 Hz) returns 220.5722 Hz with projection_gain
      0.2174 and 78.3% of the partial's amplitude lost.

Each defect is marked ``xfail(strict=True)``. The suite therefore stays
green, the defect stays pinned, and the moment a fix lands the test
XPASSes and strict mode turns that into a failure - which is the signal to
delete the marker.

The two trajectory-path invariants that already hold are asserted as
ordinary tests, so they cannot regress silently.
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

from src.res_evidence import _rephase_to_section, build_evidence
from src.res_loop import render_ledger
from src.res_synthesis import (
    _RELEASE_FADE_S,
    _corr,
    _glide_traj,
    _support_mask,
)
from src.sinusoidal_model import (
    SinusoidalTrack,
    _prepare_frames,
    _smooth_envelope,
    _trajectories,
    refine_track,
    sinusoidal_reconstruction,
)

SR = 22050
HOP = 256
NFFT = 2048
VIBRATO_F0 = 220.0
VIBRATO_DEV = 20.0
VIBRATO_RATE = 6.0
VIBRATO_HARMONICS = 8


def _make_track(times, freqs, amps, phases) -> SinusoidalTrack:
    """Build a SinusoidalTrack the way the tracker does.

    The constructor takes a single peak, not parallel arrays - passing
    ``times=``/``frequencies=`` is a TypeError.
    """
    tr = SinusoidalTrack(
        0, 0,
        {"frequency": float(freqs[0]), "amplitude": float(amps[0]),
         "phase": float(phases[0])},
        float(times[0]),
    )
    for i in range(1, len(times)):
        tr.update(i, {"frequency": float(freqs[i]),
                      "amplitude": float(amps[i]),
                      "phase": float(phases[i])}, float(times[i]))
    return tr


def _vibrato_signal(dur: float = 2.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n, dtype=np.float64)
    for m in range(1, VIBRATO_HARMONICS + 1):
        inst = m * (VIBRATO_F0 + VIBRATO_DEV * np.sin(2 * np.pi * VIBRATO_RATE * t))
        y += (0.5 / m) * np.cos(2 * np.pi * np.cumsum(inst) / SR)
    return 0.9 * y / np.max(np.abs(y))


def _sweep_signal(f0: float = 200.0, f1: float = 600.0, dur: float = 2.5):
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    inst = f0 + (f1 - f0) / dur * t
    y = 0.8 * np.cos(2 * np.pi * np.cumsum(inst) / SR)
    frame_times = np.arange(0, n - NFFT, HOP) / SR
    freqs = f0 + (f1 - f0) / dur * frame_times
    return y, frame_times, freqs


# ----------------------------------------------------------------------
# D9 - the frequency median filter is longer than the vibrato half-period
# ----------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "D9: the 9-frame (104.49 ms) median filter spans 1.25x the half period "
    "of a 6 Hz vibrato and removes 36.3% of the excursion. Needs a shorter "
    "filter (3 frames) or an intra-frame phase model."
))
def test_frequency_median_filter_preserves_vibrato_excursion() -> None:
    tt = np.arange(0, 2.5, HOP / SR)
    vib = VIBRATO_DEV * np.sin(2 * np.pi * VIBRATO_RATE * tt)
    sm = _smooth_envelope(vib, 9)
    kept = float(np.max(np.abs(sm))) / float(np.max(np.abs(vib)))
    assert kept > 0.90, f"vibrato excursion retained only {kept:.3f} (measured 0.637)"


# ----------------------------------------------------------------------
# D10 - upper harmonics shatter into stationary-phase caustics
# ----------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "D10: the 8-harmonic vibrato tone yields 161 partials, 143 of them "
    "within 25 Hz of no harmonic, with the harmonics above the 4th absent. "
    "Needs a main-lobe-safe estimator plus a parameter-overfit guard."
))
def test_vibrato_harmonics_survive_without_caustic_fragmentation() -> None:
    vib = _vibrato_signal()
    res = sinusoidal_reconstruction(
        vib, SR, n_fft=NFFT, hop_length=HOP, add_noise_layer=False,
        window_s=2.5, workers=1,
    )
    parts = res["fitted_parameters"]

    def near_harmonic(f: float) -> bool:
        return any(abs(f - m * VIBRATO_F0) <= 25.0
                   for m in range(1, VIBRATO_HARMONICS + 1))

    strays = [p for p in parts
              if not near_harmonic(float(p["frequency"]))]
    assert len(strays) <= 20, (
        f"{len(strays)} of {len(parts)} partials are caustic fragments "
        f"(measured 143 of 161)"
    )
    for m in range(1, VIBRATO_HARMONICS + 1):
        assert any(abs(float(p["frequency"]) - m * VIBRATO_F0) <= 25.0
                   for p in parts), f"harmonic {m} was not tracked at all"
    assert float(res["correlation"]) > 0.98, (
        f"vibrato correlation {res['correlation']} (measured 0.534)"
    )
    assert int(res["track_count"]) <= 20, (
        f"partial count {res['track_count']} (measured 161)"
    )


# ----------------------------------------------------------------------
# D11 - outlier rejection truncates wide linear sweeps
# ----------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "D11: _prepare_frames references the static track median, so a 200->600 Hz "
    "sweep keeps only [0.348, 2.055] s of its 2.5 s span and 255.7-528.8 Hz "
    "of its 200-600 Hz excursion. Needs a local-trend reference."
))
def test_wide_linear_sweep_keeps_its_full_tracked_span() -> None:
    y, frame_times, freqs = _sweep_signal()
    tr = _make_track(frame_times, freqs, np.full(freqs.size, 0.8),
                     np.zeros(freqs.size))
    times, fr, _amps, *_ = _prepare_frames(tr)
    assert float(times[0]) <= 0.05, (
        f"sweep starts at {times[0]:.4f} s instead of 0 s"
    )
    assert float(times[-1]) >= 2.40, (
        f"sweep ends at {times[-1]:.4f} s instead of ~2.5 s"
    )
    assert float(fr[0]) <= 205.0 and float(fr[-1]) >= 595.0, (
        f"sweep frequency span truncated to [{fr[0]:.1f}, {fr[-1]:.1f}] Hz"
    )


# ----------------------------------------------------------------------
# D12 - the bounded Brent search can leave the projection main lobe
# ----------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "D12: for a 2.5 s sustained partial the projection main lobe is only "
    "0.4000 Hz wide and the first sidelobe minimum is 0.5721 Hz away, while "
    "scipy's first bounded probes land at f_guess +- 0.4721 Hz. Measured: "
    "220.098 Hz -> 220.5722 Hz, projection_gain 0.2174, amplitude -78.3%. "
    "Needs a main-lobe-bracketed search."
))
def test_sustained_partial_does_not_slide_into_the_first_sidelobe() -> None:
    dur = 2.5
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    f_true = 220.0
    y = 0.65 * np.cos(2 * np.pi * f_true * t - 0.3)
    frame_times = np.arange(0, n - NFFT, HOP) / SR
    tr = _make_track(frame_times, np.full(frame_times.size, 220.098),
                     np.full(frame_times.size, 0.65),
                     np.zeros(frame_times.size))
    sig, params = refine_track(tr, y, SR, refine_frequency=True,
                               stft_half_window_s=NFFT / 2 / SR)
    assert abs(float(params["frequency"]) - f_true) < 0.02, (
        f"landed on {params['frequency']} Hz, the first sidelobe at "
        f"{1.4303/dur:.4f} Hz off"
    )
    assert float(params["projection_gain"]) > 0.95, (
        f"projection_gain collapsed to {params['projection_gain']}"
    )
    assert _corr(y, sig) > 0.995


# ----------------------------------------------------------------------
# Invariants that already hold - these must not regress
# ----------------------------------------------------------------------

def test_trajectory_paths_agree_across_section_offsets() -> None:
    """_trajectories and _glide_traj are replicas of each other."""
    n = int(2.5 * SR)
    off = 1.417  # a non-integer, beat-quantised section offset
    tt_local = np.linspace(0.02, 2.48, 212)
    ff = 440.0 + 18.0 * np.sin(2 * np.pi * 6.0 * tt_local)
    aa = 0.5 * np.exp(-0.35 * tt_local)
    t_local = np.arange(n, dtype=np.float64) / SR

    a_sm, th_sm, _ = _trajectories(tt_local, ff, aa, n, SR)
    a_rs, th_rs = _glide_traj(tt_local + off, ff, aa, t_local + off)
    np.testing.assert_allclose(th_sm, th_rs, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(a_sm, a_rs, atol=1e-12, rtol=1e-12)


def test_replica_whistle_and_phase_origin_invariants() -> None:
    """The analyser-renderer replica, the whistle fade bound and the
    phase re-referencing sign law."""
    assert _RELEASE_FADE_S <= 0.05

    t = np.linspace(0.0, 2.0, 2 * SR)
    mask = _support_mask(t, 0.2, 0.8, decay=4.0, amp=0.5,
                         stft_half_window_s=NFFT / 2 / SR)
    hold_s = min(1.5, float(np.log(100.0)) / 4.0)
    past = t > (0.8 + hold_s + _RELEASE_FADE_S + 1e-4)
    assert np.all(mask[past] == 0.0)

    p_stat = {"frequency": 440.0, "phase": 0.25, "phase_origin": 2.5,
              "model": "constant"}
    reph = _rephase_to_section(p_stat, 2.4)
    expected = round(0.25 + 2.0 * np.pi * 440.0 * (2.5 - 2.4), 6)
    assert abs(float(reph["phase"]) - expected) < 1e-5

    dur = 5.0
    n = int(dur * SR)
    tg = np.arange(n, dtype=np.float64) / SR
    sig = (0.45 * np.exp(-0.6 * tg) * np.cos(2 * np.pi * 261.63 * tg)
           + 0.30 * np.cos(2 * np.pi * np.cumsum(
               330.0 + 10.0 * np.sin(2 * np.pi * 5.5 * tg)) / SR))
    with tempfile.TemporaryDirectory() as tmp:
        ev = build_evidence(sig, tmp, sr=SR, max_duration=dur, window_s=2.5,
                            beat_quantized=False, use_hpss=False)
        recon = render_ledger(ev, n)
        sms = sinusoidal_reconstruction(sig, sr=SR, window_s=2.5,
                                        add_noise_layer=False, workers=1)
        assert _corr(sms["sinusoid_audio"], recon) > 0.98
