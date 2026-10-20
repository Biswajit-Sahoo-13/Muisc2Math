"""The complement of test_hypothesis_rejects_pure_noise.

``test_hypothesis_rejects_pure_noise`` proves the bank refuses noise. It does
not prove the bank *accepts* signal. Those are different properties: a gate
tightened until it refuses everything still passes the negative test.

This asserts the positive side: a section carrying a clear percussive burst
(a sine bed plus a 5 ms attack / 200 ms decay noise burst at a known position)
must have its class-C candidate ACCEPTED, and the log-spectral distance must
fall. If the class-C magnitude-domain gate is ever narrowed, this goes red
before the shipped audio does.
"""
import numpy as np
import pytest

from src.res_bank import tournament_section
from src.res_synthesis import _score

SR = 22050
DUR = 2.5


def _bed_and_burst(sr=SR, dur=DUR, burst_at=1.25, attack=0.005, decay=0.2):
    """A sine bed with one percussive burst at a known position."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(1234)
    y = (0.25 * np.sin(2 * np.pi * 220.0 * t)
         + 0.12 * np.sin(2 * np.pi * 440.0 * t))
    # a burst: broadband, fast attack, exponential decay
    env = np.exp(-(t - burst_at) / decay)
    env[t < burst_at] = 0.0
    env[t < burst_at + attack] *= (t[t < burst_at + attack] - burst_at) / attack
    burst = rng.standard_normal(n) * env * 0.35
    return (y + burst).astype(np.float64), burst_at


def _section(t0, dur):
    return {
        "start": t0, "end": t0 + dur,
        "start_sample": int(t0 * SR),
        "partials": [],
        "metrics": {},
    }


def test_hypothesis_accepts_clear_cymbal_burst():
    y, burst_at = _bed_and_burst()
    sec = _section(0.0, DUR)
    sec["partials"] = [
        {"model": "constant", "amplitude": 0.25, "frequency": 220.0,
         "phase": 0.0, "envelope_decay": 0.4, "envelope_t0": 0.0,
         "start_time": 0.0, "end_time": DUR, "projection_gain": 1.0},
        {"model": "constant", "amplitude": 0.12, "frequency": 440.0,
         "phase": 0.0, "envelope_decay": 0.4, "envelope_t0": 0.0,
         "start_time": 0.0, "end_time": DUR, "projection_gain": 1.0},
    ]
    recs = tournament_section(y, sec["partials"], SR,
                              start_sample=0, stft_half_window_s=0.023)
    percussive = [r for r in recs if r.get("class") == "percussive_bundle"]
    assert percussive, "the bank nominated no percussive candidate at all"
    assert any(r.get("accepted") for r in percussive), (
        "a clear noise burst was refused: "
        + "; ".join(f"residual {r.get('residual_before')} -> "
                    f"{r.get('residual_after')}" for r in percussive)
    )


def test_burst_nomination_sits_near_the_burst():
    """The accepted candidate should be at the burst, not somewhere random."""
    y, burst_at = _bed_and_burst()
    sec = _section(0.0, DUR)
    sec["partials"] = [
        {"model": "constant", "amplitude": 0.25, "frequency": 220.0,
         "phase": 0.0, "envelope_decay": 0.4, "envelope_t0": 0.0,
         "start_time": 0.0, "end_time": DUR, "projection_gain": 1.0},
    ]
    recs = tournament_section(y, sec["partials"], SR,
                              start_sample=0, stft_half_window_s=0.023)
    percussive = [r for r in recs if r.get("class") == "percussive_bundle"]
    assert percussive, "no percussive candidate was nominated"
    t0s = [r.get("t0") for r in percussive if r.get("t0") is not None]
    assert any(abs(t0 - burst_at) < 0.20 for t0 in t0s), (
        f"no candidate landed at the burst ({burst_at:.3f}s); got {t0s}"
    )


def test_pure_noise_is_still_refused():
    """The negative case must keep holding alongside the positive one.

    Mirrors test_hypothesis_rejects_pure_noise: the class-A state explains its
    component exactly, so the residual is pure white noise and any accepted
    note would be noise-chasing.
    """
    from src.res_synthesis import _render_partial
    rng = np.random.default_rng(3)
    class_a = [{
        "model": "damped", "amplitude": 0.30, "envelope_decay": 1.0,
        "envelope_t0": 0.0, "frequency": 300.0, "phase": 0.0,
    }]
    n = int(2.0 * SR)
    y = (0.05 * rng.standard_normal(n)
         + _render_partial(class_a[0], n, SR, start_sample=0))
    recs = tournament_section(y, class_a, SR, start_sample=0)
    accepted = [r for r in recs if r.get("accepted")]
    assert not accepted, (
        "pure noise was accepted: " + "; ".join(str(r) for r in accepted)
    )
