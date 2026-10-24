"""The fidelity benchmark: "original = 100%, ours = ?"

A single number that answers the question a listener actually asks - how
far the reconstruction sits from the original - with a pillar breakdown so
the headline cannot hide where the distance lives.

Design (independent truth, not re-derived from the engine):
- tone      : waveform correlation (x100). Shape agreement.
- spectrum  : 10^(-mrstft_db / 20). The MR-STFT distance in dB converted to
              a 0-1 agreement; 0 dB -> 1.0, 6 dB -> 0.5, 20 dB -> 0.1.
- bands     : energy-weighted per-band correlation. How much of the
              spectrum, weighed by where the original's energy actually is,
              agrees. Empty bands contribute nothing (their weight is 0).
- loudness  : exp(-|ln(rms_r / rms_y)|). 1.0 at parity, ~0.37 at 2x off,
              ~0.14 at 4x off. Catches the too-quiet-render failure that
              correlation is blind to by construction.
- fidelity  : mean of the four pillars, x100. "Ours, in percent of the
              original." distance = 100 - fidelity.

Worked examples pinned below: an identical render is 100; a shape-perfect
render at half amplitude lands at ~75 (two pillars halved); noise against
tone collapses.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.fidelity import compute_fidelity

SR = 22050


def _sine_cycles(n: int, cycles: int) -> np.ndarray:
    t = np.arange(n) / n
    return np.sin(2.0 * np.pi * cycles * t)


def test_identical_render_scores_100() -> None:
    y = _sine_cycles(SR, 100) + 0.4 * _sine_cycles(SR, 5000)
    f = compute_fidelity(y, y.copy(), SR)
    assert f["fidelity_pct"] == pytest.approx(100.0, abs=0.5)
    for p in ("tone_pct", "spectrum_pct", "band_pct", "loudness_pct"):
        assert f[p] == pytest.approx(100.0, abs=0.5)
    assert f["distance_pct"] == pytest.approx(0.0, abs=0.5)


def test_half_volume_shape_perfect_render_scores_about_75() -> None:
    """The too-quiet failure mode must be visible in the headline.

    Broadband signal so every spectral bin sits above the floor: tone 100,
    bands 100; spectrum = 10^(-6.02/20) ~ 50; loudness = 1/2. Mean ~ 75 -
    a pure scale error costs half the headline, as it should.
    """
    rng = np.random.default_rng(5)
    y = rng.standard_normal(SR)
    f = compute_fidelity(y, 0.5 * y, SR)
    assert f["tone_pct"] == pytest.approx(100.0, abs=0.5)
    assert f["spectrum_pct"] == pytest.approx(50.0, abs=3.0)
    assert f["loudness_pct"] == pytest.approx(50.0, abs=2.0)
    assert f["fidelity_pct"] == pytest.approx(75.0, abs=2.5)


def test_noise_against_tone_collapses() -> None:
    """Unrelated signals: the shape pillars collapse; loudness may partly
    agree by accident (RMS is one scalar), the headline stays far from 100."""
    rng = np.random.default_rng(3)
    y = _sine_cycles(SR, 100)
    noise = rng.standard_normal(SR)
    f = compute_fidelity(y, noise, SR)
    assert f["tone_pct"] < 5.0
    assert f["band_pct"] < 5.0
    assert f["fidelity_pct"] < 25.0
    assert f["distance_pct"] > 75.0


def test_band_pillar_is_energy_weighted() -> None:
    """A missing 5 kHz partial costs the band pillar in proportion to the
    energy the original carries there (0.4^2 / (1 + 0.4^2) ~ 13.8%)."""
    y = _sine_cycles(SR, 100) + 0.4 * _sine_cycles(SR, 5000)
    r = _sine_cycles(SR, 100).copy()
    f = compute_fidelity(y, r, SR)
    # low band weight 1/(1+0.16) = 0.862 at corr~1, high weight 0.138 at
    # corr~0 -> band_pct ~ 86 (not the unweighted 50).
    assert f["band_pct"] == pytest.approx(86.2, abs=3.0)


def test_silent_render_is_finite_and_zero_loudness() -> None:
    y = _sine_cycles(SR, 100)
    f = compute_fidelity(y, np.zeros(SR), SR)
    assert all(np.isfinite(v) for v in f.values())
    assert f["loudness_pct"] == 0.0
    assert 0.0 <= f["fidelity_pct"] < 30.0


def test_length_mismatch_is_aligned_not_fatal() -> None:
    y = _sine_cycles(SR, 100)
    f = compute_fidelity(y, y[: SR // 2], SR)
    assert f["fidelity_pct"] == pytest.approx(100.0, abs=1.0)


def test_result_is_json_able_and_complete() -> None:
    y = _sine_cycles(SR, 100)
    f = compute_fidelity(y, 0.8 * y, SR)
    assert {"tone_pct", "spectrum_pct", "band_pct", "loudness_pct",
            "fidelity_pct", "distance_pct"} <= set(f)
    json.dumps(f)
