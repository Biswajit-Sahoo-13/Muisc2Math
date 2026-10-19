"""Tests for the spectral acceptance lane and the gated sibilance burst class.

The waveform-correlation gate demands sample-exact alignment. A noise burst
can never align with real noise, so every stochastic class was rejected by
construction - the ledger showed residual deltas of exactly 0.0% for all ten
lpc_noise / filtered_noise candidates, because they were never evaluated at all.
The lane decouples perceptual modelling from phase alignment.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.res_bank import (
    _accept_candidate,
    _hf_residual_energy,
    spectral_flatness,
    sibilance_burst_hypothesis,
    tournament_section,
)

SR = 22050


def _sibilant(dur=1.0, sr=SR, seed=1):
    # The renderer emits cos(2*pi*f*t - phase), so the fixture's tonal core
    # must be a cosine too or it sits 90 degrees out of phase and correlates 0.
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    tone = 0.3 * np.cos(2 * np.pi * 300.0 * t)
    fric = 0.5 * rng.standard_normal(len(t))
    spec = np.fft.rfftfreq(len(fric), 1.0 / sr)
    fric = np.fft.irfft(np.fft.rfft(fric) * (spec >= 4000.0), n=len(fric))
    return tone + 0.5 * fric


def _tonal(dur=1.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return sum(a * np.sin(2 * np.pi * f * t) for a, f in
               ((0.5, 220.0), (0.3, 440.0), (0.2, 880.0)))


class TestSibilanceGating:
    def test_sfm_separates_noise_from_tone(self):
        assert spectral_flatness(_sibilant()) > spectral_flatness(_tonal())

    def test_proposes_burst_on_sibilant_residual(self):
        v = sibilance_burst_hypothesis(_sibilant(), SR, 0)
        assert v.partials, "sibilance gate failed on genuinely noisy highs"

    def test_declines_on_tonal_residual(self):
        v = sibilance_burst_hypothesis(_tonal(), SR, 0)
        assert not v.partials, "sibilance gate fired on a purely tonal residual"

    def test_declines_on_silence(self):
        assert not sibilance_burst_hypothesis(np.zeros(SR), SR, 0).partials

    def test_declines_on_short_input(self):
        assert not sibilance_burst_hypothesis(np.zeros(32), SR, 0).partials

    def test_marked_as_sibilance_partial(self):
        v = sibilance_burst_hypothesis(_sibilant(), SR, 0)
        assert v.partials[0].get("sibilance") is True

    def test_deterministic_for_a_fixed_seed(self):
        a = sibilance_burst_hypothesis(_sibilant(), SR, 0)
        b = sibilance_burst_hypothesis(_sibilant(), SR, 0)
        assert a.partials[0]["amplitude"] == b.partials[0]["amplitude"]


class TestHfResidualEnergy:
    def test_ignores_low_frequency(self):
        t = np.arange(SR) / SR
        low = np.sin(2 * np.pi * 100.0 * t)
        assert _hf_residual_energy(low, SR, 4000.0) == pytest.approx(0.0, abs=1e-6)

    def test_detects_high_frequency(self):
        t = np.arange(SR) / SR
        high = np.sin(2 * np.pi * 7000.0 * t)
        assert _hf_residual_energy(high, SR, 4000.0) > 1.0

    def test_zero_for_short_input(self):
        assert _hf_residual_energy(np.zeros(4), SR) == 0.0


def _base_partial(y):
    """A partial that already models the tonal core of ``y``."""
    from src.res_synthesis import _render_partial

    def _render(pls, n):
        out = np.zeros(n)
        for p in pls:
            out += _render_partial(p, n, SR)
        return out

    return [{
        "model": "constant", "frequency": 300.0, "amplitude": 0.8,
        "phase": 0.0, "envelope_decay": 0.0, "envelope_a0": 0.8,
        "envelope_t0": 0.0, "start_time": 0.0, "end_time": 1.0,
    }], _render


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


class TestSpectralAcceptanceLane:
    def test_lane_can_accept_a_corr_losing_candidate(self):
        y = _sibilant()
        base, _render = _base_partial(y)
        state = lambda: _render(base, len(y))
        r0 = state()
        cur_err = float(np.sum((y - r0) ** 2))
        cur_corr = _corr(y, r0)

        v = sibilance_burst_hypothesis(y - r0, SR, 0)
        assert v.partials, "precondition: a burst should be proposable"

        ok, _e, c_after = _accept_candidate(
            y, base, v.partials, v.param_count, v.err,
            float(np.sum(y ** 2)), cur_err, cur_corr, state, sr=SR, spectral=True,
        )
        assert ok, "spectral lane rejected a candidate that lowers LSD"
        assert cur_corr > c_after, "a burst must cost some waveform correlation"
        assert c_after >= 0.60, "the correlation floor must still be honoured"

    def test_corr_floor_rejects_a_candidate_that_would_break_it(self):
        y = _sibilant()
        base, _render = _base_partial(y)
        state = lambda: _render(base, len(y))
        r0 = state()
        v = sibilance_burst_hypothesis(y - r0, SR, 0)
        # An unreachable floor must reject even a genuine LSD improvement.
        ok, _e, _c = _accept_candidate(
            y, base, v.partials, v.param_count, v.err,
            float(np.sum(y ** 2)), float(np.sum((y - r0) ** 2)), _corr(y, r0),
            state, sr=SR, spectral=True, corr_floor=0.999999,
        )
        assert not ok

    def test_corr_gate_still_rejects_when_lane_disabled(self):
        y = _sibilant()
        base, _render = _base_partial(y)
        state = lambda: _render(base, len(y))
        r0 = state()
        cur_err = float(np.sum((y - r0) ** 2))
        cur_corr = _corr(y, r0)
        v = sibilance_burst_hypothesis(y - r0, SR, 0)
        ok, _e, _c = _accept_candidate(
            y, base, v.partials, v.param_count, v.err,
            float(np.sum(y ** 2)), cur_err, cur_corr, state, sr=SR, spectral=False,
        )
        assert not ok, "correlation gate must still reject when the lane is off"

    def test_state_restored_on_failure(self):
        y = _sibilant()
        base, _render = _base_partial(y)
        state = lambda: np.zeros(len(y))
        snapshot = len(base)
        ok, _e, _c = _accept_candidate(
            y, base, [{"model": "constant", "frequency": 300.0, "amplitude": 1.0,
                       "phase": 0.0, "envelope_decay": 0.0, "envelope_a0": 1.0,
                       "envelope_t0": 0.0, "start_time": 0.0, "end_time": 1.0}],
            1, 0.0, 1.0, 1.0, 0.9, state, sr=SR, spectral=True,
        )
        assert not ok
        assert len(base) == snapshot


class TestCorrFloorMode:
    """The mode must be a deliberate switch, not an accidental formula swap.

    ``max(corr_floor, cur_corr - drop)`` was proposed once as the "relative"
    form. It reduces to the absolute floor for every ``cur <= floor + drop``, so
    it never enables the lane on a low-correlation stem while appearing to.
    These tests pin both the correct relative form and that rejected variant.
    """

    def _f(self, mode, cur, floor=0.60, drop=0.05):
        from src.res_bank import _spectral_corr_floor
        return _spectral_corr_floor(floor, mode, cur, drop)

    def test_absolute_mode_is_the_default_and_flat(self):
        for cur in (0.711, 0.55, 0.25, 0.05):
            assert self._f("absolute", cur) == pytest.approx(0.60)

    def test_relative_mode_is_a_pure_degradation_budget(self):
        assert self._f("relative", 0.711) == pytest.approx(0.661)
        assert self._f("relative", 0.55) == pytest.approx(0.50)
        assert self._f("relative", 0.25) == pytest.approx(0.20)

    def test_relative_mode_actually_unblocks_a_low_corr_stem(self):
        assert self._f("relative", 0.25) < 0.60

    def test_max_form_is_not_relative(self):
        # The rejected variant collapses to the absolute floor below 0.65.
        for cur in (0.55, 0.25):
            assert max(0.60, cur - 0.05) == pytest.approx(0.60)
            assert self._f("relative", cur) < max(0.60, cur - 0.05)

    def test_absolute_mode_is_stricter_on_good_stems(self):
        # Absolute lets a good stem drift; relative pins it to a 0.05 budget.
        assert self._f("absolute", 0.711) < self._f("relative", 0.711)


class TestTournamentLaneWiring:
    def test_lane_off_by_default_emits_no_sibilance_record(self):
        y = _sibilant(0.5)
        base = [{
            "model": "constant", "frequency": 300.0, "amplitude": 0.28,
            "phase": 0.0, "envelope_decay": 0.0, "envelope_a0": 0.28,
            "envelope_t0": 0.0, "start_time": 0.0, "end_time": 0.5,
        }]
        recs = tournament_section(y, base, SR, 0, 0.0)
        assert not [r for r in recs if r["class"] == "sibilance_burst"]

    def test_lane_on_emits_sibilance_record(self):
        y = _sibilant(0.5)
        base = [{
            "model": "constant", "frequency": 300.0, "amplitude": 0.28,
            "phase": 0.0, "envelope_decay": 0.0, "envelope_a0": 0.28,
            "envelope_t0": 0.0, "start_time": 0.0, "end_time": 0.5,
        }]
        recs = tournament_section(y, base, SR, 0, 0.0,
                                  enable_spectral_lane=True)
        assert any(r["class"] == "sibilance_burst" for r in recs)