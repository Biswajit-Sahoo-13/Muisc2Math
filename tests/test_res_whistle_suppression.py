"""Regression tests for the no-birdie rules in res_synthesis.

Zero-decay partials above ~1 kHz hold a pure sine for their entire window, which
is audible as a whistle. High-frequency glide tracks linking noisy STFT peaks
produce sweeping "birdies" for the same reason. The support-mask release fade
was additionally 250 ms, which smears 50-80 ms syllables into one another.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.res_synthesis import (
    _MAX_GLIDE_HZ,
    _RELEASE_FADE_S,
    _WHISTLE_FREQ_HZ,
    minimum_decay_for,
    sanitize_partial,
)

SR = 22050


def _p(**kw):
    base = {
        "model": "constant",
        "frequency": 500.0,
        "amplitude": 0.2,
        "phase": 0.0,
        "envelope_decay": 0.0,
        "envelope_a0": 0.2,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 2.0,
    }
    base.update(kw)
    return base


class TestMinimumDecay:
    def test_no_floor_below_one_khz(self):
        assert minimum_decay_for(200.0) == 0.0
        assert minimum_decay_for(_WHISTLE_FREQ_HZ) == 0.0

    def test_floor_rises_with_frequency(self):
        a = minimum_decay_for(2000.0)
        b = minimum_decay_for(8000.0)
        assert 0.0 < a < b

    def test_floor_starts_at_base(self):
        assert minimum_decay_for(_WHISTLE_FREQ_HZ + 1.0) == pytest.approx(2.502, abs=1e-3)


class TestConstantHighFreqPromotion:
    def test_constant_above_one_khz_becomes_damped(self):
        q = sanitize_partial(_p(model="constant", frequency=3000.0, envelope_decay=0.0))
        assert q["model"] == "damped"
        assert q["envelope_decay"] > 0.0

    def test_constant_below_one_khz_untouched(self):
        p = _p(model="constant", frequency=500.0)
        q = sanitize_partial(p)
        assert q["model"] == "constant"
        assert q["envelope_decay"] == 0.0

    def test_existing_decay_is_respected(self):
        q = sanitize_partial(_p(model="constant", frequency=3000.0, envelope_decay=99.0))
        assert q["envelope_decay"] == pytest.approx(99.0)

    def test_positive_decay_above_one_khz_is_left_alone(self):
        # Regression guard: raising an already-fitted decay is destructive.
        # Forcing a real 1720 Hz harmonic from its fitted 1.38 up to the 3.94
        # floor cost 0.0066 correlation on a synthetic chord fixture.
        p = _p(model="damped", frequency=1720.0, envelope_decay=1.3846,
               start_time=0.0, end_time=0.9752)
        q = sanitize_partial(p)
        assert q["envelope_decay"] == pytest.approx(1.3846)
        assert q["end_time"] == pytest.approx(0.9752)


class TestSpanCapping:
    def test_short_high_transient_is_untouched(self):
        # 403 of 1104 drums partials are zero-decay above 1 kHz, but ~23 ms
        # long. Flooring them only attenuated real transients.
        q = sanitize_partial(
            _p(model="constant", frequency=6637.6, envelope_decay=0.0,
               start_time=0.0, end_time=0.023)
        )
        assert q["envelope_decay"] == 0.0
        assert q["model"] == "constant"

    def test_sustaining_high_tone_is_damped(self):
        q = sanitize_partial(
            _p(model="constant", frequency=6637.6, envelope_decay=0.0,
               start_time=0.0, end_time=0.9)
        )
        assert q["model"] == "damped"
        assert q["envelope_decay"] == pytest.approx(minimum_decay_for(6637.6))

    def test_no_span_truncation_at_all(self):
        # The 3.0/decay cap was measured at exactly 0.0000 effect and removed.
        q = sanitize_partial(
            _p(model="constant", frequency=4000.0, envelope_decay=0.0,
               start_time=0.0, end_time=2.5)
        )
        assert q["end_time"] == pytest.approx(2.5)

    def test_low_partial_span_untouched(self):
        q = sanitize_partial(_p(frequency=200.0, envelope_decay=0.0, end_time=2.5))
        assert q["end_time"] == 2.5
        assert q["model"] == "constant"


class TestGlidePreservation:
    def test_high_glide_is_preserved(self):
        # Demoting glides above 1.8 kHz cost the drums stem -0.0743 on its own.
        # Swept high-frequency content is what hats and cymbals are.
        p = _p(model="glide", frequency=_MAX_GLIDE_HZ + 200.0,
               envelope_decay=0.0, start_time=0.0, end_time=0.5,
               trajectory={"t": [0.0, 0.5], "f_hz": [3000.0, 4000.0],
                           "a": [0.1, 0.1]})
        q = sanitize_partial(p)
        assert q["model"] == "glide"
        assert "trajectory" in q

    def test_low_glide_kept(self):
        q = sanitize_partial(
            _p(model="glide", frequency=400.0,
               trajectory={"t": [0.0, 1.0], "f_hz": [300.0, 500.0], "a": [0.1, 0.1]})
        )
        assert q["model"] == "glide"
        assert "trajectory" in q

    def test_sustaining_high_glide_still_gets_decay_floor(self):
        p = _p(model="glide", frequency=_MAX_GLIDE_HZ + 200.0,
               envelope_decay=0.0, start_time=0.0, end_time=0.8,
               trajectory={"t": [0.0, 0.8], "f_hz": [3000.0, 4000.0],
                           "a": [0.1, 0.1]})
        q = sanitize_partial(p)
        assert q["envelope_decay"] > 0.0


class TestPurity:
    def test_input_not_mutated(self):
        p = _p(model="constant", frequency=5000.0, envelope_decay=0.0)
        sanitize_partial(p)
        assert p["model"] == "constant"
        assert p["envelope_decay"] == 0.0


class TestReleaseFade:
    def test_release_fade_is_short(self):
        assert _RELEASE_FADE_S <= 0.05

    def test_partials_do_not_ring_after_support(self):
        from src.res_synthesis import _support_mask

        sr = SR
        t = np.arange(sr) / sr
        end = 0.5
        decay, amp = 0.0, 1.0
        m = _support_mask(t, 0.0, end, decay, amp, 0.0)
        # Zero-decay partials take the fixed hold branch before the fade runs.
        hold = min(1.5, float(np.log(100.0)) / decay) if decay > 0.0 else 0.25
        support_end = end + hold
        assert float(np.max(m[t > support_end + _RELEASE_FADE_S])) == pytest.approx(0.0, abs=1e-9)
        assert float(np.max(m[(t > end) & (t < support_end)])) == pytest.approx(1.0)

    def test_damped_partial_fades_faster_than_legacy_250ms(self):
        from src.res_synthesis import _support_mask

        sr = SR
        t = np.arange(3 * sr) / sr
        decay = 6.0
        m = _support_mask(t, 0.0, 0.5, decay, 1.0, 0.0)
        hold = min(1.5, float(np.log(100.0)) / decay)
        support_end = 0.5 + hold
        # Legacy used a 0.25 s fade; the tail must now be materially shorter.
        assert float(np.max(m[t > support_end + _RELEASE_FADE_S])) == pytest.approx(0.0, abs=1e-9)
        assert _RELEASE_FADE_S < 0.25