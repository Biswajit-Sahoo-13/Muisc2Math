"""Regression tests for the four-stem partition in res_evidence.separate_stems.

The native fallback derived "other" as ``y_mid_high - (yh + yp)``. librosa's HPSS
uses soft masks whose two outputs sum back to the input, so that expression is
identically zero: the "other" stem was empty in every run, on every input, and
the downstream loop silently skipped it as an empty-energy stem.
"""

from __future__ import annotations

import numpy as np

from src.res_evidence import separate_stems

SR = 22050


def _test_signal(dur: float = 3.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(0)
    return (
        0.60 * np.sin(2 * np.pi * 60.0 * t)          # bass
        + 0.40 * np.sin(2 * np.pi * 440.0 * t)        # vocal-range harmonic
        + 0.20 * np.sin(2 * np.pi * 6000.0 * t)       # air / sibilance region
        + 0.02 * rng.standard_normal(len(t))          # percussive-ish noise
    )


class TestPartitionOfUnity:
    def test_stems_sum_to_input(self):
        y = _test_signal()
        s = separate_stems(y, SR, mode="native")
        assert float(np.max(np.abs(sum(s.values()) - y))) < 1e-9

    def test_all_four_stems_returned(self):
        s = separate_stems(_test_signal(), SR, mode="native")
        assert set(s) == {"bass", "drums", "vocals", "other"}

    def test_stem_lengths_match_input(self):
        y = _test_signal(2.0)
        s = separate_stems(y, SR, mode="native")
        assert all(len(v) == len(y) for v in s.values())


class TestOtherStemIsNotVacuous:
    def test_other_carries_energy(self):
        y = _test_signal()
        s = separate_stems(y, SR, mode="native")
        assert float(np.sum(s["other"] ** 2)) > 1e-9, "other stem is still empty"

    def test_other_is_not_the_hpss_residual(self):
        """The old expression is algebraically zero; assert it stays that way."""
        import librosa

        y = _test_signal()
        s = separate_stems(y, SR, mode="native")
        mid = y - s["bass"]
        yh, yp = librosa.effects.hpss(mid)
        assert float(np.max(np.abs(mid - (yh + yp)))) < 1e-9

    def test_bass_holds_low_frequencies(self):
        y = _test_signal()
        s = separate_stems(y, SR, mode="native")
        assert float(np.sqrt(np.mean(s["bass"] ** 2))) > 0.1

    def test_vocals_vocal_band_is_populated(self):
        y = _test_signal()
        s = separate_stems(y, SR, mode="native")
        assert float(np.sum(s["vocals"] ** 2)) > 1e-9

    def test_short_input_does_not_raise(self):
        y = _test_signal(0.05)
        s = separate_stems(y, SR, mode="native")
        assert set(s) == {"bass", "drums", "vocals", "other"}

    def test_silence_input_produces_silent_stems(self):
        y = np.zeros(SR)
        s = separate_stems(y, SR, mode="native")
        assert all(float(np.max(np.abs(v))) == 0.0 for v in s.values())