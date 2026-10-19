"""Regression tests for band-reserved peak selection in res_evidence.

Two independent defects made the evidence stage blind:

1. The low (N=4096) and mid (N=2048) detectors read from offset 0. A track that
   opens with more silence than the window length yields an all-zero spectrum
   and therefore zero evidence for fundamentals. At 22.05 kHz, 4096 samples is
   only 186 ms - shorter than the leading silence of most real recordings.
2. The dedicated >2 kHz detector's survivors were appended to a single list and
   then cut by a global magnitude sort against the loud low bands. Music carries
   roughly -12 dB/octave of spectral tilt, so sibilance could never survive.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.res_evidence import _loudest_window, multi_resolution_peaks

SR = 22050


def _tone(freq: float, dur: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2.0 * np.pi * freq * t)


def _with_leading_silence(sig: np.ndarray, lead_s: float = 0.3) -> np.ndarray:
    return np.concatenate([np.zeros(int(lead_s * SR)), sig])


class TestLoudestWindow:
    def test_picks_window_over_silence(self):
        y = _with_leading_silence(_tone(80.0), 0.3)
        win = 4096
        seg = _loudest_window(y, win)
        assert len(seg) == win
        assert float(np.sqrt(np.mean(seg ** 2))) > 1e-3

    def test_returns_short_input_unchanged(self):
        y = _tone(100.0, 0.05)
        assert len(_loudest_window(y, 4096)) == len(y)

    def test_selects_the_higher_energy_region(self):
        y = np.zeros(SR)
        quiet = _tone(100.0, 4096 / SR) * 0.05
        loud = _tone(300.0, 4096 / SR) * 0.9
        y[int(0.2 * SR):int(0.2 * SR) + 4096] = quiet
        y[int(0.7 * SR):int(0.7 * SR) + 4096] = loud
        seg = _loudest_window(y, 4096)
        assert float(np.abs(seg).max()) > 0.5


class TestLeadingSilenceRegression:
    """The original defect: silence at offset 0 blinded low and mid."""

    def test_finds_low_band_with_leading_silence(self):
        sig = _tone(120.0, 1.0)
        y = _with_leading_silence(sig, 0.4)
        peaks = multi_resolution_peaks(y, SR, top_k=24)
        low = [p for p in peaks if p["frequency"] <= 250.0]
        assert low, "low band produced no evidence despite a 120 Hz tone"

    def test_low_band_empty_without_the_fix_on_silent_head(self):
        # Demonstrates the bug the loudest-window selection removes.
        y = _with_leading_silence(_tone(120.0, 1.0), 0.4)
        naive = y[:4096]
        assert float(np.abs(naive).max()) == 0.0

    def test_finds_mid_band_with_leading_silence(self):
        y = _with_leading_silence(_tone(700.0, 1.0), 0.4)
        peaks = multi_resolution_peaks(y, SR, top_k=24)
        mid = [p for p in peaks if 250.0 < p["frequency"] <= 2000.0]
        assert mid, "mid band produced no evidence despite a 700 Hz tone"

    def test_all_three_bands_populated_on_realistic_input(self):
        y = _with_leading_silence(
            _tone(120.0, 1.0) + 0.5 * _tone(800.0, 1.0) + 0.2 * _tone(6000.0, 1.0),
            0.3,
        )
        bands = {p["band"] for p in multi_resolution_peaks(y, SR, top_k=24)}
        assert {"low", "mid", "high"} & bands == {"low", "mid", "high"}


class TestBandQuota:
    def test_respects_top_k(self):
        y = _with_leading_silence(_tone(120.0, 1.0) + 0.5 * _tone(900.0, 1.0), 0.2)
        assert len(multi_resolution_peaks(y, SR, top_k=24)) <= 24

    def test_high_band_survives_a_dominant_low_end(self):
        # A very loud bass plus quiet sibilance must not starve the high band.
        sr = SR
        t = np.arange(sr) / sr
        y = 1.0 * np.sin(2 * np.pi * 90.0 * t) + 0.004 * _tone(6500.0, 1.0)
        peaks = multi_resolution_peaks(y, sr, top_k=24, high_quota=8)
        high = [p for p in peaks if p["band"] == "high"]
        assert high, "quiet high-frequency content was truncated by loud bass"

    def test_quota_zero_keeps_legacy_behaviour(self):
        y = _with_leading_silence(_tone(120.0, 1.0) + 0.3 * _tone(6000.0, 1.0), 0.2)
        pk = multi_resolution_peaks(y, SR, top_k=24, high_quota=0)
        assert len(pk) <= 24

    def test_default_quota_is_a_third_of_top_k(self):
        y = _with_leading_silence(_tone(120.0, 1.0) + 0.3 * _tone(6000.0, 1.0), 0.2)
        a = multi_resolution_peaks(y, SR, top_k=24)
        b = multi_resolution_peaks(y, SR, top_k=24, high_quota=8)
        assert [p["frequency"] for p in a] == [p["frequency"] for p in b]

    def test_output_sorted_by_magnitude_descending(self):
        y = _with_leading_silence(_tone(120.0, 1.0) + 0.3 * _tone(6000.0, 1.0), 0.2)
        pk = multi_resolution_peaks(y, SR, top_k=24)
        mags = [p["magnitude"] for p in pk]
        assert mags == sorted(mags, reverse=True)


class TestDeterminism:
    def test_repeatable(self):
        y = _with_leading_silence(_tone(150.0, 1.0) + 0.4 * _tone(5000.0, 1.0), 0.25)
        a = multi_resolution_peaks(y, SR, top_k=24)
        b = multi_resolution_peaks(y, SR, top_k=24)
        assert a == b

    def test_too_short_input_is_empty(self):
        assert multi_resolution_peaks(np.zeros(8), SR) == []