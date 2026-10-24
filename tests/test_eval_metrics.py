"""Evaluation upgrade: multi-resolution STFT distance + per-band correlation.

Why: the error-budget analysis (2026-10-23) had to compute band-split
correlations and multi-resolution spectral distance by hand outside the
engine, because the referee's metric set (corr, SNR, LSD at one window,
segSNR, onset) cannot see *where* in the spectrum the error lives. The
drums stem's failure is invisible to plain LSD at n_fft=2048 yet obvious
in the bands (>=98% of 2-11 kHz unexplained, band corr 0.10-0.18).

Seams under test (public interfaces):
- src.residual.compute_error_metrics(extras=True) -> mrstft_db + band_corr
- src.res_synthesis._score -> the ledger-facing dict carries both
Default behavior (extras=False) is unchanged, so the referee hot path and
published-number provenance stay put.
"""
from __future__ import annotations

import json

import numpy as np

from src.res_synthesis import _score
from src.residual import compute_error_metrics

SR = 22050


def _sine_cycles(n: int, cycles: int) -> np.ndarray:
    """A sine with an exact integer number of cycles over n samples."""
    t = np.arange(n) / n
    return np.sin(2.0 * np.pi * cycles * t)


def test_mrstft_is_zero_for_identical_signals() -> None:
    """Independent truth: identical signals have zero spectral distance."""
    y = _sine_cycles(SR, 100)
    m = compute_error_metrics(y, y.copy(), sr=SR, extras=True)
    assert m["mrstft_db"] < 0.01


def test_mrstft_sees_what_single_window_misses() -> None:
    """A high-frequency-only error must register in the multi-resolution
    distance (the drums failure mode: LSD at one window understates it)."""
    n = SR
    y = _sine_cycles(n, 100) + 0.4 * _sine_cycles(n, 5000)
    r = _sine_cycles(n, 100).copy()  # the 5 kHz partial is missing
    m = compute_error_metrics(y, r, sr=SR, extras=True)
    assert m["mrstft_db"] > 1.0


def test_band_corr_localizes_the_missing_band() -> None:
    """Missing 5 kHz partial: low band ~perfect, high band ~zero."""
    n = SR
    y = _sine_cycles(n, 100) + 0.4 * _sine_cycles(n, 5000)
    r = _sine_cycles(n, 100).copy()
    m = compute_error_metrics(y, r, sr=SR, extras=True)
    bc = m["band_corr"]
    assert set(bc) == {"low", "mid", "high", "top"}
    assert bc["low"] > 0.99
    assert bc["high"] < 0.2


def test_band_corr_is_high_everywhere_for_a_good_render() -> None:
    """Bands that carry energy must correlate ~1 for an identical render;
    bands that are silent on BOTH sides correlate 0.0 by definition (there
    is nothing to agree about) - the documented empty-band semantics."""
    y = _sine_cycles(SR, 100) + 0.4 * _sine_cycles(SR, 5000)
    m = compute_error_metrics(y, y.copy(), sr=SR, extras=True)
    bc = m["band_corr"]
    assert bc["low"] > 0.99 and bc["high"] > 0.99
    assert bc["mid"] == 0.0 and bc["top"] == 0.0


def test_band_corr_full_spectrum_render_agrees_everywhere() -> None:
    """A render with energy in every band correlates ~1 in every band."""
    n = SR
    y = sum(0.4 * _sine_cycles(n, c) for c in (100, 500, 5000, 10000))
    m = compute_error_metrics(y, y.copy(), sr=SR, extras=True)
    assert all(v > 0.99 for v in m["band_corr"].values())


def test_default_call_has_no_extra_keys() -> None:
    """extras=False (the referee hot path) must stay byte-compatible."""
    y = _sine_cycles(SR, 100)
    m = compute_error_metrics(y, y.copy(), sr=SR)
    assert "mrstft_db" not in m
    assert "band_corr" not in m


def test_score_carries_the_new_metrics_and_is_json_able() -> None:
    """The ledger-facing score dict publishes both, JSON-clean."""
    n = SR
    y = _sine_cycles(n, 100) + 0.4 * _sine_cycles(n, 5000)
    r = _sine_cycles(n, 100).copy()
    s = _score(y, r, SR)
    assert "mrstft_db" in s and isinstance(s["mrstft_db"], float)
    assert isinstance(s["band_corr"], dict) and set(s["band_corr"]) == {
        "low", "mid", "high", "top"
    }
    json.dumps(s)  # must not raise
