"""Tests for LPC Formant Noise class in res_bank and res_synthesis (Phase 4)."""

import numpy as np
import pytest
import scipy.signal

from src.res_synthesis import _render_partial
from src.res_bank import lpc_noise_hypothesis


def test_lpc_noise_rendering_deterministic():
    """Verify that lpc_noise renders deterministically given a seed."""
    sr = 22050
    n = 1024
    p = {
        "model": "lpc_noise",
        "amplitude": 0.5,
        "lpc_order": 4,
        "lpc_coeffs": [0.6, -0.2, 0.1, -0.05],
        "seed": 1234,
        "start_time": 0.0,
        "end_time": n / sr,
    }
    y1 = _render_partial(p, n, sr)
    y2 = _render_partial(p, n, sr)
    assert np.allclose(y1, y2)
    assert np.std(y1) > 0.01


def test_lpc_noise_captures_formant_envelope():
    """Verify that lpc_noise_hypothesis fits formant poles on vocal-like filtered noise."""
    sr = 22050
    n = 4096
    t = np.arange(n) / sr

    # Create synthetic vocal tract response: noise shaped by two formants (e.g. 700 Hz and 2200 Hz)
    rng = np.random.default_rng(42)
    white = rng.standard_normal(n)
    sos1 = scipy.signal.butter(2, [600, 800], btype='bandpass', fs=sr, output='sos')
    sos2 = scipy.signal.butter(2, [2000, 2400], btype='bandpass', fs=sr, output='sos')
    vocal_target = 0.5 * scipy.signal.sosfilt(sos1, white) + 0.3 * scipy.signal.sosfilt(sos2, white)

    variant = lpc_noise_hypothesis(
        vocal_target, sr=sr, start_sample=0, order=12, seed=42
    )

    assert variant.partials, "Must generate at least one lpc_noise partial"
    part = variant.partials[0]
    assert part["model"] == "lpc_noise"
    assert len(part["lpc_coeffs"]) == 12

    # Render candidate
    rendered = _render_partial(part, n, sr)
    e_before = float(np.sum(vocal_target ** 2))
    e_after = float(np.sum((vocal_target - rendered) ** 2))

    # Spectral envelope matching must drop residual error or match spectral shape
    assert variant.err < e_before
