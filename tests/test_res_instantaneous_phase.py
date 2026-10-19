"""Tests for Slice 1: Instantaneous Phase Alignment (Hilbert transform)."""

import numpy as np
import pytest

from src.res_synthesis import _corr, _render_partial, align_instantaneous_phase


def test_align_instantaneous_phase_fixes_phase_offset():
    """Verify Hilbert transform phase alignment extracts and fixes phase error."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    f = 440.0
    true_phi = 1.25  # ~71 degrees phase offset
    target = 0.8 * np.cos(2.0 * np.pi * f * t - true_phi)
    
    # Partial with 0.0 initial phase
    p = {
        "model": "damped",
        "frequency": f,
        "amplitude": 0.8,
        "phase": 0.0,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
    }
    
    # Correlation before alignment
    r_before = _render_partial(p, n, sr)
    c_before = _corr(target, r_before)
    
    # Apply phase alignment
    d_phi = align_instantaneous_phase(target, p, sr)
    
    # Render after alignment
    r_after = _render_partial(p, n, sr)
    c_after = _corr(target, r_after)
    
    # Correlation must jump close to 1.0
    assert c_after > 0.99
    assert c_after > c_before + 0.5
    # The phase correction should be close to true_phi
    phase_diff = (p["phase"] - true_phi) % (2.0 * np.pi)
    assert phase_diff < 0.05 or abs(phase_diff - 2.0 * np.pi) < 0.05


def test_align_instantaneous_phase_preserves_aligned_partial():
    """Verify already aligned partial has near zero phase shift."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    f = 220.0
    phi = 0.5
    target = np.cos(2.0 * np.pi * f * t - phi)
    p = {
        "model": "damped",
        "frequency": f,
        "amplitude": 1.0,
        "phase": phi,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
    }
    d_phi = align_instantaneous_phase(target, p, sr)
    assert abs(d_phi) < 0.02
    assert _corr(target, _render_partial(p, n, sr)) > 0.999
