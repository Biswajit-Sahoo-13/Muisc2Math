"""Tests for Slice 3: Joint Non-Linear Parameter Refinement in res_loop."""

import numpy as np
import pytest

from src.res_synthesis import _corr, _render_partial
from src.res_loop import refine_section_parameters


def test_refine_section_parameters_resolves_overlapping_partials():
    """Verify least_squares joint refinement decouples overlapping partials."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    
    # Ground truth: two nearby tones (e.g. 440 Hz and 445 Hz beating)
    y_sec = 0.5 * np.cos(2.0 * np.pi * 440.0 * t - 0.2) + 0.4 * np.cos(2.0 * np.pi * 445.0 * t - 0.8)
    
    # Greedy estimate has slight parameter errors due to mutual interference
    partials = [
        {"frequency": 439.2, "amplitude": 0.44, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"},
        {"frequency": 445.8, "amplitude": 0.35, "phase": -0.5, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"},
    ]
    
    recon_before = sum(_render_partial(p, n, sr) for p in partials)
    c_before = _corr(y_sec, recon_before)
    
    accepted = refine_section_parameters(y_sec, partials, sr, max_nfev=30)
    assert accepted > 0
    
    recon_after = sum(_render_partial(p, n, sr) for p in partials)
    c_after = _corr(y_sec, recon_after)
    
    # Joint refinement must strictly improve correlation and push it higher
    assert c_after > c_before + 0.02
    assert c_after > 0.98


def test_refine_section_parameters_handles_empty_and_noise():
    """Verify robust handling of empty partial lists and noise-only models."""
    sr = 22050
    y_sec = np.random.randn(512)
    assert refine_section_parameters(y_sec, [], sr) == 0
    noise_part = [{"model": "filtered_noise", "amplitude": 0.5, "seed": 42}]
    assert refine_section_parameters(y_sec, noise_part, sr) == 0
