"""Tests for Clustered Joint Refinement in res_loop (Phase 4)."""

import numpy as np

from src.res_synthesis import _corr, _render_partial
from src.res_loop import refine_section_parameters


def test_clustered_refinement_leaves_isolated_partials_untouched():
    """Verify that only colliding partials (<=10 Hz apart) are refined, leaving isolated partials untouched."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr

    # Ground truth: an isolated 1000 Hz tone + two colliding tones at 440 Hz and 445 Hz
    sig_iso = 0.5 * np.cos(2.0 * np.pi * 1000.0 * t - 0.3)
    sig_coll = 0.4 * np.cos(2.0 * np.pi * 440.0 * t - 0.2) + 0.35 * np.cos(2.0 * np.pi * 445.0 * t - 0.8)
    y_sec = sig_iso + sig_coll

    # Setup partials: isolated partial has exact ground truth parameters
    p_iso = {
        "frequency": 1000.0, "amplitude": 0.5, "phase": 0.3,
        "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped",
    }
    # Colliding partials have slight greedy parameter estimation errors
    p_coll1 = {
        "frequency": 439.2, "amplitude": 0.38, "phase": 0.0,
        "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped",
    }
    p_coll2 = {
        "frequency": 445.9, "amplitude": 0.31, "phase": -0.5,
        "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped",
    }

    partials = [p_iso, p_coll1, p_coll2]

    recon_before = sum(_render_partial(p, n, sr) for p in partials)
    c_before = _corr(y_sec, recon_before)

    refined_count = refine_section_parameters(y_sec, partials, sr, max_nfev=30)
    assert refined_count == 2, f"Expected 2 colliding partials refined, got {refined_count}"

    # Isolated partial MUST NOT have changed
    assert p_iso["frequency"] == 1000.0
    assert p_iso["amplitude"] == 0.5
    assert p_iso["phase"] == 0.3

    # Colliding partials were optimized, and total correlation improved
    recon_after = sum(_render_partial(p, n, sr) for p in partials)
    c_after = _corr(y_sec, recon_after)
    assert c_after > c_before + 0.01
    assert c_after > 0.98


def test_clustered_refinement_skips_when_no_clusters():
    """When all partials are well-separated (>10 Hz), none should be refined."""
    sr = 22050
    n = 1024
    t = np.arange(n) / sr
    y_sec = 0.5 * np.cos(2 * np.pi * 200 * t) + 0.5 * np.cos(2 * np.pi * 500 * t)
    partials = [
        {"frequency": 200.0, "amplitude": 0.5, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"},
        {"frequency": 500.0, "amplitude": 0.5, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"},
    ]
    refined_count = refine_section_parameters(y_sec, partials, sr)
    assert refined_count == 0
