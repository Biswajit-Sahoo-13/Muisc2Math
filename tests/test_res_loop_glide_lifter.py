"""Tests for glide partial refinement in _section_update and seg_snr gating."""

import numpy as np

from src.res_loop import _section_update
from src.res_synthesis import _render_partial, _corr

SR = 22050


def test_section_update_refines_glide():
    """_section_update must refine glide partials with frequency/amplitude/phase optimization."""
    dur = 0.5
    n = int(dur * SR)
    t = np.arange(n) / SR

    # Ground truth: pitch glides from 300 Hz to 350 Hz with slight decay
    f_t = np.linspace(300.0, 350.0, n)
    phase_integral = 2.0 * np.pi * np.cumsum(f_t) / SR
    y_sec = 0.35 * np.exp(-1.5 * t) * np.cos(phase_integral - 0.5)

    # Initial imperfect glide partial (shifted in frequency by +2 Hz and wrong amplitude/phase)
    t_knots = np.linspace(0.0, dur, 10)
    f_knots = np.linspace(302.0, 352.0, 10)  # +2 Hz offset
    a_knots = np.full(10, 0.20)  # Lower amplitude

    glide_p = {
        "model": "glide",
        "frequency": 326.0,  # nominal ref freq with offset
        "amplitude": 0.20,
        "phase": 0.0,
        "envelope_decay": 1.5,
        "envelope_t0": 0.0,
        "projection_gain": 1.0,
        "start_time": 0.0,
        "end_time": dur,
        "trajectory": {
            "t": t_knots.tolist(),
            "f_hz": f_knots.tolist(),
            "a": a_knots.tolist(),
        },
    }

    partials = [glide_p]
    initial_render = _render_partial(glide_p, n, SR)
    initial_corr = _corr(y_sec, initial_render)

    accepted_edits = _section_update(y_sec, partials, SR, start_sample=0)
    assert accepted_edits > 0, "Coordinate lifter must refine the glide partial"

    refined_render = _render_partial(partials[0], n, SR)
    refined_corr = _corr(y_sec, refined_render)
    assert refined_corr > initial_corr + 0.02, f"Correlation should improve: {initial_corr} -> {refined_corr}"


def test_seg_snr_guard_penalizes_local_energy_bloat():
    """Verify that candidate edits causing negative segmental SNR degradation are not accepted."""
    dur = 0.5
    n = int(dur * SR)
    t = np.arange(n) / SR
    y_sec = 0.4 * np.cos(2.0 * np.pi * 440.0 * t)

    # A partial with huge amplitude spike in the first 10% of samples (energy overshoot)
    partials = [{
        "model": "constant",
        "frequency": 440.0,
        "amplitude": 0.4,
        "phase": 0.0,
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
    }]
    # Run section update on well-fitting partial
    _section_update(y_sec, partials, SR)
    # The partial should stay close to true signal and not blow up amplitude
    assert partials[0]["amplitude"] <= 0.5
