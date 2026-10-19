"""Tests for Slice 2: Overlap-Add (OLA) Boundary Smoothing in res_loop."""

import numpy as np
import pytest

from src.res_loop import apply_ola_crossfade, render_ledger


def test_apply_ola_crossfade_eliminates_boundary_discontinuity():
    """Verify OLA crossfade eliminates abrupt boundary jumps between two blocks."""
    sr = 22050
    fade_len = int(0.015 * sr)  # ~15ms fade
    n = 2048
    
    # Block 1 ending at 1.0, Block 2 starting at -1.0 (worst-case jump of 2.0)
    b1 = np.ones(n)
    b2 = -np.ones(n)
    
    # Without crossfade, pasting them creates a delta of 2.0 at boundary
    pasted = np.concatenate([b1, b2])
    assert abs(pasted[n] - pasted[n - 1]) == 2.0
    
    # With OLA crossfade between b1 tail and b2 head
    smoothed = apply_ola_crossfade([b1, b2], crossfade_samples=fade_len)
    
    # Boundary step must be smoothly bridged (max sample-to-sample difference much smaller)
    max_step = np.max(np.abs(np.diff(smoothed)))
    assert max_step < 0.05


def test_render_ledger_ola_produces_smooth_audio(tmp_path):
    """Verify render_ledger with ola=True produces clean continuous audio."""
    sr = 22050
    n = sr * 2
    ledger = {
        "schema": "res/1.1",
        "sample_rate": sr,
        "n_samples": n,
        "sections": [
            {
                "start": 0.0,
                "end": 1.0,
                "start_sample": 0,
                "partials": [{"frequency": 200.0, "amplitude": 0.8, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0}],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "start_sample": sr,
                "partials": [{"frequency": 200.0, "amplitude": 0.8, "phase": 3.1415, "envelope_decay": 0.0, "envelope_t0": 1.0}],
            },
        ],
    }
    
    out_raw = render_ledger(ledger, n, ola=False)
    out_ola = render_ledger(ledger, n, ola=True)
    
    # At boundary sample sr, raw has phase jump of pi (large diff)
    diff_raw = abs(out_raw[sr] - out_raw[sr - 1])
    diff_ola = abs(out_ola[sr] - out_ola[sr - 1])
    assert diff_ola < diff_raw
