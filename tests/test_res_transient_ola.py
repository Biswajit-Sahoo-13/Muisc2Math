"""Tests for Transient-Aware Overlap-Add (OLA) in res_loop (Phase 4)."""

import numpy as np
import pytest

from src.res_loop import apply_ola_crossfade


def test_transient_aware_ola_smooth_transition_at_drum_attack():
    """Verify that a drum/synth attack near a boundary is blended smoothly without clicks or pops."""
    sr = 22050
    fade_len = int(0.020 * sr)  # 20ms fade = 441 samples
    n = 2048

    # Block 1 is quiet background tone
    t1 = np.arange(n) / sr
    b1 = 0.1 * np.sin(2 * np.pi * 100 * t1)

    # Block 2 begins immediately with an explosive kick/snare attack
    t2 = np.arange(n) / sr
    b2 = np.zeros(n)
    b2[:100] = 1.0 * np.exp(-150.0 * t2[:100]) * np.sin(2 * np.pi * 80.0 * t2[:100])

    # In Phase 5 stabilization, smooth Hann crossfade eliminates step discontinuities
    result = apply_ola_crossfade([b1, b2], crossfade_samples=fade_len, sr=sr)

    # Verify no click/pop: sample-to-sample difference must be smooth
    boundary_region = result[n - fade_len : n + fade_len]
    max_step = np.max(np.abs(np.diff(boundary_region)))
    assert max_step < 0.10, f"Discontinuity {max_step} exceeds smooth Hann boundary bound"


def test_transient_aware_ola_smooths_when_no_transient():
    """When no transient is present, smooth Hann crossfade is maintained."""
    sr = 22050
    fade_len = int(0.015 * sr)
    n = 1024

    # Sustained static tones with a step jump at boundary
    b1 = 0.5 * np.ones(n)
    b2 = -0.5 * np.ones(n)

    result = apply_ola_crossfade([b1, b2], crossfade_samples=fade_len, sr=sr)
    max_step = np.max(np.abs(np.diff(result)))
    assert max_step < 0.05
