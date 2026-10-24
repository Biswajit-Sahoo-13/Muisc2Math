"""Unit tests for Phase 5: Stabilization and Perceptual Tuning.

Verifies:
1. OLA crossfade always uses smooth Hann crossfade (no hard cuts).
2. LPC Noise hypothesis is strictly gated by SFM and ZCR (disabled on clean 808s).
3. Clustered refinement frequency shifts are bounded to <= 2.0 Hz.
4. Acceptance gate uses Log-Spectral Distance / spectral loss alongside correlation.
"""

import numpy as np

from src.res_loop import apply_ola_crossfade, refine_section_parameters
from src.res_bank import (
    lpc_noise_hypothesis,
    spectral_flatness,
    zero_crossing_rate,
    tournament_section,
)
from src.res_synthesis import _render_partial, _corr


def test_ola_crossfade_always_smooth_hann_no_hard_cuts():
    """Verify apply_ola_crossfade always applies smooth Hann crossfade without clicks or hard cuts."""
    sr = 22050
    fade_len = int(0.015 * sr)
    n = 2048

    # Block 1 ending at 1.0, Block 2 starting with high attack and step
    t = np.arange(n) / sr
    b1 = 0.5 * np.ones(n)
    b2 = -0.5 * np.ones(n)
    # Add an energy spike to simulate a drum/synth onset at head of b2
    b2[:50] += 0.8 * np.exp(-50.0 * t[:50])

    out = apply_ola_crossfade([b1, b2], crossfade_samples=fade_len, sr=sr)

    # In smooth Hann crossfade, max sample-to-sample difference must be bounded and smooth
    max_step = np.max(np.abs(np.diff(out)))
    assert max_step < 0.08, f"Step discontinuity {max_step} exceeds smooth Hann bound"


def test_lpc_noise_gated_on_clean_808_sub_bass():
    """Verify LPC noise is strictly disabled when the residual is purely tonal (like a clean 808)."""
    sr = 22050
    n = 4096
    t = np.arange(n) / sr

    # Clean 808 sub-bass at 55 Hz
    clean_808 = 0.8 * np.sin(2.0 * np.pi * 55.0 * t)

    sfm = spectral_flatness(clean_808)
    zcr = zero_crossing_rate(clean_808)

    assert sfm < 0.05, f"808 SFM {sfm} should be very low"
    assert zcr < 0.05, f"808 ZCR {zcr} should be very low"

    # In tournament_section, a clean 808 section should not accept or run lpc_noise
    class_a = [{
        "model": "damped", "amplitude": 0.1, "envelope_decay": 0.0,
        "envelope_t0": 0.0, "frequency": 120.0, "phase": 0.0,
    }]
    recs = tournament_section(clean_808, class_a, sr=sr, start_sample=0)
    lpc_accepted = [r for r in recs if r.get("class") == "lpc_noise" and r.get("accepted")]
    assert len(lpc_accepted) == 0, "LPC noise must NOT be accepted on a clean 808 tonal signal"


def test_lpc_noise_activates_on_vocal_noise_residual():
    """Verify LPC noise activates when residual has high SFM and ZCR (speech/breath/noise)."""
    sr = 22050
    n = 4096
    rng = np.random.default_rng(123)
    # High frequency noisy residual (simulating vocal fricative / breath)
    noisy_resid = 0.3 * rng.standard_normal(n)

    sfm = spectral_flatness(noisy_resid)
    zcr = zero_crossing_rate(noisy_resid)

    assert sfm > 0.20, f"Noisy residual SFM {sfm} should be high"
    assert zcr > 0.20, f"Noisy residual ZCR {zcr} should be high"

    variant = lpc_noise_hypothesis(noisy_resid, sr=sr, start_sample=0, order=12)
    assert variant.partials, "LPC noise must generate candidate on vocal-like noise residual"


def test_clustered_refinement_dampened_frequency_shift():
    """Verify that clustered refinement frequency shifts are bounded to <= 2.0 Hz to prevent 808 phasing."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr

    # Ground truth: two beating partials at 55.0 Hz and 57.5 Hz
    y_sec = 0.6 * np.cos(2.0 * np.pi * 55.0 * t) + 0.4 * np.cos(2.0 * np.pi * 57.5 * t)

    # Initial greedy estimates with 1 Hz error
    p1 = {"frequency": 54.2, "amplitude": 0.55, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"}
    p2 = {"frequency": 58.3, "amplitude": 0.38, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"}
    partials = [p1, p2]

    f1_init = p1["frequency"]
    f2_init = p2["frequency"]

    refine_section_parameters(y_sec, partials, sr, max_nfev=30)

    # Frequency shifts must be <= 2.0 Hz
    assert abs(p1["frequency"] - f1_init) <= 2.0 + 1e-4, f"Shift {abs(p1['frequency'] - f1_init)} exceeds 2.0 Hz"
    assert abs(p2["frequency"] - f2_init) <= 2.0 + 1e-4, f"Shift {abs(p2['frequency'] - f2_init)} exceeds 2.0 Hz"


def test_acceptance_gate_evaluates_lsd():
    """Verify _accept_candidate accepts candidates that improve Log-Spectral Distance and residual error."""
    from src.res_bank import _accept_candidate

    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    y_sec = 0.5 * np.cos(2.0 * np.pi * 440.0 * t) + 0.3 * np.cos(2.0 * np.pi * 880.0 * t)

    p1 = {"frequency": 440.0, "amplitude": 0.5, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"}
    partials = [p1]

    p2 = {"frequency": 880.0, "amplitude": 0.3, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped"}
    fresh = [p2]

    def render_state():
        sig = np.zeros(n)
        for p in partials:
            sig += _render_partial(p, n, sr)
        return sig

    e_y = float(np.sum(y_sec ** 2))
    cur_err = float(np.sum((y_sec - render_state()) ** 2))
    cur_corr = _corr(y_sec, render_state())

    accepted, err_after, corr_after = _accept_candidate(
        y_sec, partials, fresh, param_count=3, vari_err=0.0,
        e_y=e_y, cur_err=cur_err, cur_corr=cur_corr, render_state=render_state, sr=sr
    )

    assert accepted, "Candidate that improves spectral envelope and error must be accepted"
    assert err_after < cur_err
    assert len(partials) == 2
