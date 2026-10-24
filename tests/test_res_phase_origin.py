"""The RES renderer must be a replica of the SMS engine's synthesis.

``docs/res_research.md`` states the invariant: "the renderer is an exact
replica of the SMS engine's synthesis, verified test-by-test (per-section
corr(mine, SMS) = 0.98-1.00)". Nothing tested it, so it silently rotted.

What broke
----------
The SMS engine analyses the signal in fixed windows and projects every
partial against that *window's* local time axis (``t_axis = arange(n)/sr``),
so the phase it returns is window-relative. The RES renderer draws every
partial on the *section's* local axis - which is what the equation sheet
documents (``x = t - t_s``). The two origins agree only when the section
boundaries coincide with the analysis windows.

Beat-quantised sections move the boundaries, so each partial was rotated by
``2*pi*f*(window origin - section origin)``: hundreds of full cycles for a
mid-range note, i.e. a random phase. The reconstruction collapsed to noise
while every individual partial was still correct.

The failure was invisible because the ledger's section metrics were computed
from the SMS engine's own render, so they reported 0.77 correlation for a
section whose actual re-render scored -0.10. These tests pin the invariant
that was never pinned: the ledger's metric must describe the ledger's
partials.
"""
from __future__ import annotations

import numpy as np

from src.res_evidence import _rephase_to_section, build_evidence
from src.res_loop import render_ledger, run_loop
from src.res_synthesis import _render_partial


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    if n < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-18 else 0.0


def _test_signal(sr: int = 22050, dur: float = 8.0) -> np.ndarray:
    """Tonal material with a clear pulse, so beat quantisation engages."""
    n = int(sr * dur)
    t = np.arange(n, dtype=np.float64) / sr
    y = 0.45 * np.sin(2.0 * np.pi * 196.0 * t)
    y += 0.22 * np.sin(2.0 * np.pi * 392.0 * t) * np.exp(-1.2 * t)
    y += 0.15 * np.sin(2.0 * np.pi * 587.0 * t) * np.exp(-2.5 * t)
    pulse = 0.5 + 0.5 * np.sign(np.sin(2.0 * np.pi * 2.0 * t))
    return y * (0.55 + 0.45 * pulse)


# ----------------------------------------------------------------------
# The re-referencing law
# ----------------------------------------------------------------------

def test_rephase_shifts_by_the_origin_difference():
    """phase_section = phase_window + 2*pi*f*(origin_window - origin_section)."""
    p = {
        "model": "constant", "phase": 0.0, "frequency": 440.0,
        "phase_origin": 2.5,
    }
    q = _rephase_to_section(p, 5.0)
    expected = 2.0 * np.pi * 440.0 * (2.5 - 5.0)
    assert abs(float(q["phase"]) - expected) < 1e-5


def test_rephase_is_identity_when_the_origins_agree():
    p = {
        "model": "constant", "phase": 0.7, "frequency": 440.0,
        "phase_origin": 2.5,
    }
    q = _rephase_to_section(p, 2.5)
    assert abs(float(q["phase"]) - 0.7) < 1e-9


def test_rephase_of_a_glide_uses_the_reference_term():
    """A glide's phase rides on (f_ref - f_guess)*t, not on f alone."""
    p = {
        "model": "glide", "phase": 0.0, "frequency": 440.0,
        "phase_ref_hz": 400.0, "phase_origin": 2.0,
        "trajectory": {"t": [2.0], "f_hz": [440.0], "a": [1.0]},
    }
    q = _rephase_to_section(p, 5.0)
    expected = 2.0 * np.pi * (440.0 - 400.0) * (2.0 - 5.0)
    assert abs(float(q["phase"]) - expected) < 1e-5


def test_rephase_leaves_legacy_partials_alone():
    """A partial with no recorded origin has no window to re-reference."""
    p = {"model": "constant", "phase": 0.3, "frequency": 440.0}
    assert _rephase_to_section(p, 5.0) is p


# ----------------------------------------------------------------------
# The ledger's metric must describe the ledger's partials
# ----------------------------------------------------------------------

def test_section_metric_matches_a_fresh_render_of_the_same_partials(tmp_path):
    """Re-rendering a section must reproduce its recorded metric.

    This is the exact shape of the bug: the ledger reported a healthy
    correlation for a section whose real re-render was uncorrelated with the
    audio, because the metric had been computed against a different signal.
    """
    sr = 22050
    y = _test_signal(sr=sr)
    ledger = build_evidence(
        y, str(tmp_path), sr=sr, max_duration=8.0, window_s=2.5,
    )
    st_hw = float(ledger.get("stft_half_window_s", 0.0))
    assert ledger["sections"], "evidence produced no sections"

    for sec in ledger["sections"]:
        w_start = int(sec["start_sample"])
        w_end = int(sec.get("end_sample", round(sec["end"] * sr)))
        y_sec = y[w_start:w_end]
        r = np.zeros(len(y_sec))
        for p in sec["partials"]:
            r += _render_partial(
                p, len(y_sec), sr, start_sample=w_start,
                stft_half_window_s=st_hw,
            )
        fresh = _corr(y_sec, r)
        recorded = float(sec["metrics"]["correlation"])
        assert abs(fresh - recorded) < 0.05, (
            f"section at {w_start}: ledger says {recorded:.4f} but the "
            f"partials actually render to {fresh:.4f}"
        )


def test_beat_quantised_sections_still_reconstruct(tmp_path):
    """A reconstruction whose section origins differ from the analysis
    windows must still correlate with the source."""
    sr = 22050
    y = _test_signal(sr=sr)
    ledger = build_evidence(
        y, str(tmp_path), sr=sr, max_duration=8.0, window_s=2.5,
    )
    n = int(ledger["n_samples"])
    recon = render_ledger(ledger, n)
    assert _corr(y[:n], recon) > 0.5, (
        "the rendered ledger does not correlate with its own source"
    )


# ----------------------------------------------------------------------
# The acceptance gate must live in the loop's own metric space
# ----------------------------------------------------------------------

def test_loop_baseline_is_its_own_render_not_the_evidence_score(tmp_path):
    """The loop must not gate its iterations on the evidence score.

    ``ledger["best"]["correlation"]`` describes the SMS engine's own render -
    a different signal from what the loop renders. Gating on it rejected
    every iteration on beat-quantised material (evidence 0.67, the loop's own
    render 0.01) and shipped the broken initial state.
    """
    sr = 22050
    y = _test_signal(sr=sr)
    ledger = build_evidence(
        y, str(tmp_path), sr=sr, max_duration=6.0, window_s=2.5,
    )
    # Inflate the evidence score: if the loop trusts it, nothing improves.
    ledger["best"]["correlation"] = 0.999
    out = tmp_path / "loop"
    out.mkdir()
    result = run_loop(ledger, y, str(out), iterations=2, target_corr=0.0)
    assert result.get("evidence_correlation") == 0.999, (
        "the evidence score must be preserved for the record"
    )
    assert float(result["best"]["correlation"]) < 0.999, (
        "the final score must be the loop's own measurement, not the "
        "evidence score it was handed"
    )
    assert float(result["best"]["correlation"]) >= float(
        result["initial_correlation"]
    ) - 1e-6, (
        "the loop must never ship a state worse than its own baseline"
    )
