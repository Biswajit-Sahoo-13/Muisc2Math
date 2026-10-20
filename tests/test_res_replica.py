"""The renderer must be a replica of the analyser - tested non-tautologically.

The research doc claims "the renderer is an exact replica of the SMS engine's
synthesis, verified test-by-test (per-section corr(mine, SMS) = 0.98-1.00)".
Nothing tested it, and the obvious test is tautological: the loop overwrites
``sec["metrics"]`` with the RES renderer's own score
(``res_loop.py`` - ``sec["metrics"] = _score(y_sec, recon_sec, sr)``), so
comparing a section's recorded metric against a fresh render of that
section's partials always agrees, whatever the renderer actually does.

The only comparison that can detect a replica gap is against the ANALYSER's
own output. So this test re-runs the analyser on a known signal, keeps its
``sinusoid_audio``, and compares it with the RES render of the very same
fitted parameters.

Two separate invariants are checked:

1. the replica itself (renderer vs analyser), and
2. that the replica holds under BEAT-QUANTISED section boundaries - where
   the section origins differ from the analysis-window origins the analyser
   fitted against, and where a partial's support can leak across a boundary.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.res_evidence import _rephase_to_section
from src.res_loop import render_ledger
from src.sinusoidal_model import sinusoidal_reconstruction

SR = 22050
WIN_S = 2.5
ST_HW = (2048 / 2.0) / SR


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-18 else 0.0


def _analyser_output(y: np.ndarray):
    """Run the analyser the way build_evidence does (on the HPSS harmonic part)."""
    import librosa

    yh, _ = librosa.effects.hpss(y)
    res = sinusoidal_reconstruction(
        yh, SR, n_fft=2048, hop_length=256, threshold=0.01, max_tracks=400,
        add_noise_layer=False, min_residual_ratio=1e-6, seed=0,
        window_s=WIN_S, glide_tolerance_hz=6.0,
    )
    return np.asarray(res["sinusoid_audio"], float), res["fitted_parameters"]


def _ledger_like(partials, bounds, n):
    """A ledger-shaped structure: overlap assignment + phase re-referencing."""
    sections = []
    for i in range(len(bounds) - 1):
        ws, we = bounds[i], bounds[i + 1]
        lo, hi = ws / SR, we / SR
        sp = []
        for p in partials:
            ps = float(p.get("start_time", p.get("envelope_t0", 0.0)))
            pe = float(p.get("end_time", p.get("envelope_t0", 0.0)))
            if pe <= lo or ps >= hi:
                continue
            sp.append(_rephase_to_section(p, lo))
        sections.append({"start": lo, "end": hi, "start_sample": ws,
                         "end_sample": we, "partials": sp})
    return {"schema": "res/1.1", "sample_rate": SR, "duration": n / SR,
            "n_samples": n, "window_s": WIN_S, "stft_half_window_s": ST_HW,
            "sections": sections}


@pytest.fixture(scope="module")
def analysed():
    n = int(SR * 6.0)
    t = np.arange(n, dtype=np.float64) / SR
    y = (0.45 * np.sin(2.0 * np.pi * 196.0 * t)
         + 0.22 * np.sin(2.0 * np.pi * 392.0 * t) * np.exp(-1.2 * t)
         + 0.15 * np.sin(2.0 * np.pi * 587.0 * t) * np.exp(-2.5 * t))
    y = y * (0.6 + 0.4 * np.sign(np.sin(2.0 * np.pi * 2.0 * t)))
    return y, _analyser_output(y), n


def test_renderer_replicates_the_analyser(analysed):
    """With aligned sections, the render must equal the analyser's own output."""
    y, (sms, partials), n = analysed
    win = int(round(WIN_S * SR))
    bounds = list(range(0, n, win))
    if bounds[-1] != n:
        bounds.append(n)
    recon = render_ledger(_ledger_like(partials, bounds, n), n)
    assert _corr(sms, recon) > 0.98, (
        f"renderer/analyser replica correlation is {_corr(sms, recon):.4f}, "
        f"below the 0.98 the research doc claims"
    )


def test_replica_holds_under_beat_quantised_sections(analysed):
    """Section origins that differ from the analysis windows must not break it.

    A partial's rendered support extends past its last tracked frame (hold
    plus release). The analyser clips that at its analysis window; a section
    assignment by frame times lets the tail leak into the next section, where
    the analyser never rendered it. That leak cost 0.06 of replica fidelity
    and 0.05 of correlation on the beat-quantised fixture.
    """
    y, (sms, partials), n = analysed
    beat = [0, int(1.1 * SR), int(2.3 * SR), int(3.4 * SR), int(4.6 * SR), n]
    beat = sorted(set(beat))
    recon = render_ledger(_ledger_like(partials, beat, n), n)
    got = _corr(sms, recon)
    assert got > 0.95, (
        f"replica correlation under beat-quantised sections is {got:.4f}; "
        f"partial support is leaking across section boundaries"
    )


def test_phase_referencing_sign_is_correct(analysed):
    """Re-referencing must not rotate a partial relative to the analyser.

    The shift is ``2*pi*f*(origin_window - origin_section)``; the opposite
    sign is off by ``4*pi*f*delta``, which for a mid-range note over a
    fraction of a second is hundreds of cycles.
    """
    _, (_, partials), n = analysed
    checked = 0
    for p in partials:
        if p.get("model") == "glide" or p.get("phase_origin") is None:
            continue
        origin = float(p["phase_origin"])
        section = origin + 0.37
        q = _rephase_to_section(p, section)
        f = float(p["frequency"])
        expected = float(p["phase"]) + 2.0 * np.pi * f * (origin - section)
        assert abs(float(q["phase"]) - expected) < 1e-5
        checked += 1
    assert checked > 0, "no non-glide partials to check"
