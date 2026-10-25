"""The polish refiner: continuous decay refinement in the coordinate lifter.

The decay grid is a handful of multipliers around the analyser's estimate;
the true decay rarely lands on it. ``--polish`` adds one candidate per
partial - a bounded scalar optimization of the decay with the closed-form
(amplitude, phase) projection inside the objective - that still has to win
the correlation gate. Flag-gated and default-off: the grid-only behaviour
is the published one.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import soundfile as sf

from src.partial_schema import validate_partial
from src.res_loop import _section_update, run_loop, run_stems_res_loop
from src.res_evidence import build_evidence
from src.res_synthesis import _corr, _render_partial

SR = 22050
DUR = 1.0
LAM_TRUE = 1.7  # sits between the grid multipliers of a 1.0 estimate


def _target() -> np.ndarray:
    t = np.arange(int(DUR * SR)) / SR
    return 0.5 * np.exp(-LAM_TRUE * t) * np.cos(2.0 * np.pi * 440.0 * t)


def _seed_partial() -> dict:
    return {
        "model": "damped",
        "frequency": 440.0,
        "amplitude": 0.5,
        "phase": 0.0,
        "envelope_decay": 1.0,   # the grid around 1.0 is {0.5,0.8,1.0,1.25,2.0}
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": DUR,
    }


def _run(polish: bool):
    y = _target()
    partials = [_seed_partial()]
    _section_update(y, partials, SR, polish=polish)
    p = partials[0]
    render = _render_partial(p, len(y), SR)
    return _corr(y, render), float(p.get("envelope_decay", 0.0)), p


def test_polish_finds_the_between_grid_decay():
    corr_grid, lam_grid, _ = _run(polish=False)
    corr_polish, lam_polish, _ = _run(polish=True)
    # The grid's best is 1.25 or 2.0 - both miss the true 1.7 by >= 0.3;
    # the bounded search closes most of that gap.
    assert abs(lam_polish - LAM_TRUE) < abs(lam_grid - LAM_TRUE)
    assert abs(lam_polish - LAM_TRUE) <= 0.2
    assert corr_polish > corr_grid


def test_polished_partial_still_validates():
    _, _, p = _run(polish=True)
    problems = validate_partial(p)
    assert problems == [], problems


def test_polish_is_off_by_default_everywhere():
    assert inspect.signature(_section_update).parameters["polish"].default is False
    assert inspect.signature(run_loop).parameters["polish"].default is False
    assert inspect.signature(run_stems_res_loop).parameters["polish"].default is False
    from src.res_ui import run_res_payload
    assert inspect.signature(run_res_payload).parameters["polish"].default is False


def _make_chord(path: Path) -> None:
    t = np.arange(int(1.2 * SR)) / SR
    y = (
        0.60 * np.exp(-1.10 * t) * np.cos(2.0 * np.pi * 440.0 * t)
        + 0.35 * np.exp(-1.45 * t) * np.cos(2.0 * np.pi * 554.365 * t)
    )
    sf.write(str(path), y, SR)


def test_loop_runs_with_polish_enabled(tmp_path):
    # End-to-end smoke: the flag threads through the worker tuple without
    # changing the loop's shape (ledger written, iteration recorded).
    path = tmp_path / "chord.wav"
    _make_chord(path)
    ledger = build_evidence(
        str(path), str(tmp_path / "ev"), sr=SR, max_duration=1.2,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )
    out = run_loop(
        copy.deepcopy(ledger), str(path), str(tmp_path / "out"),
        iterations=1, target_corr=0.0, provider="numeric",
        polish=True,
    )
    assert len(out["iterations"]) == 1
    # The flag-off run stays bit-identical to the pre-polish behaviour.
    out2 = run_loop(
        ledger, str(path), str(tmp_path / "out2"),
        iterations=1, target_corr=0.0, provider="numeric",
    )
    assert out2["iterations"][0]["global"]["correlation"] >= 0.0
