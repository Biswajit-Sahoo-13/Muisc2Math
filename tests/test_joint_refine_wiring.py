"""P1.3 wiring: the joint TRF refiner joins the production loop.

``refine_section_parameters`` (clustered TRF least squares for partials
within 10 Hz - the detuned-808 / unison-beating fix) existed since Phase 4
but had no production caller: the lifter refines each partial alone, which
cannot decouple beating collisions. This pins the wiring: the section
worker runs it between the lifter and the tournament when the caller opts
in (``run_loop(joint_refine=True)``, CLI ``--joint-refine``), and the
beating fixture must measurably improve.

Seam under test: ``_process_section_task`` (the worker's public tuple
contract, exercised the way run_loop invokes it).
"""
from __future__ import annotations

import numpy as np

from src.res_loop import _process_section_task

SR = 22050


def _beating_section():
    """Two 808-style layers 4 Hz apart: the classic unison/beat collision."""
    n = SR
    t = np.arange(n) / SR
    y = (
        0.5 * np.cos(2 * np.pi * 220.0 * t - 0.2)
        + 0.4 * np.cos(2 * np.pi * 224.0 * t - 0.8)
    )
    partials = [
        {"frequency": 219.2, "amplitude": 0.44, "phase": 0.0,
         "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped",
         "start_time": 0.0, "end_time": 1.0},
        {"frequency": 224.8, "amplitude": 0.35, "phase": -0.5,
         "envelope_decay": 0.0, "envelope_t0": 0.0, "model": "damped",
         "start_time": 0.0, "end_time": 1.0},
    ]
    sec = {
        "start": 0.0, "end": 1.0, "start_sample": 0, "partial_count": 2,
        "partials": partials,
        "metrics": {"correlation": 0.5, "snr_db": 0.0,
                    "log_spectral_distance_db": 20.0, "seg_snr_db": 0.0,
                    "onset_corr": 0.0},
    }
    return y, sec


def test_joint_refine_improves_a_beating_section() -> None:
    y, sec = _beating_section()
    base, _, _ = _process_section_task(
        (sec, y, SR, 0, 0.0, False, False, False, 2, 12, False)
    )
    y2, sec2 = _beating_section()
    refined, _, _ = _process_section_task(
        (sec2, y2, SR, 0, 0.0, False, False, False, 2, 12, True)
    )
    c_base = float(base["metrics"]["correlation"])
    c_ref = float(refined["metrics"]["correlation"])
    assert c_ref > c_base + 0.02, (
        f"joint refine must decouple the beat: {c_base:.3f} -> {c_ref:.3f}"
    )


def test_joint_refine_never_loses_correlation() -> None:
    """On a clean section the refiner must be a no-op or a win."""
    y, sec = _beating_section()
    refined, _, _ = _process_section_task(
        (sec, y, SR, 0, 0.0, False, False, False, 2, 12, True)
    )
    assert float(refined["metrics"]["correlation"]) >= 0.5


def test_worker_tuple_without_the_new_slot_still_works() -> None:
    """Old-width tuples (pre joint_refine) must keep working: the worker
    contract has been extended positionally three times already."""
    y, sec = _beating_section()
    out, _, _ = _process_section_task(
        (sec, y, SR, 0, 0.0, False, False, False, 2, 12)
    )
    assert "metrics" in out
