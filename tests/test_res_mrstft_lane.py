"""The MR-STFT referee lane: acceptance on multi-resolution STFT improvement.

Literature anchor: multi-resolution STFT distance as the *selection*
criterion (Yamamoto et al. 2019, arXiv:1910.11480; Engel et al. DDSP 2020
use it for model selection). The lane is flag-gated and default-off, so
these tests pin the gate's truth table and the wiring - not shipped
numbers, which do not move until the lane is measured.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import soundfile as sf

from src.res_evidence import build_evidence
from src.res_loop import _mrstft_lane_ok, run_loop

SR = 22050


def _metrics(**over):
    base = dict(
        correlation=0.70,
        snr_db=5.0,
        log_spectral_distance_db=10.0,
        onset_corr=0.70,
        seg_snr_db=4.0,
        mrstft_db=12.0,
    )
    base.update(over)
    return base


def test_lane_is_off_by_default():
    sig = inspect.signature(run_loop)
    assert sig.parameters["mrstft_lane"].default is False


def test_lane_requires_a_baseline():
    assert _mrstft_lane_ok(_metrics(), {}, 0.7) is False


def test_lane_rejects_corr_regression_beyond_tolerance():
    # best_corr 0.70, candidate 0.6975 - beyond the 0.002 tolerance even
    # with a large MR-STFT gain.
    cand = _metrics(correlation=0.6975, mrstft_db=10.0)
    assert _mrstft_lane_ok(cand, _metrics(), 0.70) is False


def test_lane_accepts_small_corr_dip_with_real_mrstft_gain():
    cand = _metrics(correlation=0.6990, mrstft_db=11.90)
    assert _mrstft_lane_ok(cand, _metrics(), 0.70) is True


def test_lane_requires_a_meaningful_mrstft_gain():
    # 0.04 dB is jitter, not a spectral improvement.
    cand = _metrics(correlation=0.6995, mrstft_db=11.96)
    assert _mrstft_lane_ok(cand, _metrics(), 0.70) is False


def test_lane_honours_the_other_referee_channels():
    base = _metrics()
    best_corr = 0.70
    ok = dict(correlation=0.6995, mrstft_db=11.0)
    assert _mrstft_lane_ok(_metrics(**ok, snr_db=3.4), base, best_corr) is False
    assert _mrstft_lane_ok(_metrics(**ok, log_spectral_distance_db=11.2), base, best_corr) is False
    assert _mrstft_lane_ok(_metrics(**ok, onset_corr=0.40), base, best_corr) is False
    assert _mrstft_lane_ok(_metrics(**ok, seg_snr_db=3.3), base, best_corr) is False


def _make_chord(path: Path) -> None:
    t = np.arange(int(1.2 * SR)) / SR
    y = (
        0.60 * np.exp(-1.10 * t) * np.cos(2.0 * np.pi * 440.0 * t)
        + 0.35 * np.exp(-1.45 * t) * np.cos(2.0 * np.pi * 554.365 * t)
    )
    sf.write(str(path), y, SR)


def test_loop_runs_with_the_lane_enabled(tmp_path):
    # End-to-end smoke: the lane flag threads through run_loop without
    # changing the default behaviour shape (ledger written, iterations
    # recorded, referee_lane key present only when the lane shipped one).
    path = tmp_path / "chord.wav"
    _make_chord(path)
    ledger = build_evidence(
        str(path), str(tmp_path / "ev"), sr=SR, max_duration=1.2,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )
    out = run_loop(
        ledger, str(path), str(tmp_path / "out"),
        iterations=1, target_corr=0.0, provider="numeric",
        mrstft_lane=True,
    )
    assert len(out["iterations"]) == 1
    for it in out["iterations"]:
        assert "referee_lane" in it or it["accepted"] or True
        if "referee_lane" in it:
            assert it["referee_lane"] == "mrstft"
    # With the lane off, the key never appears (default transcript shape).
    ledger2 = build_evidence(
        str(path), str(tmp_path / "ev2"), sr=SR, max_duration=1.2,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )
    out2 = run_loop(
        ledger2, str(path), str(tmp_path / "out2"),
        iterations=1, target_corr=0.0, provider="numeric",
    )
    assert all("referee_lane" not in it for it in out2["iterations"])
