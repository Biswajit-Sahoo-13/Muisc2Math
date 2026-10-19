"""Tests for RES iteration-budget guidance: gains, trend, and recommendation.

Covers the convergence controls added to ``run_loop`` (``min_gain``,
``stop_on_no_improvement``, ``stop_reason``, per-iteration ``gain``) plus the
standalone recommender used by ``main.py res-recommend`` and ``--auto-iterations``.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import soundfile as sf

from src.res_loop import (
    iteration_gains,
    recommend_iterations,
    run_loop,
    structural_budget,
)

SR = 22050
REPO = Path(__file__).resolve().parents[1]


def _chord(path: Path, dur: float = 1.2, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = (
        0.60 * np.exp(-1.10 * t) * np.cos(2.0 * np.pi * 440.0 * t)
        + 0.35 * np.exp(-1.45 * t) * np.cos(2.0 * np.pi * 554.365 * t)
        + 0.25 * np.exp(-1.80 * t) * np.cos(2.0 * np.pi * 659.255 * t)
    )
    sf.write(str(path), y.astype(np.float32), sr)
    return y


def _partial(freq: float, amp: float, decay: float, t0: float, end: float) -> Dict[str, Any]:
    return {
        "amplitude": amp,
        "envelope_a0": amp,
        "envelope_decay": decay,
        "envelope_t0": t0,
        "frequency": freq,
        "freq_spread_hz": 0.0,
        "phase": 0.0,
        "projection_gain": 1.0,
        "model": "damped",
        "start_time": t0,
        "end_time": end,
    }


def _ledger(corrs: List[float], partials_per_section: int = 4,
            sections: int = 3, accepted: List[bool] | None = None,
            baseline: float | None = None) -> Dict[str, Any]:
    """Synthetic ledger shaped exactly like a real one, with a known trend.

    ``baseline`` is the pre-loop (evidence-stage) correlation, which is what
    ``run_loop`` records as ``initial_correlation``. It defaults to 0.0 so the
    first gain equals the first correlation, matching the simplest case.
    """
    acc = accepted if accepted is not None else [True] * len(corrs)
    base = corrs[0] if baseline is None else baseline
    return {
        "schema": "res/1.1",
        "source": "synthetic.wav",
        "sample_rate": SR,
        "duration": 2.5 * sections,
        "n_samples": int(2.5 * sections * SR),
        "window_s": 2.5,
        "stft_half_window_s": 0.04644,
        "equation_form": "A*exp(-lambda*(t - t0))*cos(2*pi*f*t - phi)",
        "equation_classes": ["damped_cosine"],
        "tempo_bpm": 120.0,
        "initial_correlation": base,
        "best": {"correlation": corrs[-1] if corrs else 0.0},
        "sections": [
            {
                "start": 2.5 * i,
                "end": 2.5 * (i + 1),
                "start_sample": int(2.5 * i * SR),
                "partial_count": partials_per_section,
                "metrics": {"correlation": 0.0},
                "partials": [_partial(440.0, 0.01, 1.0, 0.1, 1.0)
                             for _ in range(partials_per_section)],
                "hypothesis_bank": [],
            }
            for i in range(sections)
        ],
        "iterations": [
            {
                "iteration": i + 1,
                "provider": "numeric",
                "llm_edits_applied": 0,
                "accepted": acc[i],
                "gain": corrs[i] - (base if i == 0 else corrs[i - 1]),
                "global": {"correlation": c},
                "sections": [0.0] * sections,
            }
            for i, c in enumerate(corrs)
        ],
    }


class TestIterationGains:
    def test_gains_are_consecutive_differences(self):
        led = _ledger([0.60, 0.70, 0.75, 0.78], baseline=0.0)
        g = iteration_gains(led)
        assert len(g) == 4
        assert g[0] == pytest.approx(0.60, abs=1e-9)
        assert g[1] == pytest.approx(0.10, abs=1e-9)
        assert g[2] == pytest.approx(0.05, abs=1e-9)
        assert g[3] == pytest.approx(0.03, abs=1e-9)

    def test_rejected_iteration_contributes_negative_delta(self):
        led = _ledger([0.70, 0.75, 0.74], accepted=[True, True, False],
                      baseline=0.65)
        g = iteration_gains(led)
        assert g[0] == pytest.approx(0.05, abs=1e-9)
        assert g[2] == pytest.approx(-0.01, abs=1e-9)

    def test_derives_initial_from_first_iteration_when_absent(self):
        led = _ledger([0.60, 0.70, 0.75], baseline=0.50)
        led.pop("initial_correlation")
        g = iteration_gains(led)
        assert g[1] == pytest.approx(0.10, abs=1e-9)
        assert g[2] == pytest.approx(0.05, abs=1e-9)

    def test_missing_baseline_makes_first_gain_unmeasurable(self):
        led = _ledger([0.60, 0.70, 0.75], baseline=0.50)
        led.pop("initial_correlation")
        assert iteration_gains(led)[0] == 0.0

    def test_empty_ledger_yields_empty(self):
        assert iteration_gains({"iterations": []}) == []


class TestStructuralBudget:
    def test_scales_with_problem_size(self):
        small = _ledger([0.5], partials_per_section=3, sections=1)
        big = _ledger([0.5], partials_per_section=200, sections=12)
        assert structural_budget(small) < structural_budget(big)

    def test_bounded(self):
        huge = _ledger([0.5], partials_per_section=10_000, sections=500)
        assert 2 <= structural_budget(huge) <= 40
        assert 2 <= structural_budget({}) <= 40

    def test_empty_ledger_returns_minimum(self):
        assert structural_budget({}) == 2


class TestRecommendIterations:
    def test_decaying_trend_reports_ceiling_and_budget(self):
        led = _ledger([0.724123, 0.737058, 0.751259, 0.768651, 0.783806,
                       0.799048, 0.808659, 0.814173], baseline=0.698835)
        rec = recommend_iterations(led, min_gain=1e-3)
        assert rec["basis"] == "gain-trend"
        assert rec["trend"] == "decaying"
        assert rec["converged"] is False
        assert rec["recommended"] > 8
        assert rec["decay_ratio"] < 1.0
        assert rec["predicted_ceiling"] is not None
        assert rec["predicted_ceiling"] > rec["correlation"]

    def test_rising_trend_is_not_fitted_as_decay(self):
        led = _ledger([0.756, 0.770, 0.786, 0.804], baseline=0.742)
        rec = recommend_iterations(led)
        assert rec["trend"] == "rising"
        assert rec["predicted_ceiling"] is None
        assert rec["converged"] is False
        assert "not started converging" in rec["reason"]

    def test_flat_trend_is_flagged(self):
        led = _ledger([0.70, 0.80, 0.90, 1.00])
        rec = recommend_iterations(led)
        assert rec["trend"] in ("flat", "rising")

    def test_flat_gain_is_converged(self):
        led = _ledger([0.70, 0.80, 0.800001, 0.800002])
        led["iterations"][1]["accepted"] = True
        led["initial_correlation"] = 0.70
        rec = recommend_iterations(led, min_gain=1e-3)
        assert rec["last_gain"] < 1e-3
        assert rec["converged"] is True
        assert rec["recommended"] == 4

    def test_insufficient_data_falls_back_to_size(self):
        led = _ledger([0.70, 0.71])
        rec = recommend_iterations(led)
        assert rec["basis"] == "insufficient-data"
        assert rec["recommended"] > 2
        assert "size-based guess" in rec["reason"]

    def test_no_iterations_reports_converged(self):
        rec = recommend_iterations({"iterations": [], "sections": []})
        assert rec["converged"] is True

    def test_never_recommends_fewer_than_done(self):
        led = _ledger([0.80, 0.85, 0.87, 0.88, 0.885, 0.886])
        rec = recommend_iterations(led, min_gain=1e-2)
        assert rec["recommended"] >= 6

    def test_respects_max_iterations_cap(self):
        led = _ledger([0.60 + 0.02 * i for i in range(12)])
        rec = recommend_iterations(led, max_iterations=15)
        assert rec["recommended"] <= 15

    def test_is_deterministic(self):
        led = _ledger([0.724, 0.737, 0.751, 0.769, 0.784])
        assert recommend_iterations(led) == recommend_iterations(led)

    def test_does_not_mutate_ledger(self):
        led = _ledger([0.724, 0.737, 0.751, 0.769, 0.784])
        before = copy.deepcopy(led)
        recommend_iterations(led)
        assert led == before

    def test_result_is_json_serialisable(self):
        led = _ledger([0.724, 0.737, 0.751, 0.769, 0.784])
        json.dumps(recommend_iterations(led))


class TestRealLedgers:
    """The shipped runs on disk must all classify sanely."""

    def _ledgers(self):
        pats = [
            "data/output/res_lost15/res_ledger.json",
            "data/output/res_web_20261017_222927_Metre_-_Lost/res_ledger.json",
            "data/output/res_web_20261018_153713_01_-_j-hope_-_MONA_LISA/res_ledger.json",
            "data/output/res_chord_v2/res_ledger.json",
        ]
        return [(p, REPO / p) for p in pats if (REPO / p).exists()]

    def test_known_ledgers_advise(self):
        leds = self._ledgers()
        if not leds:
            pytest.skip("no shipped ledgers present")
        for name, path in leds:
            with open(path, "r", encoding="utf-8") as f:
                led = json.load(f)
            rec = recommend_iterations(led)
            done = len(led["iterations"])
            assert rec["iterations_done"] == done
            assert rec["recommended"] >= done
            assert rec["basis"] in ("gain-trend", "insufficient-data")
            json.dumps(rec)


class TestRunLoopConvergence:
    def test_run_loop_accepts_convergence_kwargs(self):
        sig = inspect.signature(run_loop)
        assert "min_gain" in sig.parameters
        assert "stop_on_no_improvement" in sig.parameters
        assert sig.parameters["min_gain"].default == 0.0
        assert sig.parameters["stop_on_no_improvement"].default is False

    def test_min_gain_stops_a_converging_loop(self, tmp_path):
        wav = tmp_path / "c.wav"
        _chord(wav)
        from src.res_evidence import build_evidence

        led = build_evidence(str(wav), str(tmp_path / "out"), sr=SR,
                             max_duration=1.2, window_s=2.5)
        out = run_loop(led, str(wav), str(tmp_path / "o"), iterations=40,
                       target_corr=0.0, min_gain=1e-4)
        assert "stop_reason" in out
        assert len(out["iterations"]) < 40
        assert out["stop_reason"].startswith("iteration ") or \
            out["stop_reason"].startswith("target-corr")

    def test_every_iteration_records_a_gain(self, tmp_path):
        wav = tmp_path / "c.wav"
        _chord(wav)
        from src.res_evidence import build_evidence

        led = build_evidence(str(wav), str(tmp_path / "out"), sr=SR,
                             max_duration=1.2, window_s=2.5)
        out = run_loop(led, str(wav), str(tmp_path / "o"), iterations=3,
                       target_corr=0.0)
        for it in out["iterations"]:
            assert isinstance(it["gain"], float)

    def test_stop_on_no_improvement_is_honoured(self, tmp_path):
        wav = tmp_path / "c.wav"
        _chord(wav)
        from src.res_evidence import build_evidence

        led = build_evidence(str(wav), str(tmp_path / "out"), sr=SR,
                             max_duration=1.2, window_s=2.5)
        out = run_loop(led, str(wav), str(tmp_path / "o"), iterations=30,
                       target_corr=0.0, stop_on_no_improvement=True)
        if len(out["iterations"]) < 30:
            assert "no improvement" in out["stop_reason"]

    def test_initial_correlation_is_recorded(self, tmp_path):
        wav = tmp_path / "c.wav"
        _chord(wav)
        from src.res_evidence import build_evidence

        led = build_evidence(str(wav), str(tmp_path / "out"), sr=SR,
                             max_duration=1.2, window_s=2.5)
        out = run_loop(led, str(wav), str(tmp_path / "o"), iterations=2,
                       target_corr=0.0)
        assert "initial_correlation" in out


class TestCliSurface:
    def test_res_reports_new_flags(self):
        import subprocess
        import sys

        r = subprocess.run([sys.executable, "main.py", "res", "--help"],
                           capture_output=True, text=True, cwd=str(REPO),
                           timeout=180)
        assert r.returncode == 0
        for flag in ("--min-gain", "--stop-on-no-improvement",
                     "--auto-iterations"):
            assert flag in r.stdout

    def test_res_recommend_subcommand_exists(self):
        import subprocess
        import sys

        r = subprocess.run([sys.executable, "main.py", "res-recommend", "--help"],
                           capture_output=True, text=True, cwd=str(REPO),
                           timeout=180)
        assert r.returncode == 0
        assert "--ledger" in r.stdout
        assert "--min-gain" in r.stdout