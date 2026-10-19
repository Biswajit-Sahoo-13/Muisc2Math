"""Tests for the browser-facing RES payload (src.res_ui)."""

from __future__ import annotations

import io
import math
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 22050


def _chord_wav(tmp_path: Path, dur: float = 1.1) -> bytes:
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for f0, a, lam in [(330.0, 0.30, 1.0), (430.0, 0.20, 1.4)]:
        for m in range(1, 5):
            y += (a / m) * np.exp(-lam * t) * np.sin(2 * np.pi * f0 * m * t)
    y += 0.002 * np.random.default_rng(0).standard_normal(y.size)
    path = tmp_path / "chord.wav"
    sf.write(str(path), y, SR)
    return path.read_bytes()


def test_run_res_payload_end_to_end(tmp_path):
    from src.res_ui import run_res_payload

    # fresh sandbox so the repo's data/output stays untouched
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)

    blob = _chord_wav(tmp_path)
    p = run_res_payload(
        blob, "chord.wav", duration=1.0, iterations=2, bank=True,
        root=root,
    )
    assert p["trail"], "payload must carry the iteration trail"
    assert len(p["trail"]) == 2
    assert p["best"]["correlation"] >= p["start_corr"] - 1e-9
    assert p["best"]["correlation"] > 0.8
    assert p["sections"] and p["sections"][0]["partials"] > 0
    assert "section 0" in p["equations"]
    for rec in p["accepted"]:
        assert rec["class"] in {"harmonic_string", "percussive_bundle", "filtered_noise", "lpc_noise"}
        assert rec["section"] < len(p["sections"])
        assert rec["residual_cut"] > 0.0
    assert Path(p["ledger_path"]).exists()
    assert p["best_bytes"].startswith(b"RIFF")

    # original span WAV: parse it back and check the length matches duration
    with wave.open(io.BytesIO(p["orig_bytes"]), "rb") as wf:
        frames = wf.getnframes()
        assert abs(frames / wf.getframerate() - p["duration"]) < 0.05
    assert p["duration"] <= 1.0 + 1e-6
