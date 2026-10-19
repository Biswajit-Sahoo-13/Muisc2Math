"""Streamlit AppTest boot + RES tab rendering checks (app.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 22050

APP = Path(__file__).resolve().parent.parent / "app.py"


def _sandbox_payload(tmp_path: Path):
    """A real RES run in a sandboxed root, safe to inject into the job registry."""
    from src.res_ui import run_res_payload

    t = np.arange(int(1.1 * SR)) / SR
    y = np.zeros_like(t)
    for f0, a, lam in [(330.0, 0.30, 1.0), (430.0, 0.20, 1.4)]:
        for m in range(1, 5):
            y += (a / m) * np.exp(-lam * t) * np.sin(2 * np.pi * f0 * m * t)
    y += 0.002 * np.random.default_rng(0).standard_normal(y.size)
    wav = tmp_path / "chord.wav"
    sf.write(str(wav), y, SR)

    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    return run_res_payload(
        wav.read_bytes(), "chord.wav",
        duration=1.0, iterations=1, bank=True, root=root,
    )


def test_app_boots_without_exception(tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]


def test_res_results_render_through_ui(tmp_path):
    import src.res_ui as rui
    from streamlit.testing.v1 import AppTest

    payload = _sandbox_payload(tmp_path)
    rui._JOBS.clear()
    rui._JOBS["test:done"] = {"state": "done", "payload": payload}
    try:
        at = AppTest.from_file(APP, default_timeout=600)
        at.run()
        assert at.exception == [], [str(e.value) for e in at.exception]
        bkeys = [b.key for b in at.download_button if b.key]
        assert "res_dl_wav" in bkeys and "res_dl_ledger" in bkeys, bkeys
        labels = [m.label for m in at.metric]
        for want in ("correlation", "SNR", "LSD", "segmental SNR", "onset corr"):
            assert want in labels, labels
    finally:
        rui._JOBS.clear()
