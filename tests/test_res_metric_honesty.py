"""Guard the published metrics against the measurement bug that hid them.

Reading the source FLAC through ``sf.read(...).mean(axis=1)`` (stereo->mono by
averaging, at the file's native 44.1 kHz) and comparing it against a 22.05 kHz
mono render produces a correlation near 0.05 instead of the true ~0.72, because
the two signals are neither at the same rate nor phase-aligned. Every published
RES number must be measured the same way the engine measures it: the source
resampled once through ``load_audio(target_sr=...)`` and compared sample-exact.
"""
import numpy as np
import pytest
import soundfile as sf

from src.analysis import load_audio

SR = 22050
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
SRC = ROOT / "data/uploads/20261018_153713_01_-_j-hope_-_MONA_LISA.flac"


def _corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_naive_stereo_average_mismeasures_the_metric():
    """The bug must be demonstrable, otherwise nobody reproduces it."""
    if not SRC.exists():
        pytest.skip("source not available")
    y, _ = load_audio(str(SRC), target_sr=SR)
    y = y[:15 * SR]
    raw, native_sr = sf.read(str(SRC), always_2d=False)
    assert native_sr != SR, "source is expected to differ from the render rate"

    naive = raw.mean(axis=1) if raw.ndim > 1 else raw
    naive = naive[:15 * SR]
    truth = _corr(y, y)
    wrong = _corr(naive, y)
    assert truth == pytest.approx(1.0)
    assert abs(wrong - truth) > 0.3, (
        "precondition: naive channel averaging must actually mismeasure"
    )


def test_render_correlates_with_the_source_using_engine_conventions():
    """The published combined correlation must be reproducible from disk."""
    out = ROOT / "data/output/res_demucs15_v2/res_best.wav"
    if not out.exists():
        pytest.skip("no Demucs run on disk")
    y, _ = load_audio(str(SRC), target_sr=SR)
    y = y[:15 * SR]
    r, rs = sf.read(str(out), always_2d=False)
    if r.ndim > 1:
        r = r.mean(axis=1)
    assert rs == SR, "render sample rate must match the engine rate"
    c = _corr(r, y)
    assert 0.60 < c < 0.85, f"combined correlation {c:.4f} outside the measured band"


def test_ledger_and_disk_agree():
    """The ledger's headline number must match the shipped audio."""
    import json
    d = ROOT / "data/output/res_demucs15_v2"
    if not (d / "res_multitrack_ledger.json").exists():
        pytest.skip("no Demucs run on disk")
    led = json.load(open(d / "res_multitrack_ledger.json"))
    y, _ = load_audio(str(SRC), target_sr=SR)
    y = y[: led["n_samples"]]
    r, _ = sf.read(str(d / "res_best.wav"), always_2d=False)
    if r.ndim > 1:
        r = r.mean(axis=1)
    disk = _corr(r, y)
    claimed = float(led["best"]["correlation"])
    assert abs(disk - claimed) < 0.02, (
        f"ledger claims {claimed:.4f} but the shipped audio measures {disk:.4f}"
    )