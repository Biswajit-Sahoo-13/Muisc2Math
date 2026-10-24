"""Tests for HPSS pre-separation and beat-quantized windowing in res_evidence.py."""

from pathlib import Path
import numpy as np
import soundfile as sf

from src.res_evidence import build_evidence

SR = 22050


def _make_drum_plus_chords(path: Path, dur: float = 5.0) -> np.ndarray:
    """Create a test track with sustained chords and periodic drum hits."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    # Sustained chords
    harm = (
        0.3 * np.cos(2 * np.pi * 220.0 * t)
        + 0.2 * np.cos(2 * np.pi * 330.0 * t)
        + 0.15 * np.cos(2 * np.pi * 440.0 * t)
    )
    # Rhythmic drum clicks every 0.5s (120 bpm)
    perc = np.zeros(n)
    rng = np.random.default_rng(42)
    beat_interval = int(0.5 * SR)
    for hit_idx in range(0, n, beat_interval):
        hit_len = min(int(0.02 * SR), n - hit_idx)
        perc[hit_idx:hit_idx + hit_len] += 0.5 * rng.standard_normal(hit_len)
    y = harm + perc
    sf.write(str(path), y, SR)
    return y


def test_build_evidence_with_hpss(tmp_path):
    path = tmp_path / "drum_chord.wav"
    _make_drum_plus_chords(path, dur=3.0)

    ledger_hpss = build_evidence(
        str(path), str(tmp_path / "ev_hpss"),
        sr=SR, max_duration=3.0, window_s=1.0,
        use_hpss=True, beat_quantized=False,
    )
    assert ledger_hpss["hpss_applied"] is True
    assert "sections" in ledger_hpss
    assert len(ledger_hpss["sections"]) >= 2
    for sec in ledger_hpss["sections"]:
        assert sec["partial_count"] >= 1
        assert "metrics" in sec


def test_build_evidence_beat_quantization(tmp_path):
    path = tmp_path / "drum_chord.wav"
    _make_drum_plus_chords(path, dur=4.0)

    # Window nominally 2.0s with beat quantization
    ledger = build_evidence(
        str(path), str(tmp_path / "ev_beat"),
        sr=SR, max_duration=4.0, window_s=2.0,
        use_hpss=True, beat_quantized=True,
    )
    assert ledger["beat_quantized"] is True
    sections = ledger["sections"]
    assert len(sections) >= 2
    # Check that boundaries align with detected beat grid rather than rigid 2.000s
    boundary_t = sections[0]["end"]
    # At 120 bpm (0.5s interval), beats occur around 0.5, 1.0, 1.5, 2.0, etc.
    # The section end should be near one of the beats (multiple of ~0.5s)
    rem = abs(boundary_t % 0.5)
    assert rem < 0.08 or abs(rem - 0.5) < 0.08


def test_build_evidence_fallback_on_static_signal(tmp_path):
    """Static chord has low beat confidence; beat_quantized gracefully falls back to fixed windowing."""
    path = tmp_path / "pure_chord.wav"
    dur = 2.0
    t = np.arange(int(dur * SR)) / SR
    y = 0.3 * np.cos(2 * np.pi * 220 * t) + 0.2 * np.cos(2 * np.pi * 440 * t)
    sf.write(str(path), y, SR)

    ledger = build_evidence(
        str(path), str(tmp_path / "ev_static"),
        sr=SR, max_duration=dur, window_s=1.0,
        beat_quantized=True,
    )
    # Should produce 2 regular sections
    assert len(ledger["sections"]) == 2
    assert ledger["sections"][0]["start"] == 0.0
    assert abs(ledger["sections"][0]["end"] - 1.0) < 0.05
