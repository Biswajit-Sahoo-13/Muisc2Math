"""Tests for Phase 2: Speed optimizations (incremental residual and evidence cache)."""

import numpy as np
import pytest

from src.res_evidence import build_evidence, get_cached_section_evidence, clear_evidence_cache
from src.res_loop import _section_update
from src.res_synthesis import _corr


def test_section_update_incremental_produces_valid_result():
    """Verify incremental _section_update improves or maintains correlation."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    y_sec = np.sin(2.0 * np.pi * 220.0 * t) + 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    
    # Slight error in amplitudes and phases
    partials = [
        {"frequency": 220.0, "amplitude": 0.8, "phase": 0.2, "envelope_decay": 0.0, "envelope_t0": 0.0},
        {"frequency": 440.0, "amplitude": 0.3, "phase": -0.1, "envelope_decay": 0.0, "envelope_t0": 0.0},
    ]
    
    # Initial correlation
    recon_init = 0.8 * np.sin(2.0 * np.pi * 220.0 * t - 0.2) + 0.3 * np.sin(2.0 * np.pi * 440.0 * t + 0.1)
    corr_init = _corr(y_sec, recon_init)
    
    accepted = _section_update(y_sec, partials, sr)
    assert accepted > 0
    
    # Check new correlation
    from src.res_synthesis import _render_partial
    recon_after = sum(_render_partial(p, n, sr) for p in partials)
    corr_after = _corr(y_sec, recon_after)
    assert corr_after > corr_init


def test_evidence_cache_reuses_computed_evidence(tmp_path):
    """Verify evidence caching caches STFT/spectral evidence per section."""
    sr = 22050
    n = sr
    t = np.arange(n) / sr
    y = np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "test_cache.wav"
    import soundfile as sf
    sf.write(str(audio_path), y, sr)
    
    clear_evidence_cache()
    
    ledger = build_evidence(str(audio_path), str(tmp_path / "out"), sr=sr, max_duration=1.0, window_s=0.5)
    
    # Check that sections carry or can retrieve cached evidence
    sec0 = ledger["sections"][0]
    ev = get_cached_section_evidence(sec0, y[sec0["start_sample"]:sec0["start_sample"] + int(0.5 * sr)], sr)
    assert ev is not None
    assert "fft_mag" in ev
    
    # Second fetch should hit memory cache without recomputing
    ev2 = get_cached_section_evidence(sec0, y[sec0["start_sample"]:sec0["start_sample"] + int(0.5 * sr)], sr)
    assert ev2 is ev
