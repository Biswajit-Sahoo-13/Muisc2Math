"""Tests for Phase 2: Per-section early stopping and parallel processing."""

import numpy as np
import pytest

from src.res_loop import run_loop, _process_section_task


def test_early_stopping_freezes_converged_section(tmp_path):
    """Verify section with consecutive gains < 0.002 is marked converged."""
    sr = 22050
    n = sr
    t = np.arange(n) / sr
    y = np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "test_freeze.wav"
    import soundfile as sf
    sf.write(str(audio_path), y, sr)
    
    # Perfectly fitted section that will have 0 gain
    ledger = {
        "schema": "res/1.1",
        "source": str(audio_path),
        "sample_rate": sr,
        "duration": 1.0,
        "n_samples": n,
        "window_s": 1.0,
        "best": {"correlation": 0.999, "snr_db": 50.0, "log_spectral_distance_db": 0.1, "seg_snr_db": 50.0, "onset_corr": 1.0},
        "sections": [
            {
                "start": 0.0,
                "end": 1.0,
                "start_sample": 0,
                "partial_count": 1,
                "partials": [
                    {"frequency": 440.0, "amplitude": 1.0, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0}
                ],
                "metrics": {"correlation": 0.999, "snr_db": 50.0, "log_spectral_distance_db": 0.1, "seg_snr_db": 50.0, "onset_corr": 1.0},
            }
        ],
    }
    
    out_dir = tmp_path / "out"
    res = run_loop(ledger, str(audio_path), str(out_dir), iterations=3, target_corr=0.0, bank=False, workers=1)
    
    sec0 = res["sections"][0]
    # Should be flagged converged because gains in iters 1 and 2 were < 0.002
    assert sec0.get("converged") is True


def test_parallel_section_processing_consistency(tmp_path):
    """Verify run_loop with workers=2 yields valid consistent results."""
    sr = 22050
    n = 2 * sr
    t = np.arange(n) / sr
    y = np.sin(2.0 * np.pi * 440.0 * t)
    audio_path = tmp_path / "test_parallel.wav"
    import soundfile as sf
    sf.write(str(audio_path), y, sr)
    
    sec_def = lambda s, e: {
        "start": s,
        "end": e,
        "start_sample": int(s * sr),
        "partial_count": 1,
        "partials": [
            {"frequency": 440.0, "amplitude": 0.8, "phase": 0.1, "envelope_decay": 0.0, "envelope_t0": s}
        ],
        "metrics": {"correlation": 0.8, "snr_db": 10.0, "log_spectral_distance_db": 5.0, "seg_snr_db": 10.0, "onset_corr": 0.8},
    }
    
    ledger = {
        "schema": "res/1.1",
        "source": str(audio_path),
        "sample_rate": sr,
        "duration": 2.0,
        "n_samples": n,
        "window_s": 1.0,
        "best": {"correlation": 0.8, "snr_db": 10.0, "log_spectral_distance_db": 5.0, "seg_snr_db": 10.0, "onset_corr": 0.8},
        "sections": [sec_def(0.0, 1.0), sec_def(1.0, 2.0)],
    }
    
    out_dir = tmp_path / "out"
    res = run_loop(ledger, str(audio_path), str(out_dir), iterations=1, bank=False, workers=2)
    
    assert len(res["sections"]) == 2
    assert res["best"]["correlation"] > 0.8
