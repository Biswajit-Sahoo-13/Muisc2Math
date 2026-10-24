"""Tests for Phase 2: Time alignment in res_loop."""

import numpy as np

from src.error_correction import find_best_lag, apply_lag
from src.res_synthesis import _corr
from src.res_loop import run_loop


def test_time_lag_alignment_direct():
    """Verify that find_best_lag and apply_lag correct artificial time shift."""
    sr = 22050
    t = np.arange(2048) / sr
    target = np.sin(2.0 * np.pi * 440.0 * t)
    
    # 8-sample artificial lag
    lag_true = 8
    shifted_target = np.roll(target, lag_true)
    
    lag_found, _ = find_best_lag(shifted_target, target, sr, max_lag_s=0.01)
    assert lag_found == lag_true
    
    aligned = apply_lag(target, lag_found)
    assert _corr(shifted_target, aligned) > 0.99


def test_section_lag_recorded_in_loop(tmp_path):
    """Verify res_loop applies find_best_lag per section and records lag_samples."""
    sr = 22050
    n = sr  # 1 second
    t = np.arange(n) / sr
    y = np.sin(2.0 * np.pi * 220.0 * t) + 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    
    # Save test audio
    audio_path = tmp_path / "test_lag.wav"
    import soundfile as sf
    sf.write(str(audio_path), y, sr)
    
    # Create ledger with 1 section having a small initial timing offset
    ledger = {
        "schema": "res/1.1",
        "source": str(audio_path),
        "sample_rate": sr,
        "duration": 1.0,
        "n_samples": n,
        "window_s": 1.0,
        "best": {"correlation": 0.5, "snr_db": 0.0, "log_spectral_distance_db": 20.0, "seg_snr_db": 0.0, "onset_corr": 0.0},
        "sections": [
            {
                "start": 0.0,
                "end": 1.0,
                "start_sample": 0,
                "partial_count": 2,
                "partials": [
                    {"frequency": 220.0, "amplitude": 1.0, "phase": 0.5, "envelope_decay": 0.0, "envelope_t0": 0.0},
                    {"frequency": 440.0, "amplitude": 0.5, "phase": 0.5, "envelope_decay": 0.0, "envelope_t0": 0.0},
                ],
                "metrics": {"correlation": 0.5, "snr_db": 0.0, "log_spectral_distance_db": 20.0, "seg_snr_db": 0.0, "onset_corr": 0.0},
            }
        ],
    }
    
    out_dir = tmp_path / "out"
    res = run_loop(ledger, str(audio_path), str(out_dir), iterations=1, bank=False, workers=1)
    
    sec0 = res["sections"][0]
    # Check that lag_samples was evaluated and stored in the section
    assert "lag_samples" in sec0
    assert isinstance(sec0["lag_samples"], int)
    assert sec0["metrics"]["correlation"] > 0.85
