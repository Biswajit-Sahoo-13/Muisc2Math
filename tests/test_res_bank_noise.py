"""Tests for Phase 2: Stochastic Noise Channel (Class C++) in res_bank."""

import numpy as np

from src.res_synthesis import _render_partial
from src.res_bank import filtered_noise_hypothesis, tournament_section


def test_filtered_noise_render_deterministic():
    """Verify filtered_noise model renders deterministically and matches envelope."""
    sr = 22050
    n = 2048
    p = {
        "model": "filtered_noise",
        "seed": 12345,
        "amplitude": 0.25,
        "band_centers": [100.0, 500.0, 2000.0, 8000.0],
        "band_mags": [0.01, 0.5, 0.8, 0.2],
        "envelope_decay": 0.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": float(n) / sr,
    }
    
    r1 = _render_partial(p, n, sr)
    r2 = _render_partial(p, n, sr)
    
    assert len(r1) == n
    np.testing.assert_allclose(r1, r2, rtol=1e-10, atol=1e-10)
    assert np.std(r1) > 0.01


def test_filtered_noise_hypothesis_creation():
    """Verify filtered_noise_hypothesis analyzes residual and creates valid candidate."""
    sr = 22050
    n = 4096
    rng = np.random.default_rng(999)
    # Target is high-frequency noise (e.g. hi-hat or breath)
    white = rng.standard_normal(n)
    fft_w = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    # High-pass filter above 4000 Hz
    fft_w[freqs < 4000] *= 0.05
    resid = np.fft.irfft(fft_w, n=n) * 0.1
    
    cand = filtered_noise_hypothesis(resid, sr, start_sample=0, seed=42)
    assert cand is not None
    assert len(cand.partials) == 1
    p = cand.partials[0]
    assert p["model"] == "filtered_noise"
    assert p["amplitude"] > 0.0
    assert len(p["band_centers"]) >= 4
    assert len(p["band_mags"]) == len(p["band_centers"])
    
    # Render candidate and verify it reduces residual error
    rendered = _render_partial(p, n, sr)
    err_before = float(np.sum(resid ** 2))
    err_after = float(np.sum((resid - rendered) ** 2))
    assert err_after < err_before


def test_tournament_accepts_filtered_noise_on_noise_residual():
    """Verify tournament_section accepts filtered_noise when residual is stochastic."""
    sr = 22050
    n = 4096
    rng = np.random.default_rng(42)
    
    # Target: 1 sinusoid + broadband noise
    t = np.arange(n) / sr
    sine = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    noise = 0.15 * rng.standard_normal(n)
    y_sec = sine + noise
    
    # Partials initially only describe the sinusoid
    partials = [
        {"frequency": 440.0, "amplitude": 0.5, "phase": 0.0, "envelope_decay": 0.0, "envelope_t0": 0.0}
    ]
    
    records = tournament_section(y_sec, partials, sr, start_sample=0, k=1, k_hits=1, enable_noise=True)
    
    # Check if filtered_noise was nominated in records
    noise_recs = [r for r in records if r.get("class") == "filtered_noise"]
    assert len(noise_recs) >= 1
    # Check if a filtered_noise partial was appended to partials
    noise_partials = [p for p in partials if p.get("model") == "filtered_noise"]
    assert len(noise_partials) >= 1
