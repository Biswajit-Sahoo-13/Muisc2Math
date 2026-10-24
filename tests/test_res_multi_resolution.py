"""Tests for Slice 4: Multi-Resolution Spectral Evidence in res_evidence."""

import numpy as np

from src.res_evidence import (
    multi_resolution_peaks,
    get_cached_section_evidence,
    clear_evidence_cache,
)


def test_multi_resolution_resolves_close_sub_bass():
    """Verify large window (<250Hz) resolves close sub-bass 808 frequencies."""
    sr = 22050
    duration = 0.5  # ~11025 samples
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Two close sub-bass frequencies that standard 512 FFT cannot separate
    f1, f2 = 45.0, 58.0
    sig = 0.6 * np.cos(2 * np.pi * f1 * t) + 0.5 * np.cos(2 * np.pi * f2 * t)

    peaks = multi_resolution_peaks(sig, sr)
    freqs = [p["frequency"] for p in peaks if p.get("band") == "low" or p["frequency"] < 250.0]

    # Must resolve both f1 and f2 within +-3 Hz
    has_f1 = any(abs(f - f1) <= 3.0 for f in freqs)
    has_f2 = any(abs(f - f2) <= 3.0 for f in freqs)

    assert has_f1, f"Failed to resolve {f1} Hz in low band: {freqs}"
    assert has_f2, f"Failed to resolve {f2} Hz in low band: {freqs}"


def test_multi_resolution_captures_high_frequency_transients():
    """Verify small window (>2kHz) captures transient hi-hat content."""
    sr = 22050
    n = 4096
    t = np.arange(n) / sr

    # Low drone + short high-frequency burst at 6000 Hz
    sig = 0.3 * np.cos(2 * np.pi * 100.0 * t)
    # Burst between sample 1000 and 1200 (~10 ms)
    sig[1000:1200] += 0.8 * np.cos(2 * np.pi * 6000.0 * t[1000:1200])

    peaks = multi_resolution_peaks(sig, sr)
    high_peaks = [p for p in peaks if p.get("band") == "high" or p["frequency"] > 2000.0]

    assert len(high_peaks) > 0
    assert any(abs(p["frequency"] - 6000.0) < 150.0 for p in high_peaks)


def test_get_cached_section_evidence_includes_multi_resolution():
    """Verify get_cached_section_evidence integrates multi-resolution peaks."""
    clear_evidence_cache()
    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    sig = 0.5 * np.cos(2 * np.pi * 65.0 * t) + 0.3 * np.cos(2 * np.pi * 3500.0 * t)

    sec = {"start": 0.0, "start_sample": 0}
    ev = get_cached_section_evidence(sec, sig, sr)

    assert "multi_resolution_peaks" in ev
    assert len(ev["multi_resolution_peaks"]) > 0
    assert ev["energy"] > 0.0
