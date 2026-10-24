"""Unit tests for Phase 6: Tri-Mode Stem Separation & Multi-Track Ledger.

Verifies:
1. Native DSP stem separation forms a zero-phase complementary partition of unity (< 1e-12 error).
2. High-resolution 808 glide tracking captures pitch drops as model='glide' with >= 0.95 correlation.
3. Multi-track ledger hybrid schema (res/2.0-multitrack) provides per-stem metrics and renders via render_ledger.
4. Tri-mode stem resolution (Directory, Demucs detection, Native DSP fallback).
"""

import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from src.res_evidence import separate_stems
from src.sinusoidal_model import sinusoidal_reconstruction
from src.res_synthesis import _corr
from src.res_loop import render_ledger
from src.math_track_engine import multitrack_ledger_to_math_tracks


def test_native_dsp_stem_separation_partition_of_unity():
    """Verify native DSP stem separation forms an exact zero-phase partition of unity (sum == y)."""
    sr = 22050
    dur = 2.0
    n = int(dur * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(42)

    # Composite signal: sub-bass (55 Hz), vocal tone (440 Hz), and broadband transient noise
    y = 0.6 * np.sin(2.0 * np.pi * 55.0 * t) + 0.3 * np.cos(2.0 * np.pi * 440.0 * t) + 0.1 * rng.standard_normal(n)

    stems = separate_stems(y, sr, mode="native")
    assert set(stems.keys()) == {"bass", "drums", "vocals", "other"}

    # Sum of stems must match original to machine precision (< 1e-12)
    reconstructed_y = stems["bass"] + stems["drums"] + stems["vocals"] + stems["other"]
    max_err = float(np.max(np.abs(y - reconstructed_y)))
    assert max_err < 1e-12, f"Partition of unity error {max_err} exceeds 1e-12 threshold"


def test_gliding_808_captured_as_glide_with_high_correlation():
    """Verify gliding 808 (55 Hz -> 45 Hz) is captured as model='glide' with >= 0.95 correlation."""
    sr = 22050
    dur = 2.5
    n = int(dur * sr)
    t = np.arange(n) / sr

    # True gliding 808 pitch drop with exponential amplitude envelope
    f_true = 55.0 - 10.0 * (1.0 - np.exp(-3.0 * t))
    phase_true = 2.0 * np.pi * np.cumsum(f_true) / sr
    y_808 = 0.8 * np.exp(-1.2 * t) * np.sin(phase_true)

    # Reconstruct with fine-resolution FFT (N=4096) and glide_tolerance_hz=2.5
    res = sinusoidal_reconstruction(
        y_808, sr, n_fft=4096, hop_length=128, threshold=0.005,
        glide_tolerance_hz=2.5, window_s=2.5
    )

    partials = res["fitted_parameters"]
    assert len(partials) >= 1
    # Leading partial must be model 'glide'
    p0 = partials[0]
    assert p0["model"] == "glide", f"Leading partial model is {p0['model']}, expected 'glide'"
    assert "trajectory" in p0, "Glide partial must contain frequency trajectory"

    c = _corr(y_808, res["sinusoid_audio"])
    assert c >= 0.95, f"Gliding 808 correlation {c:.4f} is under 0.95 target"


def test_multitrack_ledger_hybrid_schema_compatibility():
    """Verify multi-track ledger carries top-level 'stems' and renders backward-compatibly with render_ledger."""
    sr = 22050
    dur = 1.0
    n = int(dur * sr)
    np.arange(n) / sr

    # Mock multitrack ledger with bass and vocal stems
    p_bass = {
        "model": "damped", "frequency": 55.0, "amplitude": 0.8,
        "phase": 0.0, "envelope_decay": 0.5, "envelope_t0": 0.0,
        "stem": "bass"
    }
    p_vocal = {
        "model": "damped", "frequency": 440.0, "amplitude": 0.3,
        "phase": 0.0, "envelope_decay": 1.0, "envelope_t0": 0.0,
        "stem": "vocals"
    }

    ledger = {
        "schema": "res/2.0-multitrack",
        "sample_rate": sr,
        "duration": dur,
        "n_samples": n,
        "stems": {
            "bass": {
                "metrics": {"correlation": 0.99, "snr_db": 18.5, "log_spectral_distance_db": 3.2},
                "sections": [{"start": 0.0, "end": dur, "partials": [p_bass]}],
            },
            "vocals": {
                "metrics": {"correlation": 0.92, "snr_db": 11.2, "log_spectral_distance_db": 4.5},
                "sections": [{"start": 0.0, "end": dur, "partials": [p_vocal]}],
            },
            "drums": {"metrics": {"correlation": 0.80, "snr_db": 5.0}, "sections": []},
            "other": {"metrics": {"correlation": 0.75, "snr_db": 4.0}, "sections": []},
        },
        "sections": [
            {
                "start": 0.0,
                "end": dur,
                "start_sample": 0,
                "partial_count": 2,
                "partials": [p_bass, p_vocal],
            }
        ],
        "best": {"correlation": 0.95, "snr_db": 14.0, "log_spectral_distance_db": 3.8},
    }

    # render_ledger must render the combined mix smoothly
    recon = render_ledger(ledger, n, ola=False)
    assert len(recon) == n
    assert np.max(np.abs(recon)) > 0.1

    # Math-DAW engine can convert multitrack ledger to independent soloable tracks
    tracks = multitrack_ledger_to_math_tracks(ledger)
    assert len(tracks) >= 2
    track_names = [tr.name for tr in tracks]
    assert "bass" in track_names
    assert "vocals" in track_names


def test_tri_mode_stem_resolution_directory_mode():
    """Verify Mode 1 (Directory): loads pre-existing stems if provided."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr
    y = np.sin(2 * np.pi * 100 * t)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sf.write(str(tmp_path / "bass.wav"), 0.5 * y, sr)
        sf.write(str(tmp_path / "drums.wav"), 0.2 * y, sr)
        sf.write(str(tmp_path / "vocals.wav"), 0.3 * y, sr)
        sf.write(str(tmp_path / "other.wav"), 0.1 * y, sr)

        stems = separate_stems(y, sr, mode="auto", stems_dir=str(tmp_path))
        assert len(stems) == 4
        assert np.allclose(stems["bass"], 0.5 * y, atol=1e-4)
