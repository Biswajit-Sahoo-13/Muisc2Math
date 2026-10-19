"""B1: attack-phase refinement.

The discovery path fits amplitude and phase by joint least squares over the
whole note. A note's attack carries a small fraction of its energy, so the
full-track solution barely constrains the onset phase: a 0.3 ms offset is
invisible in the body but a large *relative* error at t=0, where the signal is
near zero.

The fix re-estimates the phase on a short leading window, holding frequency
and decay fixed, and keeps the result only if it does not degrade the body.
"""
import numpy as np
import pytest

from src.synthesis import generate_benchmark_chord, render_component
from src.tracking import discover_equations_from_audio

SR = 44100
ATTACK_S = 0.040


def _discover(sig, sr=SR, n_fft=4096, hop_length=1024, threshold=0.01):
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "probe.wav"
        import soundfile as sf
        sf.write(str(p), sig.astype(np.float64), sr)
        return discover_equations_from_audio(
            p, n_fft=n_fft, hop_length=hop_length, threshold=threshold)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_synthetic_triad_attack_error_is_small():
    """A triad with known phase must be recovered with a small attack error.

    This test failed with a 19.6x attack/body ratio until the benchmark
    fixture stopped clipping: the "attack error" turned out to be 95 clipped
    samples at the constructive-interference peaks, not a phase defect. The
    invariant stands: the attack region must not carry disproportionate error.
    """
    with tempfile_dir() as td:
        from pathlib import Path
        import soundfile as sf
        p = Path(td) / "chord.wav"
        generate_benchmark_chord(p, duration=2.0, sr=SR)
        sig, _ = sf.read(str(p), always_2d=False)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        disc = _discover(sig)
        assert len(disc["components"]) >= 3

        orig = sig.astype(np.float64)
        recon = np.zeros(len(orig))
        for c in disc["components"]:
            s = render_component(c, 2.0, SR)
            recon[:len(s)] += s[:len(s)]

        k = int(ATTACK_S * SR)
        attack_rms = _rms(orig[:k] - recon[:k])
        body = slice(int(0.3 * SR), int(1.5 * SR))
        body_rms = _rms(orig[body] - recon[body])
        assert attack_rms < 3.0 * body_rms, (
            f"attack rms {attack_rms:.6f} vs body rms {body_rms:.6f}: "
            f"attack error is not bounded"
        )


import contextlib
import tempfile


@contextlib.contextmanager
def tempfile_dir():
    with tempfile.TemporaryDirectory() as td:
        yield td
