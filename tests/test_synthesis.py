"""Tests for forward synthesis engine."""

import unittest
import numpy as np
from pathlib import Path
import tempfile
from src.synthesis import (
    compute_phase_integration,
    compute_envelope,
    generate_benchmark_chord,
)


class TestSynthesis(unittest.TestCase):
    def test_constant_frequency_phase_integration(self):
        sr = 44100
        dur = 0.1
        n = int(dur * sr)
        f0 = 440.0
        f_traj = np.full(n, f0)
        theta = compute_phase_integration(f_traj, sr, initial_phase=0.0)

        t = np.arange(n) / sr
        expected_theta = 2.0 * np.pi * f0 * t

        # Should match analytical phase closely
        max_err = np.max(np.abs(theta - expected_theta))
        self.assertLess(max_err, 1e-10)

    def test_vibrato_phase_derivative(self):
        """Numerical derivative of phase integration must match instantaneous frequency."""
        sr = 44100
        dur = 0.5
        n = int(dur * sr)
        t = np.arange(n) / sr
        f0 = 440.0
        # 5 Hz vibrato with depth 10 Hz
        f_traj = f0 + 10.0 * np.sin(2.0 * np.pi * 5.0 * t)
        theta = compute_phase_integration(f_traj, sr, initial_phase=0.0)

        # d(theta)/dt = 2*pi*f(t)
        d_theta_dt = np.diff(theta) * sr / (2.0 * np.pi)
        f_mid = 0.5 * (f_traj[:-1] + f_traj[1:])
        err = np.max(np.abs(d_theta_dt - f_mid))
        self.assertLess(err, 1e-6)

    def test_exponential_envelope(self):
        dur = 1.0
        sr = 1000
        t = np.arange(int(dur * sr)) / sr
        a0 = 0.8
        decay = 2.0
        env = compute_envelope({"type": "exponential", "a0": a0, "decay_rate": decay}, t, dur)
        self.assertAlmostEqual(env[0], a0, places=6)
        expected_end = a0 * np.exp(-decay * t[-1])
        self.assertAlmostEqual(env[-1], expected_end, places=6)

    def test_benchmark_chord_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "math_chord.wav"
            model = generate_benchmark_chord(wav_path, duration=0.5, sr=22050)
            self.assertTrue(wav_path.exists())
            self.assertGreater(wav_path.stat().st_size, 1000)
            self.assertEqual(len(model["components"]), 3)


if __name__ == "__main__":
    unittest.main()
