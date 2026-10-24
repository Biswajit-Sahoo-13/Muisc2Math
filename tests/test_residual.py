"""Tests for error metrics, residual projection, and iterative refinement."""

import unittest
import numpy as np
from src.residual import (
    compute_error_metrics,
    estimate_sinusoid_by_projection,
    iterative_residual_refinement,
)


class TestResidual(unittest.TestCase):
    def test_identical_signal_metrics(self):
        sr = 44100
        dur = 0.5
        t = np.arange(int(dur * sr)) / sr
        x = 0.7 * np.sin(2.0 * np.pi * 440.0 * t)

        metrics = compute_error_metrics(x, x, sr=sr)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=6)
        self.assertGreater(metrics["snr_db"], 100.0)
        self.assertAlmostEqual(metrics["spectral_mse"], 0.0, places=6)

    def test_estimate_sinusoid_by_projection(self):
        sr = 44100
        dur = 1.0
        t = np.arange(int(dur * sr)) / sr
        f0 = 550.0
        a_true = 0.45
        phi_true = 0.65  # radians

        x = a_true * np.sin(2.0 * np.pi * f0 * t + phi_true)
        a_est, phi_est = estimate_sinusoid_by_projection(x, sr, f0)

        self.assertAlmostEqual(a_est, a_true, delta=0.01)
        # Check phase difference modulo 2*pi
        diff_phi = (phi_est - phi_true + np.pi) % (2.0 * np.pi) - np.pi
        self.assertAlmostEqual(abs(diff_phi), 0.0, delta=0.02)

    def test_iterative_residual_refinement(self):
        """Simulates finding and adding a missing harmonic (e.g. 440 + 880 Hz)."""
        sr = 44100
        dur = 1.0
        t = np.arange(int(dur * sr)) / sr
        f1, a1 = 440.0, 0.70
        f2, a2 = 880.0, 0.35

        # Ground truth has both
        x = a1 * np.sin(2.0 * np.pi * f1 * t) + a2 * np.sin(2.0 * np.pi * f2 * t)

        # Incomplete model has only 440 Hz
        initial_model = {
            "model_name": "partial_model",
            "sample_rate": sr,
            "duration": dur,
            "components": [
                {
                    "id": "comp_1",
                    "frequency": f1,
                    "amplitude": a1,
                    "phase": 0.0,
                    "envelope": {"type": "constant", "a0": a1},
                }
            ],
        }

        refined_model, x_hat, final_metrics = iterative_residual_refinement(
            original_audio=x,
            sr=sr,
            initial_model=initial_model,
            max_iterations=5,
            target_snr_db=30.0,
            verbose=False,
        )

        # Refined model should now contain the second component
        self.assertGreaterEqual(len(refined_model["components"]), 2)
        # SNR should jump significantly (> 30 dB)
        self.assertGreaterEqual(final_metrics["snr_db"], 25.0)


if __name__ == "__main__":
    unittest.main()
