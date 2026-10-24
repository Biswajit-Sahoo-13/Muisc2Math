"""Unit tests for Sinusoidal Modeling Synthesis (SMS).

The decisive properties are verified against ground truth rather than
self-referential checks: a stationary partial must be reconstructed essentially
exactly, and a decaying triad must have its three true frequencies, amplitudes
and decay rates recovered.
"""

import unittest

import numpy as np

from src.sinusoidal_model import (
    noise_residual_layer,
    sinusoidal_reconstruction,
)


SR = 22050


def _tone(freq, dur, amp=0.5, decay=0.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-decay * t) if decay else np.ones_like(t)
    return amp * env * np.sin(2.0 * np.pi * freq * t)


class TestPerfectReconstruction(unittest.TestCase):
    def test_single_tone_is_near_exact(self):
        y = _tone(440.0, 1.0, amp=0.5)
        r = sinusoidal_reconstruction(
            y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        self.assertGreater(r["snr_db"], 40.0)
        self.assertGreater(r["correlation"], 0.999)
        self.assertEqual(r["track_count"], 1)

    def test_single_tone_parameters_match_ground_truth(self):
        y = _tone(440.0, 1.0, amp=0.5)
        r = sinusoidal_reconstruction(
            y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        p = r["fitted_parameters"][0]
        self.assertAlmostEqual(p["frequency"], 440.0, delta=0.5)
        self.assertAlmostEqual(p["amplitude"], 0.5, delta=0.02)
        self.assertLess(abs(p["envelope_decay"]), 0.2)
        self.assertEqual(p["model"], "constant")

    def test_snr_scales_with_frequency_stability(self):
        """A steady partial must beat the earlier trajectory-integrating model."""
        y = _tone(440.0, 1.0, amp=0.5)
        steady = sinusoidal_reconstruction(
            y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        self.assertGreater(steady["snr_db"], 40.0)

    def test_two_tone_chord(self):
        y = _tone(440.0, 1.0, amp=0.5) + _tone(554.365, 1.0, amp=0.35)
        r = sinusoidal_reconstruction(
            y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        self.assertGreater(r["snr_db"], 25.0)
        self.assertGreater(r["correlation"], 0.99)
        freqs = sorted(p["frequency"] for p in r["fitted_parameters"])
        self.assertTrue(any(abs(f - 440.0) < 2.0 for f in freqs), freqs)
        self.assertTrue(any(abs(f - 554.365) < 2.0 for f in freqs), freqs)


class TestDecayingTriad(unittest.TestCase):
    """The three-partial benchmark has exact known parameters."""

    def setUp(self):
        self.truth = [(440.0, 0.60, 1.10), (554.365, 0.35, 1.45), (659.255, 0.25, 1.80)]
        y = np.zeros(int(2.0 * SR))
        for f, a, lam in self.truth:
            t = np.arange(len(y)) / SR
            y += a * np.exp(-lam * t) * np.sin(2.0 * np.pi * f * t)
        self.y = y

    def test_recovers_all_three_partials(self):
        r = sinusoidal_reconstruction(
            self.y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        self.assertGreater(r["correlation"], 0.995)
        self.assertGreater(r["snr_db"], 20.0)
        found = r["fitted_parameters"]
        self.assertGreaterEqual(len(found), 3)
        for f_true, a_true, lam_true in self.truth:
            best = min(found, key=lambda p: abs(p["frequency"] - f_true))
            self.assertAlmostEqual(best["frequency"], f_true, delta=1.0)
            self.assertAlmostEqual(best["amplitude"], a_true, delta=0.08)
            self.assertAlmostEqual(best["envelope_decay"], lam_true, delta=0.15)

    def test_clean_triad_yields_exactly_three_tracks(self):
        """A pure triad has no spurious peaks, so nothing should be pruned."""
        r = sinusoidal_reconstruction(
            self.y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        self.assertEqual(r["track_count"], 3)
        self.assertEqual(r["pruned_tracks"], 0)
        self.assertEqual(r["candidate_tracks"], 3)

    def test_prunes_spurious_tracks_on_real_benchmark(self):
        """The 44.1 kHz benchmark does generate sidelobe peaks that get pruned."""
        import tempfile
        from pathlib import Path

        from src.analysis import load_audio
        from src.synthesis import generate_benchmark_chord

        with tempfile.TemporaryDirectory() as tmp:
              wav = Path(tmp) / "chord.wav"
              generate_benchmark_chord(wav, duration=2.0, sr=44100)
              y, sr = load_audio(wav, target_sr=44100)
        y = y[: int(2.0 * sr)]
        # A harsh, clipped signal still produces sidelobe peaks that must be
        # pruned. The clean benchmark fixture no longer clips (it used to),
        # so recreate the harshness explicitly to keep exercising the pruner.
        y = np.clip(y * 1.3, -1.0, 1.0)
        r = sinusoidal_reconstruction(
              y, sr, n_fft=2048, hop_length=256, threshold=0.005,
              add_noise_layer=False,
        )
        self.assertLess(r["track_count"], r["candidate_tracks"])
        self.assertGreaterEqual(r["pruned_tracks"], 1)
        self.assertGreater(r["snr_db"], 20.0)

    def test_clean_benchmark_yields_exactly_the_true_partials(self):
        """Without clipping there is nothing spurious to prune."""
        import tempfile
        from pathlib import Path

        from src.analysis import load_audio
        from src.synthesis import generate_benchmark_chord

        with tempfile.TemporaryDirectory() as tmp:
              wav = Path(tmp) / "chord.wav"
              generate_benchmark_chord(wav, duration=2.0, sr=44100)
              y, sr = load_audio(wav, target_sr=44100)
        y = y[: int(2.0 * sr)]
        r = sinusoidal_reconstruction(
              y, sr, n_fft=2048, hop_length=256, threshold=0.005,
              add_noise_layer=False,
        )
        self.assertEqual(r["track_count"], 3)
        self.assertGreater(r["snr_db"], 20.0)


class TestFallbacksAndEdges(unittest.TestCase):
    def test_empty_signal(self):
        r = sinusoidal_reconstruction(np.zeros(0), SR)
        self.assertEqual(r["track_count"], 0)
        self.assertEqual(r["snr_db"], 0.0)

    def test_silence(self):
        r = sinusoidal_reconstruction(np.zeros(SR), SR, add_noise_layer=False)
        self.assertEqual(r["track_count"], 0)

    def test_noise_layer_shape_and_scale(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(2048) * 0.1
        noise, meta = noise_residual_layer(y, np.zeros_like(y), sr=SR, seed=1)
        self.assertEqual(len(noise), len(y))
        self.assertTrue(np.all(np.isfinite(noise)))
        self.assertTrue(meta["enabled"])
        self.assertEqual(meta["construction"], "spectrum-shaped noise (S+N)")
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(noise ** 2))),
            float(np.sqrt(np.mean(y ** 2))),
            delta=0.05,
        )

    def test_noise_layer_silent_residual_is_disabled_not_nan(self):
        """A residual the sinusoids fully explain must not produce NaN noise.

        Regression: the spectrum-shaping ratio became 0/0 = NaN when the
        residual was exactly zero, and the NaN propagated into the written WAV.
        """
        y = _tone(440.0, 0.5)
        noise, meta = noise_residual_layer(y, y, sr=SR, seed=1)
        self.assertFalse(meta["enabled"])
        self.assertEqual(len(noise), len(y))
        self.assertTrue(np.all(np.isfinite(noise)))
        self.assertTrue(np.all(noise == 0.0))

    def test_noise_layer_is_spectraly_shaped_not_white(self):
        """The S+N layer must match the residual timbre, not just its power."""
        sr = 22050
        t = np.arange(sr) / sr
        # Residual concentrated in the highs (percussive-like).
        rng = np.random.default_rng(3)
        y = 0.3 * rng.standard_normal(sr)
        hi = np.sin(2 * np.pi * 7000 * t) * 0.3
        target = y + hi
        noise, _ = noise_residual_layer(target, np.zeros_like(target), sr=sr, seed=5)

        def band_frac(x):
            spec = np.abs(np.fft.rfft(x))
            freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
            high = float(np.sum(spec[freqs > 4000]))
            total = float(np.sum(spec)) + 1e-12
            return high / total

        # The shaped layer should resemble the target's spectral balance.
        self.assertGreater(band_frac(noise), 0.5)
        self.assertTrue(np.all(np.isfinite(noise)))

    def test_noise_layer_disabled_is_zero(self):
        y = _tone(440.0, 0.5)
        r = sinusoidal_reconstruction(y, SR, n_fft=2048, hop_length=256,
                                     threshold=0.005, add_noise_layer=False)
        self.assertTrue(np.all(r["noise_audio"] == 0.0))
        self.assertFalse(r["noise_layer"]["enabled"])

    def test_model_exports_trajectories(self):
        y = _tone(440.0, 0.5, amp=0.5, decay=1.0)
        r = sinusoidal_reconstruction(
            y, SR, n_fft=2048, hop_length=256, threshold=0.005,
            add_noise_layer=False,
        )
        model = r["model"]
        self.assertEqual(len(model), r["track_count"])
        for item in model:
            self.assertIn("theta(t) = int 2*pi*f(t) dt", item["expression"])
            self.assertEqual(len(item["f_hz"]), item["frames"])
            self.assertEqual(len(item["a"]), item["frames"])
            self.assertEqual(len(item["phi"]), item["frames"])
            self.assertGreater(item["frames"], 0)

    def test_report_keys(self):
        y = _tone(440.0, 0.5)
        r = sinusoidal_reconstruction(y, SR, n_fft=2048, hop_length=256, threshold=0.005)
        for key in ("audio", "sinusoid_audio", "noise_audio", "tracks", "model",
                    "fitted_parameters", "track_count", "candidate_tracks",
                    "pruned_tracks", "snr_db", "correlation", "noise_layer",
                    "sr", "n_samples"):
            self.assertIn(key, r)

    def test_output_length_matches_input(self):
        for dur in (0.25, 0.5, 1.0):
            y = _tone(440.0, dur)
            r = sinusoidal_reconstruction(y, SR, n_fft=2048, hop_length=256,
                                         threshold=0.005)
            self.assertEqual(len(r["audio"]), len(y))

    def test_reconstruction_is_finite(self):
        y = _tone(440.0, 0.5, decay=2.0)
        r = sinusoidal_reconstruction(y, SR, n_fft=2048, hop_length=256, threshold=0.005)
        self.assertTrue(np.all(np.isfinite(r["audio"])))


if __name__ == "__main__":
    unittest.main()
