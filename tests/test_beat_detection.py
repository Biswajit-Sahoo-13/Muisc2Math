"""Unit tests for multi-band spectral-flux beat detection."""
import numpy as np
import unittest

from src.beat_detection import (
    compute_spectral_flux,
    estimate_tempo,
    detect_beats,
    detect_onsets,
)


class TestBeatDetection(unittest.TestCase):
    def _make_click_train(self, bpm=120.0, dur=6.0, sr=22050):
        t = np.arange(int(sr * dur)) / sr
        y = np.zeros_like(t)
        period = 60.0 / bpm
        rng = np.random.default_rng(42)
        for b in np.arange(0.0, dur, period):
            idx = int(b * sr)
            w = np.hanning(64)
            if idx + len(w) <= len(y):
                y[idx : idx + len(w)] += w * rng.normal(size=len(w))
        return y, sr

    def test_spectral_flux_shape(self):
        y, sr = self._make_click_train()
        env, times = compute_spectral_flux(y, sr=sr)
        self.assertEqual(env.shape, times.shape)
        self.assertGreater(env.size, 0)
        self.assertTrue(np.all(np.isfinite(env)))

    def test_tempo_estimation(self):
        y, sr = self._make_click_train(bpm=120.0, dur=8.0)
        env, _ = compute_spectral_flux(y, sr=sr)
        tempo, lag, conf = estimate_tempo(env, sr=sr)
        self.assertAlmostEqual(tempo, 120.0, delta=6.0)
        self.assertGreater(lag, 0)

    def test_detect_beats_returns_structure(self):
        y, sr = self._make_click_train()
        res = detect_beats(y, sr=sr)
        self.assertIn("tempo", res)
        self.assertIn("beat_times", res)
        self.assertIn("onset_times", res)
        self.assertGreater(len(res["beat_times"]), 0)
        self.assertTrue(np.all(np.diff(res["beat_times"]) > 0))

    def test_detect_beats_tempo_120(self):
        y, sr = self._make_click_train(bpm=120.0, dur=8.0)
        res = detect_beats(y, sr=sr)
        self.assertAlmostEqual(res["tempo"], 120.0, delta=6.0)

    def test_detect_onsets_empty_safe(self):
        y = np.zeros(1024, dtype=np.float64)
        env, _ = compute_spectral_flux(y, sr=22050)
        frames, times = detect_onsets(env, sr=22050)
        self.assertIsInstance(frames, np.ndarray)
        self.assertIsInstance(times, np.ndarray)


if __name__ == "__main__":
    unittest.main()