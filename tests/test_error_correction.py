"""Tests for closed-loop error correction and A/B comparison."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.error_correction import (
    apply_lag,
    correct_reconstruction,
    energy_guard,
    evaluate_equation,
    find_best_lag,
    least_squares_gain,
    match_rms_level,
    plot_ab_comparison,
    refine_segmented,
    rms,
    waveform_correlation,
)

SR = 22050


def _sine(f, dur, sr=SR, amp=0.5, decay=1.0, phase=0.0):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.exp(-decay * t) * np.sin(2.0 * np.pi * f * t + phase)


class TestHelpers(unittest.TestCase):
    def test_rms_and_level_matching(self):
        a = _sine(220, 0.5, amp=0.5)
        self.assertAlmostEqual(rms(a), rms(a), places=9)
        self.assertEqual(rms(np.zeros(0)), 0.0)
        scaled, gain = match_rms_level(0.2 * a, a)
        self.assertAlmostEqual(gain, 5.0, delta=0.05)
        self.assertAlmostEqual(rms(scaled), rms(a), delta=1e-6)
        untouched, g2 = match_rms_level(a, np.zeros(100))
        self.assertEqual(g2, 1.0)
        self.assertIs(untouched, a)

    def test_energy_guard_bounds(self):
        a = _sine(220, 1.0, amp=0.5)
        self.assertTrue(energy_guard(a, a))
        self.assertTrue(energy_guard(2.0 * a, a))
        self.assertFalse(energy_guard(0.01 * a, a))
        self.assertFalse(energy_guard(np.zeros_like(a), a))
        self.assertTrue(energy_guard(np.zeros(100), np.zeros(100)))

    def test_evaluate_equation(self):
        n = SR
        eq = "0.5*exp(-1*t)*sin(2*pi*220*t)"
        y = evaluate_equation(eq, n, SR)
        self.assertEqual(len(y), n)
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreater(float(np.max(np.abs(y))), 0.3)

    def test_evaluate_empty_and_zero_length(self):
        self.assertEqual(len(evaluate_equation("", 0, SR)), 0)
        self.assertEqual(len(evaluate_equation("", 100, SR)), 100)
        self.assertTrue(np.all(evaluate_equation("", 50, SR) == 0.0))

    def test_correlation_bounds(self):
        a = _sine(220, 0.5)
        self.assertAlmostEqual(waveform_correlation(a, a), 1.0, places=6)
        self.assertAlmostEqual(waveform_correlation(a, -a), -1.0, places=6)
        self.assertEqual(waveform_correlation(np.zeros(100), a), 0.0)
        self.assertEqual(waveform_correlation(a, np.zeros(100)), 0.0)

    def test_apply_lag_shifts_later_for_positive(self):
        y = np.concatenate([np.zeros(10), np.ones(10)])
        shifted = apply_lag(y, 5)
        self.assertEqual(len(shifted), len(y))
        self.assertEqual(shifted[5], 0.0)
        self.assertEqual(shifted[15], 1.0)
        self.assertTrue(np.all(apply_lag(y, 0) == y))

    def test_apply_lag_negative_advances(self):
        y = np.concatenate([np.ones(10), np.zeros(10)])
        shifted = apply_lag(y, -5)
        self.assertEqual(len(shifted), 15)
        self.assertEqual(shifted[0], 1.0)

    def test_find_best_lag_detects_offset(self):
        ref = _sine(300, 1.0)
        lag = 64
        est = np.concatenate([np.zeros(lag), ref[:-lag]])
        found, _ = find_best_lag(ref, est, SR, max_lag_s=0.02)
        self.assertTrue(abs(abs(found) - lag) <= 2, f"found {found}, expected ~{lag}")

    def test_find_best_lag_zero_for_aligned(self):
        ref = _sine(300, 1.0)
        found, corr = find_best_lag(ref, ref, SR, max_lag_s=0.02)
        self.assertEqual(found, 0)
        self.assertGreater(corr, 0.99)

    def test_least_squares_gain(self):
        est = _sine(200, 1.0)
        ref = 0.5 * est
        self.assertAlmostEqual(least_squares_gain(ref, est), 0.5, places=3)
        self.assertEqual(least_squares_gain(ref, np.zeros_like(est)), 1.0)

    def test_least_squares_gain_clamped(self):
        ref = 100.0 * _sine(200, 1.0)
        est = _sine(200, 1.0)
        self.assertLessEqual(least_squares_gain(ref, est), 2.0)


class TestCorrectReconstruction(unittest.TestCase):
    def test_recovers_a_missing_partial(self):
        f0, dur = 220.0, 1.5
        partial1 = _sine(f0, dur, amp=0.6, decay=1.0)
        partial2 = _sine(660.0, dur, amp=0.3, decay=3.0)
        original = partial1 + partial2
        estimate = partial1.copy()

        rep = correct_reconstruction(
            original, estimate, SR, rounds=2, terms_per_round=3, align_lag=False
        )
        corrected = rep["corrected_audio"]
        self.assertEqual(len(corrected), len(original))
        self.assertGreater(rep["metrics_after"]["snr_db"], rep["metrics_before"]["snr_db"])
        self.assertTrue(rep["total_snr_gain_db"] > 1.0)
        self.assertGreater(waveform_correlation(original, corrected), 0.8)
        self.assertIn("residual_audio", rep)
        self.assertIn("equations", rep)

    def test_level_mismatch_is_corrected(self):
        original = _sine(330, 1.0, amp=0.5)
        estimate = 0.2 * original
        rep = correct_reconstruction(original, estimate, SR, rounds=1, align_lag=False)
        self.assertAlmostEqual(rep["initial_level_gain"], 5.0, delta=0.1)
        self.assertGreater(rep["metrics_before"]["snr_db"], 40.0)
        self.assertGreater(waveform_correlation(original, rep["corrected_audio"]), 0.95)

    def test_energy_guard_prevents_silent_solution(self):
        """SNR alone is gameable by emitting silence; the guard must stop it."""
        original = _sine(220, 2.0, amp=0.5)
        # A deliberately over-loud estimate whose residual invites over-correction.
        estimate = 6.0 * original
        rep = correct_reconstruction(original, estimate, SR, rounds=3, align_lag=False)
        out = rep["corrected_audio"]
        self.assertGreater(rms(out), 0.35 * rms(original))
        self.assertLess(rms(out), 3.0 * rms(original))

    def test_time_misalignment_is_corrected(self):
        original = _sine(300, 2.0, amp=0.6)
        lag = 120
        estimate = np.concatenate([np.zeros(lag), original[:-lag]])
        rep = correct_reconstruction(original, estimate, SR, rounds=0, align_lag=True, max_lag_s=0.05)
        self.assertNotEqual(rep["lag_samples"], 0)
        self.assertGreater(waveform_correlation(original, rep["corrected_audio"]), 0.9)

    def test_already_perfect_is_not_degraded(self):
        original = _sine(440, 1.0)
        rep = correct_reconstruction(original, original.copy(), SR, rounds=2)
        self.assertGreaterEqual(
            rep["metrics_after"]["snr_db"], rep["metrics_before"]["snr_db"] - 1e-6
        )
        self.assertGreater(rep["metrics_after"]["snr_db"], 40.0)

    def test_report_shape(self):
        original = _sine(440, 1.0)
        rep = correct_reconstruction(original, np.zeros_like(original), SR, rounds=1)
        for key in (
            "lag_samples", "lag_ms", "gain", "rounds", "equations",
            "metrics_before", "metrics_after", "total_snr_gain_db",
            "total_correlation_gain", "corrected_audio", "residual_audio",
        ):
            self.assertIn(key, rep)
        for m in (rep["metrics_before"], rep["metrics_after"]):
            for key in ("snr_db", "spectral_mse", "log_spectral_distance_db", "correlation"):
                self.assertIn(key, m)

    def test_empty_and_short_inputs(self):
        rep = correct_reconstruction(np.zeros(0), np.zeros(0), SR, rounds=1)
        self.assertEqual(len(rep["corrected_audio"]), 0)


class TestSegmentedRefinement(unittest.TestCase):
    def _segments(self, beat, dur, n=8):
        return [
            (int(i * beat * SR), int((i + 1) * beat * SR))
            for i in range(min(n, int(dur / beat)))
        ]

    def test_segmented_pass_improves_match(self):
        dur, beat = 4.0, 0.5
        t = np.arange(int(dur * SR)) / SR
        original = np.zeros_like(t)
        for i in range(int(dur / beat)):
            s, e = i * beat, (i + 1) * beat
            m = (t >= s) & (t < e)
            original[m] = 0.5 * np.exp(-2.0 * (t[m] - s)) * np.sin(2 * np.pi * 330.0 * (t[m] - s))
        estimate = np.zeros_like(t)

        seg = refine_segmented(original, estimate, SR, self._segments(beat, dur), rounds=2)
        self.assertGreater(seg["segment_count"], 0)
        self.assertGreater(seg["metrics_after"]["snr_db"], -40.0)
        self.assertGreater(
            waveform_correlation(original, seg["corrected_audio"]), 0.3
        )
        self.assertIn("layers", seg)
        self.assertIn("start_time", seg["layers"][0])

    def test_correct_reconstruction_accepts_segments(self):
        dur, beat = 4.0, 0.5
        t = np.arange(int(dur * SR)) / SR
        original = np.zeros_like(t)
        for i in range(int(dur / beat)):
            s, e = i * beat, (i + 1) * beat
            m = (t >= s) & (t < e)
            original[m] = 0.5 * np.exp(-2.0 * (t[m] - s)) * np.sin(2 * np.pi * 330.0 * (t[m] - s))
        estimate = np.zeros_like(t)
        rep = correct_reconstruction(
            original, estimate, SR, rounds=0, segments=self._segments(beat, dur)
        )
        self.assertIn("segmented", rep)
        self.assertTrue(rep["segmented"]["accepted"])
        self.assertGreater(rep["metrics_after"]["snr_db"], rep["metrics_before"]["snr_db"])


class TestComparisonPlot(unittest.TestCase):
    def test_writes_png(self):
        dur = 1.0
        t = np.arange(int(dur * SR)) / SR
        original = 0.5 * np.sin(2 * np.pi * 220 * t)
        estimate = 0.4 * np.sin(2 * np.pi * 220 * t)
        corrected = original.copy()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ab.png"
            plot_ab_comparison(original, estimate, corrected, SR, out, title="t")
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 2000)

    def test_plot_with_metrics(self):
        dur = 0.5
        y = _sine(330, dur)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ab2.png"
            plot_ab_comparison(
                y, y * 0.5, y, SR, out,
                metrics_before={"snr_db": 1.0, "correlation": 0.1,
                                "spectral_mse": 0.2, "log_spectral_distance_db": 9.0},
                metrics_after={"snr_db": 12.0, "correlation": 0.9,
                               "spectral_mse": 0.01, "log_spectral_distance_db": 3.0},
            )
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()