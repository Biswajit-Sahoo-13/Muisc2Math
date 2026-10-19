"""Tests for spectral analysis, parabolic interpolation, and sinusoidal tracking."""

import unittest
import numpy as np
import scipy.signal as signal
from src.analysis import compute_normalized_stft, find_parabolic_peaks
from src.tracking import fit_log_linear_envelope, track_sinusoids, tracks_to_equations


class TestAnalysis(unittest.TestCase):
    def test_parabolic_interpolation_sub_bin_accuracy(self):
        sr = 44100
        n_fft = 4096
        # Bin spacing is 44100 / 4096 = 10.7666 Hz
        # Choose frequency right in the middle between bins 41 (441.43 Hz) and 42 (452.20 Hz)
        f_true = 446.81
        a_true = 0.75
        dur = 0.5
        t = np.arange(int(dur * sr)) / sr
        y = a_true * np.sin(2.0 * np.pi * f_true * t)

        w = signal.get_window("hann", n_fft)
        coherent_gain = 0.5 * np.sum(w)
        frame = y[:n_fft] * w
        raw_mags = np.abs(np.fft.rfft(frame))

        # Naive peak picking
        naive_bin = np.argmax(raw_mags[1:]) + 1
        naive_f = naive_bin * sr / n_fft
        naive_err = abs(naive_f - f_true)

        # Parabolic interpolation
        peaks = find_parabolic_peaks(raw_mags, coherent_gain, sr, n_fft, threshold=0.01)
        self.assertGreater(len(peaks), 0)
        best_peak = peaks[0]
        interp_f = best_peak["frequency"]
        interp_a = best_peak["amplitude"]
        interp_err = abs(interp_f - f_true)

        # Parabolic interpolation must dramatically outperform naive peak bin
        self.assertLess(interp_err, 0.25)  # within 0.25 Hz
        self.assertLess(interp_err, naive_err / 10.0)  # at least 10x better than naive binning

        # Amplitude recovery test: within ~3% of true amplitude across worst-case bin offset
        self.assertAlmostEqual(interp_a, a_true, delta=0.04)

    def test_log_linear_envelope_fit(self):
        dur = 2.0
        times = np.linspace(0.0, dur, 50)
        a0_true = 0.65
        lambda_true = 1.35
        amps = a0_true * np.exp(-lambda_true * times)

        fit = fit_log_linear_envelope(times, amps)
        self.assertEqual(fit["type"], "exponential")
        self.assertAlmostEqual(fit["a0"], a0_true, places=2)
        self.assertAlmostEqual(fit["decay_rate"], lambda_true, places=2)
        self.assertGreater(fit["r_squared"], 0.999)

    def test_constant_envelope_fit(self):
        dur = 1.0
        times = np.linspace(0.0, dur, 30)
        a0_true = 0.5
        amps = np.full_like(times, a0_true)

        fit = fit_log_linear_envelope(times, amps)
        self.assertEqual(fit["type"], "constant")
        self.assertAlmostEqual(fit["a0"], a0_true, places=2)
        self.assertAlmostEqual(fit["decay_rate"], 0.0, places=2)

    def test_tracking_on_synthetic_sine(self):
        sr = 44100
        dur = 1.0
        t = np.arange(int(dur * sr)) / sr
        f0 = 440.0
        y = 0.8 * np.sin(2.0 * np.pi * f0 * t)

        stft_res = compute_normalized_stft(y, sr=sr, n_fft=2048, hop_length=512)
        tracks = track_sinusoids(stft_res, threshold=0.01)

        self.assertGreaterEqual(len(tracks), 1)
        strongest = tracks[0]
        self.assertAlmostEqual(strongest.mean_frequency, f0, delta=1.0)
        self.assertAlmostEqual(strongest.peak_amplitude, 0.8, delta=0.05)


if __name__ == "__main__":
    unittest.main()
