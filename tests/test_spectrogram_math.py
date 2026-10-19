"""Unit tests for Spectrogram Math transforms and Polyphonic Note Extraction."""

import unittest
import numpy as np
from src.spectrogram_math import (
    compute_cqt_spectrogram,
    compute_reassigned_spectrogram,
    extract_notes_cqt_polyphonic,
)


class TestSpectrogramMath(unittest.TestCase):
    """Test suite for CQT, Reassigned Spectrogram, and Polyphonic CQT Extraction."""

    def setUp(self):
        self.sr = 22050
        self.dur = 1.0
        self.t = np.arange(int(self.sr * self.dur)) / self.sr

    def test_compute_cqt_spectrogram(self):
        """CQT should produce logarithmic frequency bins matching musical scales."""
        y = 0.5 * np.sin(2 * np.pi * 440.0 * self.t)
        res = compute_cqt_spectrogram(y, sr=self.sr, n_bins=60, bins_per_octave=12)
        self.assertIn("cqt_mag", res)
        self.assertIn("frequencies", res)
        self.assertEqual(res["cqt_mag"].shape[0], 60)
        self.assertGreater(res["cqt_mag"].shape[1], 0)

    def test_compute_reassigned_spectrogram(self):
        """Reassigned spectrogram should return sharpened magnitude array."""
        y = 0.5 * np.sin(2 * np.pi * 440.0 * self.t)
        res = compute_reassigned_spectrogram(y, sr=self.sr, n_fft=1024, hop_length=256)
        self.assertIn("magnitude", res)
        self.assertIn("freqs", res)
        self.assertIn("times", res)
        self.assertEqual(res["magnitude"].shape[0], 1024 // 2 + 1)

    def test_polyphonic_note_preservation(self):
        """Polyphonic CQT extraction must detect multiple notes simultaneously without argmax collapse."""
        # Create a 2-note dyad: A3 (220 Hz) and E4 (329.6 Hz)
        y = (
            0.5 * np.sin(2 * np.pi * 220.0 * self.t) +
            0.5 * np.sin(2 * np.pi * 329.63 * self.t)
        )
        notes = extract_notes_cqt_polyphonic(
            y, sr=self.sr, n_bins=72, bins_per_octave=12, peak_threshold=0.1
        )
        self.assertGreater(len(notes), 0, "No notes extracted from polyphonic dyad")

        detected_pitches = [n.pitch_hz for n in notes]
        # Should find pitches near 220 and 329.63
        has_220 = any(abs(p - 220.0) < 15.0 for p in detected_pitches)
        has_330 = any(abs(p - 329.63) < 20.0 for p in detected_pitches)
        self.assertTrue(has_220, f"Failed to detect 220 Hz in polyphonic mix. Found: {detected_pitches}")
        self.assertTrue(has_330, f"Failed to detect 330 Hz in polyphonic mix. Found: {detected_pitches}")


if __name__ == "__main__":
    unittest.main()
