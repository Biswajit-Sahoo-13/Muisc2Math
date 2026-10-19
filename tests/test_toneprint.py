"""Tests for Shazam-style fingerprinting and precise tone identification."""
import unittest
import numpy as np

from src.toneprint import ToneprintDB, identify_tones


def _sine(freq, start, end, sr, amp=0.6, decay=0.0):
    t = np.arange(int((end - start) * sr)) / sr
    env = np.exp(-decay * t)
    return amp * env * np.sin(2.0 * np.pi * freq * t)


class TestToneprint(unittest.TestCase):
    def test_monophonic_melody_tones(self):
        sr = 22050
        notes = [(440.0, 0.0, 0.6), (523.25, 0.6, 1.2), (659.25, 1.2, 1.8)]
        y = np.zeros(int(1.8 * sr))
        for f, s, e in notes:
            seg = _sine(f, s, e, sr)
            y[int(s * sr):int(s * sr) + len(seg)] += seg
        hits = identify_tones(y, sr=sr, segment=0.2)
        midis = [h.pitch_midi for h in hits]
        self.assertIn(69, midis)
        self.assertIn(72, midis)
        self.assertIn(76, midis)
        by_midi = {h.pitch_midi: h for h in hits}
        self.assertAlmostEqual(by_midi[69].pitch_hz, 440.0, delta=3.0)
        self.assertAlmostEqual(by_midi[72].pitch_hz, 523.25, delta=3.0)
        self.assertAlmostEqual(by_midi[76].pitch_hz, 659.25, delta=3.0)

    def test_polyphonic_chord_tones(self):
        sr = 22050
        dur = 1.5
        t = np.arange(int(dur * sr)) / sr
        y = (
            0.5 * np.sin(2.0 * np.pi * 440.0 * t)
            + 0.5 * np.sin(2.0 * np.pi * 554.37 * t)
            + 0.5 * np.sin(2.0 * np.pi * 659.25 * t)
        )
        hits = identify_tones(y, sr=sr, segment=0.3)
        found = {h.pitch_midi for h in hits}
        self.assertTrue({69, 73, 76} <= found, f"missing tones: {found}")

    def test_fingerprint_identifies_time_shifted_excerpt(self):
        sr = 22050
        t = np.arange(int(4.0 * sr)) / sr
        ref = (
            0.5 * np.sin(2.0 * np.pi * 440.0 * t) * (t < 2.0)
            + 0.5 * np.sin(2.0 * np.pi * 659.25 * t) * (t >= 2.0)
        )
        db = ToneprintDB()
        stored = db.add("four_second_demo", ref, sr=sr)
        self.assertGreater(stored, 10)
        query = ref[int(1.0 * sr):int(2.5 * sr)].copy()
        rng = np.random.default_rng(7)
        query = query + 0.005 * rng.standard_normal(query.shape)
        results = db.match(query, sr=sr, top_k=1)
        self.assertTrue(results)
        self.assertEqual(results[0]["name"], "four_second_demo")
        self.assertGreater(results[0]["confidence"], 0.2)
        self.assertAlmostEqual(results[0]["offset"], 1.0, delta=0.12)

    def test_silence_returns_empty(self):
        y = np.zeros(22050, dtype=np.float64)
        self.assertEqual(identify_tones(y, sr=22050), [])


if __name__ == "__main__":
    unittest.main()
