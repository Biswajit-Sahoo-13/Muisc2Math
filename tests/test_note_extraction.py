"""Tests for note extraction (pyin) and harmonic grouping."""

import unittest
import numpy as np
import tempfile
from pathlib import Path

from src.synthesis import generate_benchmark_melody, render_model, save_audio
from src.analysis import load_audio, compute_normalized_stft
from src.tracking import track_sinusoids
from src.note_extraction import extract_melody_pyin, group_partials_into_notes, NoteEvent


class TestNoteExtraction(unittest.TestCase):
    def test_pyin_melody_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            melody_wav = Path(tmpdir) / "math_melody.wav"
            # Notes: A4 (440 Hz, MIDI 69), C5 (523.25 Hz, MIDI 72), E5 (659.25 Hz, MIDI 76)
            generate_benchmark_melody(melody_wav, duration=3.0, sr=22050)

            y, sr = load_audio(melody_wav, target_sr=22050)
            notes = extract_melody_pyin(y, sr=sr, min_note_duration=0.1)

            self.assertGreaterEqual(len(notes), 2)
            detected_midis = [n.pitch_midi for n in notes]

            # Verify presence of the target pitches
            self.assertTrue(any(abs(m - 69) <= 1 for m in detected_midis), f"A4 (69) not in {detected_midis}")
            self.assertTrue(any(abs(m - 72) <= 1 for m in detected_midis), f"C5 (72) not in {detected_midis}")
            self.assertTrue(any(abs(m - 76) <= 1 for m in detected_midis), f"E5 (76) not in {detected_midis}")

    def test_harmonic_grouping(self):
        # Create a synthetic note event at 220 Hz (A3)
        note = NoteEvent(
            pitch_midi=57,
            pitch_hz=220.0,
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
        )

        # Mock tracks representing fundamental (220 Hz), 2nd harmonic (440 Hz), 3rd harmonic (660 Hz)
        class MockTrack:
            def __init__(self, f, a, t0, dur):
                self.mean_frequency = f
                self.peak_amplitude = a
                self.times = [t0, t0 + dur]
                self.duration = dur

        tracks = [
            MockTrack(220.2, 0.8, 0.0, 1.0),
            MockTrack(439.8, 0.4, 0.0, 1.0),
            MockTrack(660.1, 0.2, 0.0, 1.0),
            MockTrack(1000.0, 0.1, 0.0, 1.0),  # Non-harmonic
        ]

        grouped_notes = group_partials_into_notes(tracks, [note], max_harmonic_num=8)
        self.assertEqual(len(grouped_notes), 1)
        partials = grouped_notes[0].partials
        self.assertEqual(len(partials), 3)

        harmonic_nums = [p["harmonic_number"] for p in partials]
        self.assertIn(1, harmonic_nums)
        self.assertIn(2, harmonic_nums)
        self.assertIn(3, harmonic_nums)


if __name__ == "__main__":
    unittest.main()
