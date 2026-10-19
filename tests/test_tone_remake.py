"""Tests for tone-to-equation song recreation."""
import unittest
import numpy as np
import soundfile as sf
import tempfile
from pathlib import Path

from src.tone_remake import recreate_song_from_tones, tone_records_to_bytebeat


class TestToneRemake(unittest.TestCase):
    def test_melody_remake(self):
        sr = 22050
        notes = [(440.0, 0.0, 0.6), (523.25, 0.6, 1.2), (659.25, 1.2, 1.8)]
        y = np.zeros(int(1.8 * sr))
        for f, s, e in notes:
            n = int((e - s) * sr)
            t = np.arange(n) / sr
            y[int(s * sr):int(s * sr) + n] += 0.6 * np.sin(2.0 * np.pi * f * t)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "melody.wav"
            sf.write(str(src), y, sr)
            res = recreate_song_from_tones(src, tmp, max_duration=2.0, sr=sr)
            self.assertEqual(res["status"], "success")
            self.assertGreaterEqual(res["tone_count"], 2)
            self.assertGreaterEqual(res["track_count"], 2)
            midis = {t["pitch_midi"] for t in res["tones"]}
            self.assertTrue({69, 72, 76} <= midis, f"missing tones: {midis}")
            wav = Path(res["remake_wav"])
            self.assertTrue(wav.exists())
            audio, _ = sf.read(str(wav))
            self.assertFalse(np.all(np.asarray(audio) == 0.0))
            bytebeat = Path(res["bytebeat_txt"])
            self.assertTrue(bytebeat.exists())
            code = bytebeat.read_text(encoding="utf-8")
            self.assertIn("Math.sin", code)
            self.assertIn("T>=", code)

    def test_bytebeat_export_layers(self):
        tones = [
            {"pitch_midi": 69, "pitch_hz": 440.0, "start_time": 0.0,
             "end_time": 1.0, "formula": "fm_synthesis_bell",
             "A": 0.6, "decay_lambda": 2.0, "timbre_param": 1.2},
            {"pitch_midi": 45, "pitch_hz": 110.0, "start_time": 0.0,
             "end_time": 1.0, "formula": "pure_sine",
             "A": 0.5, "decay_lambda": 1.0, "timbre_param": 1.0},
        ]
        code = tone_records_to_bytebeat(tones, 120.0, 0.5, {0, 2}, 4)
        self.assertIn("440.0", code)
        self.assertIn("HP=[1,0,1,0]", code)
        self.assertIn("Math.tanh", tone_records_to_bytebeat(
            [{**tones[0], "formula": "soft_square_tanh"}], 120.0, 0.5))


if __name__ == "__main__":
    unittest.main()
