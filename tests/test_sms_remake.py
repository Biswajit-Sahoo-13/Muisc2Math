"""Round-trip tests for the SMS song reconstruction entry point."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.tone_remake import recreate_song_from_sinusoids


class TestSmsRemake(unittest.TestCase):
    def test_reconstructs_triad_from_song_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "triad.wav"
            sr = 22050
            t = np.arange(int(2.0 * sr)) / sr
            y = (
                0.60 * np.exp(-1.10 * t) * np.sin(2 * np.pi * 440.0 * t)
                + 0.35 * np.exp(-1.45 * t) * np.sin(2 * np.pi * 554.365 * t)
                + 0.25 * np.exp(-1.80 * t) * np.sin(2 * np.pi * 659.255 * t)
            )
            sf.write(str(src), y, sr)

            res = recreate_song_from_sinusoids(src, tmp_path, max_duration=2.0, sr=sr)

            self.assertEqual(res["status"], "success")
            self.assertGreaterEqual(res["track_count"], 3)
            self.assertGreater(res["snr_db"], 20.0)
            self.assertGreater(res["correlation"], 0.99)

            wav = Path(res["remake_wav"])
            self.assertTrue(wav.exists())
            audio, _ = sf.read(str(wav))
            self.assertEqual(len(audio), len(y))
            self.assertFalse(np.all(np.asarray(audio) == 0.0))

    def test_model_json_is_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "tone.wav"
            sr = 22050
            t = np.arange(int(1.0 * sr)) / sr
            sf.write(str(src), 0.5 * np.sin(2 * np.pi * 440.0 * t), sr)

            res = recreate_song_from_sinusoids(src, tmp_path, max_duration=1.0, sr=sr)
            data = json.loads(Path(res["model_json"]).read_text(encoding="utf-8"))

            self.assertEqual(data["model"], "sinusoidal-modeling-synthesis")
            self.assertIn("theta_k(t)", data["equation"])
            self.assertIn("snr_db", data)
            self.assertIn("correlation", data)
            self.assertEqual(len(data["tracks"]), res["track_count"])
            for track in data["tracks"]:
                self.assertEqual(len(track["f_hz"]), track["frames"])
                self.assertEqual(len(track["a"]), track["frames"])

    def test_no_noise_layer_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "tone.wav"
            sr = 22050
            t = np.arange(int(1.0 * sr)) / sr
            sf.write(str(src), 0.5 * np.sin(2 * np.pi * 440.0 * t), sr)

            res = recreate_song_from_sinusoids(
                src, tmp_path, max_duration=1.0, sr=sr, add_noise_layer=False
            )
            self.assertIsNone(res["noise_layer"])
            self.assertGreater(res["snr_db"], 40.0)

    def test_plot_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "tone.wav"
            sr = 22050
            t = np.arange(int(0.5 * sr)) / sr
            sf.write(str(src), 0.5 * np.sin(2 * np.pi * 440.0 * t), sr)

            res = recreate_song_from_sinusoids(src, tmp_path, max_duration=0.5, sr=sr)
            plot = Path(res["comparison_plot"])
            self.assertTrue(plot.exists())
            self.assertGreater(plot.stat().st_size, 2000)


if __name__ == "__main__":
    unittest.main()