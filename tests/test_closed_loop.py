"""Phase 3: Closed-Loop Verification test.

Validates the full cycle:
1. Generate synthetic benchmark (Math -> Audio: math_chord.wav)
2. Reverse engineer components (Audio -> Math: equations.json)
3. Re-synthesize from discovered equations (Math -> Audio: rendered_from_equations.wav)
4. Evaluate residual error, SNR, and discovered parameter accuracy
"""

import unittest
import tempfile
from pathlib import Path

from src.synthesis import generate_benchmark_chord, render_model, save_audio
from src.analysis import load_audio
from src.tracking import discover_equations_from_audio
from src.residual import compute_error_metrics


class TestClosedLoop(unittest.TestCase):
    def test_closed_loop_chord_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chord_wav = tmp_path / "math_chord.wav"
            eq_json = tmp_path / "equations.json"
            recon_wav = tmp_path / "rendered_from_equations.wav"

            # 1. Forward Synthesis: Generate benchmark chord (A4, C#5, E5 with exponential decay)
            generate_benchmark_chord(chord_wav, duration=2.0, sr=44100)
            self.assertTrue(chord_wav.exists())

            # 2. Reverse Engineering: Discover equations from audio
            discovered_model = discover_equations_from_audio(
                chord_wav,
                n_fft=4096,
                hop_length=1024,
                threshold=0.01,
                out_json_path=eq_json,
            )
            self.assertTrue(eq_json.exists())

            # Verify discovered components
            comps = discovered_model["components"]
            self.assertGreaterEqual(len(comps), 3)

            # Check if all 3 frequencies were discovered within 1.0 Hz
            gt_freqs = [440.0, 554.365, 659.255]
            disc_freqs = [c["frequency"] for c in comps]

            for gf in gt_freqs:
                closest_f = min(disc_freqs, key=lambda f: abs(f - gf))
                self.assertLess(abs(closest_f - gf), 1.0, f"Expected {gf} Hz, closest discovered was {closest_f} Hz")

            # 3. Re-synthesis: Render audio solely from equations.json
            recon_audio, sr = render_model(eq_json, duration=2.0, sr=44100)
            save_audio(recon_wav, recon_audio, sr)
            self.assertTrue(recon_wav.exists())

            # 4. Error analysis: Compare original vs re-synthesized audio
            orig_audio, _ = load_audio(chord_wav, target_sr=sr)
            metrics = compute_error_metrics(orig_audio, recon_audio, sr=sr)

            # Spectral error should be low
            self.assertLess(metrics["spectral_mse"], 0.05)
            # Log-spectral distance should be low (< 10 dB)
            self.assertLess(metrics["log_spectral_distance_db"], 10.0)


if __name__ == "__main__":
    unittest.main()
