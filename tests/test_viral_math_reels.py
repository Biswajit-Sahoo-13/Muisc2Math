"""Unit tests for Viral Math Phonk & Reel Equations Suite."""

import unittest
from pathlib import Path
import numpy as np
from src.viral_math_reels import (
    REEL_EQUATIONS,
    synthesize_exact_sinc_drop,
    build_viral_math_phonk_project,
    render_all_viral_reels,
)
from src.math_track_engine import render


class TestViralMathReels(unittest.TestCase):
    """Test suite for viral reel equations and sonifications."""

    def test_reel_equations_registry(self):
        """All viral reel equations must be present in the registry."""
        required = [
            "7_10_sinc_drop",
            "1_10_tresillo_rhythm",
            "2_10_modulo_cowbell",
            "3_10_logistic_808",
            "4_10_shepard_tone",
            "5_10_tangent_screech",
            "6_10_parabolic_pulse",
            "8_10_arctan_saw",
            "9_10_fm_pluck",
        ]
        for key in required:
            self.assertIn(key, REEL_EQUATIONS)
            self.assertIn("latex", REEL_EQUATIONS[key])
            self.assertIn("description", REEL_EQUATIONS[key])

    def test_synthesize_exact_sinc_drop(self):
        """#7/10 SINC DROP must synthesize finite non-silent audio with zero DC offset."""
        audio, t, raw = synthesize_exact_sinc_drop(duration=2.0, bpm=140.0, f0=55.0, sr=22050)
        self.assertEqual(len(audio), int(2.0 * 22050))
        self.assertFalse(np.all(audio == 0.0), "Audio is silence")
        self.assertLessEqual(np.max(np.abs(audio)), 1.0)
        # DC offset should be minimal
        self.assertAlmostEqual(float(np.mean(audio)), 0.0, places=2)

    def test_build_viral_math_phonk_project(self):
        """Viral Math Phonk project must have 5 tracks and render without errors."""
        project = build_viral_math_phonk_project(bpm=140.0, duration=2.0, sr=22050)
        self.assertEqual(len(project.tracks), 5)
        master, tracks = render(project)
        self.assertEqual(len(master), int(2.0 * 22050))
        self.assertFalse(np.all(master == 0.0))
        self.assertLessEqual(np.max(np.abs(master)), 1.0)

    def test_render_all_viral_reels(self):
        """render_all_viral_reels must output WAVs, plot PNG, and project JSON."""
        res = render_all_viral_reels(output_dir="data/output", duration=2.0, bpm=140.0, sr=22050)
        self.assertEqual(res["status"], "success")
        self.assertTrue(Path(res["sinc_drop_wav"]).exists())
        self.assertTrue(Path(res["sinc_drop_plot"]).exists())
        self.assertTrue(Path(res["complete_phonk_wav"]).exists())
        self.assertTrue(Path(res["project_json"]).exists())


if __name__ == "__main__":
    unittest.main()
