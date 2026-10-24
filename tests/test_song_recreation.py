"""Unit tests for song recreation and multi-track project building."""

import unittest
import numpy as np
from src.song_recreation import (
    build_recreated_project_for_metre_lost,
)
from src.math_track_engine import render


class TestSongRecreation(unittest.TestCase):
    """Test suite for song recreation engine."""

    def test_build_recreated_project(self):
        """Recreated project for Metre - Lost must have 5 tracks and render without errors."""
        project = build_recreated_project_for_metre_lost(bpm=120.0, duration=2.0, sr=22050)
        self.assertEqual(len(project.tracks), 5)
        self.assertEqual(project.bpm, 120.0)

        # Render project
        master, tracks = render(project)
        self.assertEqual(len(tracks), 5)
        self.assertEqual(len(master), int(2.0 * 22050))
        self.assertFalse(np.all(master == 0.0), "Master output is all silence")
        self.assertLessEqual(np.max(np.abs(master)), 1.0)


if __name__ == "__main__":
    unittest.main()
