"""Tests for the DAW-style arrangement timeline graph and realtime player."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.math_track_engine import Project, demo_sinc_drop, render
from src.math_track_visualize import (
    build_timeline_figure,
    build_timeline_player_html,
    plot_project_timeline,
    save_timeline_player,
)
import matplotlib.pyplot as plt


class TestTimeline(unittest.TestCase):
    def test_build_timeline_figure(self):
        proj = demo_sinc_drop()
        fig = build_timeline_figure(proj)
        try:
            self.assertEqual(len(fig.axes), 1)
            ax = fig.axes[0]
            labels = [t.get_text() for t in ax.get_yticklabels()]
            self.assertEqual(labels, [t.name for t in proj.tracks])
            # Blocks were drawn (timeline patches + lane strips)
            self.assertGreater(len(ax.patches), len(proj.tracks))
        finally:
            plt.close(fig)

    def test_save_timeline_png(self):
        proj = demo_sinc_drop()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "timeline.png"
            plot_project_timeline(proj, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_realtime_player_html(self):
        proj = Project(bpm=120.0, duration=2.0, sr=22050)
        proj.add("lead", "0.5*sin(ph)*exp(-2*u)*g", pattern=[69, None, 72, None], gain=1.0)
        proj.add("hats", "0.1*rnd*exp(-30*u)", pattern=[None, 42, None, 42], gain=1.0)
        master, _ = render(proj)
        html = build_timeline_player_html(proj, master, proj.sr, uid="test")
        self.assertIn("lead", html)
        self.assertIn("data:audio/wav;base64,", html)
        self.assertIn("requestAnimationFrame", html)
        self.assertIn("test_play", html)
        self.assertNotIn("undefined", html)

    def test_realtime_player_js_syntax_valid(self):
        if shutil.which("node") is None:
            self.skipTest("node not available")
        proj = demo_sinc_drop()
        master, _ = render(proj)
        html = build_timeline_player_html(proj, master, proj.sr, uid="test")
        start = html.index("<script>") + len("<script>")
        end = html.index("</script>")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(html[start:end])
            path = f.name
        res = subprocess.run(["node", "--check", path], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_save_timeline_player_file(self):
        proj = demo_sinc_drop()
        master, _ = render(proj)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "player.html"
            save_timeline_player(proj, master, proj.sr, out)
            self.assertTrue(out.exists())
            content = out.read_text(encoding="utf-8")
            self.assertIn("<canvas", content)
            self.assertIn("data:audio/wav;base64,", content)


if __name__ == "__main__":
    unittest.main()
