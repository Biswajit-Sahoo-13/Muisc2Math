"""Tests for the Math-DAW multi-track equation engine.

Validates:
1. Time variable computation (t, u, s, beat)
2. Pattern → frequency/gate arrays
3. Phase-continuous oscillator (cumsum)
4. Safe expression evaluation
5. Equation rendering & DC removal
6. Master bus soft-clipping
7. JSON project round-trip (save/load)
8. Demo project rendering
"""

import unittest
import json
import tempfile
import numpy as np
from pathlib import Path

from src.safe_eval import safe_eval, SafeEvalError
from src.math_track_engine import (
    Project, Track, render, render_to_wav,
    midi2freq, freq2midi,
    demo_sinc_drop, demo_math_remix, demo_ambient,
)


class TestMidi(unittest.TestCase):
    """Tests for MIDI ↔ frequency conversion."""

    def test_midi_69_is_a4(self):
        self.assertAlmostEqual(midi2freq(69), 440.0)

    def test_midi_60_is_middle_c(self):
        self.assertAlmostEqual(midi2freq(60), 261.6256, places=3)

    def test_freq2midi_round_trip(self):
        for midi_note in [36, 48, 60, 69, 72, 84]:
            freq = midi2freq(midi_note)
            result = freq2midi(freq)
            self.assertEqual(result, midi_note)


class TestTimeVars(unittest.TestCase):
    """Tests for the Project time variable computation."""

    def test_time_array_length(self):
        p = Project(bpm=120, duration=2.0, sr=44100)
        t, u, s, beat = p.time_vars()
        expected_len = int(44100 * 2.0)
        self.assertEqual(len(t), expected_len)
        self.assertEqual(len(u), expected_len)
        self.assertEqual(len(s), expected_len)

    def test_beat_duration(self):
        p = Project(bpm=120, duration=1.0, sr=1000)
        _, _, _, beat = p.time_vars()
        self.assertAlmostEqual(beat, 0.5)  # 120 BPM = 0.5s per beat

    def test_u_range(self):
        """u should be in [0, 1) within each beat."""
        p = Project(bpm=120, duration=2.0, sr=44100)
        _, u, _, _ = p.time_vars()
        self.assertGreaterEqual(np.min(u), 0.0)
        self.assertLess(np.max(u), 1.0)

    def test_step_index_increments(self):
        """s should increment at each beat boundary."""
        p = Project(bpm=120, duration=2.0, sr=1000)
        t, _, s, beat = p.time_vars()
        # At t=0.5s (beat boundary), step should change from 0 to 1
        idx_half = int(0.5 * 1000)
        self.assertEqual(s[0], 0.0)
        self.assertEqual(s[idx_half], 1.0)


class TestPatternArrays(unittest.TestCase):
    """Tests for MIDI pattern → frequency/gate arrays."""

    def test_null_pattern(self):
        """No pattern → 440Hz, gate always on."""
        p = Project(bpm=120, duration=1.0, sr=1000)
        _, _, s, _ = p.time_vars()
        freq, gate = p.pattern_arrays(s, None)
        self.assertTrue(np.all(freq == 440.0))
        self.assertTrue(np.all(gate == 1.0))

    def test_pattern_with_rest(self):
        """None entries should produce gate=0 and freq=0."""
        p = Project(bpm=120, duration=1.0, sr=1000)
        _, _, s, _ = p.time_vars()
        pattern = [60, None, 60, None]
        freq, gate = p.pattern_arrays(s, pattern)
        # On the first beat (step 0), should have MIDI 60 frequency
        self.assertAlmostEqual(freq[0], midi2freq(60), places=1)
        self.assertEqual(gate[0], 1.0)

    def test_pattern_loops(self):
        """Pattern should loop after its length."""
        p = Project(bpm=120, duration=4.0, sr=1000)
        _, _, s, _ = p.time_vars()
        pattern = [60, 72]  # Two-step pattern
        freq, gate = p.pattern_arrays(s, pattern)
        # Steps 0 and 2 should have MIDI 60; steps 1 and 3 should have MIDI 72
        beat_dur = 0.5  # 120 BPM
        idx_step0 = 0
        idx_step1 = int(beat_dur * 1000)
        idx_step2 = int(2 * beat_dur * 1000)
        self.assertAlmostEqual(freq[idx_step0], midi2freq(60), places=1)
        self.assertAlmostEqual(freq[idx_step1], midi2freq(72), places=1)
        self.assertAlmostEqual(freq[idx_step2], midi2freq(60), places=1)


class TestSafeEval(unittest.TestCase):
    """Tests for the safe expression evaluator."""

    def test_basic_arithmetic(self):
        result = safe_eval("2 + 3")
        self.assertEqual(result, 5)

    def test_numpy_array_ops(self):
        t = np.linspace(0, 1, 100)
        result = safe_eval("t * 2.0", {"t": t})
        np.testing.assert_array_almost_equal(result, t * 2.0)

    def test_sin_function(self):
        t = np.linspace(0, 2 * np.pi, 100)
        result = safe_eval("sin(t)", {"t": t})
        np.testing.assert_array_almost_equal(result, np.sin(t))

    def test_sinc_function(self):
        t = np.linspace(-5, 5, 100)
        result = safe_eval("sinc(t)", {"t": t})
        np.testing.assert_array_almost_equal(result, np.sinc(t))

    def test_complex_equation(self):
        """Test a real equation from the viral demos."""
        t = np.linspace(0, 1, 1000)
        u = t
        ph = np.cumsum(2.0 * np.pi * 440.0 / 1000 * np.ones_like(t))
        result = safe_eval("0.55*sin(ph)*exp(-2.5*u)", {"ph": ph, "u": u})
        expected = 0.55 * np.sin(ph) * np.exp(-2.5 * u)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rejects_import(self):
        with self.assertRaises(SafeEvalError):
            safe_eval("__import__('os')")

    def test_rejects_attribute_access(self):
        with self.assertRaises(SafeEvalError):
            safe_eval("t.__class__", {"t": np.zeros(10)})

    def test_rejects_subscript(self):
        with self.assertRaises(SafeEvalError):
            safe_eval("t[0]", {"t": np.zeros(10)})

    def test_pi_constant(self):
        result = safe_eval("pi")
        self.assertAlmostEqual(result, np.pi)

    def test_ternary_expression(self):
        t = np.array([0.3, 0.5, 0.8])
        result = safe_eval("1.0 if t > 0.5 else 0.0", {"t": t})
        expected = np.where(t > 0.5, 1.0, 0.0)
        np.testing.assert_array_almost_equal(result, expected)


class TestRendering(unittest.TestCase):
    """Tests for the equation rendering pipeline."""

    def test_single_sine_track(self):
        """A simple sine wave should produce non-zero output."""
        p = Project(bpm=120, duration=0.5, sr=22050)
        p.add("sine", "sin(ph)", pattern=[69], gain=1.0)
        master, tracks = render(p)
        self.assertEqual(len(tracks), 1)
        self.assertGreater(np.max(np.abs(master)), 0.01)

    def test_dc_removal(self):
        """A constant equation should have its DC offset removed."""
        p = Project(bpm=120, duration=0.5, sr=22050)
        p.add("dc_test", "1.0", gain=1.0)
        master, _ = render(p)
        # After DC removal and tanh, mean should be near zero
        self.assertAlmostEqual(np.mean(master), 0.0, places=2)

    def test_multi_track_sum(self):
        """Multiple tracks should be summed together."""
        p = Project(bpm=120, duration=0.5, sr=22050)
        p.add("track1", "sin(ph)", pattern=[69], gain=0.5)
        p.add("track2", "sin(ph)", pattern=[69], gain=0.5)
        _, tracks = render(p)
        self.assertEqual(len(tracks), 2)

    def test_master_peak_normalized(self):
        """Master output should be peak-normalized to ≤ 0.9."""
        p = Project(bpm=120, duration=1.0, sr=22050)
        p.add("loud", "sin(ph)", pattern=[69], gain=5.0)
        master, _ = render(p)
        self.assertLessEqual(np.max(np.abs(master)), 0.91)

    def test_gate_silences_rests(self):
        """Gate should silence samples during rest steps (None in pattern)."""
        p = Project(bpm=120, duration=1.0, sr=22050)
        p.add("gated", "sin(ph) * g", pattern=[69, None], gain=1.0)
        master, tracks = render(p)
        _, track_audio = tracks[0]
        # During the second beat (rest), audio should be zero
        beat_samples = int(0.5 * 22050)  # 120 BPM = 0.5s per beat
        rest_region = track_audio[beat_samples:2*beat_samples]
        self.assertAlmostEqual(np.max(np.abs(rest_region)), 0.0, places=5)

    def test_phase_continuity(self):
        """Phase should not have discontinuities at step boundaries."""
        p = Project(bpm=120, duration=1.0, sr=44100)
        p.add("pitch_change", "sin(ph)", pattern=[60, 72], gain=1.0)
        _, tracks = render(p)
        _, audio = tracks[0]
        # Check for large jumps (clicks) at beat boundary
        beat_idx = int(0.5 * 44100)
        # The difference between adjacent samples should be bounded
        diff = np.abs(np.diff(audio[beat_idx-5:beat_idx+5]))
        # A click would produce a diff > 1.0; smooth transition should be < 0.5
        self.assertLess(np.max(diff), 0.5)

    def test_error_in_equation_produces_silence(self):
        """A bad equation should produce silence, not crash."""
        p = Project(bpm=120, duration=0.5, sr=22050)
        p.add("bad", "invalid_func(t)", gain=1.0)
        master, tracks = render(p)
        _, audio = tracks[0]
        self.assertTrue(np.all(audio == 0.0))


class TestProjectJSON(unittest.TestCase):
    """Tests for project JSON serialization round-trip."""

    def test_save_load_round_trip(self):
        p = Project(bpm=140, duration=4.0, seed=42)
        p.add("kick", "sinc(26*(u-0.08))", [36, 36, 36, 36], 1.0)
        p.add("bass", "0.5*sin(ph)*exp(-2*u)", [33, None, 31, None], 0.8)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "test_project.json"
            p.save(json_path)

            loaded = Project.load(json_path)
            self.assertEqual(loaded.bpm, 140)
            self.assertEqual(loaded.duration, 4.0)
            self.assertEqual(loaded.seed, 42)
            self.assertEqual(len(loaded.tracks), 2)
            self.assertEqual(loaded.tracks[0].name, "kick")
            self.assertEqual(loaded.tracks[0].equation, "sinc(26*(u-0.08))")
            self.assertEqual(loaded.tracks[1].pattern, [33, None, 31, None])
            self.assertAlmostEqual(loaded.tracks[1].gain, 0.8)

    def test_rendered_audio_matches_after_reload(self):
        """Audio from a saved/loaded project should be identical."""
        p = demo_sinc_drop()
        master1, _ = render(p)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.json"
            p.save(path)
            p2 = Project.load(path)
            master2, _ = render(p2)

        np.testing.assert_array_almost_equal(master1, master2)


class TestDemoProjects(unittest.TestCase):
    """Tests that all demo presets render without errors."""

    def test_sinc_drop_renders(self):
        p = demo_sinc_drop()
        master, tracks = render(p)
        self.assertEqual(len(tracks), 4)
        self.assertGreater(np.max(np.abs(master)), 0.1)

    def test_math_remix_renders(self):
        p = demo_math_remix()
        master, tracks = render(p)
        self.assertEqual(len(tracks), 4)
        self.assertGreater(np.max(np.abs(master)), 0.1)

    def test_ambient_renders(self):
        p = demo_ambient()
        master, tracks = render(p)
        self.assertEqual(len(tracks), 3)
        self.assertGreater(np.max(np.abs(master)), 0.1)

    def test_sinc_drop_wav_output(self):
        p = demo_sinc_drop()
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "test.wav"
            render_to_wav(p, wav_path)
            self.assertTrue(wav_path.exists())
            self.assertGreater(wav_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
