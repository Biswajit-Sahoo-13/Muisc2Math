"""Tests for the exciter: the broadband strike of a struck body.

A percussive hit is a strike exciting a set of resonances. The modes are
the resonances; these tests pin the strike.
"""
import os
import unittest

import numpy as np

from src.res_percussive import broadband_envelope, resonant_burst_hypothesis
from src.res_synthesis import _render_partial

SR = 22050


class TestBroadbandEnvelope(unittest.TestCase):
    def test_the_median_ignores_a_narrow_peak(self):
        # A spectrum that is flat except for one very loud narrow peak:
        # the peak is a mode, not broadband energy, so the band median
        # must not follow it.
        bins = np.linspace(10.0, 10000.0, 512)
        mag = np.ones_like(bins) * 0.1
        peak = int(np.argmin(np.abs(bins - 300.0)))
        mag[peak] = 50.0
        centers, mags = broadband_envelope(mag, bins, band_lo=45.0, band_hi=9000.0)
        self.assertTrue(centers)
        self.assertLess(max(mags), 1.0)

    def test_an_empty_spectrum_gives_nothing(self):
        centers, mags = broadband_envelope(np.array([]), np.array([]))
        self.assertEqual(centers, [])
        self.assertEqual(mags, [])

    def test_bands_are_log_spaced_and_ordered(self):
        bins = np.linspace(10.0, 10000.0, 1024)
        mag = np.ones_like(bins)
        centers, mags = broadband_envelope(mag, bins, band_lo=45.0, band_hi=9000.0)
        self.assertEqual(len(centers), len(mags))
        self.assertTrue(all(centers[i] < centers[i + 1] for i in range(len(centers) - 1)))


class TestResonantBurst(unittest.TestCase):
    def _burst_target(self, n=4096, hit=1024, length=256):
        """A seeded noise burst: the thing the exciter must explain."""
        rng = np.random.default_rng(7)
        np.arange(n) / SR
        noise = rng.standard_normal(n)
        env = np.zeros(n)
        env[hit:hit + length] = np.exp(-np.arange(length) / 60.0)
        return noise * env, hit

    def test_it_recovers_a_burst_it_can_explain(self):
        target, hit = self._burst_target()
        bins = np.fft.rfftfreq(target.size, 1.0 / SR)
        mag = np.abs(np.fft.rfft(target))
        centers, mags = broadband_envelope(mag, bins)
        self.assertTrue(centers, "the spectrum and the bins must agree in length")
        p = resonant_burst_hypothesis(
            target, SR, 0, 1024.0 / SR, hit, centers, mags,
        )
        self.assertIsNotNone(p, "the exciter must be proposed for a noise burst")
        self.assertEqual(p["model"], "filtered_noise")
        self.assertTrue(p["unit_step"], "a strike must not sound before the hit")
        self.assertGreater(p["amplitude"], 0.0)

    def test_it_refuses_a_target_it_cannot_explain(self):
        # A pure tone is not a broadband strike.
        n = 4096
        t = np.arange(n) / SR
        target = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
        bins = np.fft.rfftfreq(n, 1.0 / SR)
        mag = np.abs(np.fft.rfft(target))
        centers, mags = broadband_envelope(mag, bins)
        p = resonant_burst_hypothesis(
            target, SR, 0, 1024.0 / SR, 1024, centers, mags,
        )
        self.assertIsNone(p, "a pure tone must not acquire a broadband strike")

    def test_it_needs_a_spectrum(self):
        p = resonant_burst_hypothesis(
            np.zeros(4096), SR, 0, 1024.0 / SR, 1024, [], [],
        )
        self.assertIsNone(p)


class TestRendererHonoursOneSidedAndDecay(unittest.TestCase):
    """The renderer must agree with what the bank asks for."""

    def setUp(self):
        # The one-sidedness and the decay of a stochastic channel ship with
        # the exciter, which is behind a switch until it is measured.
        self._old = os.environ.get("M2M_EXCITER")
        os.environ["M2M_EXCITER"] = "1"

    def tearDown(self):
        if self._old is None:
            os.environ.pop("M2M_EXCITER", None)
        else:
            os.environ["M2M_EXCITER"] = self._old

    def _noise_partial(self, **extra):
        p = {
            "amplitude": 0.5,
            "envelope_decay": 0.0,
            "envelope_t0": 0.5,
            "model": "filtered_noise",
            "start_time": 0.5,
            "end_time": 0.55,
            "band_centers": [100.0, 1000.0, 5000.0],
            "band_mags": [1.0, 1.0, 1.0],
            "seed": 3,
        }
        p.update(extra)
        return p

    def test_a_one_sided_noise_channel_is_silent_before_its_onset(self):
        n = 22050
        sig = _render_partial(
            self._noise_partial(unit_step=True), n, SR, start_sample=0,
            stft_half_window_s=1024.0 / SR,
        )
        onset = int(0.5 * SR)
        before = float(np.max(np.abs(sig[: onset - 200])))
        self.assertEqual(before, 0.0, "a one-sided noise channel leaked before its onset")
        self.assertGreater(float(np.max(np.abs(sig[onset:]))), 0.0)

    def test_a_decay_rings_the_burst_down(self):
        n = 22050
        decay = 40.0
        sig = _render_partial(
            self._noise_partial(unit_step=True, envelope_decay=decay), n, SR,
            start_sample=0, stft_half_window_s=1024.0 / SR,
        )
        onset = int(0.5 * SR)
        early = float(np.sqrt(np.mean(sig[onset:onset + 100] ** 2)))
        late = float(np.sqrt(np.mean(sig[onset + 2000:onset + 2400] ** 2)))
        self.assertGreater(early, late, "the burst did not ring down")

    def test_a_stationary_noise_channel_is_unchanged(self):
        # The noise floor class wants a stationary channel, so a zero decay
        # must leave the amplitude alone.
        n = 22050
        sig = _render_partial(
            self._noise_partial(), n, SR, start_sample=0,
            stft_half_window_s=1024.0 / SR,
        )
        onset = int(0.5 * SR)
        early = float(np.sqrt(np.mean(sig[onset:onset + 100] ** 2)))
        late = float(np.sqrt(np.mean(sig[onset + 2000:onset + 2400] ** 2)))
        self.assertAlmostEqual(early, late, delta=0.35 * max(early, 1e-12))


if __name__ == "__main__":
    unittest.main()
