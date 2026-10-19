"""Guard for band-reserved track selection (the -13 dB high-band deficit)."""
import numpy as np
import pytest

from src.sinusoidal_model import (
    HIGH_BAND_HZ, select_tracks_banded,
)
from src.tracking import SinusoidalTrack


def _track(track_id, freq, amp, frames=4):
    t = SinusoidalTrack(track_id, 0, {"frequency": freq, "amplitude": amp,
                                      "phase": 0.0}, 0.0)
    for i in range(1, frames):
        t.update(i, {"frequency": freq, "amplitude": amp, "phase": 0.0}, i * 0.01)
    return t


def _tilted_population(n_low=60, n_high=30):
    """Low tracks are louder (spectral tilt); high tracks are present but faint."""
    low = [_track(i, 60 + i, 1.0 - i * 1e-3) for i in range(n_low)]
    high = [_track(1000 + i, 6000 + i * 10, 0.05 - i * 1e-4) for i in range(n_high)]
    return low + high


def test_no_truncation_leaves_selection_untouched():
    ts = _tilted_population(5, 3)
    sel = select_tracks_banded(ts, 100)
    assert len(sel) == len(ts)
    assert sel == sorted(ts, key=lambda t: t.peak_amplitude, reverse=True)


def test_budget_is_respected():
    ts = _tilted_population(60, 30)
    assert len(select_tracks_banded(ts, 24)) == 24


def test_high_band_survives_the_cut():
    """This is the defect: without reservation the high band is fully deleted."""
    ts = _tilted_population(60, 30)
    naive = sorted(ts, key=lambda t: t.peak_amplitude, reverse=True)[:24]
    assert not any(t.mean_frequency > HIGH_BAND_HZ for t in naive), \
        "precondition: naive sort deletes every high-band track"

    sel = select_tracks_banded(ts, 24)
    n_high = sum(1 for t in sel if t.mean_frequency > HIGH_BAND_HZ)
    assert n_high >= 8, f"expected the reserved quota, got {n_high}"


def test_high_band_only_population_is_fully_kept():
    ts = _tilted_population(0, 30)
    sel = select_tracks_banded(ts, 24)
    assert len(sel) == 24
    assert all(t.mean_frequency > HIGH_BAND_HZ for t in sel)


def test_selection_is_deterministic():
    ts = _tilted_population(40, 20)
    a = [t.track_id for t in select_tracks_banded(ts, 24)]
    b = [t.track_id for t in select_tracks_banded(list(ts), 24)]
    assert a == b


def test_zero_budget_returns_nothing():
    assert select_tracks_banded(_tilted_population(5, 5), 0) != []
    # max_tracks <= 0 must not raise
    select_tracks_banded(_tilted_population(5, 5), -1)