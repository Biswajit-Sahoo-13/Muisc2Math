"""The note layer: frequency-native detection (beat pairs + 12-TET snap).

Pins the first increment of the user's "detect the frequency, write the
equation from it" idea: beat-pair detection over fitted partials, with
the carrier's 12-TET snap recorded. The renderer is untouched (pairs are
recorded, not re-rendered), so analyser/renderer parity is safe.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.note_layer import (
    detect_beat_pairs,
    detect_note_events,
    hz_to_note,
    snap_carrier,
    sub_bass_note_partials,
)


def _p(f, amp=0.3, decay=1.0, t0=0.0, t1=1.0, model="damped"):
    return {"frequency": f, "amplitude": amp, "envelope_decay": decay,
            "start_time": t0, "end_time": t1, "model": model}


def test_hz_to_note_snaps_and_reports_cents() -> None:
    n = hz_to_note(440.0)
    assert n["note"] == "A4" and n["note_hz"] == 440.0 and n["cents"] == 0.0
    # 30 cents flat of A4: still snapped to A4, deviation recorded.
    flat = hz_to_note(440.0 * 2 ** (-30 / 1200))
    assert flat["note"] == "A4"
    assert flat["cents"] == pytest_approx(-30.0)


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=0.5)


def test_detect_beat_pairs_finds_a_detuned_unison() -> None:
    """Two layers 4 Hz apart at 220 Hz (~31 cents) - the 808 unison case."""
    partials = [
        _p(220.0, amp=0.5, decay=1.2, t0=0.0, t1=1.0),
        _p(224.0, amp=0.4, decay=2.0, t0=0.05, t1=1.0),
        _p(880.0, amp=0.1),  # far away, must not pair
    ]
    pairs = detect_beat_pairs(partials)
    assert len(pairs) == 1
    r = pairs[0]
    assert r["f_carrier_hz"] == 222.0
    assert r["f_beat_hz"] == 4.0  # intensity beat rate = f2 - f1
    assert r["lam1"] == 1.2 and r["lam2"] == 2.0  # decays kept independent
    assert r["carrier_note"] in ("A3",)  # 222 Hz -> A3 +17c


def test_detect_beat_pairs_uses_a_cents_window_not_just_hz() -> None:
    """At sub-bass, 15 Hz is 470 cents - it must NOT merge different notes.
    A 10 Hz gap at 40 Hz (~410 cents) is two notes, not a beat pair."""
    partials = [_p(40.0, amp=0.5, t0=0.0, t1=1.0),
                _p(50.0, amp=0.5, t0=0.0, t1=1.0)]
    assert detect_beat_pairs(partials) == []


def test_detect_beat_pairs_ignores_non_overlapping_supports() -> None:
    partials = [_p(220.0, t0=0.0, t1=0.3), _p(223.0, t0=0.7, t1=1.0)]
    assert detect_beat_pairs(partials) == []


def test_detect_beat_pairs_skips_noise_and_glides() -> None:
    partials = [
        _p(220.0, t0=0.0, t1=1.0),
        _p(223.0, t0=0.0, t1=1.0),
        _p(221.0, model="filtered_noise", t0=0.0, t1=1.0),
        {"frequency": 222.0, "amplitude": 0.5, "model": "glide",
         "start_time": 0.0, "end_time": 1.0},
    ]
    pairs = detect_beat_pairs(partials)
    assert len(pairs) == 1  # only the two damped partials pair


def test_snappable_flag_marks_true_pitch_carriers() -> None:
    """A carrier within a quarter tone of a note is snappable; deeper than
    that it is a glide or a non-440 tuning (every frequency is within 50
    cents of SOME note, so the tolerance must be tighter than that)."""
    near = [_p(441.5, t0=0.0, t1=1.0), _p(444.5, t0=0.0, t1=1.0)]
    far = [_p(453.5, t0=0.0, t1=1.0), _p(456.5, t0=0.0, t1=1.0)]  # ~455 Hz
    assert detect_beat_pairs(near)[0]["snappable"] is True
    assert detect_beat_pairs(far)[0]["snappable"] is False


def test_strongest_pairs_sort_first() -> None:
    partials = [
        _p(220.0, amp=0.2, t0=0.0, t1=1.0), _p(222.0, amp=0.2, t0=0.0, t1=1.0),
        _p(440.0, amp=0.9, t0=0.0, t1=1.0), _p(442.0, amp=0.9, t0=0.0, t1=1.0),
    ]
    pairs = detect_beat_pairs(partials)
    assert len(pairs) == 2
    assert pairs[0]["f_carrier_hz"] == 441.0  # the louder pair first


# --- step 2: 12-TET snap of a beat pair (the carrier, never the layers) ---

def test_snap_carrier_moves_the_center_keeps_the_beat() -> None:
    """The snap moves the pair's CENTER to the grid and preserves the beat
    - the detune (the percept) is untouched, symmetric around the master."""
    pair = {"f_carrier_hz": 443.0, "f_beat_hz": 3.0,
            "carrier_note": "A4", "carrier_note_hz": 440.0,
            "carrier_cents_off": 11.8}
    out = snap_carrier(pair)
    assert out["applied"] is True
    assert out["f1_snapped_hz"] == pytest.approx(438.5, abs=0.01)
    assert out["f2_snapped_hz"] == pytest.approx(441.5, abs=0.01)
    assert (out["f2_snapped_hz"] - out["f1_snapped_hz"]) == pytest.approx(3.0)


def test_snap_carrier_refuses_deep_offpitch_carriers() -> None:
    """A carrier deeper than a quarter tone off is a glide/tuning issue,
    not a snap: returned unchanged with applied=False."""
    pair = {"f_carrier_hz": 455.0, "f_beat_hz": 3.0,
            "carrier_note": "A#4", "carrier_note_hz": 466.16,
            "carrier_cents_off": -40.0}
    out = snap_carrier(pair)
    assert out["applied"] is False
    assert out["f1_snapped_hz"] == pair["f_carrier_hz"] - 1.5


def test_snap_carrier_json_clean() -> None:
    import json
    pair = {"f_carrier_hz": 443.0, "f_beat_hz": 3.0,
            "carrier_note": "A4", "carrier_note_hz": 440.0}
    json.dumps(snap_carrier(pair))


# --- step 3: sub-bass note events (the 808 as notes, not smeared glides) --

def test_detect_note_events_finds_the_808_notes() -> None:
    """Two sub-bass notes with the measured attack drop: the detector must
    segment them and report f_attack > f_sustain per note."""
    sr = SR = 22050
    total = sr  # 1 s
    y = np.zeros(total)

    def add_note(t0, dur, f_start, f_end):
        i0 = int(t0 * sr)
        i1 = i0 + int(dur * sr)
        seg_t = np.arange(i1 - i0) / sr
        f = f_start + (f_end - f_start) * np.minimum(seg_t / 0.25, 1.0)
        phase = 2 * np.pi * np.cumsum(f) / sr
        env = np.exp(-0.8 * seg_t)
        y[i0:i1] += 0.8 * env * np.sin(phase)

    add_note(0.05, 0.45, 43.4, 42.0)
    add_note(0.60, 0.35, 43.4, 42.0)
    events = detect_note_events(y, sr, f_lo=30.0, f_hi=150.0)
    assert len(events) == 2
    for ev in events:
        assert ev["f_attack_hz"] == pytest.approx(43.4, abs=0.6)
        assert ev["f_sustain_hz"] == pytest.approx(42.0, abs=0.6)
        assert ev["f_attack_hz"] > ev["f_sustain_hz"]
    assert events[1]["t0_s"] == pytest.approx(0.60, abs=0.05)


# --- the climb: rebuild the sub-bass from the detected notes --------------

def test_sub_bass_note_partials_outrender_a_smeared_glide() -> None:
    """The engine's glides smeared 32.8-79.5 Hz across what is really one
    43.4->42.0 Hz note. Rebuilding the sub-bass from the detected note
    events must beat the smear in the band that carries the energy."""
    from src.res_synthesis import _render_partial

    sr = SR = 22050
    total = sr  # 1 s
    y = np.zeros(total)

    def add_note(t0, dur, f_start, f_end):
        i0 = int(t0 * sr)
        i1 = i0 + int(dur * sr)
        seg_t = np.arange(i1 - i0) / sr
        f = f_start + (f_end - f_start) * np.minimum(seg_t / 0.25, 1.0)
        phase = 2 * np.pi * np.cumsum(f) / sr
        env = np.exp(-0.8 * seg_t)
        y[i0:i1] += 0.8 * env * np.sin(phase)

    add_note(0.05, 0.45, 43.4, 42.0)

    # The smear: one glide sweeping far past the true excursion.
    smear = [{
        "model": "glide", "amplitude": 0.5, "projection_gain": 1.0,
        "phase": 0.0, "frequency": 55.0,
        "trajectory": {"t": [0.0, 0.25, 0.5, 0.75, 1.0],
                       "f_hz": [33.0, 45.0, 79.5, 50.0, 40.0],
                       "a": [0.5, 0.8, 1.0, 0.6, 0.3]},
        "start_time": 0.0, "end_time": 1.0,
        "envelope_decay": 0.0, "envelope_t0": 0.0,
    }]

    rebuilt = sub_bass_note_partials(y, sr, f_lo=30.0, f_hi=150.0)
    assert rebuilt, "the note events must produce glide partials"
    n = total

    def band_err(partials):
        r = np.zeros(n)
        for p in partials:
            r += _render_partial(p, n, sr)
        X = np.fft.rfft(y); f = np.fft.rfftfreq(n, 1.0 / sr)
        m = (f >= 30.0) & (f < 150.0)
        yb = np.fft.irfft(X * m, n=n)
        rb = np.fft.irfft(np.fft.rfft(r) * m, n=n)
        return float(np.sum((yb - rb) ** 2))

    assert band_err(rebuilt) < 0.5 * band_err(smear), (
        "note-event rebuild must at least halve the smear's band error"
    )
