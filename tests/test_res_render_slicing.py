"""Equivalence guard for the support-window slicing in ``_render_partial``.

Slicing is only sound for the pointwise (deterministic) models: the stochastic
models draw from an RNG stream positioned at the start of the buffer, run an
IIR filter whose state depends on that leading history, and normalise by an RMS
taken over the whole buffer, so none of them may be sliced.
"""
import numpy as np
import pytest

import src.res_synthesis as rs

SR = 22050
N = SR  # 1.0 s render window


def _sine(**kw):
    p = {
        "model": "constant",
        "amplitude": 0.3,
        "frequency": 440.0,
        "phase": 0.25,
        "envelope_decay": 0.0,
        "envelope_t0": 0.1,
        "start_time": 0.10,
        "end_time": 0.40,
    }
    p.update(kw)
    return p


def _glide(tt=(0.05, 0.20, 0.35), **kw):
    p = {
        "model": "glide",
        "amplitude": 0.2,
        "frequency": 600.0,
        "phase": 0.1,
        "envelope_decay": 0.0,
        "envelope_t0": 0.05,
        "start_time": float(tt[0]),
        "end_time": float(tt[-1]),
        "projection_gain": 0.8,
        "trajectory": {
            "t": list(tt),
            "f_hz": [600.0, 900.0, 750.0],
            "a": [0.1, 0.2, 0.1],
        },
    }
    p.update(kw)
    return p


def _cases():
    return [
        ("sine flat", _sine(), 0, 0.0),
        ("sine decay", _sine(envelope_decay=3.0), 0, 0.0),
        ("sine steep decay", _sine(envelope_decay=40.0), 0, 0.0),
        ("sine stft", _sine(), 0, 0.25),
        ("sine stft decay", _sine(envelope_decay=3.0), 0, 0.25),
        ("sine offset section", _sine(start_time=0.55, end_time=0.85), SR // 2, 0.0),
        ("sine offset stft", _sine(start_time=0.55, end_time=0.85), SR // 2, 0.25),
        ("sine at buffer start", _sine(start_time=0.0, end_time=0.2), 0, 0.0),
        ("sine at buffer end", _sine(start_time=0.8, end_time=1.0), 0, 0.0),
        ("glide", _glide(), 0, 0.0),
        ("glide stft", _glide(), 0, 0.25),
        ("glide offset", _glide(tt=(0.55, 0.70, 0.85)), SR // 2, 0.25),
        ("glide decay", _glide(envelope_decay=2.0), 0, 0.0),
        ("no support", _sine(start_time=None, end_time=None), 0, 0.0),
        ("zero amp", _sine(amplitude=0.0), 0, 0.0),
        # stochastic models must remain untouched by the slicing path
        ("filtered_noise", {
            "model": "filtered_noise", "amplitude": 0.1, "seed": 7,
            "envelope_decay": 1.0, "start_time": 0.1, "end_time": 0.5,
            "band_centers": [300.0, 1200.0, 4000.0],
            "band_mags": [1.0, 0.5, 0.2],
        }, 0, 0.0),
        ("lpc_noise", {
            "model": "lpc_noise", "amplitude": 0.1, "seed": 11,
            "envelope_decay": 1.0, "start_time": 0.1, "end_time": 0.5,
            "lpc_coeffs": [0.9, -0.4, 0.2],
        }, 0, 0.0),
    ]


@pytest.mark.parametrize("label,partial,start,stft", _cases(),
                         ids=[c[0] for c in _cases()])
def test_sliced_matches_full(monkeypatch, label, partial, start, stft):
    sliced = rs._render_partial(dict(partial), N, SR, start, stft)
    monkeypatch.setattr(rs, "_support_window", lambda *a, **k: None)
    full = rs._render_partial(dict(partial), N, SR, start, stft)
    assert sliced.shape == full.shape == (N,)
    assert np.max(np.abs(sliced - full)) < 1e-12, label


def test_window_actually_narrows_the_work():
    """The optimisation must be active, not silently disabled."""
    p = rs.sanitize_partial(_sine())
    i0, i1 = rs._support_window(p, N, SR, 0, 0.0)
    assert i0 > 0, "should skip the leading silence"
    assert i1 < N, "should skip the trailing silence"
    assert (i1 - i0) / N < 0.6, "a 0.3 s partial should not touch the full buffer"


def test_release_tail_is_inside_the_window():
    """A long decay hold keeps signal alive well past end_time."""
    p = _sine(envelope_decay=1.0, end_time=0.40)
    n_long = 3 * SR          # must be long enough to hold the whole tail
    monkey = rs._support_window
    rs._support_window = lambda *a, **k: None
    full = rs._render_partial(dict(p), n_long, SR, 0, 0.0)
    rs._support_window = monkey
    nz = np.nonzero(np.abs(full) > 1e-15)[0]
    hold = min(1.5, float(np.log(100.0)) / 1.0)
    # the hold alone is 1.5 s, so live signal must run to ~1.90 s
    assert nz.max() / SR > 0.40 + hold - 0.05, "release tail must survive past end_time"
    i0, i1 = rs._support_window(rs.sanitize_partial(p), n_long, SR, 0, 0.0)
    assert i0 <= nz.min() and i1 > nz.max(), "window must cover every live sample"


def test_window_respects_start_sample_offset():
    """A section-offset render must slice in local sample indices."""
    start = SR // 2                      # section begins at global t = 0.5 s
    p = _sine(start_time=0.55, end_time=0.85)
    i0, i1 = rs._support_window(rs.sanitize_partial(p), N, SR, start, 0.0)
    g_lo = (i0 + start) / SR
    g_hi = (i1 + start) / SR
    assert g_lo <= 0.55, "window must start at or before the partial support"
    assert g_hi > 0.85, "window must extend past end_time for the release"
    # local index must be section-relative, not global
    assert i0 < N // 4, "slicing must not run off the end of the section buffer"


def test_window_is_empty_when_support_is_outside_the_buffer():
    start = SR // 2                      # buffer covers global t in [0.5, 1.5]
    # release hold reaches 0.10 + 0.25 = 0.35 s, still clear of the buffer
    p = _sine(start_time=0.02, end_time=0.10)
    i0, i1 = rs._support_window(rs.sanitize_partial(p), N, SR, start, 0.0)
    assert i1 <= i0, "a support entirely before the section yields an empty window"


def test_release_tail_can_reach_into_a_later_section():
    """A partial ending before a section may still be audible via its tail."""
    start = SR // 2                      # buffer covers global t in [0.5, 1.5]
    p = _sine(start_time=0.10, end_time=0.40, envelope_decay=0.0)
    i0, i1 = rs._support_window(rs.sanitize_partial(p), N, SR, start, 0.0)
    assert (i1 + start) / SR > 0.65, "0.25 s hold from 0.40 reaches 0.65 s"
    monkey = rs._support_window
    rs._support_window = lambda *a, **k: None
    full = rs._render_partial(dict(p), N, SR, start, 0.0)
    rs._support_window = monkey
    assert np.max(np.abs(full)) > 1e-6, "the tail must actually be non-silent"