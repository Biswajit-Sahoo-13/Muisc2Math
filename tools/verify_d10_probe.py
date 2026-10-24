# -*- coding: utf-8 -*-
"""Standalone measurement probe for defect D10 (vibrato caustic fragmentation).

Reimplements the exact tone from tests/test_res_trajectory_defects.py
(``_vibrato_signal``, lines 88-95) inline -- no import of the test package --
and runs the shipped reconstruction on it, printing the four D10 quantities:

  1. partial count and stray count (partials not within +/-25 Hz of m*F0)
  2. per-harmonic tracking (harmonics 1..8 within +/-25 Hz)
  3. reconstruction correlation (test demands > 0.98)
  4. track_count (test demands <= 20)

Ground rules:
  - program is a pure measurement; it changes nothing
  - every number printed is a real run over the shipped code path
Run: .venv\\Scripts\\python.exe tools\\verify_d10_probe.py
"""

from __future__ import annotations

import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.sinusoidal_model import sinusoidal_reconstruction  # noqa: E402

# --- constants mirroring the D10 test block (lines 60-66) -----------------
SR = 22050
HOP = 256
NFFT = 2048
VIBRATO_F0 = 220.0
VIBRATO_DEV = 20.0
VIBRATO_RATE = 6.0
VIBRATO_HARMONICS = 8
FREQ_TOL = 25.0
DURATION = 2.5
WINDOW_S = 2.5


def _vibrato_signal() -> np.ndarray:
    """Verbatim reimplementation of tests/test_res_trajectory_defects.py
    ``_vibrato_signal`` (lines 88-95)."""
    t = np.arange(int(DURATION * SR)) / SR
    inst = VIBRATO_F0 + VIBRATO_DEV * np.sin(
        2.0 * np.pi * VIBRATO_RATE * t
    )
    y = np.zeros_like(t)
    for m in range(1, VIBRATO_HARMONICS + 1):
        inst_h = m * inst
        phase = 2.0 * np.pi * np.cumsum(inst_h) / SR
        y += (0.5 / m) * np.cos(phase)
    return 0.9 * y / np.max(np.abs(y))


def main() -> None:
    n = int(DURATION * SR)
    t = np.arange(n) / SR
    _ = t  # durations only; phase accumulation is inside _vibrato_signal
    vib = _vibrato_signal()

    print(f"reconstruction input: {n} samples @ {SR} Hz "
          f"({n / SR:.3f} s), NFFT={NFFT}, HOP={HOP}")

    res = sinusoidal_reconstruction(
        vib,
        SR,
        n_fft=NFFT,
        hop_length=HOP,
        add_noise_layer=False,
        window_s=WINDOW_S,
        workers=1,
    )

    partial_freqs = [float(track["frequency"]) for track in res["fitted_parameters"]]
    freqs = np.sort(np.asarray(partial_freqs, dtype=float))

    near = lambda f: any(
        abs(f - m * VIBRATO_F0) <= FREQ_TOL for m in range(1, VIBRATO_HARMONICS + 1)
    )
    strays = np.array([f for f in freqs if not near(f)], dtype=float)

    # per-harmonic tracking: which harmonics 1..8 have at least one
    # partial within +/-FREQ_TOL across the whole set
    tracked = {}
    for m in range(1, VIBRATO_HARMONICS + 1):
        hits = [f for f in freqs if abs(f - m * VIBRATO_F0) <= FREQ_TOL]
        tracked[m] = sorted(hits)[:3]  # show up to 3 partials per harmonic

    corr = float(res["correlation"])
    track_count = int(res["track_count"])

    print("")
    print("=== D10 probe measurements (real run over shipped code) ===")
    print(f"partials           : {len(freqs)}")
    print(f"strays (>25 Hz off): {len(strays)}")
    print(f"correlation        : {corr:.4f}  (test demands > 0.98)")
    print(f"track_count        : {track_count}  (test demands <= 20)")
    print("")
    print("--- per-harmonic partials within +/-25 Hz ---")
    for m in range(1, VIBRATO_HARMONICS + 1):
        hits = tracked[m]
        status = "tracked" if hits else "ABSENT"
        col = ", ".join(f"{h:.2f}" for h in hits) if hits else "-"
        print(f"  harmonic {m} ({m * VIBRATO_F0:7.2f} Hz nominal): {status}: {col}")
    missing = [m for m in range(1, VIBRATO_HARMONICS + 1) if not tracked[m]]
    print("")
    print(f"harmonics absent    : {missing if missing else 'none'}")


if __name__ == "__main__":
    main()
