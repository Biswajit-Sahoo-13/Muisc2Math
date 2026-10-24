"""Diagnose the vibrato harmonic case: what does the analyser produce?"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sinusoidal_model import (  # noqa: E402
    _heterodyne_refine_glide,
)

SR = 22050
HOP = 256
NFFT = 2048
F0 = 220.0
DEV = 20.0
RATE = 6.0
NH = 8


def vibrato_signal(dur: float = 2.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n, dtype=np.float64)
    for m in range(1, NH + 1):
        inst = m * (F0 + DEV * np.sin(2 * np.pi * RATE * t))
        y += (0.5 / m) * np.cos(2 * np.pi * np.cumsum(inst) / SR)
    return 0.9 * y / np.max(np.abs(y))


vib = vibrato_signal()

# What does the *tracker* see?
from src.analysis import compute_normalized_stft  # noqa: E402
from src.tracking import track_sinusoids  # noqa: E402

stft = compute_normalized_stft(vib, sr=SR, n_fft=NFFT, hop_length=HOP)
tracks = track_sinusoids(stft, threshold=0.01, min_track_duration_frames=3)
print(f"tracker produced {len(tracks)} tracks")
for tr in sorted(tracks, key=lambda t: -t.peak_amplitude)[:12]:
    print(f"  id={tr.track_id:3d} n={len(tr.times):3d} "
          f"f=[{min(tr.frequencies):8.1f},{max(tr.frequencies):8.1f}] "
          f"med={np.median(tr.frequencies):8.1f} "
          f"dur={tr.duration:.2f} amp={tr.peak_amplitude:.4f}")

# Does the heterodyne correction recover the truth for a strong track?
print("\nheterodyne probe on the strongest track:")
tr = max(tracks, key=lambda t: t.peak_amplitude)
times = np.asarray(tr.times)
freqs = np.asarray(tr.frequencies)
corr = _heterodyne_refine_glide(vib, SR, times, freqs)
if corr is None:
    print("  correction REJECTED (None)")
else:
    print(f"  estimated span {freqs.max()-freqs.min():.2f} Hz -> "
          f"corrected span {corr.max()-corr.min():.2f} Hz (truth 40.00 Hz)")
    # true trajectory at the frame times
    truth = F0 + DEV * np.sin(2 * np.pi * RATE * times)
    print(f"  rms error before: {np.sqrt(np.mean((freqs-truth)**2)):8.2f} Hz")
    print(f"  rms error after : {np.sqrt(np.mean((corr-truth)**2)):8.2f} Hz")
