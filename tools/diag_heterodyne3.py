"""Where does the de-chirped baseband's phase go wrong with 8 harmonics?"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import compute_normalized_stft  # noqa: E402
from src.tracking import track_sinusoids  # noqa: E402

SR, HOP, NFFT = 22050, 256, 2048
F0, DEV, RATE = 220.0, 20.0, 6.0


def make(nh: int) -> np.ndarray:
    n = int(2.5 * SR)
    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n)
    for m in range(1, nh + 1):
        inst = m * (F0 + DEV * np.sin(2 * np.pi * RATE * t))
        y += (0.5 / m) * np.cos(2 * np.pi * np.cumsum(inst) / SR)
    return 0.9 * y / np.max(np.abs(y))


y = make(8)
stft = compute_normalized_stft(y, sr=SR, n_fft=NFFT, hop_length=HOP)
tracks = track_sinusoids(stft, threshold=0.01, min_track_duration_frames=3)
tr = max(tracks, key=lambda t_: t_.peak_amplitude)
times = np.asarray(tr.times)
freqs = np.asarray(tr.frequencies)

n = len(y)
grid = np.arange(n) / SR
f_interp = np.interp(grid, times, freqs)
theta = 2.0 * np.pi * np.cumsum(f_interp) / SR
base = hilbert(y) * np.exp(-1j * theta)

sos = butter(4, 40.0 / (SR / 2.0), btype="low", output="sos")
lp = sosfilt(sos, base)

amp = np.abs(lp)
print(f"baseband amplitude: min {amp.min():.5f} med {np.median(amp):.5f} "
      f"max {amp.max():.5f}")
print(f"fundamental amplitude in the source: 0.33")

# spectrum of the de-chirped signal BEFORE low-pass, 0..600 Hz
sp = np.abs(np.fft.fft(base * np.hanning(n)))
fa = np.fft.fftfreq(n, 1.0 / SR)
for lo, hi in ((0, 40), (40, 120), (120, 200), (200, 260), (260, 400),
               (400, 700)):
    sel = (fa >= lo) & (fa < hi)
    print(f"  de-chirped band {lo:4d}-{hi:4d} Hz: max |X| = {sp[sel].max():.4f}")

# Is the fundamental's own de-chirped component the only thing in 0-40 Hz?
# Compare with a de-chirp built from the TRUE trajectory.
truth_f = F0 + DEV * np.sin(2 * np.pi * RATE * grid)
theta_true = 2.0 * np.pi * np.cumsum(truth_f) / SR
base_true = hilbert(y) * np.exp(-1j * theta_true)
spt = np.abs(np.fft.fft(base_true * np.hanning(n)))
for lo, hi in ((0, 40), (40, 120), (120, 200), (200, 260), (260, 400),
               (400, 700)):
    sel = (fa >= lo) & (fa < hi)
    print(f"  TRUE de-chirp band {lo:4d}-{hi:4d} Hz: max |X| = {spt[sel].max():.4f}")

# Now: with the TRUE de-chirp, the baseband must be a clean DC.
lpt = sosfilt(sos, base_true)
ph = np.unwrap(np.angle(lpt))
d = np.gradient(ph, 1.0 / SR) / (2 * np.pi)
print(f"\nTRUE de-chirp -> residual freq rms {np.sqrt(np.mean(d**2)):.4f} Hz "
      f"(should be ~0)")
print(f"baseband amplitude with TRUE de-chirp: min {np.abs(lpt).min():.5f} "
      f"med {np.median(np.abs(lpt)):.5f} max {np.abs(lpt).max():.5f}")
