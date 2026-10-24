"""Probe the heterodyne de-chirp against the known vibrato truth."""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfilt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import compute_normalized_stft  # noqa: E402
from src.tracking import track_sinusoids  # noqa: E402

SR, HOP, NFFT = 22050, 256, 2048
F0, DEV, RATE, NH = 220.0, 20.0, 6.0, 8

n = int(2.5 * SR)
t = np.arange(n, dtype=np.float64) / SR
y = np.zeros(n)
for m in range(1, NH + 1):
    inst = m * (F0 + DEV * np.sin(2 * np.pi * RATE * t))
    y += (0.5 / m) * np.cos(2 * np.pi * np.cumsum(inst) / SR)
y = 0.9 * y / np.max(np.abs(y))

stft = compute_normalized_stft(y, sr=SR, n_fft=NFFT, hop_length=HOP)
tracks = track_sinusoids(stft, threshold=0.01, min_track_duration_frames=3)
tr = max(tracks, key=lambda t_: t_.peak_amplitude)
times = np.asarray(tr.times)
freqs = np.asarray(tr.frequencies)
truth = F0 + DEV * np.sin(2 * np.pi * RATE * times)
want = truth - freqs                      # the correction we should measure
print(f"track: {len(times)} frames  f=[{freqs.min():.1f},{freqs.max():.1f}]")
print(f"correction we want: rms {np.sqrt(np.mean(want**2)):.3f} Hz, "
      f"span {want.max()-want.min():.2f} Hz")

grid = np.arange(n) / SR
f_interp = np.interp(grid, times, freqs)
theta = 2.0 * np.pi * np.cumsum(f_interp) / SR
analytic = hilbert(y)
base = analytic * np.exp(-1j * theta)

for cutoff in (40.0, 88.0, 150.0, 300.0):
    sos = butter(4, cutoff / (SR / 2.0), btype="low", output="sos")
    lp = sosfilt(sos, base)
    phase = np.unwrap(np.angle(lp))
    corr = np.gradient(phase, 1.0 / SR) / (2.0 * np.pi)
    got = np.interp(times, grid, corr)
    # what is actually in the baseband?
    sp = np.abs(np.fft.fft(lp * np.hanning(len(lp))))
    faxis = np.fft.fftfreq(len(lp), 1.0 / SR)
    band = sp[(faxis > 0) & (faxis < 400)]
    peak_hz = faxis[(faxis > 0) & (faxis < 400)][np.argmax(band)]
    print(f"cutoff {cutoff:6.1f} Hz -> measured rms {np.sqrt(np.mean(got**2)):8.3f} Hz "
          f"(want {np.sqrt(np.mean(want**2)):.3f})  span {got.max()-got.min():7.2f} "
          f"(want {want.max()-want.min():.2f})  baseband peak {peak_hz:6.1f} Hz")
