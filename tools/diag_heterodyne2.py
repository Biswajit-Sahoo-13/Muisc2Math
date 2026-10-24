"""Isolate the heterodyne: fundamental only vs full harmonic stack."""
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


def probe(y, cutoff=40.0):
    stft = compute_normalized_stft(y, sr=SR, n_fft=NFFT, hop_length=HOP)
    tracks = track_sinusoids(stft, threshold=0.01, min_track_duration_frames=3)
    tr = max(tracks, key=lambda t_: t_.peak_amplitude)
    times = np.asarray(tr.times)
    freqs = np.asarray(tr.frequencies)
    truth = F0 + DEV * np.sin(2 * np.pi * RATE * times)
    want = truth - freqs
    n = len(y)
    grid = np.arange(n) / SR
    f_interp = np.interp(grid, times, freqs)
    theta = 2.0 * np.pi * np.cumsum(f_interp) / SR
    base = hilbert(y) * np.exp(-1j * theta)
    sos = butter(4, cutoff / (SR / 2.0), btype="low", output="sos")
    lp = sosfilt(sos, base)
    phase = np.unwrap(np.angle(lp))
    corr = np.gradient(phase, 1.0 / SR) / (2.0 * np.pi)
    got = np.interp(times, grid, corr)
    # the baseband's own spectrum, to see what else is in the passband
    sp = np.abs(np.fft.fft(lp * np.hanning(n)))
    faxis = np.fft.fftfreq(n, 1.0 / SR)
    sel = (faxis > 0.2) & (faxis < cutoff)
    order = np.argsort(sp[sel])[::-1][:5]
    peaks = [(round(float(faxis[sel][i]), 2), round(float(sp[sel][i] / sp[sel].max()), 3))
             for i in order]
    c = float(np.corrcoef(got, want)[0, 1]) if got.std() > 0 else 0.0
    print(f"  want rms {np.sqrt(np.mean(want**2)):6.3f} | got rms "
          f"{np.sqrt(np.mean(got**2)):6.3f} | corr {c:+.3f} | "
          f"band peaks {peaks}")


for nh in (1, 2, 4, 8):
    print(f"harmonics=1..{nh}")
    probe(make(nh))
