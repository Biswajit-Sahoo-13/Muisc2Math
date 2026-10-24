"""Test zero-phase filtering + amplitude masking for the heterodyne."""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfilt, sosfiltfilt

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


def probe(y, cutoff, zero_phase, mask_frac):
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
    lp = sosfiltfilt(sos, base) if zero_phase else sosfilt(sos, base)
    amp = np.abs(lp)
    phase = np.unwrap(np.angle(lp))
    corr = np.gradient(phase, 1.0 / SR) / (2.0 * np.pi)
    thr = mask_frac * np.median(amp)
    corr = np.where(amp >= thr, corr, np.nan)
    got = np.interp(times, grid, corr)
    ok = np.isfinite(got)
    c = float(np.corrcoef(got[ok], want[ok])[0, 1])
    rms = float(np.sqrt(np.mean((got[ok] - want[ok]) ** 2)))
    return c, rms, float(np.sqrt(np.mean(want[ok] ** 2)))


for nh in (1, 8):
    y = make(nh)
    for cutoff in (30.0, 40.0, 60.0):
        for zp in (False, True):
            for mf in (0.0, 0.5):
                c, rms, wr = probe(y, cutoff, zp, mf)
                tag = "zero-phase" if zp else "causal     "
                print(f"nh={nh} cut={cutoff:5.1f} {tag} mask={mf:.1f} -> "
                      f"corr {c:+.4f}  rms {rms:7.3f} (want rms {wr:6.3f})")
