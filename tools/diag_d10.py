"""Report the four D10 metrics exactly as the pinned test measures them."""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sinusoidal_model import sinusoidal_reconstruction  # noqa: E402

SR, HOP, NFFT = 22050, 256, 2048
F0, DEV, RATE, NH = 220.0, 20.0, 6.0, 8


def vibrato_signal(dur: float = 2.5) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n, dtype=np.float64) / SR
    y = np.zeros(n)
    for m in range(1, NH + 1):
        inst = m * (F0 + DEV * np.sin(2 * np.pi * RATE * t))
        y += (0.5 / m) * np.cos(2 * np.pi * np.cumsum(inst) / SR)
    return 0.9 * y / np.max(np.abs(y))


vib = vibrato_signal()
t0 = time.time()
res = sinusoidal_reconstruction(vib, SR, n_fft=NFFT, hop_length=HOP,
                               add_noise_layer=False, window_s=2.5, workers=1)
elapsed = time.time() - t0
parts = res["fitted_parameters"]


def near_harmonic(f: float) -> bool:
    return any(abs(f - m * F0) <= 25.0 for m in range(1, NH + 1))


strays = [p for p in parts if not near_harmonic(float(p["frequency"]))]
found = sorted({
    m for m in range(1, NH + 1)
    if any(abs(float(p["frequency"]) - m * F0) <= 25.0 for p in parts)
})
print(f"partials        : {len(parts):4d}   (test wants <= 20)")
print(f"strays          : {len(strays):4d}   (test wants <= 20)")
print(f"harmonics found : {found}  (test wants 1..8)")
print(f"correlation     : {res['correlation']:.4f}  (test wants > 0.98)")
print(f"elapsed         : {elapsed:.1f}s")
print(f"\nmodel counts    : "
      f"{ {m: sum(1 for p in parts if p['model'] == m) for m in ('glide','constant')} }")
top = sorted(parts, key=lambda p: -float(p["amplitude"]))[:10]
print("top partials by amplitude:")
for p in top:
    print(f"   f={float(p['frequency']):9.2f} amp={float(p['amplitude']):.5f} "
          f"gain={float(p['projection_gain']):.3f} {p['model']}")
