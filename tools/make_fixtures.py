"""Write the real native-DSP stem fixtures to disk.

The scoreboard's "stem" fixtures were previously reconstructions written by
an earlier run (res_stem_<name>.wav is the *rendered* stem, not the input),
which is a much weaker test than reconstructing real source audio. These are
the actual separated stems, so every fixture is genuine source material.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import load_audio  # noqa: E402
from src.res_evidence import separate_stems  # noqa: E402

SR = 22050
DUR = 15.0
SRC = ROOT / "01 - j-hope - MONA LISA.flac"
OUT = ROOT / "data" / "output" / "_fixtures"
OUT.mkdir(parents=True, exist_ok=True)

y, _ = load_audio(str(SRC), target_sr=SR)
y = y[: int(DUR * SR)]
stems = separate_stems(y, SR, mode="native")
for name, sig in stems.items():
    sig = np.asarray(sig, dtype=np.float64)
    p = OUT / f"stem_{name}.wav"
    sf.write(str(p), np.clip(sig, -1.0, 1.0), SR)
    print(f"{name:<8} rms={np.sqrt((sig**2).mean()):.5f} -> {p.name}")

# sanity: partition of unity
tot = sum(np.asarray(v, dtype=np.float64) for v in stems.values())
err = float(np.max(np.abs(tot - y[: len(tot)])))
print(f"partition of unity max error = {err:.3e}")
