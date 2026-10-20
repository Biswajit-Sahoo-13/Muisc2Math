"""Check a stem-mode ledger: does a fresh render match the recorded metric?"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import load_audio  # noqa: E402
from src.res_synthesis import _render_partial  # noqa: E402


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-18 else 0.0


for path in sys.argv[1:]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / path
    ledger = json.loads(p.read_text(encoding="utf-8"))
    sr = int(ledger["sample_rate"])
    n = int(ledger["n_samples"])
    src = ledger["source"]
    if isinstance(src, (list, np.ndarray)):
        y = np.asarray(src, dtype=np.float64)
    elif isinstance(src, str) and src.lstrip()[:1] in "[(":
        # json.dump turned the ndarray into its repr string, truncated for
        # long arrays -> fall back to the stem wav the run wrote.
        stem_wav = p.parent.parent / f"res_{p.parent.name}.wav"
        if stem_wav.exists():
            y, _ = load_audio(str(stem_wav), target_sr=sr)
        else:
            y = np.array(src.strip("[]()").split(), dtype=np.float64)
    else:
        y, _ = load_audio(src, target_sr=sr)
    y = y[:n]
    st_hw = float(ledger.get("stft_half_window_s", 0.0))
    print(f"=== {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} ===")
    print(f"  best: {ledger['best']}")
    for sec in ledger["sections"]:
        ws = int(sec["start_sample"])
        we = int(sec.get("end_sample", round(sec["end"] * sr)))
        y_sec = y[ws:we]
        r = np.zeros(len(y_sec))
        glides = 0
        for pp in sec["partials"]:
            if pp.get("model") == "glide":
                glides += 1
            r += _render_partial(pp, len(y_sec), sr, start_sample=ws,
                                 stft_half_window_s=st_hw)
        rec = float(sec["metrics"]["correlation"])
        fresh = corr(y_sec, r)
        print(f"  sec {ws:<7} n={len(sec['partials']):<4} glides={glides:<3} "
              f"recorded={rec:.4f} fresh={fresh:.4f} "
              f"delta={fresh - rec:+.4f} "
              f"rms_y={np.sqrt((y_sec**2).mean()):.5f} "
              f"rms_r={np.sqrt((r**2).mean()):.5f} "
              f"n_y={y_sec.size} n_r={r.size}")
