"""Decompose the residual: (a) the over-sustained tail, (b) percussive demand.

(a) For every partial: how much of the RENDERED energy falls in
    [end_time, end_time + hold_s] - the span the support mask holds open past
    the last tracked frame - and how much of the TARGET's energy is there.
(b) Per section: how many transients the flux detector actually sees against
    the cap, and how many modes the detected hits would need against the
    per-hit mode cap.

Read-only. Emits one line per fixture and a per-fixture summary.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import load_audio  # noqa: E402
from src.res_bank import detect_hits  # noqa: E402
from src.res_synthesis import _render_partial  # noqa: E402

SR = 22050
MAX_MODES = 6
FIXTURES = {
    "mix_lost": "Metre - Lost.mp3",
    "stem_bass": "data/output/_fixtures/stem_bass.wav",
    "stem_drums": "data/output/_fixtures/stem_drums.wav",
    "stem_vocals": "data/output/_fixtures/stem_vocals.wav",
    "stem_other": "data/output/_fixtures/stem_other.wav",
}

print("=" * 78)
print("(a) OVER-SUSTAINED TAIL: rendered energy in [end_time, end_time+hold_s]")
print("    vs the target's energy in the same span")
print("=" * 78)
print(f"{'fixture':<13} {'tailE_render':>13} {'tailE_target':>13} "
      f"{'render share':>13} {'partials':>9} {'>20%':>6}")

for name, rel in FIXTURES.items():
    run = ROOT / f"data/output/_scoreboard/after/{name}"
    led = json.loads((run / "res_ledger.json").read_text(encoding="utf-8"))
    y, _ = load_audio(str(ROOT / rel), target_sr=SR)
    st_hw = float(led.get("stft_half_window_s", 0.0))
    tail_r = tail_t = 0.0
    tot_r = 0.0
    n_tail = n_big = 0
    for sec in led["sections"]:
        w0, w1 = int(sec["start_sample"]), int(sec["end_sample"])
        n = w1 - w0
        t = (np.arange(n) + w0) / SR
        y_sec = y[w0:w1]
        for p in sec["partials"]:
            if p.get("model") in ("filtered_noise", "lpc_noise"):
                continue
            sig = _render_partial(p, n, SR, start_sample=w0,
                                  stft_half_window_s=st_hw)
            et = float(p.get("end_time", p.get("envelope_t0", 0.0)))
            dec = float(p.get("envelope_decay", 0.0))
            hold = min(1.5, math.log(100.0) / dec) if dec > 0 else 0.25
            m = (t >= et) & (t < et + hold)
            if not m.any():
                continue
            er = float(np.sum(sig[m] ** 2))
            et_y = float(np.sum(y_sec[m] ** 2))
            tail_r += er
            tail_t += et_y
            tot_r += float(np.sum(sig ** 2))
            n_tail += 1
            if er > 0.20 * float(np.sum(sig ** 2)):
                n_big += 1
    share = 100.0 * tail_r / max(tot_r, 1e-20)
    print(f"{name:<13} {tail_r:>13.4f} {tail_t:>13.4f} "
          f"{share:>12.2f}% {n_tail:>9} {n_big:>6}")

print()
print("=" * 78)
print("(b) PERCUSSIVE DEMAND: transients present vs the cap, modes needed vs cap")
print("=" * 78)
print(f"{'fixture':<13} {'hits/section (k=2)':>20} {'hits/section (k=12)':>21} "
      f"{'modes needed':>13} {'cap/section':>12}")

for name, rel in FIXTURES.items():
    run = ROOT / f"data/output/_scoreboard/after/{name}"
    led = json.loads((run / "res_ledger.json").read_text(encoding="utf-8"))
    y, _ = load_audio(str(ROOT / rel), target_sr=SR)
    st_hw = float(led.get("stft_half_window_s", 0.0))
    h2, h12, modes = [], [], []
    for sec in led["sections"]:
        w0, w1 = int(sec["start_sample"]), int(sec["end_sample"])
        n = w1 - w0
        y_sec = y[w0:w1]
        r = np.zeros(n)
        for p in sec["partials"]:
            r += _render_partial(p, n, SR, start_sample=w0,
                                 stft_half_window_s=st_hw)
        resid = y_sec - r
        h2.append(len(detect_hits(resid, SR, w0, k=2)))
        h12.append(len(detect_hits(resid, SR, w0, k=12)))
        # a hit needs up to MAX_MODES modes; count how many the bundle would
        # actually produce, from the accepted percussive records
        modes.append(sum(int(rec.get("m_count") or 0)
                         for rec in (sec.get("hypothesis_bank") or [])
                         if rec.get("class") == "percussive_bundle"))
    print(f"{name:<13} {str(h2):>20} {str(h12):>21} {str(modes):>13} "
          f"{2 * MAX_MODES:>12}")

print()
print("cap/section is k_hits(12) * MAX_MODES(6) = 72 modes; a 2.5 s section at")
print("120 BPM carries 5 beats, so the old cap of 2 hits/section = 12 modes.")
