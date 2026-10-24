"""The fidelity benchmark: how far each reconstruction sits from its source.

For every run directory given (a RES output with res_ledger.json and
res_best.wav), reloads the ORIGINAL source the ledger points at, crops it
to the analysed span, and scores the render on the original = 100% scale
(src/fidelity.py: tone / spectrum / bands / loudness pillars).

Usage:
    python tools/fidelity_report.py out/eval_mix_parity out/eval_drums_after
    python tools/fidelity_report.py data/output/_scoreboard/after/*

Prints one row per run: the headline ("ours = X% of the original") and the
pillar breakdown, so the distance is attributable, not just quoted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from src.analysis import load_audio  # noqa: E402
from src.fidelity import compute_fidelity  # noqa: E402


def _load_render_source(run_dir: Path):
    ledger = json.loads(
        (run_dir / "res_ledger.json").read_text(encoding="utf-8")
    )
    src = Path(ledger["source"])
    if not src.exists():
        alt = Path("data/uploads") / src.name
        src = alt if alt.exists() else None
    if src is None:
        return None, ledger
    y, _ = load_audio(str(src), target_sr=int(ledger["sample_rate"]))
    n = int(ledger.get("n_samples", 0)) or len(y)
    return y[:n], ledger


def main(argv: list[str]) -> int:
    run_dirs = [Path(a) for a in argv if (Path(a) / "res_ledger.json").exists()]
    skipped = [a for a in argv if Path(a) not in run_dirs]
    if not run_dirs:
        print("no run directories with res_ledger.json found")
        return 2
    for s in skipped:
        print(f"(skipped {s}: no res_ledger.json)")

    rows = []
    for run_dir in run_dirs:
        y, ledger = _load_render_source(run_dir)
        if y is None:
            print(f"(skipped {run_dir}: source audio not on disk)")
            continue
        wav = run_dir / "res_best.wav"
        if not wav.exists():
            print(f"(skipped {run_dir}: no res_best.wav)")
            continue
        r, sr = sf.read(str(wav), always_2d=False)
        if r.ndim > 1:
            r = r.mean(axis=1)
        f = compute_fidelity(y[: len(r)], np.asarray(r, dtype=np.float64), sr)
        rows.append((run_dir.name, f, sr))

    width = max((len(name) for name, _, _ in rows), default=10)
    hdr = (f"{'run':<{width}} | {'OURS':>6} | {'dist':>5} | "
           f"{'tone':>5} | {'spect':>5} | {'bands':>5} | {'loud':>5} | sr")
    print("THE FIDELITY BENCHMARK - original = 100%, ours = ...")
    print(hdr)
    print("-" * len(hdr))
    for name, f, sr in sorted(rows, key=lambda t: -t[1]["fidelity_pct"]):
        print(
            f"{name:<{width}} | {f['fidelity_pct']:>5.1f}% | "
            f"{f['distance_pct']:>4.1f} | {f['tone_pct']:>5.1f} | "
            f"{f['spectrum_pct']:>5.1f} | {f['band_pct']:>5.1f} | "
            f"{f['loudness_pct']:>5.1f} | {sr}"
        )
    print()
    print("pillars: tone = waveform corr; spectrum = MR-STFT agreement;")
    print("bands = energy-weighted band corr; loudness = RMS parity.")
    print("distance = 100 - ours = how far the reconstruction sits from the original.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
