"""Frozen-fixture scoreboard for the RES engine.

Every claim about "the engine got better" must be backed by a row in
``data/output/scoreboard.json`` that names the exact command, the ledger it
came from and the five referee metrics. This tool runs the fixed fixture set
through ``main.py res`` and records the result.

Usage
-----
    python tools/bottleneck_scoreboard.py --label baseline
    python tools/bottleneck_scoreboard.py --label after-f1 --fixtures stem_bass
    python tools/bottleneck_scoreboard.py --show

Fixtures are frozen on disk so every run measures the *same* audio; nothing
here re-separates stems or re-downloads anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD = ROOT / "data" / "output" / "scoreboard.json"
RUN_ROOT = ROOT / "data" / "output" / "_scoreboard"

# Frozen fixtures. They already exist in the repo's own output tree, so the
# scoreboard is reproducible without a source-separation step.
#
# The stem fixtures are the REAL separated stems (written by
# tools/make_fixtures.py), not the res_stem_<name>.wav files - those are the
# *rendered* stems from an earlier run, so reconstructing them is a much
# weaker test than reconstructing source audio.
FIXTURES: Dict[str, str] = {
    "stem_bass": "data/output/_fixtures/stem_bass.wav",
    "stem_drums": "data/output/_fixtures/stem_drums.wav",
    "stem_vocals": "data/output/_fixtures/stem_vocals.wav",
    "stem_other": "data/output/_fixtures/stem_other.wav",
    "mix_lost": "Metre - Lost.mp3",
    "mix_monalisa": "01 - j-hope - MONA LISA.flac",
}

METRICS = ("correlation", "snr_db", "log_spectral_distance_db",
           "seg_snr_db", "onset_corr")


def _python() -> str:
    venv = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv if venv.exists() else Path(sys.executable))


def _ledger_metrics(ledger: Dict[str, Any]) -> Dict[str, Any]:
    best = ledger.get("best", {}) or {}
    row: Dict[str, Any] = {m: best.get(m) for m in METRICS}
    sections = ledger.get("sections", []) or []
    row["partials"] = sum(
        len(s.get("partials", []) or []) for s in sections
    )
    row["sections"] = len(sections)
    row["stop_reason"] = ledger.get("stop_reason")
    bank_total = bank_accepted = 0
    for s in sections:
        for rec in (s.get("hypothesis_bank") or []):
            bank_total += 1
            bank_accepted += 1 if rec.get("accepted") else 0
    row["bank_records"] = bank_total
    row["bank_accepted"] = bank_accepted
    iters = ledger.get("iterations") or []
    row["iterations_run"] = len(iters)
    return row


def run_fixture(
    label: str,
    name: str,
    rel_path: str,
    duration: float,
    iterations: int,
    extra: Optional[List[str]] = None,
) -> Dict[str, Any]:
    src = ROOT / rel_path
    if not src.exists():
        return {
            "fixture": name, "status": "missing", "source": rel_path,
            "error": f"fixture not found: {src}",
        }
    out_dir = RUN_ROOT / label / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _python(), str(ROOT / "main.py"), "res",
        "-i", str(src), "-o", str(out_dir),
        "-d", str(duration), "--iterations", str(iterations),
    ]
    if extra:
        cmd.extend(extra)
    t0 = time.time()
    proc = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    elapsed = time.time() - t0
    ledger_path = out_dir / "res_ledger.json"
    row: Dict[str, Any] = {
        "fixture": name,
        "status": "ok" if proc.returncode == 0 else "failed",
        "label": label,
        "source": rel_path,
        "command": " ".join(cmd),
        "seconds": round(elapsed, 2),
        "returncode": proc.returncode,
        "ledger": str(ledger_path.relative_to(ROOT))
        if ledger_path.exists() else None,
    }
    if not ledger_path.exists():
        row["error"] = (proc.stdout or "")[-2000:] + "\n" + (proc.stderr or "")[-2000:]
        return row
    with ledger_path.open("r", encoding="utf-8") as fh:
        ledger = json.load(fh)
    row.update(_ledger_metrics(ledger))
    row["initial_correlation"] = ledger.get("initial_correlation")
    return row


def load() -> Dict[str, Any]:
    if SCOREBOARD.exists():
        with SCOREBOARD.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"schema": "res-scoreboard/1.0", "labels": {}}


def save(data: Dict[str, Any]) -> None:
    SCOREBOARD.parent.mkdir(parents=True, exist_ok=True)
    tmp = SCOREBOARD.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    tmp.replace(SCOREBOARD)


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "   n/a"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def show(data: Dict[str, Any]) -> None:
    labels = list(data.get("labels", {}).keys())
    if not labels:
        print("scoreboard is empty")
        return
    fixtures: List[str] = []
    for label in labels:
        for name in data["labels"][label]:
            if name not in fixtures:
                fixtures.append(name)
    header = f"{'fixture':<12}"
    for label in labels:
        header += f" | {label:<34}"
    print(header)
    print("-" * len(header))
    for name in fixtures:
        line = f"{name:<12}"
        for label in labels:
            row = data["labels"][label].get(name)
            if not row:
                line += f" | {'-':<34}"
                continue
            if row.get("status") != "ok":
                line += f" | {row.get('status', '?'):<34}"
                continue
            line += (
                f" | c{_fmt(row.get('correlation'))}"
                f" s{_fmt(row.get('snr_db'), '.1f')}"
                f" l{_fmt(row.get('log_spectral_distance_db'), '.1f')}"
                f" g{_fmt(row.get('seg_snr_db'), '.1f')}"
                f" p{row.get('partials')}"
            )
        print(line)
    print()
    print("c=correlation  s=snr_db  l=log_spectral_distance_db  "
          "g=seg_snr_db  p=partials")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", help="label for this run")
    ap.add_argument("--fixtures", nargs="*", help="fixture subset")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="extra args forwarded to main.py res")
    ap.add_argument("--noise", action="store_true",
                    help="enable the stochastic noise class (--noise)")
    ap.add_argument("--spectral-lane", action="store_true",
                    help="enable the spectral acceptance lane")
    ap.add_argument("--k-hits", type=int, default=None,
                    help="percussive hits the bank may nominate per section")
    ap.add_argument("--k-notes", type=int, default=None,
                    help="harmonic notes the bank may nominate per section")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args(argv)

    extra = list(args.extra)
    if args.noise:
        extra.append("--noise")
    if args.spectral_lane:
        extra.append("--spectral-lane")
    if args.k_hits is not None:
        extra.extend(["--k-hits", str(args.k_hits)])
    if args.k_notes is not None:
        extra.extend(["--k-notes", str(args.k_notes)])

    data = load()
    if args.show or not args.label:
        show(data)
        return 0

    names = args.fixtures or list(FIXTURES.keys())
    rows: Dict[str, Any] = {}
    for name in names:
        rel = FIXTURES.get(name)
        if rel is None:
            print(f"[!] unknown fixture: {name}", file=sys.stderr)
            continue
        print(f"[*] {args.label}/{name} ...", flush=True)
        row = run_fixture(args.label, name, rel, args.duration,
                          args.iterations, extra)
        rows[name] = row
        if row.get("status") == "ok":
            print(f"    corr={_fmt(row.get('correlation'))} "
                  f"lsd={_fmt(row.get('log_spectral_distance_db'), '.2f')} "
                  f"seg={_fmt(row.get('seg_snr_db'), '.2f')} "
                  f"partials={row.get('partials')} "
                  f"({row.get('seconds')}s)", flush=True)
        else:
            print(f"    FAILED: {row.get('error', '')[:400]}", flush=True)

    data.setdefault("labels", {})[args.label] = rows
    data["schema"] = "res-scoreboard/1.0"
    save(data)
    show(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
