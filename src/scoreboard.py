"""The scoreboard: what the engine measured, and whether it is any good.

A *label* is one named run of the engine over the frozen fixture set. A *row*
is one fixture's result inside a label. Both live in
``data/output/scoreboard.json`` and point at the ledger that produced them.

This module owns that file and the checks over it, so the CLI verifier and the
web UI read the same rules from one place.

Two verdicts, deliberately not conflated:

* **consistent** - the row's numbers agree with the ledger it names. Catches a
  stale ledger match under the same label.
* **improved** - the run beat its own initial render: ``best`` never falls
  below ``initial_correlation``, and ``best`` is exactly the best accepted
  iteration (or the initial, when nothing was accepted).

The second is the load-bearing one. A baseline whose gate could never fire
records ``best`` *below* ``initial`` and matches no iteration at all -
consistent with its own ledger, yet a shipped regression. A label may declare
itself a captured regression via ``expect_improvement``, so it is reported
rather than failing the tool; every other label must improve.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD = ROOT / "data" / "output" / "scoreboard.json"
RUN_ROOT = ROOT / "data" / "output" / "_scoreboard"
TOL = 1e-6

# The frozen fixture set. The stem fixtures are the REAL separated stems
# (written by tools/make_fixtures.py); the mix fixtures are whole songs.
FIXTURES: Dict[str, str] = {
    "mix_lost": "Metre - Lost.mp3",
    "stem_bass": "data/output/_fixtures/stem_bass.wav",
    "stem_drums": "data/output/_fixtures/stem_drums.wav",
    "stem_vocals": "data/output/_fixtures/stem_vocals.wav",
    "stem_other": "data/output/_fixtures/stem_other.wav",
}
FIXTURE_ORDER = ["mix_lost", "stem_bass", "stem_drums", "stem_other",
                 "stem_vocals"]


def python_exe(root: Optional[Path] = None) -> str:
    """The interpreter every scoreboard run must use, so numbers compare."""
    root = Path(root) if root is not None else ROOT
    venv = root / ".venv" / "Scripts" / "python.exe"
    return str(venv if venv.exists() else Path(sys.executable))


# ----------------------------------------------------------------- the file
def load(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path is not None else SCOREBOARD
    if not p.exists():
        return {"labels": {}, "expect_improvement": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save(data: Dict[str, Any], path: Optional[Path] = None) -> None:
    p = Path(path) if path is not None else SCOREBOARD
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ------------------------------------------------------------------- checks
def _resolve(root: Path, rel: str) -> Path:
    """Resolve a stored relative path on any platform.

    Scoreboards written on Windows contain backslashes, which are a legal
    filename character on POSIX and so do not resolve there. Normalise before
    joining.
    """
    return Path(root) / Path(str(rel).replace("\\", "/"))


def _posix(rel: str) -> str:
    """Store a path the same way on every platform."""
    return str(rel).replace("\\", "/")


def max_accepted_correlation(ledger: Dict[str, Any]) -> Optional[float]:
    """Highest correlation among the iterations the loop actually accepted."""
    vals = []
    for it in (ledger.get("iterations") or []):
        if not it.get("accepted"):
            continue
        c = (it.get("global") or {}).get("correlation")
        if c is not None:
            vals.append(float(c))
    return max(vals) if vals else None


def check_row(ledger: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    """Return {check_name: True | False | None}. None = not checkable."""
    best = float((ledger.get("best") or {})["correlation"])
    init = ledger.get("initial_correlation")
    init = float(init) if init is not None else None

    out: Dict[str, Optional[bool]] = {}
    out["row"] = abs(best - float(row["correlation"])) < 1e-9

    row_init = row.get("initial_correlation")
    if row_init is None or init is None:
        out["init_row"] = None
    else:
        out["init_row"] = abs(init - float(row_init)) < 1e-9

    out["best_ge_init"] = True if init is None else best >= init - TOL

    best_accepted = max_accepted_correlation(ledger)
    if best_accepted is None:
        # nothing was accepted: "best" must be the untouched initial render
        out["best_is_max"] = (True if init is None
                               else abs(best - init) < TOL)
    else:
        target = max(init if init is not None else 0.0, best_accepted)
        out["best_is_max"] = abs(best - target) < TOL
    return out


def verify(root: Optional[Path] = None) -> Dict[str, Any]:
    """Check every label. Returns per-label verdicts and the overall verdict."""
    root = Path(root) if root is not None else ROOT
    sb_path = root / "data" / "output" / "scoreboard.json"
    d = load(sb_path)
    expectations = d.get("expect_improvement") or {}

    labels: Dict[str, Any] = {}
    consistent = True
    for label, rows in sorted((d.get("labels") or {}).items()):
        label_consistent = True
        label_improved = True
        details: List[Dict[str, Any]] = []
        for name, row in sorted(rows.items()):
            p = _resolve(root, row["ledger"])
            if not p.exists():
                consistent = False
                label_consistent = False
                label_improved = False
                details.append({"fixture": name, "error": "missing ledger",
                                "checks": {}})
                continue
            ledger = json.loads(p.read_text(encoding="utf-8"))
            checks = check_row(ledger, row)
            if any(checks.get(k) is False for k in ("row", "init_row")):
                consistent = False
                label_consistent = False
            if any(checks.get(k) is False for k in ("best_ge_init", "best_is_max")):
                label_improved = False
            details.append({
                "fixture": name,
                "correlation": float(row.get("correlation", 0.0) or 0.0),
                "initial": row.get("initial_correlation"),
                "checks": checks,
                "consistent": label_consistent,
            })
        labels[label] = {
            "consistent": label_consistent,
            "improved": label_improved,
            "expect_improvement": expectations.get(label, True),
            "rows": details,
        }

    unexpected = [lbl for lbl, v in labels.items()
                  if not v["improved"] and v["expect_improvement"]]
    return {
        "labels": labels,
        "consistent": consistent,
        "unexpected_regressions": unexpected,
        "ok": consistent and not unexpected,
    }


# ------------------------------------------------------------------- runner
def run_label(
    label: str,
    fixtures: Optional[Sequence[str]] = None,
    duration: float = 10.0,
    iterations: int = 6,
    root: Optional[Path] = None,
    extra: Optional[Sequence[str]] = None,
    progress=None,
) -> Dict[str, Any]:
    """Run one label over the fixture set and record it in the scoreboard.

    This is the harness: it drives the *same* CLI the user would type, so a
    label is reproducible by hand as well as from the app.
    """
    root = Path(root) if root is not None else ROOT
    names = list(fixtures) if fixtures else list(FIXTURES.keys())
    py = python_exe(root)
    rows: Dict[str, Any] = {}
    if progress is not None:
        progress(0, len(names), "starting", None)
    for i, name in enumerate(names):
        rel = FIXTURES.get(name)
        if rel is None:
            continue
        src = root / rel
        if not src.exists():
            rows[name] = {"fixture": name, "status": "missing",
                          "source": rel, "error": f"fixture not found: {src}"}
            continue
        out = RUN_ROOT / label / name
        out.mkdir(parents=True, exist_ok=True)
        if progress is not None:
            progress(i, len(names), name, None)
        cmd = [py, str(root / "main.py"), "res", "-i", str(src),
               "-o", str(out), "-d", str(duration),
               "--iterations", str(iterations)]
        if extra:
            cmd.extend(extra)
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(root), capture_output=True,
                              text=True, encoding="utf-8", errors="replace")
        elapsed = round(time.time() - t0, 2)
        led = out / "res_ledger.json"
        row: Dict[str, Any] = {
            "fixture": name, "status": "ok" if proc.returncode == 0 else "failed",
            "label": label, "source": rel, "command": " ".join(cmd),
            "seconds": elapsed, "returncode": proc.returncode,
            "ledger": _posix(str(led.relative_to(root))) if led.exists() else None,
        }
        if not led.exists():
            row["error"] = ((proc.stdout or "")[-1500:] + "\n"
                            + (proc.stderr or "")[-1500:])
            rows[name] = row
        else:
            L = json.loads(led.read_text(encoding="utf-8"))
            best = L.get("best", {}) or {}
            for m in ("correlation", "snr_db", "log_spectral_distance_db",
                      "seg_snr_db", "onset_corr"):
                row[m] = best.get(m)
            secs = L.get("sections", []) or []
            row["partials"] = sum(len(s.get("partials", []) or []) for s in secs)
            row["sections"] = len(secs)
            row["stop_reason"] = L.get("stop_reason")
            row["initial_correlation"] = L.get("initial_correlation")
            rows[name] = row
        if progress is not None:
            progress(i + 1, len(names), name, rows[name].get("correlation"))

    sb = load(root / "data" / "output" / "scoreboard.json")
    sb.setdefault("labels", {})[label] = rows
    save(sb, root / "data" / "output" / "scoreboard.json")
    return rows
