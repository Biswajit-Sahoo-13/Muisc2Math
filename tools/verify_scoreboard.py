"""Verify the scoreboard's rows match the ledgers they point at.

Two kinds of check:

1. **Consistency** - the row's numbers agree with the ledger it names. This
   catches a stale ledger match under the same label.
2. **Improvement** - the run actually improved on its own initial render:
   ``best`` never falls below ``initial_correlation``, and ``best`` is exactly
   the best accepted iteration (or the initial, when nothing was accepted).

The second check is the load-bearing one. A baseline whose gate could never
fire records ``best`` *below* ``initial`` and matches no iteration at all -
"consistent" with its own ledger, yet a shipped regression. Rows that fail it
are reported, not silenced.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOREBOARD = ROOT / "data" / "output" / "scoreboard.json"
TOL = 1e-6


def _max_accepted_correlation(ledger):
    """Highest correlation among iterations the loop actually accepted."""
    vals = []
    for it in (ledger.get("iterations") or []):
        if not it.get("accepted"):
            continue
        c = (it.get("global") or {}).get("correlation")
        if c is not None:
            vals.append(float(c))
    return max(vals) if vals else None


def _check_row(ledger, row):
    """Return {check_name: True | False | None}. None = not checkable."""
    best = float((ledger.get("best") or {})["correlation"])
    init = ledger.get("initial_correlation")
    init = float(init) if init is not None else None

    out = {}
    out["row"] = abs(best - float(row["correlation"])) < 1e-9

    row_init = row.get("initial_correlation")
    if row_init is None or init is None:
        out["init_row"] = None
    else:
        out["init_row"] = abs(init - float(row_init)) < 1e-9

    out["best_ge_init"] = True if init is None else best >= init - TOL

    best_accepted = _max_accepted_correlation(ledger)
    if best_accepted is None:
        # nothing was accepted: "best" must be the untouched initial render
        out["best_is_max"] = (
            True if init is None else abs(best - init) < TOL
        )
    else:
        target = max(init if init is not None else 0.0, best_accepted)
        out["best_is_max"] = abs(best - target) < TOL
    return out


def main() -> int:
    if not SCOREBOARD.exists():
        print(f"no scoreboard at {SCOREBOARD}")
        return 2
    d = json.loads(SCOREBOARD.read_text(encoding="utf-8"))
    consistent = True
    improved: dict[str, bool] = {}
    for label, rows in sorted(d["labels"].items()):
        print(f"--- label {label} ---")
        label_ok = True
        for name, row in sorted(rows.items()):
            p = ROOT / row["ledger"]
            if not p.exists():
                print(f"  {name:<12} MISSING LEDGER {p}")
                consistent = False
                label_ok = False
                continue
            ledger = json.loads(p.read_text(encoding="utf-8"))
            checks = _check_row(ledger, row)
            # Two different verdicts, and they must not be conflated:
            #   consistent - the row's numbers agree with the ledger it names
            #   improved   - the run beat its own initial render
            # A baseline whose gate could never fire is CONSISTENT and NOT
            # improved; that is the finding, not a tooling failure.
            consistency_failed = [k for k in ("row", "init_row")
                                  if checks.get(k) is False]
            improvement_failed = [k for k in ("best_ge_init", "best_is_max")
                                  if checks.get(k) is False]
            if consistency_failed:
                consistent = False
                label_ok = False
            flags = " ".join(
                f"{k}={'Y' if v is True else ('-' if v is None else 'N')}"
                for k, v in checks.items()
            )
            init = ledger.get("initial_correlation")
            src = str(ledger.get("source", "")).replace("\\", "/").split("/")[-1]
            verdict = "ok" if not consistency_failed else (
                "INCONSISTENT:" + ",".join(consistency_failed))
            if improvement_failed:
                verdict += "  (did not improve: " + ",".join(improvement_failed) + ")"
            print(f"  {name:<12} best={float(ledger['best']['correlation']):.6f} "
                  f"init={init if init is None else f'{init:.6f}'} "
                  f"{flags}  {verdict}  src={src}")
        improved[label] = label_ok
    print()
    print("CONSISTENT (every row matches its ledger):", consistent)
    for label, ok in sorted(improved.items()):
        print(f"  {label:<10} consistent={ok}")
    return 0 if consistent else 1


if __name__ == "__main__":
    sys.exit(main())
