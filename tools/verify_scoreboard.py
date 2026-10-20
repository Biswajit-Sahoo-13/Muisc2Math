"""Verify the scoreboard's rows match the ledgers they point at.

The rules live in ``src/scoreboard.py`` so this CLI and the web UI apply the
same checks. Run with no arguments to verify; ``--label NAME`` to verify one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoreboard import ROOT, verify  # noqa: E402


def main() -> int:
    result = verify()
    for label, v in sorted(result["labels"].items()):
        print(f"--- label {label} ---")
        for r in v["rows"]:
            if r.get("error"):
                print(f"  {r['fixture']:<12} ERROR {r['error']}")
                continue
            flags = " ".join(
                f"{k}={'Y' if val is True else ('-' if val is None else 'N')}"
                for k, val in r["checks"].items())
            verdict = "ok"
            if any(r["checks"].get(k) is False
                   for k in ("row", "init_row")):
                verdict = "INCONSISTENT"
            if any(r["checks"].get(k) is False
                   for k in ("best_ge_init", "best_is_max")):
                verdict += "  (did not improve)"
            init = r["initial"]
            init_txt = "-" if init is None else f"{init:.6f}"
            print(f"  {r['fixture']:<12} best={r['correlation']:.6f} "
                  f"init={init_txt} {flags}  {verdict}")
        note = "  (captured regression: expected)" \
            if v["expect_improvement"] is False else ""
        print(f"  -> consistent={v['consistent']} improved={v['improved']}{note}")
    print()
    print("CONSISTENT (every row matches its ledger):", result["consistent"])
    if result["unexpected_regressions"]:
        print("UNEXPECTED REGRESSION (these labels must improve): "
              + ", ".join(result["unexpected_regressions"]))
    print("VERDICT:", "OK" if result["ok"] else "FAIL")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
