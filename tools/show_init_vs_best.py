"""Print initial vs best correlation for every scoreboard row, straight from
the ledgers."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "data/output/scoreboard.json").read_text(encoding="utf-8"))
print(f"{'label':<10} {'fixture':<12} {'initial':>10} {'best':>10} "
      f"{'delta':>9}  verdict")
for label, rows in sorted(d["labels"].items()):
    for name, row in sorted(rows.items()):
        L = json.loads((ROOT / row["ledger"]).read_text(encoding="utf-8"))
        init = L.get("initial_correlation")
        best = float(L["best"]["correlation"])
        acc = [it for it in (L.get("iterations") or []) if it.get("accepted")]
        if init is None:
            verdict = "?"
        elif best >= float(init) - 1e-9:
            verdict = "improved"
        else:
            verdict = "SHIPPED BELOW ITS OWN INITIAL RENDER"
        delta = "" if init is None else f"{best - float(init):+.6f}"
        print(f"{label:<10} {name:<12} "
              f"{('-' if init is None else f'{float(init):.6f}'):>10} "
              f"{best:>10.4f} {delta:>9}  {verdict} "
              f"(accepted iterations: {len(acc)})")
