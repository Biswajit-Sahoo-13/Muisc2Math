"""Rebuild a scoreboard label from the ledgers of a completed run."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SB = ROOT / "data/output/scoreboard.json"
METRICS = ("correlation", "snr_db", "log_spectral_distance_db",
           "seg_snr_db", "onset_corr")

TARGET = sys.argv[1] if len(sys.argv) > 1 else "after"
SOURCE = sys.argv[2] if len(sys.argv) > 2 else TARGET


def rebuild(label: str, fixtures) -> dict:
    rows = {}
    for name, rel in fixtures.items():
        led = ROOT / f"data/output/_scoreboard/{SOURCE}/{name}/res_ledger.json"
        if not led.exists():
            continue
        L = json.loads(led.read_text(encoding="utf-8"))
        row = {"fixture": name, "status": "ok", "label": label,
               "source": str(L.get("source", "")),
               "ledger": str(led.relative_to(ROOT)),
               "command": (f".venv\\Scripts\\python.exe main.py res -i <fixture> "
                           f"-o data/output/_scoreboard/{label}/{name} "
                           f"-d 10 --iterations 6")}
        best = L.get("best", {}) or {}
        for m in METRICS:
            row[m] = best.get(m)
        secs = L.get("sections", []) or []
        row["partials"] = sum(len(s.get("partials", []) or []) for s in secs)
        row["sections"] = len(secs)
        row["stop_reason"] = L.get("stop_reason")
        bt = ba = 0
        for s in secs:
            for rec in (s.get("hypothesis_bank") or []):
                bt += 1
                ba += 1 if rec.get("accepted") else 0
        row["bank_records"] = bt
        row["bank_accepted"] = ba
        row["iterations_run"] = len(L.get("iterations") or [])
        row["initial_correlation"] = L.get("initial_correlation")
        rows[name] = row
    return rows


FIXTURES = {
    "stem_bass": "data/output/_fixtures/stem_bass.wav",
    "stem_drums": "data/output/_fixtures/stem_drums.wav",
    "stem_vocals": "data/output/_fixtures/stem_vocals.wav",
    "stem_other": "data/output/_fixtures/stem_other.wav",
    "mix_lost": "Metre - Lost.mp3",
}

d = json.loads(SB.read_text(encoding="utf-8"))
d.setdefault("labels", {})[TARGET] = rebuild(TARGET, FIXTURES)
SB.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
print("labels:", sorted(d["labels"]))

