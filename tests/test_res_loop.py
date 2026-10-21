"""Unit tests for the RES (Reverse Engineer a Song) evidence + loop."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from src.res_evidence import build_evidence, evidence_prompt
from src.res_loop import (
    _parse_json_edits,
    _render_partial,
    apply_edits,
    llm_edits_via_command,
    render_ledger,
    run_loop,
)

SR = 22050


def _make_chord(path: Path, sr: int = SR, dur: float = 1.2) -> np.ndarray:
    """Three decaying partials - simplest verifiable equation ground truth."""
    t = np.arange(int(dur * sr)) / sr
    y = (
        0.60 * np.exp(-1.10 * t) * np.cos(2.0 * np.pi * 440.0 * t)
        + 0.35 * np.exp(-1.45 * t) * np.cos(2.0 * np.pi * 554.365 * t)
        + 0.25 * np.exp(-1.80 * t) * np.cos(2.0 * np.pi * 659.255 * t)
    )
    sf.write(str(path), y, sr)
    return y


def _make_ledger(path: Path, out_dir: Path) -> dict:
    return build_evidence(
        str(path), str(out_dir), sr=SR, max_duration=1.2,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )


def test_build_evidence_ledger_shape(tmp_path):
    path = tmp_path / "chord.wav"
    _make_chord(path)
    ledger = _make_ledger(path, tmp_path / "evidence")

    assert ledger["schema"] == "res/1.1"
    assert len(ledger["sections"]) == 2
    assert ledger["iterations"] == []
    for sec in ledger["sections"]:
        assert sec["partial_count"] == len(sec["partials"]) >= 3
        assert 0.0 <= sec["metrics"]["correlation"] <= 1.0
        assert "snr_db" in sec["metrics"]
        assert "log_spectral_distance_db" in sec["metrics"]
    assert (tmp_path / "evidence" / "res_evidence_initial.json").exists()


def test_evidence_prompt_lists_sections_and_form(tmp_path):
    path = tmp_path / "chord.wav"
    _make_chord(path)
    ledger = _make_ledger(path, tmp_path / "evidence")

    prompt = evidence_prompt(ledger)
    assert "section 0" in prompt
    assert "corr=" in prompt
    assert ledger["equation_form"] in prompt
    # The prompt must carry the part ids the edit schema uses. The list is
    # sorted by amplitude and truncated, so without the id a proposer can
    # only guess which partial a line refers to.
    ids = re.findall(r"part_(\d{3}):", prompt)
    assert ids, "the prompt must show the part ids"
    widest = max(len(s["partials"]) for s in ledger["sections"])
    assert all(int(i) < widest for i in ids)


def test_render_ledger_reproduces_chord(tmp_path):
    path = tmp_path / "chord.wav"
    y = _make_chord(path)
    ledger = _make_ledger(path, tmp_path / "evidence")

    recon = render_ledger(ledger, len(y))
    std = float(np.std(y))
    corr = float(np.corrcoef(y, recon)[0, 1]) if std > 1e-12 else 0.0
    assert corr > 0.90


def test_apply_edits_schema_and_receipt():
    ledger = {
        "sections": [{
            "start": 0.0, "end": 1.0,
            "partials": [{
                "model": "constant", "amplitude": 0.5, "phase": 0.0,
                "envelope_decay": 1.0, "frequency": 440.0,
            }],
        }]
    }
    edits = [
        {"section": 0, "id": "part_000", "op": "refit_amplitude", "value": 0.9},
        {"section": 0, "id": "part_000", "op": "refit_phase", "value": 1.5},
        {"section": 0, "id": "part_000", "op": "set_frequency_offset", "value": 10.0},
        {"section": 0, "id": "part_017", "op": "set_decay", "value": 2.0},
        {"section": 5, "id": "part_000", "op": "drop", "value": 0.0},
        {"section": 0, "id": "part_000", "op": "not_an_op", "value": 1.0},
    ]
    assert apply_edits(ledger, edits) == 3
    p = ledger["sections"][0]["partials"][0]
    assert p["amplitude"] == 0.9
    assert p["phase"] == 1.5
    assert p["frequency"] == 450.0
    assert p["envelope_decay"] == 1.0


def test_parse_json_edits_tolerates_fences():
    text = "Sure! Here are my edits:\n```json\n" \
           '[{"section": 1, "id": "part_002", "op": "set_decay", "value": 2.5}]' \
           "\n```\n"
    edits = _parse_json_edits(text)
    assert edits == [{"section": 1, "id": "part_002", "op": "set_decay", "value": 2.5}]


def test_llm_edits_via_command_template_and_failures(tmp_path):
    script = (
        "import sys, json\n"
        "path = sys.argv[1]\n"
        "txt = open(path, encoding='utf-8').read()\n"
        "val = 2.5 if 'correlation' in txt else 0.0\n"
        "print('```json')\n"
        "print(json.dumps([{'section': 0, 'id': 'part_000', "
        "'op': 'set_decay', 'value': val}]))\n"
        "print('```')\n"
    )
    edits = llm_edits_via_command(
        "correlation=0.10",
        [sys.executable, "-c", script, "{prompt_file}"],
    )
    assert edits == [{"section": 0, "id": "part_000", "op": "set_decay", "value": 2.5}]

    failing = llm_edits_via_command(
        "x", [sys.executable, "-c", "import sys; sys.exit(2)"]
    )
    assert failing == []


def test_render_respects_support_window():
    """A short-support partial must not ring across the entire timeline."""
    sr = SR
    n = int(2.0 * sr)
    p = {"model": "constant", "frequency": 800.0, "amplitude": 0.5,
         "envelope_decay": 10.0, "envelope_t0": 1.2, "phase": 0.0,
         "start_time": 1.2, "end_time": 1.4}
    sig = _render_partial(p, n, sr, stft_half_window_s=1024.0 / sr)
    before = float(np.sqrt(np.mean(sig[: int(0.8 * sr)] ** 2)))
    inside = float(np.sqrt(np.mean(sig[int(1.3 * sr): int(1.4 * sr)] ** 2)))
    after = float(np.sqrt(np.mean(sig[int(2.0 * sr) - 100:] ** 2)))
    assert inside > 0.05
    assert before < 0.01 * inside + 1e-9
    assert after < 0.05 * inside + 1e-9


def test_run_loop_improves_and_writes_outputs(tmp_path):
    path = tmp_path / "chord.wav"
    y = _make_chord(path)
    ledger = _make_ledger(path, tmp_path / "res")
    initial = ledger["best"]["correlation"]

    ledger = run_loop(
        ledger, str(path), str(tmp_path / "res"),
        iterations=2, target_corr=0.0, provider="numeric",
    )
    assert len(ledger["iterations"]) == 2
    assert ledger["best"]["correlation"] >= initial
    assert ledger["best"]["correlation"] > 0.999
    for iter_rec in ledger["iterations"]:
        assert set(iter_rec) >= {"iteration", "provider", "accepted", "global"}

    leads = ledger["outputs"]
    for key in ("ledger", "best_wav", "final_wav", "equations_txt"):
        assert Path(leads[key]).exists()
    eq_text = Path(leads["equations_txt"]).read_text(encoding="utf-8")
    assert "RES equation model" in eq_text
    data = json.load(open(Path(leads["ledger"]), encoding="utf-8"))
    assert data["best"]["correlation"] >= initial


def test_run_loop_keeps_section_local_rendering(tmp_path):
    """Two tones in disjoint sections: per-section corr must stay high.

    Regression test for the global-time offset: partials fitted in the second
    section carry global times, so a renderer that treats the slice as t=0
    collapses the section correlation to ~0.
    """
    sr = SR
    dur = 1.2
    n = int(dur * sr)
    y = np.zeros(n)
    t1 = np.arange(n) / sr
    y += 0.5 * np.exp(-1.0 * t1) * np.cos(2.0 * np.pi * 440.0 * t1)
    start2 = int(0.7 * sr)
    t2 = (np.arange(n) - start2) / sr
    mask2 = t2 >= 0.0
    seg = 0.45 * np.exp(-1.2 * np.maximum(t2, 0.0)) * np.cos(2.0 * np.pi * 660.0 * t2)
    y[mask2] += seg[mask2]
    path = tmp_path / "two_tones.wav"
    sf.write(str(path), y, sr)

    ledger = build_evidence(
        str(path), str(tmp_path / "ev"), sr=sr, max_duration=dur,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )
    assert len(ledger["sections"]) == 2
    assert ledger["sections"][1]["partial_count"] >= 1

    ledger = run_loop(
        ledger, str(path), str(tmp_path / "res"),
        iterations=2, target_corr=0.0, provider="numeric",
    )
    for sec in ledger["sections"]:
        assert sec["metrics"]["correlation"] > 0.90, sec["metrics"]
        assert sec["partial_count"] >= 1
    assert ledger["best"]["correlation"] > 0.90
