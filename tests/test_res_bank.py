"""Unit tests for the RES v2 hypothesis bank (class B: harmonic_string)."""

from __future__ import annotations

import math

import numpy as np
import soundfile as sf

from src.res_bank import (
    acceptance_margin,
    detect_hits,
    f0_candidates,
    freq_m,
    percussive_bundle,
    tournament_section,
)
from src.res_evidence import build_evidence
from src.res_loop import _render_partial, run_loop, _score

SR = 22050


def _inharmonic_note(
    f0: float,
    B: float,
    m_max: int,
    sr: int = SR,
    dur: float = 2.0,
    base_amp: float = 0.20,
    lam: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for m in range(1, m_max + 1):
        fm = m * f0 * math.sqrt(1.0 + B * m * m)
        y += (base_amp / m) * np.exp(-lam * t) * np.cos(
            2.0 * np.pi * fm * t
        )
    rng = np.random.default_rng(seed)
    return y + 0.002 * rng.standard_normal(y.size)


def test_freq_m_relation():
    assert freq_m(110.0, 0.0, 4) == 440.0
    assert abs(freq_m(110.0, 2e-4, 4) - 440.0 * math.sqrt(1.0 + 32e-4)) < 1e-9


def test_acceptance_margin_noise_floor():
    # fitting p random params to white noise shaves ~ p/n of the residual
    n, p = 55000, 20
    err = 0.3
    m = acceptance_margin(1.0, err, p, n)
    assert m > 3.5 * p / n * err


def test_f0_candidates_finds_fundamental():
    y = _inharmonic_note(200.0, 0.0, 6)
    cands = f0_candidates(y, SR, k=2)
    assert cands
    assert abs(cands[0] - 200.0) < 3.0


def test_f0_candidates_cancellation_recovers_two_sources():
    y = _inharmonic_note(150.0, 0.0, 5) + _inharmonic_note(
        220.0, 0.0, 4, base_amp=0.12, seed=1
    )
    cands = f0_candidates(y, SR, k=3)
    near_150 = any(abs(c - 150.0) < 3.0 for c in cands)
    near_220 = any(abs(c - 220.0) < 4.5 for c in cands)
    assert near_150 and near_220, cands


def _hit_fixture(dur: float = 1.5, hit_t: float = 0.7):
    sr = SR
    t = np.arange(int(dur * sr)) / sr
    y = 0.10 * np.exp(-0.5 * t) * np.sin(2.0 * np.pi * 200.0 * t)
    h = int(hit_t * sr)
    th = np.arange(int(0.1 * sr)) / sr
    y[h:h + th.size] += 0.6 * np.exp(-35.0 * th) * np.sin(2.0 * np.pi * 180.0 * th)
    y[h:h + th.size] += 0.4 * np.exp(-35.0 * th) * np.sin(2.0 * np.pi * 320.0 * th)
    return y, hit_t


def test_percussive_bundle_matches_hit():
    y, hit_t = _hit_fixture()
    vari = percussive_bundle(y, SR, 0, 0.0085, int(hit_t * SR))
    assert vari.partials
    freqs = [float(p["frequency"]) for p in vari.partials]
    assert any(abs(f - 180.0) < 25.0 for f in freqs), freqs
    assert any(abs(f - 320.0) < 35.0 for f in freqs), freqs
    assert all(
        float(p["envelope_decay"]) in (30.0, 80.0, 200.0)
        for p in vari.partials
    )
    assert all(
        float(p["start_time"]) >= hit_t - 0.02
        and float(p["end_time"]) <= hit_t + 0.2
        for p in vari.partials
    )
    assert vari.err < float(np.sum(y ** 2))


def test_tournament_records_bundles_for_hits():
    y, hit_t = _hit_fixture()
    class_a = [{
        "model": "damped", "amplitude": 0.10, "envelope_decay": 0.5,
        "envelope_t0": 0.0, "frequency": 200.0, "phase": -1.5708,
    }]
    recs = tournament_section(y, class_a, SR, start_sample=0)
    bundles = [r for r in recs if r["class"] == "percussive_bundle"]
    assert bundles, recs
    assert any(
        abs((r.get("t0") or -1.0) - hit_t) < 0.05 for r in bundles
    ), bundles


def test_detect_hits_locates_onsets():
    y, hit_t = _hit_fixture()
    hits = detect_hits(y, SR, 0, k=2)
    assert hits
    assert any(abs(h / SR - hit_t) < 0.03 for h in hits), hits


def test_score_reports_segmental_and_onset():
    n = int(1.0 * SR)
    t = np.arange(n) / SR
    y = 0.4 * np.exp(-0.3 * t) * np.sin(2.0 * np.pi * 220.0 * t)
    m_bad = _score(y, np.zeros(n), SR)
    m_good = _score(y, y, SR)
    for key in ("seg_snr_db", "onset_corr"):
        assert key in m_bad and key in m_good
    assert m_good["seg_snr_db"] > m_bad["seg_snr_db"] + 10.0
    assert m_good["onset_corr"] > 0.9
    assert m_bad["onset_corr"] < 0.5


def test_ledger_records_carry_classes(tmp_path):
    y, _ = _hit_fixture(dur=1.2)
    path = tmp_path / "hit.wav"
    sf.write(str(path), y, SR)
    ledger = build_evidence(
        str(path), str(tmp_path / "ev"), sr=SR, max_duration=1.2,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )
    ledger = run_loop(
        ledger, str(path), str(tmp_path / "res"),
        iterations=1, target_corr=0.0, bank=True,
    )
    assert ledger["schema"] == "res/1.1"
    for sec in ledger["sections"]:
        for r in sec.get("hypothesis_bank") or []:
            assert r.get("class") in {"harmonic_string", "percussive_bundle", "filtered_noise", "lpc_noise"}
            assert "accepted" in r and "param_count" in r
    accepted = [
        r for s in ledger["sections"]
        for r in (s.get("hypothesis_bank") or []) if r.get("accepted")
    ]
    assert accepted
    if any(r.get("class") == "percussive_bundle" for r in accepted):
        assert "percussive_bundle" in ledger["equation_classes"]


def test_hypothesis_recovers_inharmonic_string():
    f0_true, B_true = 180.0, 5e-4
    y = _inharmonic_note(f0_true, B_true, 6)
    # class A state: one half-fitted unrelated partial
    class_a = [{
        "model": "damped", "amplitude": 0.05, "envelope_decay": 1.5,
        "envelope_t0": 0.0, "frequency": 700.0, "phase": 0.0,
    }]

    def render(parts):
        r = np.zeros(len(y))
        for p in parts:
            r += _render_partial(
                p, len(y), SR, start_sample=0, stft_half_window_s=0.0
            )
        return r

    corr_before = float(np.corrcoef(y, render(class_a))[0, 1])
    recs = tournament_section(y, class_a, SR, start_sample=0)
    assert recs, "tournament must at least record candidates"
    accepted = [r for r in recs if r["accepted"]]
    assert accepted, recs
    # The bank may win with a subharmonic or other groupings; the truth at
    # f0=180 must still appear among the recovered candidates.
    hit = [
        r for r in accepted
        if r.get("f0") is not None and abs(r["f0"] - f0_true) < 0.03 * f0_true
    ]
    assert hit, recs
    assert hit[0]["residual_after"] < hit[0]["residual_before"]
    assert hit[0]["B"] < 2e-3
    corr_after = float(np.corrcoef(y, render(class_a))[0, 1])
    assert corr_after > corr_before + 0.05


def test_hypothesis_rejects_pure_noise():
    rng = np.random.default_rng(3)
    noise = 0.05 * rng.standard_normal(int(2.0 * SR))
    # class A state that explains its component exactly: the residual is
    # then purely white noise, so any accepted note would be noise-chasing.
    class_a = [{
        "model": "damped", "amplitude": 0.30, "envelope_decay": 1.0,
        "envelope_t0": 0.0, "frequency": 300.0, "phase": 0.0,
    }]
    y = noise + _render_partial(
        class_a[0], len(noise), SR, start_sample=0
    )
    n_before = len(class_a)
    recs = tournament_section(y, class_a, SR, start_sample=0)
    assert all(not r["accepted"] for r in recs), recs
    assert len(class_a) == n_before


def test_tournament_is_monotone():
    y = _inharmonic_note(196.0, 2e-4, 8) + _inharmonic_note(
        246.9, 1e-4, 4, base_amp=0.10
    )
    class_a = [{
        "model": "damped", "amplitude": 0.08, "envelope_decay": 2.0,
        "envelope_t0": 0.0, "frequency": 350.0, "phase": 0.4,
    }]
    resid_before = y - _render_partial(
        class_a[0], len(y), SR, start_sample=0
    )
    e_before = float(np.sum(resid_before ** 2))
    recs = tournament_section(y, class_a, SR, start_sample=0)
    for r in recs:
        if r["accepted"]:
            assert r["residual_after"] < r["residual_before"]
    resid_after = y - sum(
        _render_partial(p, len(y), SR, start_sample=0) for p in class_a
    )
    e_after = float(np.sum(resid_after ** 2))
    assert e_after < e_before


def test_bank_disabled_leaves_sections_clean(tmp_path):
    path = tmp_path / "note.wav"
    y = _inharmonic_note(196.0, 1e-4, 5, dur=1.2, base_amp=0.4)
    sf.write(str(path), y, SR)
    ledger = build_evidence(
        str(path), str(tmp_path / "ev"), sr=SR, max_duration=1.2,
        window_s=0.6, threshold=0.01, max_tracks=12,
    )
    assert ledger["schema"] == "res/1.1"
    ledger = run_loop(
        ledger, str(path), str(tmp_path / "res"),
        iterations=1, target_corr=0.0, bank=False,
    )
    for sec in ledger["sections"]:
        assert sec.get("hypothesis_bank") == []
