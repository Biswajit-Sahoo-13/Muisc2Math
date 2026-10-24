"""Energy parity: per-section optimal gain refit on the shipped render.

Why: the referee optimises correlation, which is scale-invariant, so nothing
in the loop enforces loudness. Measured on the drums fixture the best render
ships at 2.1% of the source energy (SNR 0.45 dB) - the equations are right,
the volume is wrong. The parity refit is one scalar per section, fitted by
least squares (the gain that minimises that section's residual energy), so
it maximises SNR by construction and leaves correlation untouched wherever
the render is already at the right level (gain == 1).

Seams under test (public interfaces only):
- src.res_synthesis.apply_energy_parity  (the gain math)
- src.res_loop.run_loop(energy_parity=)  (the wiring into the shipped render)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.res_evidence import build_evidence
from src.res_loop import run_loop
from src.res_synthesis import apply_energy_parity

SR = 22050


def _sine(n: int, freq: float = 220.0) -> np.ndarray:
    t = np.arange(n) / SR
    return np.sin(2.0 * np.pi * freq * t)


def _make_chord(path: Path) -> None:
    """Same decaying triad the loop tests use."""
    t = np.arange(int(1.2 * SR)) / SR
    y = (
        0.60 * np.exp(-1.10 * t) * np.cos(2.0 * np.pi * 440.0 * t)
        + 0.35 * np.exp(-1.45 * t) * np.cos(2.0 * np.pi * 554.365 * t)
        + 0.25 * np.exp(-1.80 * t) * np.cos(2.0 * np.pi * 659.255 * t)
    )
    sf.write(str(path), y, SR)


# --- the gain math (worked examples: g* = <y, r> / <r, r>) ---------------

def test_parity_matches_each_section_energy() -> None:
    """Two sections rendered 3x and 1x too quiet come back at unity energy.

    Independent expectation: the least-squares gain equals ||y_i|| / ||r_i||
    per section, so 3.0 and 1.0 here.
    """
    n = SR
    y = np.concatenate([3.0 * _sine(n // 2), 1.0 * _sine(n - n // 2)])
    r = _sine(n)
    scaled, gains = apply_energy_parity(
        y, r, [(0, n // 2), (n // 2, n)]
    )
    assert np.allclose(gains, [3.0, 1.0], atol=1e-9)
    e_y = np.sum(y ** 2)
    e_r = np.sum(scaled ** 2)
    assert abs(e_r / e_y - 1.0) < 1e-9


def test_parity_leaves_a_correct_render_untouched() -> None:
    """Where the render is already at level, the gain is exactly 1.0."""
    r = 0.8 * _sine(SR)
    scaled, gains = apply_energy_parity(r.copy(), r, [(0, SR)])
    assert gains[0] == 1.0
    assert np.array_equal(scaled, r)


def test_parity_skips_silent_render_sections() -> None:
    """A section whose render is (near) zero must not NaN or explode."""
    n = SR
    y = _sine(n)
    r = np.zeros(n)
    r[n // 2:] = 0.5 * _sine(n - n // 2)
    scaled, gains = apply_energy_parity(y, r, [(0, n // 2), (n // 2, n)])
    assert np.all(np.isfinite(scaled))
    assert np.all(np.isfinite(gains))
    assert gains[0] == 1.0  # skipped, not scaled


def test_parity_never_inverts_the_render() -> None:
    """An anti-correlated section floors the gain at 0, never negative."""
    n = SR
    y = -_sine(n)
    r = _sine(n)
    scaled, gains = apply_energy_parity(y, r, [(0, n)])
    assert gains[0] >= 0.0


def test_parity_respects_the_max_gain_argument() -> None:
    """A pathological near-zero render is bounded by the caller's cap."""
    n = SR
    y = _sine(n)
    r = 1e-9 * _sine(n)
    scaled, gains = apply_energy_parity(y, r, [(0, n)], max_gain=10.0)
    assert gains[0] <= 10.0 + 1e-12
    assert np.all(np.isfinite(scaled))


# --- rms mode: loudness parity, not SNR parity ---------------------------

def test_parity_rms_mode_matches_source_rms() -> None:
    """mode='rms' scales each span to the source's RMS, not to the
    least-squares optimum. The LS gain provably lands the render at
    (correlation) x source RMS - SNR-optimal but sounding too quiet on
    low-corr stems; RMS parity is the 'sounds like the original' choice."""
    n = SR
    y = _sine(n)
    r = 0.3 * _sine(n)
    scaled, gains = apply_energy_parity(y, r, [(0, n)], mode="rms")
    assert gains[0] == pytest.approx(1.0 / 0.3, rel=1e-9)
    assert float(np.sqrt(np.mean(scaled ** 2))) == pytest.approx(
        float(np.sqrt(np.mean(y ** 2))), rel=1e-9
    )


def test_parity_rms_mode_loudness_pillar_reaches_100() -> None:
    """On the fidelity benchmark the loudness pillar must saturate."""
    from src.fidelity import compute_fidelity

    n = SR
    rng = np.random.default_rng(11)
    y = rng.standard_normal(n)
    r = 0.3 * y
    scaled, _ = apply_energy_parity(y, r, [(0, n)], mode="rms")
    f = compute_fidelity(y, scaled, SR)
    assert f["loudness_pct"] == pytest.approx(100.0, abs=0.5)


def test_parity_default_mode_stays_ls() -> None:
    """The shipped default is the least-squares gain (SNR-optimal). On a
    partially-correlated render the LS gain is strictly quieter than the
    RMS gain - it lands the render at (correlation) x source RMS."""
    n = SR
    y = _sine(n)
    r = 0.3 * _sine(n)
    scaled, gains = apply_energy_parity(y, r, [(0, n)])
    # Identical shape: LS gain = rms_y/rms_r = rms gain.
    assert gains[0] == pytest.approx(1.0 / 0.3, rel=1e-9)
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(n)
    y2 = _sine(n) + 0.5 * noise
    r2 = _sine(n)  # corr ~ 0.89
    _, ls_gains = apply_energy_parity(y2, r2, [(0, n)])
    _, rms_gains = apply_energy_parity(y2, r2, [(0, n)], mode="rms")
    assert rms_gains[0] > ls_gains[0]  # RMS mode pushes louder than LS


def test_parity_rms_mode_rejects_unknown_mode() -> None:
    n = SR
    with pytest.raises(ValueError):
        apply_energy_parity(_sine(n), _sine(n), [(0, n)], mode="bogus")


# --- the wiring (public run_loop interface) ------------------------------

def _run(tmp_path: Path, parity: bool, parity_mode: str = "ls") -> dict:
    tag = f"{parity}_{parity_mode}"
    wav = tmp_path / f"chord_{tag}.wav"
    _make_chord(wav)
    ledger = build_evidence(
        str(wav), str(tmp_path / f"ev_{tag}"),
        sr=SR, max_duration=1.2, window_s=0.6, threshold=0.01, max_tracks=12,
    )
    return run_loop(
        ledger, str(wav), str(tmp_path / f"out_{tag}"),
        iterations=1, target_corr=0.0, provider="numeric",
        bank=False, energy_parity=parity, parity_mode=parity_mode,
    )


def test_run_loop_applies_parity_and_maximises_snr(tmp_path) -> None:
    plain = _run(tmp_path, parity=False)
    fixed = _run(tmp_path, parity=True)

    rec = fixed["energy_parity"]
    assert rec["applied"] is True
    assert len(rec["gains"]) == len(fixed["sections"])
    assert all(np.isfinite(g) and g >= 0.0 for g in rec["gains"])

    # Per-section least-squares gains maximise the summed SNR by
    # construction, so parity can only improve it on the same render.
    assert fixed["best"]["snr_db"] >= plain["best"]["snr_db"] - 1e-6
    # Correlation is scale-invariant per section; the global figure moves
    # only by the gain spread, which stays small on tonal material.
    assert (
        fixed["best"]["correlation"] >= plain["best"]["correlation"] - 0.05
    )


def test_run_loop_parity_is_off_by_default(tmp_path) -> None:
    """Published numbers must not move until the flag is passed."""
    plain = _run(tmp_path, parity=False)
    assert not (plain.get("energy_parity") or {}).get("applied")


def test_run_loop_rms_mode_is_recorded_and_loud(tmp_path) -> None:
    """parity_mode='rms' shows in the ledger and lands the render at the
    source loudness (the LS mode provably lands it at corr x source RMS)."""
    ls = _run(tmp_path, parity=True, parity_mode="ls")
    rms = _run(tmp_path, parity=True, parity_mode="rms")
    assert (rms["energy_parity"] or {}).get("mode") == "per-section rms"
    assert (ls["energy_parity"] or {}).get("mode") == "per-section least squares"

    from src.fidelity import compute_fidelity

    y, _ = _load(tmp_path, "chord_True_rms.wav")
    r_ls, _ = sf.read(str(tmp_path / "out_True_ls" / "res_best.wav"))
    r_rms, _ = sf.read(str(tmp_path / "out_True_rms" / "res_best.wav"))
    n = min(len(y), len(r_ls), len(r_rms))
    f_ls = compute_fidelity(y[:n], r_ls[:n], SR)
    f_rms = compute_fidelity(y[:n], r_rms[:n], SR)
    assert f_rms["loudness_pct"] > f_ls["loudness_pct"]


def _load(tmp_path: Path, name: str):
    import soundfile as sf

    y, sr = sf.read(str(tmp_path / name), always_2d=False)
    return y, sr
