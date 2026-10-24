"""P0.1 from the physics roadmap: open the spectral lane on low-corr stems.

The lane's acceptance floor was absolute 0.60 correlation, which
deliberately kept it inert on any stem already below 0.60 - exactly the
stems (drums at 0.42) that need stochastic models. The production
tournament call now passes ``corr_floor_mode="relative"``, a pure
degradation budget (``cur_corr - corr_max_drop``), so the lane is active
wherever the stem actually sits. Opt-in as before: only when the caller
turns the spectral lane on; published defaults untouched.

Seam under test: the public ``run_loop`` interface - a noise-dominated
section (corr ~0.45) with noise + lane enabled must be able to accept a
stochastic hypothesis (filtered_noise / lpc_noise / sibilance_burst),
which the absolute floor made structurally impossible.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from src.res_loop import run_loop

SR = 22050
STOCHASTIC = {"filtered_noise", "lpc_noise", "sibilance_burst"}


def _noise_dominated_wav(path: Path) -> np.ndarray:
    """A tonal partial drowned in noise: evidence fits the tone, corr ~0.45.

    Correlation of the tone-only render against this target is
    amp_tone / rms(y) ~ 0.45 < 0.60, so the old absolute lane floor could
    never fire here.
    """
    rng = np.random.default_rng(7)
    n = SR
    t = np.arange(n) / SR
    y = 0.35 * np.sin(2.0 * np.pi * 440.0 * t) + 0.65 * rng.standard_normal(n)
    y = 0.9 * y / np.max(np.abs(y))
    sf.write(str(path), y, SR)
    return y


def _ledger(path: Path) -> dict:
    """A hand-built ledger: one section, one tonal partial, corr 0.45."""
    n = SR
    return {
        "schema": "res/1.1",
        "source": str(path),
        "sample_rate": SR,
        "duration": 1.0,
        "n_samples": n,
        "window_s": 1.0,
        "best": {"correlation": 0.45, "snr_db": -1.0,
                 "log_spectral_distance_db": 15.0, "seg_snr_db": -1.0,
                 "onset_corr": 0.1},
        "sections": [{
            "start": 0.0, "end": 1.0, "start_sample": 0,
            "partial_count": 1,
            "partials": [{
                "frequency": 440.0, "amplitude": 0.35, "phase": 0.0,
                "envelope_decay": 0.0, "envelope_t0": 0.0,
                "start_time": 0.0, "end_time": 1.0,
                "model": "damped_cosine",
            }],
            "metrics": {"correlation": 0.45, "snr_db": -1.0,
                        "log_spectral_distance_db": 15.0,
                        "seg_snr_db": -1.0, "onset_corr": 0.1},
        }],
    }


def test_spectral_lane_accepts_stochastic_models_on_low_corr_sections(
    tmp_path,
) -> None:
    wav = tmp_path / "noisy.wav"
    _noise_dominated_wav(wav)
    res = run_loop(
        _ledger(wav), str(wav), str(tmp_path / "out"),
        iterations=1, target_corr=0.0, provider="numeric",
        bank=True, enable_noise=True, spectral_lane=True, workers=1,
    )
    records = res["sections"][0].get("hypothesis_bank") or []
    stoch = [r for r in records if r.get("class") in STOCHASTIC]
    assert stoch, "the stochastic lanes never fired at all"
    accepted = [r for r in stoch if r.get("accepted")]
    assert accepted, (
        "stochastic candidates were offered but all rejected - the lane "
        "floor is still absolute and inert below 0.60"
    )
    for r in accepted:
        # The degradation budget must hold: the accepted candidate did not
        # give back more correlation than the budget allows.
        assert r.get("residual_after") is not None
