"""The fidelity benchmark: "original = 100%, ours = ?".

One headline number answering how far a reconstruction sits from the
original, plus a pillar breakdown so the headline cannot hide where the
distance lives:

- tone      : waveform correlation x100 (shape agreement).
- spectrum  : 10^(-mrstft_db/20) x100 - the multi-resolution spectral
              distance converted to a 0-100 agreement (0 dB -> 100,
              6 dB -> 50, 20 dB -> 10).
- bands     : original-energy-weighted per-band correlation x100 - how
              much of the spectrum, weighed by where the original's energy
              actually is, agrees.
- loudness  : exp(-|ln(rms ratio)|) x100 - 100 at parity, ~37 at 2x off,
              ~14 at 4x off. Catches the too-quiet-render failure that
              correlation is blind to by construction (the referee's own
              metric is scale-invariant; measured, a drums stem once
              shipped at 2.1% of the source energy).
- fidelity  : mean of the four pillars. "Ours, in percent of the
              original." distance_pct = 100 - fidelity.

The pillars are independent sources of truth (docstring of
tests/test_fidelity.py pins worked examples); nothing here re-derives the
engine's own arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.residual import (
    _BAND_EDGES_HZ,
    _BAND_NAMES,
    align_signals,
    compute_error_metrics,
)


def compute_fidelity(y: np.ndarray, recon: np.ndarray, sr: int) -> Dict[str, Any]:
    """Score ``recon`` against ``y`` on the original = 100% scale."""
    x = np.asarray(y, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    x, r = align_signals(x, r)
    n = len(x)

    # Tone: Pearson correlation with the engine's degenerate guards.
    if n < 2:
        tone = 0.0
    else:
        sx, srms = float(np.std(x)), float(np.std(r))
        tone = float(np.corrcoef(x, r)[0, 1]) if (sx > 1e-12 and srms > 1e-12) else 0.0

    m = compute_error_metrics(x, r, sr=sr, extras=True)
    spectrum = 10.0 ** (-float(m["mrstft_db"]) / 20.0)

    # Bands: weights from the ORIGINAL's per-band energy, so a band the
    # original never excites cannot drag the score down.
    freqs = np.fft.rfftfreq(n, d=1.0 / sr) if n >= 2 else np.zeros(1)
    edges = (0.0,) + tuple(_BAND_EDGES_HZ) + (float(freqs[-1]) + 1.0,)
    corr_by_band = m["band_corr"]
    weights = []
    for i in range(len(_BAND_NAMES)):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        e = float(np.sum(np.abs(np.fft.rfft(x)[: mask.size][mask]) ** 2)) if n >= 2 else 0.0
        weights.append(e)
    total_w = float(sum(weights))
    if total_w > 0.0:
        band = sum(w * float(corr_by_band[name])
                   for w, name in zip(weights, _BAND_NAMES)) / total_w
    else:
        band = 0.0

    # Loudness: RMS ratio on a log scale.
    rms_y = float(np.sqrt(np.mean(x ** 2))) if n else 0.0
    rms_r = float(np.sqrt(np.mean(r ** 2))) if n else 0.0
    if rms_y <= 1e-12:
        loudness = 1.0 if rms_r <= 1e-12 else 0.0
    elif rms_r <= 1e-12:
        loudness = 0.0
    else:
        loudness = float(np.exp(-abs(np.log(rms_r / rms_y))))

    fidelity = 100.0 * float(np.mean([tone, spectrum, band, loudness]))
    fidelity = min(max(fidelity, 0.0), 100.0)
    return {
        "tone_pct": round(100.0 * tone, 2),
        "spectrum_pct": round(100.0 * spectrum, 2),
        "band_pct": round(100.0 * band, 2),
        "loudness_pct": round(100.0 * loudness, 2),
        "fidelity_pct": round(fidelity, 2),
        "distance_pct": round(100.0 - fidelity, 2),
    }
