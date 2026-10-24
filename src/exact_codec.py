"""The exact mode: equations + the stored remainder = 100% reconstruction.

The one law from the research sweep: 100% comes from identity and exact
coding of the residual, never from modeling. The parametric class (this
engine) is the only codec family with no lossless member - no published
standard pairs a sinusoidal core with an exact residual stream.

This module is that missing member, first increment: after the equations
fit the song, the residual `y - render` is quantized to a fixed bit depth
and FLAC-compressed in memory. Reconstruction is render + decoded
residual, exact to within the quantization bound (peak / 2^(bits-1)).

The ladder this creates: equations alone (compact, ~70% on a mastered
mix) -> equations + residual (exact, ~100.0%) - scalable between, which
is the "100% and 0%" product no parametric codec ships.
"""
from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
import soundfile as sf


def _quantize(residual: np.ndarray, bits: int):
    # Scale by 2^(bits-1) so the FLAC PCM round-trip is EXACT: writing
    # q/2^(bits-1) as PCM_bits and reading it back returns q/2^(bits-1)
    # with no second rounding (a 32767 scale would not survive the
    # container's 32768 grid).
    peak = float(np.max(np.abs(residual))) if residual.size else 0.0
    if peak <= 1e-12:
        peak = 1e-12
    scale = float(2 ** (bits - 1))
    q = np.clip(np.round(residual / peak * scale), -scale, scale - 1)
    return q.astype(np.int64), peak, scale


def _flac_subtype(bits: int) -> str:
    return "PCM_S8" if bits == 8 else f"PCM_{bits}"


def encode_exact_residual(
    y: np.ndarray,
    recon: np.ndarray,
    sr: int,
    bits: int = 16,
) -> Dict[str, Any]:
    """FLAC-compress the residual; return the payload and honest stats.

    ``flac_bytes`` is the in-memory FLAC of the quantized residual -
    the entire cost of the exact mode beyond the equations themselves.
    """
    x = np.asarray(y, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    n = min(x.size, r.size)
    residual = x[:n] - r[:n]
    q, peak, scale = _quantize(residual, bits)
    buf = io.BytesIO()
    sf.write(buf, q.astype(np.float64) / scale, sr, format="FLAC",
             subtype=_flac_subtype(bits))
    flac_bytes = buf.getvalue()
    resid_q = q / scale * peak
    err_after = float(np.max(np.abs(residual[:n] - resid_q[:n])))
    return {
        "flac_bytes": flac_bytes,
        "sr": sr,
        "n_samples": int(n),
        "bits": bits,
        "residual_peak": round(peak, 8),
        "quantization_error_max": round(err_after, 10),
        "raw_residual_bytes": int(n * (bits // 8)),
        "flac_vs_raw_pct": round(100.0 * len(flac_bytes) / max(int(n * (bits // 8)), 1), 1),
    }


def exact_reconstruct(
    recon: np.ndarray,
    payload: Dict[str, Any],
) -> np.ndarray:
    """Reconstruct y = render + decoded residual (exact to quantization)."""
    buf = io.BytesIO(payload["flac_bytes"])
    dec, _ = sf.read(buf, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    n = min(r.size, dec.size)
    out = r.copy()
    out[:n] = out[:n] + dec[:n] * float(payload["residual_peak"])
    return out
