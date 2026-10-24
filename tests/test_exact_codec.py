"""The exact mode: equations + stored residual = 100% reconstruction.

Pins the missing member of the parametric family: the render plus a
FLAC-compressed residual reconstructs the source to within the
quantization bound (max error <= peak / 2^(bits-1)), i.e. correlation
~1.0 - the first exact-mode ladder rung for an equation engine.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.exact_codec import encode_exact_residual, exact_reconstruct
from src.fidelity import compute_fidelity

SR = 22050


def _chord(n: int = SR) -> np.ndarray:
    t = np.arange(n) / SR
    return (0.5 * np.sin(2 * np.pi * 220.0 * t)
            + 0.3 * np.sin(2 * np.pi * 277.18 * t)
            + 0.2 * np.exp(-1.0 * t) * np.sin(2 * np.pi * 330.0 * t))


def test_exact_reconstruct_hits_100_percent_fidelity() -> None:
    """Render + stored residual must reconstruct the source at ~100% -
    the fidelity benchmark's own scale, from the engine's equations."""
    y = _chord()
    recon = 0.55 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)
    payload = encode_exact_residual(y, recon, SR)
    back = exact_reconstruct(recon, payload)
    f = compute_fidelity(y, back, SR)
    assert f["fidelity_pct"] == pytest.approx(100.0, abs=0.01)
    assert f["tone_pct"] == pytest.approx(100.0, abs=0.01)


def test_reconstruction_error_is_within_the_quantization_bound() -> None:
    y = _chord()
    recon = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)
    payload = encode_exact_residual(y, recon, SR, bits=16)
    back = exact_reconstruct(recon, payload)
    bound = payload["residual_peak"] / 2 ** 15 + 1e-12
    assert float(np.max(np.abs(y[: len(back)] - back))) <= bound


def test_silent_residual_compresses_to_nothing() -> None:
    """A perfect render leaves a zero residual: the exact mode costs
    ~nothing because there is nothing left to store."""
    y = _chord()
    payload = encode_exact_residual(y, y.copy(), SR)
    assert len(payload["flac_bytes"]) < 2000  # FLAC of pure zeros


def test_stats_are_json_able_and_honest() -> None:
    y = _chord()
    recon = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)
    payload = encode_exact_residual(y, recon, SR)
    for k in ("flac_bytes", "residual_peak", "raw_residual_bytes",
              "flac_vs_raw_pct", "quantization_error_max"):
        assert k in payload
    json.dumps({k: v for k, v in payload.items() if k != "flac_bytes"})


def test_bits_are_a_real_dial() -> None:
    """Fewer bits = smaller payload, larger bounded error."""
    y = _chord()
    recon = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)
    p16 = encode_exact_residual(y, recon, SR, bits=16)
    p8 = encode_exact_residual(y, recon, SR, bits=8)
    assert len(p8["flac_bytes"]) < len(p16["flac_bytes"])
    assert p8["quantization_error_max"] > p16["quantization_error_max"]
    back8 = exact_reconstruct(recon, p8)
    f8 = compute_fidelity(y, back8, SR)
    # 8-bit residual: peak/256 quantization - audible but still far above
    # what the equations alone reconstruct.
    assert f8["fidelity_pct"] > 95.0


def test_exact_roundtrip_converges_to_the_source() -> None:
    """Encoding the previous reconstruction's residual again must converge
    toward the source: pass 2's residual is pass 1's quantization error
    (<= peak/65536), so its own quantization is negligible by comparison."""
    y = _chord()
    recon = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)
    payload = encode_exact_residual(y, recon, SR)
    back = exact_reconstruct(recon, payload)
    payload2 = encode_exact_residual(y, back, SR)
    back2 = exact_reconstruct(back, payload2)
    err1 = float(np.max(np.abs(y[: len(back)] - back)))
    err2 = float(np.max(np.abs(y[: len(back2)] - back2)))
    assert err2 <= err1 + 1e-12
    assert err2 <= 1e-8
