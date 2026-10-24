"""Measure the two drifts the audit points at:

  (i)  the release fade: res_synthesis uses 0.03 s, sinusoidal_model's
       inline mask uses 0.25 s. Does matching them take the replica to 1.000?

  (ii) the shelf: the analyser's amplitude trajectory is np.interp-clamped
       past the last tracked frame, and a glide's phase integral freezes to
       DC over the same span. The RES renderer reproduces both clamps for
       glides (_glide_traj uses np.interp) but already extrapolates the
       fitted decay on the damped path.

For each variant: the replica (corr between the analyser's own
sinusoid_audio and the RES render of the same partials) and the fidelity
(corr between the reference and the RES render).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402

from src.analysis import load_audio  # noqa: E402
from src.res_evidence import _rephase_to_section  # noqa: E402
from src.res_loop import render_ledger  # noqa: E402
import src.res_synthesis as rs  # noqa: E402
from src.sinusoidal_model import sinusoidal_reconstruction  # noqa: E402

SR = 22050
WIN_S = 2.5
ST_HW = (2048 / 2.0) / SR
BEAT = [0, 64512, 111616, 174592, 220500]


def corr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = min(a.size, b.size); a, b = a[:n], b[:n]
    a = a - a.mean(); b = b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-18 else 0.0


_orig_glide = rs._glide_traj
_orig_fade = rs._RELEASE_FADE_S


def glide_extrapolated(tt, ff, aa, t):
    """Continue the glide's phase integral past the last frame instead of
    freezing it to DC, and keep the amplitude's decay running."""
    a_env, integral = _orig_glide(tt, ff, aa, t)
    if tt.size == 0:
        return a_env, integral
    f_smooth = rs._smooth_envelope(ff, 9)
    after = t > tt[-1]
    if np.any(after):
        integral = np.where(
            after,
            integral + 2.0 * np.pi * float(f_smooth[-1]) * (t - tt[-1]),
            integral,
        )
    return a_env, integral


def analyse(rel):
    y, _ = load_audio(str(ROOT / rel), target_sr=SR)
    y = y[: int(10.0 * SR)]
    yh, _ = librosa.effects.hpss(y)
    res = sinusoidal_reconstruction(
        yh, SR, n_fft=2048, hop_length=256, threshold=0.01, max_tracks=400,
        add_noise_layer=False, min_residual_ratio=1e-6, seed=0,
        window_s=WIN_S, glide_tolerance_hz=6.0,
    )
    return y, np.asarray(res["sinusoid_audio"], float), res["fitted_parameters"]


def build(partials, bounds, n):
    sections = []
    for i in range(len(bounds) - 1):
        ws, we = bounds[i], bounds[i + 1]
        lo, hi = ws / SR, we / SR
        sp = []
        for p in partials:
            ps = float(p.get("start_time", p.get("envelope_t0", 0.0)))
            pe = float(p.get("end_time", p.get("envelope_t0", 0.0)))
            if pe <= lo or ps >= hi:
                continue
            sp.append(_rephase_to_section(p, lo))
        sections.append({"start": lo, "end": hi, "start_sample": ws,
                         "end_sample": we, "partials": sp})
    return {"schema": "res/1.1", "sample_rate": SR, "duration": n / SR,
            "n_samples": n, "window_s": WIN_S, "stft_half_window_s": ST_HW,
            "sections": sections}


for rel in ("Metre - Lost.mp3", "data/output/_fixtures/stem_bass.wav"):
    y, sms, partials = analyse(rel)
    n = len(y)
    print(f"=== {rel} ===")
    variants = [
        ("current (fade 0.03, glide clamps)", 0.03, False),
        ("release fade 0.25 (match analyser)", 0.25, False),
        ("glide integral extrapolated", 0.03, True),
        ("both", 0.25, True),
    ]
    for label, fade, glide in variants:
        rs._RELEASE_FADE_S = fade
        rs._glide_traj = glide_extrapolated if glide else _orig_glide
        import src.res_loop as rl
        rl._render_partial = rs._render_partial
        try:
            win = int(round(WIN_S * SR))
            fixed = list(range(0, n, win))
            if fixed[-1] != n:
                fixed.append(n)
            out = []
            for bname, bounds in (("fixed", fixed), ("beat", BEAT[:5])):
                rec = render_ledger(build(partials, bounds, n), n)
                out.append((bname, corr(sms, rec), corr(y, rec)))
            print(f"  {label:<38} " + "   ".join(
                f"{bn}: replica={rp:.4f} fidelity={fd:.4f}"
                for bn, rp, fd in out))
        finally:
            rs._RELEASE_FADE_S = _orig_fade
            rs._glide_traj = _orig_glide
    print()
