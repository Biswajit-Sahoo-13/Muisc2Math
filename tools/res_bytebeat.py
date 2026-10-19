"""Ledger -> dollchan bytebeat (Funcbeat) equation exporter.

Emits the reconstruction as one JS function ready to paste into the Bytebeat
Composer: mode "Funcbeat", sample rate = ledger sr. Semantics mirror the
engine exactly (src/res_loop._render_partial): damped partials carry a
one-sided exp envelope, support windows with the SMS attack ramp and release
fade; glides carry pre-integrated phase + amplitude knots (linear interp,
matching np.interp).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TWO_PI = 6.283185307179586


def _f(v: float, k: int) -> str:
    """Compact but lossless-enough number format."""
    s = f"{v:.{k}g}"
    if "e" in s:
        s = f"{v:.{k}f}".rstrip("0").rstrip(".")
    return s


def _wrap(ph: float) -> float:
    return (float(ph) + math.pi) % (2 * math.pi) - math.pi


def _hold(lam: float) -> float:
    return min(1.5, math.log(100.0) / lam) if lam > 0.0 else 0.25


def _mask_parts(st: float, et: float, lam: float, hw: float):
    """(lo-bound, ramp_den, ramp_exists, ST0, S1, E) constants of the mask."""
    if hw > 0.0:
        pre = max(0.0, st - hw)
        ramp = (st - hw) >= 0.0
        st0 = pre
        den = max(st - pre, 1e-9)
    else:
        ramp = False
        st0 = st
        den = max(min(0.02, 0.5 * max(st, 1e-6)), 1e-9)
    e = et + _hold(lam)
    return st0, den, ramp, st, e


def collect(ledger: dict, ratio: float):
    hw = float(ledger.get("stft_half_window_s") or 0.0)
    scored = []
    for si, sec in enumerate(ledger["sections"]):
        ts = float(sec["start"])
        sec_lo, sec_hi = ts, float(sec["end"])
        for p in sec["partials"]:
            a = abs(float(p.get("amplitude", 0.0)))
            lam = float(p.get("envelope_decay", 0.0))
            st, et = p.get("start_time"), p.get("end_time")
            span = (float(et) - float(st)) if st is not None and et is not None else (sec_hi - ts)
            e = a * a * min(span, 1.0 / max(2.0 * lam, 1e-9))
            scored.append((si, ts, p, e, sec_lo, sec_hi))
    total = sum(e for _, _, _, e, _, _ in scored) or 1.0
    keep = [
        (si, ts, p, e, sec_lo, sec_hi)
        for si, ts, p, e, sec_lo, sec_hi in scored
        if e > total * ratio
    ]
    return keep, total, hw


def _glide_knots(p: dict, ts: float):
    from src.res_loop import _glide_traj  # exact engine smoothing/integration

    tr = p["trajectory"]
    tt = np.asarray(tr["t"], dtype=np.float64)
    ff = np.asarray(tr["f_hz"], dtype=np.float64)
    aa = np.asarray(tr["a"], dtype=np.float64)
    gain = float(p.get("projection_gain", 1.0))
    a_env, integral = _glide_traj(tt, ff, aa, tt)
    fr = float(p.get("frequency", 0.0)) - float(np.median(ff))
    # Knots carry ONLY the smoothed integral; the live linear term
    # 2*pi*fr*(T - ts) and the phase constant are applied in the emitted
    # math so the dtheta keeps running after np.interp clamps (engine-exact).
    return tt, gain * a_env, integral, TWO_PI * fr, ts, -float(p.get("phase", 0.0))


def simulate_exported(keep, hw, sr: int, dur: float) -> np.ndarray:
    """Numpy replica of the emitted JS math (rounding included), for check."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n)
    for si, ts, p, _, sec_lo, sec_hi in keep:
        i_lo = int(sec_lo * sr)
        i_hi = min(n, int(sec_hi * sr))
        if i_hi <= i_lo:
            continue
        tsec = t[i_lo:i_hi]
        if p.get("model") == "glide":
            tt, a_k, th_k, k2, s0, c0 = _glide_knots(p, ts)
            st, et = float(p["start_time"]), float(p["end_time"])
            lam_g = float(p.get("envelope_decay", 0.0))
            st0, den, ramp, _, e = _mask_parts(st, et, lam_g, hw)
            mask = np.clip((tsec - st0) / den, 0.0, 1.0) if ramp else np.ones_like(tsec)
            mask *= 1.0 - np.clip((tsec - e) / 0.25, 0.0, 1.0)
            sel = mask > 0.0
            if not sel.any():
                continue
            out[i_lo:i_hi][sel] += np.interp(tsec[sel], tt, a_k) * mask[sel] * np.cos(
                np.interp(tsec[sel], tt, th_k) + k2 * (tsec[sel] - s0) + c0
            )
            continue
        a = float(np.round(float(p["amplitude"]), 6))
        lam = float(np.round(float(p["envelope_decay"]), 6))
        t0 = float(np.round(float(p["envelope_t0"]), 6))
        f = float(np.round(float(p["frequency"]), 6))
        ph = _wrap(-float(p.get("phase", 0.0)))
        st, et = p.get("start_time"), p.get("end_time")
        env = a * np.exp(-lam * np.maximum(tsec - t0, 0.0))
        sig = env * np.cos(TWO_PI * f * (tsec - ts) + ph)
        if st is not None and et is not None:
            st0, den, ramp, s_, e = _mask_parts(float(st), float(et), lam, hw)
            mask = np.clip((tsec - st0) / den, 0.0, 1.0) if ramp else np.ones_like(tsec)
            mask *= 1.0 - np.clip((tsec - e) / 0.25, 0.0, 1.0)
            sig *= mask
        out[i_lo:i_hi] += sig
    return np.clip(out, -1.0, 1.0)


def emit(ledger: dict, ratio: float) -> str:
    keep, _, hw = collect(ledger, ratio)
    dur = float(ledger.get("duration") or 15.0)
    best = ledger["best"]
    out = []
    out.append(f"// RES equations: '{ledger.get('source', 'song')}'")
    out.append(
        f"// engine corr={best['correlation']:.4f} | {len(keep)} subtone"
        f" equations | {len(ledger.get('sections', []))} sections | {dur:.0f} s"
    )
    out.append("// dollchan: type FUNCBEAT, sample rate 22050")
    out.append("return function(t, srate) {")
    out.append("var s=0,T=t;")
    out.append(
        "function lp(ta,ya,z){var i=0,n=ta.length-1;"
        "if(z<=ta[0])return ya[0];while(i<n&&ta[i+1]<=z)++i;"
        "if(i>=n)return ya[n];return ya[i]+(ya[i+1]-ya[i])*(z-ta[i])/(ta[i+1]-ta[i]);}"
    )
    glide_idx = 0
    for si, ts, p, _, sec_lo, sec_hi in keep:
        if p.get("model") == "glide":
            tt, a_k, th_k, k2, s0, c0 = _glide_knots(p, ts)
            st, et = float(p["start_time"]), float(p["end_time"])
            lam_g = float(p.get("envelope_decay", 0.0))
            st0, den, ramp, _, e = _mask_parts(st, et, lam_g, hw)
            out.append(
                f"var T{glide_idx}=[{','.join(_f(v, 6) for v in tt)}],"
                f"A{glide_idx}=[{','.join(_f(v, 6) for v in a_k)}],"
                f"H{glide_idx}=[{','.join(_f(v, 8) for v in th_k)}];"
            )
            m = "1"
            if ramp:
                m = f"(T<{_f(st, 6)}?max(T-{st0:.6f},0)*{_f(1.0 / den, 8)}:1)"
            m += f"*(T<{_f(e, 6)}?1:T>={_f(e + 0.25, 6)}?0:1-(T-{_f(e, 6)})*4)"
            hi = min(e + 0.25, sec_hi)
            lo = max(st0, sec_lo) if ramp else sec_lo
            out.append(
                f"T>{_f(lo, 6)}&&T<{_f(hi, 6)}&&(s+=lp(T{glide_idx},A{glide_idx},T)"
                f"*({m})"
                f"*cos(lp(T{glide_idx},H{glide_idx},T)+{_f(k2, 8)}*(T-{s0:.6f})+{_f(_wrap(c0), 8)}));"
            )
            glide_idx += 1
            continue
        a = float(p["amplitude"])
        lam = float(p["envelope_decay"])
        t0 = float(p["envelope_t0"])
        f = float(p["frequency"])
        ph = _wrap(-float(p.get("phase", 0.0)))
        st, et = p.get("start_time"), p.get("end_time")
        env = (
            f"*exp(-{_f(lam, 6)}*max(T-{t0:.6f},0))" if lam > 0 else ""
        )
        core = f"{_f(a, 6)}{env}"
        if st is None or et is None:
            out.append(
                f"T>={_f(sec_lo, 6)}&&T<{_f(sec_hi, 6)}&&"
                f"(s+={core}*cos({_f(TWO_PI * f, 8)}*(T-{ts:.6f})+{_f(ph, 5)}));"
            )
            continue
        st0, den, ramp, s_, e = _mask_parts(float(st), float(et), lam, hw)
        m = "1"
        if ramp:
            m = f"(T<{_f(st, 6)}?max(T-{st0:.6f},0)*{_f(1.0 / den, 8)}:1)"
        m += f"*(T<{_f(e, 6)}?1:T>={_f(e + 0.25, 6)}?0:1-(T-{_f(e, 6)})*4)"
        hi = min(e + 0.25, sec_hi)
        lo = max(st0, sec_lo) if ramp else sec_lo
        out.append(
            f"T>{_f(lo, 6)}&&T<{_f(hi, 6)}&&(s+={core}*({m})"
            f"*cos({_f(TWO_PI * f, 8)}*(T-{ts:.6f})+{_f(ph, 5)}));"
        )
    out.append("var o=s>1?1:s<-1?-1:s;return o;")
    out.append("};")
    # Funcbeat convention: the pasted code is evaluated once and MUST return
    # the sample function fn(t_seconds, srate) -> [-1, 1] (the `return` is
    # emitted by the header above, before the trailing declarations).
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--ratio", type=float, default=0.0002)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    text = emit(ledger, args.ratio)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"// bytes: {len(text.encode('utf-8'))}")
    if args.out:
        print(f"// written: {args.out}")
    if args.check and args.out:
        from src.analysis import load_audio
        import soundfile as sf

        keep, _, hw = collect(ledger, args.ratio)
        sr = int(ledger["sample_rate"])
        dur = float(ledger.get("duration") or 15.0)
        sim = simulate_exported(keep, hw, sr, dur)
        y_ref, _ = load_audio(Path(args.out).with_name("res_best.wav"), target_sr=sr)
        n = min(len(y_ref), len(sim))
        corr = float(np.corrcoef(sim[:n], y_ref[:n])[0, 1])
        src_file = ledger.get("source")
        cs = 0.0
        if src_file and Path(src_file).exists():
            y_src, _ = load_audio(src_file, target_sr=sr)
            m = min(len(y_src), n)
            if m > 10:
                cs = float(np.corrcoef(sim[:m], y_src[:m])[0, 1])
        print(f"// check: sim-vs-engine corr={corr:.4f} | sim-vs-song corr={cs:.4f}")


if __name__ == "__main__":
    main()
