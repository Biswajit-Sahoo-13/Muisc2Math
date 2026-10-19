"""Ledger -> dollchan Floatbeat exporter.

Floatbeat in this fork compiles as `return 0,<code>;` (one comma-expression,
no statements, no function return) and is called per sample as fn(t) with
t = raw sample index at the composer rate. So everything is emitted as a
single expression: per-sample assignments T/S run down the comma chain, all
constants live once-initialized global literal arrays (gA*, gTx*, ...), and
glide interpolation is a nested ternary chain identical to np.interp
(clamped at both ends).

Semantics mirror tools.res_bytebeat exactly (same masks, same rounding).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from res_bytebeat import (  # noqa: E402
    TWO_PI,
    _f,
    _mask_parts,
    _wrap,
    collect,
    _glide_knots,
)

DIV_DEFAULT = 48000


def _lerp_chain(arr: str, kn: str, n: int) -> str:
    """Nested ternary == np.interp(T, kn_arr, arr), constant beyond both ends."""
    if n == 1:
        return "(" + arr + "[0])"
    parts = ["(T<=(" + kn + "[0])?" + arr + "[0]:("]
    for i in range(n - 1):
        j = i + 1
        body = (
            arr + "[" + str(i) + "]+(" + arr + "[" + str(j) + "]-" + arr + "[" + str(i) + "])"
            "*(T-" + kn + "[" + str(i) + "])/(" + kn + "[" + str(j) + "]-" + kn + "[" + str(i) + "])"
        )
        tail = ":(" if i < n - 2 else ":"
        parts.append("T<=(" + kn + "[" + str(j) + "])?" + body + tail)
    parts.append(arr + "[" + str(n - 1) + "]")
    parts.append(")" * n)
    return "".join(parts)


def _glide_term(idx: int, p: dict, ts: float, hw: float, sec_lo: float, sec_hi: float) -> str:
    tt, a_k, th_k, k2, s0, c0 = _glide_knots(p, ts)
    st, et = float(p["start_time"]), float(p["end_time"])
    lam_g = float(p.get("envelope_decay", 0.0))
    st0, den, ramp, _, e = _mask_parts(st, et, lam_g, hw)
    if ramp:
        mask = f"(T<{_f(st, 6)}?max(T-{st0:.6f},0)*{_f(1.0 / den, 8)}:1)"
    else:
        mask = "1"
    mask += f"*(T<{_f(e, 6)}?1:T>={_f(e + 0.25, 6)}?0:1-(T-{_f(e, 6)})*4)"
    hi = min(e + 0.25, sec_hi)
    lo = max(st0, sec_lo) if ramp else sec_lo
    na, nh = len(a_k), len(th_k)
    return (
        "gA" + str(idx) + "=window.gA" + str(idx) + "||[" + ",".join(_f(v, 6) for v in a_k) + "],"
        "gTx" + str(idx) + "=window.gTx" + str(idx) + "||[" + ",".join(_f(v, 6) for v in tt) + "],"
        "gH" + str(idx) + "=window.gH" + str(idx) + "||[" + ",".join(_f(v, 8) for v in th_k) + "],"
        "gHt" + str(idx) + "=window.gHt" + str(idx) + "||[" + ",".join(_f(v, 6) for v in tt) + "],"
        "_gA" + str(idx) + "=" + _lerp_chain("gA" + str(idx), "gTx" + str(idx), na) + ","
        "_gH" + str(idx) + "=" + _lerp_chain("gH" + str(idx), "gHt" + str(idx), nh) + ","
        "T>" + _f(lo, 6) + "&&T<" + _f(hi, 6) + "&&"
        "(s+=_gA" + str(idx) + "*(" + mask + ")"
        "*cos(_gH" + str(idx) + "+" + _f(k2, 8) + "*(T-" + f"{s0:.6f}" + ")+" + _f(_wrap(c0), 5) + "))"
    )


def emit(ledger: dict, ratio: float, div: int = DIV_DEFAULT) -> str:
    keep, _, hw = collect(ledger, ratio)
    dur = float(ledger.get("duration") or 15.0)
    best = ledger["best"]
    out = [
        f"// RES Floatbeat: '{ledger.get('source', 'song')}'",
        f"// engine corr={best['correlation']:.4f} | {len(keep)} subtone"
        f" equations | {len(ledger.get('sections', []))} sections | {dur:.0f} s",
        f"// dollchan: type FLOATBEAT, sample rate {div} (divider is baked in: keep them equal)",
    ]
    chunks = [f"T=t/{div},s=0"]
    glide_idx = 0
    for si, ts, p, _, sec_lo, sec_hi in keep:
        if p.get("model") == "glide":
            chunks.append(_glide_term(glide_idx, p, ts, hw, float(sec_lo), float(sec_hi)))
            glide_idx += 1
            continue
        a = float(p["amplitude"])
        lam = float(p["envelope_decay"])
        t0 = float(p["envelope_t0"])
        f = float(p["frequency"])
        ph = _wrap(-float(p.get("phase", 0.0)))
        st, et = p.get("start_time"), p.get("end_time")
        env = f"*exp(-{_f(lam, 6)}*max(T-{t0:.6f},0))" if lam > 0 else ""
        core = f"{_f(a, 6)}{env}"
        if st is None or et is None:
            chunks.append(
                f"T>=({_f(sec_lo, 6)})&&T<({_f(sec_hi, 6)})&&"
                f"(s+={core}*cos({_f(TWO_PI * f, 8)}*(T-{ts:.6f})+{_f(ph, 5)}))"
            )
            continue
        st0, den, ramp, _, e = _mask_parts(float(st), float(et), lam, hw)
        m = "1"
        if ramp:
            m = f"(T<{_f(st, 6)}?max(T-{st0:.6f},0)*{_f(1.0 / den, 8)}:1)"
        m += f"*(T<{_f(e, 6)}?1:T>={_f(e + 0.25, 6)}?0:1-(T-{_f(e, 6)})*4)"
        hi = min(e + 0.25, sec_hi)
        lo = max(st0, sec_lo) if ramp else sec_lo
        chunks.append(
            f"T>{_f(lo, 6)}&&T<({_f(hi, 6)})&&"
            f"(s+={core}*({m})"
            f"*cos({_f(TWO_PI * f, 8)}*(T-{ts:.6f})+{_f(ph, 5)}))"
        )
    chunks.append("s")
    out.append(",".join(chunks))
    # Floatbeat: pasted text must be ONE expression (composer wraps it as
    # `return 0,<code>;`). No trailing semicolon, no `return`, no var.
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--ratio", type=float, default=0.0002)
    ap.add_argument("--sample-rate", type=int, default=DIV_DEFAULT)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    text = emit(ledger, args.ratio, args.sample_rate)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(f"// bytes: {len(text.encode('utf-8'))}")
    if args.out:
        print(f"// written: {args.out}")
    if args.check and args.out:
        _check(ledger, args, text)


def _check(ledger: dict, args, text: str):
    import subprocess
    import tempfile

    import numpy as np
    from res_bytebeat import simulate_exported

    sr = 22050
    dur = float(ledger.get("duration") or 15.0)
    keep, _, hw = collect(ledger, args.ratio)
    sim = simulate_exported(keep, hw, sr, dur).astype(np.float32)
    tmp = Path(tempfile.mkdtemp(prefix="fbchk_", dir=r"C:\Users\KIIT\AppData\Local\Temp\opencode"))
    sim_path = tmp / "sim.f32"
    code_path = tmp / "code.js"  # checker runs on the 22050 grid for exact parity
    checked = text.replace(f"T=t/{args.sample_rate}", f"T=t/{sr}", 1)
    sim_path.write_bytes(sim.tobytes())
    code_path.write_text(checked, encoding="utf-8")
    harness = tmp / "harness.js"
    harness.write_text(
        "const fs=require('fs');"
        f"const code=fs.readFileSync({json.dumps(str(code_path))},'utf8');"
        "const ps=Object.getOwnPropertyNames(Math);const vs=ps.map(k=>Math[k]);ps.push('int','window');vs.push(Math.floor,globalThis);"
        "const fn=new Function(...ps,'t',`return 0,\\n${code};`).bind(globalThis,...vs);"
        "fn(0);"
        f"const sim=fs.readFileSync({json.dumps(str(sim_path))});"
        "const n=sim.length/4;"
        "const out=Buffer.alloc(n*4);"
        "for(let i=0;i<n;i++){"
        "const v=fn(i);"
        "out.writeFloatLE(Math.max(Math.min(v,1),-1),i*4);"
        "}"
        "let sa=0,sb=0,sc=0;"
        "for(let i=0;i<n;i++){"
        "const a=sim.readFloatLE(i*4),b=out.readFloatLE(i*4);sa+=a*a;sb+=b*b;sc+=a*b;}"
        "const ab=(Math.sqrt(sa)*Math.sqrt(sb))||1;"
        "console.log('engine-vs-sim corr='+(sc/ab).toFixed(6));"
        "console.log('engine rms='+Math.sqrt(sa/n).toFixed(4)+' sim rms='+Math.sqrt(sb/n).toFixed(4));",
        encoding="utf-8",
    )
    res = subprocess.run(["node", str(harness)], capture_output=True, text=True, timeout=600)
    print(res.stdout.strip())
    if res.returncode != 0:
        print(res.stderr.strip()[:2000])


if __name__ == "__main__":
    main()
