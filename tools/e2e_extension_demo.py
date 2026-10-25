"""End-to-end demo run: MONA LISA through the exact browser-extension path.

Builds the same upload the extension produces (mono 22.05 kHz PCM16 WAV -
what extension/wav.js decodeToMono + encodeWav emit), starts the real API
server on a loopback port, POSTs with the Standard preset query, and
prints the result summary. Run from the repo root:

    .venv/Scripts/python.exe tools/e2e_extension_demo.py [seconds] [preset]

The request target is the literal loopback host this script just started;
nothing else is ever contacted.
"""

from __future__ import annotations

import http.client
import io
import json
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from http.server import ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HOST = "127.0.0.1"
SR = 22050
PRESETS = {
    "fast": "iterations=2&noise=0&lane=0&parity=ls",
    "standard": "iterations=6&noise=1&lane=1&parity=rms",
    "full": "iterations=8&noise=1&lane=1&parity=rms&polish=1&mrstft=1&joint=1",
    # The 70.9%-recipe adaptation for 15 s: Demucs stems + everything on.
    "full-stems": ("iterations=6&noise=1&lane=1&parity=rms&polish=1"
                   "&mrstft=1&joint=1&stems=auto"),
}


def extension_style_wav(src_path: Path, seconds: float) -> bytes:
    """Mono 22.05 kHz PCM16 crop - byte-identical in form to the extension's upload."""
    import librosa

    y, _ = librosa.load(str(src_path), sr=SR, mono=True)
    y = y[: int(seconds * SR)]
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def main() -> None:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    preset = sys.argv[2] if len(sys.argv) > 2 else "standard"
    src = Path("01 - j-hope - MONA LISA.flac")
    if not src.exists():
        src = Path("data/uploads/20261024_204840_01_-_j-hope_-_MONA_LISA.flac")
    assert src.exists(), "MONA LISA source not found"

    body = extension_style_wav(src, seconds)
    print(f"[e2e] built extension-style upload: {len(body)} bytes "
          f"({seconds:.0f}s mono 22050 PCM16)")

    from src.res_server import _Handler

    httpd = ThreadingHTTPServer((HOST, 0), _Handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        conn = http.client.HTTPConnection(HOST, port, timeout=3600)
        conn.request("GET", "/api/health")
        r = conn.getresponse()
        print(f"[e2e] health: {r.status} {r.read().decode('utf-8')}")

        q = (f"/api/recreate?name=mona_lisa_{seconds:.0f}s.wav"
             f"&duration={seconds:.0f}&{PRESETS[preset]}&audio=1")
        print(f"[e2e] POST {q} (preset={preset}) - engine running, "
              f"this takes minutes...")
        t0 = time.time()
        conn.request("POST", q, body=body)
        r = conn.getresponse()
        payload = json.loads(r.read().decode("utf-8"))
        wall = time.time() - t0
    finally:
        httpd.shutdown()
        httpd.server_close()

    print(f"[e2e] HTTP {r.status}, wall {wall:.0f}s, engine {payload.get('elapsed_s')}s")
    if not payload.get("ok"):
        print(json.dumps(payload, indent=2)[:2000])
        return
    m = payload["metrics"]
    f = payload.get("fidelity") or {}
    print("=== MONA LISA via the browser-extension path ===")
    print(f"  correlation : {m.get('correlation')}")
    print(f"  snr_db      : {m.get('snr_db')}")
    print(f"  lsd_db      : {m.get('log_spectral_distance_db')}")
    print(f"  onset_corr  : {m.get('onset_corr')}")
    print(f"  mrstft_db   : {m.get('mrstft_db')}")
    print(f"  fidelity    : {f.get('fidelity_pct')}%  "
          f"(tone {f.get('tone_pct')} / spectrum {f.get('spectrum_pct')} / "
          f"bands {f.get('band_pct')} / loudness {f.get('loudness_pct')})")
    print(f"  equations   : {payload['equation_count']} "
          f"({payload['equation_bytes']} bytes = "
          f"{100 * payload['equation_bytes'] / (payload['duration'] * SR * 2):.2f}% of source)")
    print(f"  classes     : {payload['classes']}")
    print(f"  accepted    : {payload['accepted_hypotheses']} hypotheses")
    print(f"  trail       : " + " -> ".join(
        f"{t['corr']:.4f}" for t in payload["trail"]))
    print(f"  txt preview : {payload['equations'][:300]!r}")
    out = Path("out") / "_e2e_extension_result.json"
    out.parent.mkdir(exist_ok=True)
    slim = {k: v for k, v in payload.items() if k != "best_wav_b64"}
    slim["best_wav_bytes"] = payload.get("best_wav_bytes")
    out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    print(f"[e2e] full result (minus audio) written to {out}")


if __name__ == "__main__":
    main()
