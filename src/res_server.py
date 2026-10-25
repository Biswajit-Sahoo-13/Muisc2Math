"""Local HTTP front door for the RES engine — what the browser extension
(``extension/``) talks to.

The client POSTs a mono 22.05 kHz WAV (the extension encodes it in the
browser: decode → OfflineAudioContext resample → PCM16), the engine
recreates the audio as editable equations, and the JSON response carries
the metrics, the four-pillar fidelity and the full equation-sheet text so
the client can offer the ``.txt`` download without any file server.

Deliberately stdlib-only (no FastAPI/Flask dependency): the surface is two
endpoints.

- ``GET /api/health`` - liveness probe the popup pings on open.
- ``POST /api/recreate`` - raw WAV bytes as the request body; options are
  query parameters (``name``, ``duration``, ``iterations``, ``noise``,
  ``lane``, ``parity``, ``polish``, ``mrstft``, ``joint``, ``stems``,
  ``audio``). Response: JSON with metrics, fidelity, trail, equations.

Security posture: binds 127.0.0.1 by default (``--host`` to expose, at
your own risk), runs ONE job at a time (the engine is CPU-heavy; a second
request gets 409), caps the body size, and answers CORS for any origin —
safe because there is no auth, no secrets, no state beyond analysing the
audio you send, and the server makes no outbound requests of any kind.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from typing import Any, Dict

# 40 MB of WAV is ~11 minutes at 22.05 kHz mono 16-bit - far past anything
# the engine should analyse in one request.
MAX_BODY_BYTES = 40 * 1024 * 1024

# The out-of-the-box preset mirrors the README's full-fidelity example:
# noise layer + spectral lane + loudness parity. The experimental flags
# (polish, mrstft lane, joint refine) stay opt-in.
_DEFAULTS = {
    "duration": 15.0,
    "iterations": 6,
    "noise": 1,
    "lane": 1,
    "parity": "rms",
    "polish": 0,
    "mrstft": 0,
    "joint": 0,
    "audio": 1,
}

JOB_LOCK = threading.Lock()


def _clamp(name: str, value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def recreate(body: bytes, query: Dict[str, list]) -> Dict[str, Any]:
    """Run the RES loop on one uploaded WAV; return the extension payload.

    Split from the HTTP layer so tests can call it directly.
    """
    from src.res_ui import run_res_payload

    q = {k: v[-1] for k, v in query.items() if v}

    def _flag(key: str) -> bool:
        return str(q.get(key, _DEFAULTS[key])) in ("1", "true", "yes", "on")

    duration = _clamp("duration", float(q.get("duration", _DEFAULTS["duration"])), 2.0, 60.0)
    iterations = int(_clamp("iterations", float(q.get("iterations", _DEFAULTS["iterations"])), 1.0, 12.0))
    parity = str(q.get("parity", _DEFAULTS["parity"]))
    if parity not in ("ls", "rms"):
        parity = "rms"
    stems = str(q.get("stems", "none"))
    if stems not in ("none", "auto", "demucs"):
        stems = "none"
    name = str(q.get("name", "capture.wav"))[:80] or "capture.wav"
    if not name.lower().endswith(".wav"):
        name += ".wav"

    t0 = time.time()
    # An empty progress dict lets run_res_payload keep its iteration history
    # in stems mode too (the multitrack ledger has no "iterations" list, and
    # the trail otherwise comes back empty).
    payload = run_res_payload(
        body,
        name,
        duration=duration,
        iterations=iterations,
        bank=True,
        stem_mode=stems,
        enable_noise=_flag("noise"),
        spectral_lane=_flag("lane"),
        energy_parity=True,
        parity_mode=parity,
        polish=_flag("polish"),
        mrstft_lane=_flag("mrstft"),
        joint_refine=_flag("joint"),
        progress={},
    )
    equations = payload.get("equations") or ""
    sections = payload.get("sections") or []
    out: Dict[str, Any] = {
        "ok": True,
        "name": name,
        "duration": payload.get("duration", duration),
        "start_corr": payload.get("start_corr"),
        "metrics": payload.get("best") or {},
        "fidelity": payload.get("fidelity"),
        "trail": payload.get("trail") or [],
        "sections": sections,
        "equations": equations,
        "equation_count": sum(int(s.get("partials", 0)) for s in sections),
        "equation_bytes": len(equations.encode("utf-8")),
        "classes": payload.get("equation_classes") or [],
        "accepted_hypotheses": len(payload.get("accepted") or []),
        "separation_note": payload.get("separation_note"),
        "outputs": {
            "ledger": payload.get("ledger_path"),
            "best_wav": payload.get("best_path"),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    if _flag("audio"):
        best = payload.get("best_bytes") or b""
        if best:
            out["best_wav_b64"] = base64.b64encode(best).decode("ascii")
            out["best_wav_bytes"] = len(best)
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "M2M-RES/1.0"

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, obj: Dict[str, Any]) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server naming
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/api/health":
            self._json(200, {"ok": True, "engine": "m2m-res"})
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/recreate":
            self._json(404, {"ok": False, "error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(400, {"ok": False, "error": "empty body - POST the WAV bytes"})
            return
        if length > MAX_BODY_BYTES:
            self._json(413, {"ok": False, "error": "body too large (max 40 MB)"})
            return
        body = self.rfile.read(length)
        if len(body) < 44 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
            self._json(400, {"ok": False, "error": "body is not a WAV file"})
            return
        if not JOB_LOCK.acquire(blocking=False):
            self.send_response(409)
            self._cors()
            self.send_header("Retry-After", "30")
            data = json.dumps({"ok": False, "error": "engine busy - one job at a time"}).encode("utf-8")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        try:
            query = parse_qs(parsed.query)
            result = recreate(body, query)
            self._json(200, result)
        except ValueError as exc:
            self._json(400, {"ok": False, "error": f"bad request: {exc}"})
        except Exception as exc:  # the engine must never crash the server
            self._json(500, {"ok": False, "error": f"engine error: {exc}"})
        finally:
            JOB_LOCK.release()

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[res-server] {self.address_string()} {fmt % args}", flush=True)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the API server until Ctrl-C. One job at a time, localhost by default."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    print(f"[res-server] M2M RES engine API on http://{host}:{port}")
    print("[res-server]   health   : curl http://%s:%d/api/health" % (host, port))
    print("[res-server]   recreate : curl -X POST --data-binary @clip.wav "
          "'http://%s:%d/api/recreate?name=clip.wav&iterations=6'" % (host, port))
    print("[res-server]   (one job at a time; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[res-server] stopped")
    finally:
        httpd.server_close()
