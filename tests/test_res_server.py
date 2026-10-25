"""The local API server the browser extension talks to.

One real engine run goes through the actual HTTP path (socket, handler,
JSON) with the same kind of payload the extension POSTs - a mono 22.05 kHz
PCM16 WAV the browser encoder produces (extension/wav.js, whose output is
cross-checked bit-exactly by extension/selftest.cjs). The remaining tests
pin the HTTP contract: health, bad body, wrong path, one-job-at-a-time.

The request helpers use http.client against the LITERAL host 127.0.0.1
(never a dynamic URL): the only server these tests will ever talk to is
the throwaway instance they start themselves on a random loopback port.
"""

from __future__ import annotations

import http.client
import io
import json
import threading
import wave

import numpy as np

from http.server import ThreadingHTTPServer

from src.res_server import JOB_LOCK, _Handler, recreate

SR = 22050
# Fixed by construction: the test server's loopback host. No other host,
# scheme or redirect is ever requested here.
HOST = "127.0.0.1"


def _wav_bytes(seconds: float = 2.0) -> bytes:
    """A decaying chord as mono PCM16 WAV - the extension's upload format."""
    t = np.arange(int(seconds * SR)) / SR
    y = (
        0.6 * np.exp(-1.1 * t) * np.cos(2 * np.pi * 440.0 * t)
        + 0.35 * np.exp(-1.5 * t) * np.cos(2 * np.pi * 554.365 * t)
    )
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class _Server:
    def __init__(self):
        self.httpd = ThreadingHTTPServer((HOST, 0), _Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def _request(port: int, method: str, path: str, body: bytes | None = None,
             timeout: int = 240):
    conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
    try:
        conn.request(method, path, body=body)
        r = conn.getresponse()
        return r.status, json.loads(r.read().decode("utf-8"))
    finally:
        conn.close()


def test_health():
    with _Server() as s:
        code, j = _request(s.port, "GET", "/api/health")
        assert code == 200 and j["ok"] is True and j["engine"] == "m2m-res"


def test_http_contract_errors():
    with _Server() as s:
        # Wrong path on both verbs.
        assert _request(s.port, "GET", "/nope")[0] == 404
        assert _request(s.port, "POST", "/nope", b"x")[0] == 404
        # Not a WAV body.
        code, j = _request(s.port, "POST", "/api/recreate", b"this is not a wav")
        assert code == 400 and j["ok"] is False
        # Empty body.
        code, j = _request(s.port, "POST", "/api/recreate", b"")
        assert code == 400


def test_busy_lock_returns_409():
    with _Server() as s:
        acquired = JOB_LOCK.acquire(blocking=False)
        assert acquired
        try:
            code, j = _request(s.port, "POST", "/api/recreate", _wav_bytes(2.0))
            assert code == 409 and j["ok"] is False
        finally:
            JOB_LOCK.release()


def test_recreate_end_to_end_over_http():
    with _Server() as s:
        q = ("/api/recreate?name=test_chord.wav&duration=2&iterations=1"
             "&noise=0&lane=0&parity=rms&audio=1")
        code, j = _request(s.port, "POST", q, _wav_bytes(2.0))
        assert code == 200, j
        assert j["ok"] is True
        assert j["duration"] == 2.0
        assert j["equations"], "equation sheet text must come back"
        assert j["equation_count"] >= 1
        assert "correlation" in j["metrics"]
        assert j["fidelity"] is not None and "fidelity_pct" in j["fidelity"]
        assert j["best_wav_b64"], "the recreation preview WAV must come back"
        assert j["elapsed_s"] > 0
        assert isinstance(j["trail"], list) and len(j["trail"]) == 1


def test_recreate_query_clamping():
    # Direct (non-HTTP) call pins the option clamping: junk parity falls
    # back to rms, absurd iterations clamp, the name gains .wav.
    j = recreate(
        _wav_bytes(2.0),
        {"duration": ["999"], "iterations": ["99"], "parity": ["junk"],
         "name": ["clip"], "audio": ["0"], "noise": ["0"], "lane": ["0"]},
    )
    assert j["ok"] is True
    # duration=999 clamps to the 60 s cap, but the reported duration is
    # what was actually analysed - the 2 s clip.
    assert j["duration"] == 2.0
    assert j["name"] == "clip.wav"
    assert "best_wav_b64" not in j  # audio=0 keeps the payload lean
    assert len(j["trail"]) == 12    # iterations clamped from 99 to 12


def test_serve_cli_parses(capsys):
    import sys

    import pytest

    import main as m2m_main

    old = sys.argv
    sys.argv = ["m2m", "serve", "--help"]
    try:
        with pytest.raises(SystemExit) as exc:
            m2m_main.main()
        assert exc.value.code == 0
        assert "--port" in capsys.readouterr().out
    finally:
        sys.argv = old
