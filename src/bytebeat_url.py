"""Bytebeat share-link encoding (dollchan / bytebeat-composer URL fragments).

Wire format (verified against the upstream composer and a known reference link):

    fragment = "4" + base64(payload)          with all "=" padding stripped
    payload  = mode_index (1 byte)
             | sample_rate   (float32, little endian, 4 bytes)
             | deflate_raw   (raw DEFLATE of the UTF-8 code)

``mode_index`` selects the synthesis mode. The fragment travels in the URL
hash, so the code itself is never sent to a server.

All of this is pure Python ``zlib``/``base64``; no Node.js and no browser are
required to produce a link.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, Dict

BYTEBEAT_MODES = ("Bytebeat", "Signed Bytebeat", "Floatbeat", "Funcbeat")

DEFAULT_MODE = "Floatbeat"
DEFAULT_SAMPLE_RATE = 8000
DEFAULT_BASE_URL = "https://dollchan.net/bytebeat/"

_PREFIX = "4"
_OLD_PREFIX = "v3b64"


class BytebeatURLError(ValueError):
    """Raised when a bytebeat URL fragment cannot be decoded."""


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _raw_inflate(data: bytes) -> bytes:
    return zlib.decompress(data, -zlib.MAX_WBITS)


def bytebeat_url_fragment(
    code: str,
    mode: str = DEFAULT_MODE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> str:
    """Encode a bytebeat function into a shareable URL fragment.

    Parameters
    ----------
    code : str
        The bytebeat function source.
    mode : str
        One of ``BYTEBEAT_MODES``. Unknown values fall back to Floatbeat.
    sample_rate : int
        Sample rate stored in the fragment (8000 for classic bytebeat).
    """
    if mode not in BYTEBEAT_MODES:
        mode = DEFAULT_MODE
    payload = (
        bytes([BYTEBEAT_MODES.index(mode)])
        + struct.pack("<f", float(sample_rate))
        + _raw_deflate(str(code).encode("utf-8"))
    )
    return _PREFIX + base64.b64encode(payload).decode("ascii").rstrip("=")


def bytebeat_share_url(
    code: str,
    mode: str = DEFAULT_MODE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """Build the full shareable URL for a bytebeat function."""
    return f"{base_url}#{bytebeat_url_fragment(code, mode, sample_rate)}"


def decode_bytebeat_fragment(fragment: str) -> Dict[str, Any]:
    """Decode a URL fragment (with or without the leading ``#``).

    Returns a dict with ``mode``, ``sample_rate`` and ``code``. Raises
    ``BytebeatURLError`` for unrecognized or corrupt input.
    """
    text = str(fragment).lstrip("#").strip()
    if not text:
        raise BytebeatURLError("empty fragment")

    if text.startswith(_PREFIX):
        body = text[len(_PREFIX):]
        payload = base64.b64decode(body + "=" * (-len(body) % 4))
        if len(payload) < 5:
            raise BytebeatURLError("payload too short")
        mode_index = payload[0]
        if mode_index >= len(BYTEBEAT_MODES):
            raise BytebeatURLError(f"unknown mode index {mode_index}")
        sample_rate = struct.unpack("<f", payload[1:5])[0]
        code = _raw_inflate(payload[5:]).decode("utf-8")
        return {
            "mode": BYTEBEAT_MODES[mode_index],
            "sample_rate": sample_rate,
            "code": code,
        }

    if text.startswith(_OLD_PREFIX):
        body = text[len(_OLD_PREFIX):]
        payload = base64.b64decode(body + "=" * (-len(body) % 4))
        decoded = _raw_inflate(payload).decode("utf-8")
        if decoded.startswith("{"):
            import json

            data = json.loads(decoded)
            code = data.get("formula", data.get("code", ""))
            return {"mode": None, "sample_rate": None, "code": code}
        return {"mode": None, "sample_rate": None, "code": decoded}

    raise BytebeatURLError("unrecognized url data")


def decode_bytebeat_url(url: str) -> Dict[str, Any]:
    """Decode the fragment of a full bytebeat URL."""
    return decode_bytebeat_fragment(str(url).split("#", 1)[-1])


__all__ = [
    "BYTEBEAT_MODES",
    "BytebeatURLError",
    "bytebeat_share_url",
    "bytebeat_url_fragment",
    "decode_bytebeat_fragment",
    "decode_bytebeat_url",
]