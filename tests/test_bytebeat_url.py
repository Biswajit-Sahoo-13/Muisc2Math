"""Tests for bytebeat share-link encoding/decoding."""
import unittest

from src.bytebeat_url import (
    BYTEBEAT_MODES,
    BytebeatURLError,
    bytebeat_share_url,
    bytebeat_url_fragment,
    decode_bytebeat_fragment,
    decode_bytebeat_url,
)

# Real reference link shared by the user, with the exact code shown in its
# screenshot (mode Bytebeat, 8000 Hz).
REFERENCE_FRAGMENT = "4AAAA+kUzNNDSKLGzM68pqQFSZpraJkC+GpBpaAwRAAA"
REFERENCE_CODE = "10*(t>>7|t|t>>6)+4*(t&t>>13|t>>6)"

VIDEO_EQUATION = "x=t/8000*2,(3.2*Math.exp(-.5*(x%4))*(1-(2*((x%1)-1))**2)-1.6)/8"


class TestDecodeReference(unittest.TestCase):
    def test_decodes_known_reference_fragment(self):
        decoded = decode_bytebeat_fragment(REFERENCE_FRAGMENT)
        self.assertEqual(decoded["code"], REFERENCE_CODE)
        self.assertEqual(decoded["mode"], "Bytebeat")
        self.assertAlmostEqual(decoded["sample_rate"], 8000.0, places=3)

    def test_decodes_reference_url(self):
        decoded = decode_bytebeat_url(f"https://dollchan.net/bytebeat/#{REFERENCE_FRAGMENT}")
        self.assertEqual(decoded["code"], REFERENCE_CODE)

    def test_reference_round_trips_through_encoder(self):
        frag = bytebeat_url_fragment(REFERENCE_CODE, mode="Bytebeat", sample_rate=8000)
        self.assertEqual(frag, REFERENCE_FRAGMENT)


class TestRoundTrip(unittest.TestCase):
    def test_round_trip_all_modes(self):
        for mode in BYTEBEAT_MODES:
            with self.subTest(mode=mode):
                frag = bytebeat_url_fragment(VIDEO_EQUATION, mode=mode, sample_rate=8000)
                decoded = decode_bytebeat_fragment(frag)
                self.assertEqual(decoded["mode"], mode)
                self.assertEqual(decoded["code"], VIDEO_EQUATION)
                self.assertAlmostEqual(decoded["sample_rate"], 8000.0, places=3)

    def test_round_trip_unicode_code(self):
        code = "// \u03c0 \u266a\n" + VIDEO_EQUATION
        frag = bytebeat_url_fragment(code, mode="Funcbeat")
        self.assertEqual(decode_bytebeat_fragment(frag)["code"], code)

    def test_round_trip_empty_code(self):
        frag = bytebeat_url_fragment("")
        self.assertEqual(decode_bytebeat_fragment(frag)["code"], "")

    def test_round_trip_multiline_code(self):
        code = "S=8000,\nB=60/140*S,\n(K=Math.exp(-(t%B)/140)*Math.sin(6.283*48*t/S))/3"
        frag = bytebeat_url_fragment(code)
        self.assertEqual(decode_bytebeat_fragment(frag)["code"], code)

    def test_fragment_has_no_padding_and_correct_prefix(self):
        frag = bytebeat_url_fragment(VIDEO_EQUATION)
        self.assertTrue(frag.startswith("4"))
        self.assertNotIn("=", frag)
        self.assertTrue(frag[1:].isascii())

    def test_share_url_shape(self):
        url = bytebeat_share_url(VIDEO_EQUATION, mode="Floatbeat", sample_rate=8000)
        self.assertTrue(url.startswith("https://dollchan.net/bytebeat/#4"))
        self.assertEqual(decode_bytebeat_url(url)["code"], VIDEO_EQUATION)

    def test_accepts_hash_prefixed_fragment(self):
        self.assertEqual(
            decode_bytebeat_fragment("#" + REFERENCE_FRAGMENT)["code"], REFERENCE_CODE
        )

    def test_custom_base_url(self):
        url = bytebeat_share_url("sin(1)", base_url="http://localhost:8080/bb/")
        self.assertTrue(url.startswith("http://localhost:8080/bb/#4"))
        self.assertEqual(decode_bytebeat_url(url)["code"], "sin(1)")


class TestLegacyAndErrors(unittest.TestCase):
    def test_legacy_v3b64_plain_code(self):
        import base64
        import zlib

        deflated = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        blob = deflated.compress(b"t*(t>>7|t)") + deflated.flush()
        frag = "v3b64" + base64.b64encode(blob).decode("ascii")
        self.assertEqual(decode_bytebeat_fragment(frag)["code"], "t*(t>>7|t)")

    def test_legacy_v3b64_json_with_formula_key(self):
        import base64
        import zlib

        blob_in = b'{"formula":"t%5E(t%7C9)"}'
        deflated = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        blob = deflated.compress(blob_in) + deflated.flush()
        frag = "v3b64" + base64.b64encode(blob).decode("ascii")
        self.assertEqual(decode_bytebeat_fragment(frag)["code"], "t%5E(t%7C9)")

    def test_unknown_mode_index_rejected(self):
        import base64
        import struct
        import zlib

        deflated = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        blob = deflated.compress(b"t") + deflated.flush()
        payload = bytes([99]) + struct.pack("<f", 8000.0) + blob
        frag = "4" + base64.b64encode(payload).decode("ascii").rstrip("=")
        with self.assertRaises(BytebeatURLError):
            decode_bytebeat_fragment(frag)

    def test_corrupt_payload_raises(self):
        import base64

        frag = "4" + base64.b64encode(b"\x00\x01").decode("ascii").rstrip("=")
        with self.assertRaises(BytebeatURLError):
            decode_bytebeat_fragment(frag)

    def test_empty_and_unrecognized_fragments(self):
        with self.assertRaises(BytebeatURLError):
            decode_bytebeat_fragment("")
        with self.assertRaises(BytebeatURLError):
            decode_bytebeat_fragment("#nonsense")


class TestDefaults(unittest.TestCase):
    def test_unknown_mode_falls_back_to_floatbeat(self):
        decoded = decode_bytebeat_fragment(bytebeat_url_fragment("t", mode="Nonsense"))
        self.assertEqual(decoded["mode"], "Floatbeat")

    def test_custom_sample_rate_survives(self):
        frag = bytebeat_url_fragment("t", sample_rate=44100)
        self.assertAlmostEqual(decode_bytebeat_fragment(frag)["sample_rate"], 44100.0, places=1)


if __name__ == "__main__":
    unittest.main()