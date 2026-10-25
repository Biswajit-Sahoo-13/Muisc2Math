/* Self-test for extension/wav.js encodeWav (node; the decode path needs
 * browser APIs and is exercised by using the extension).
 *
 * Writes a known sine to out/_wav_selftest.wav; the Python check below
 * (run by CI-adjacent verification, see tests/test_res_server.py for the
 * engine side) reads it back with the stdlib wave module and asserts the
 * samples round-trip bit-exactly.
 *
 *   node extension/selftest.cjs
 */
"use strict";

const fs = require("fs");
const path = require("path");
const { encodeWav } = require("./wav.js");

const SR = 22050;
const N = 2205; // 100 ms
const samples = new Float32Array(N);
for (let i = 0; i < N; i++) {
  samples[i] = 0.5 * Math.sin((2 * Math.PI * 440 * i) / SR);
}

const buf = Buffer.from(encodeWav(samples, SR));
const out = path.join(__dirname, "..", "out");
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(path.join(out, "_wav_selftest.wav"), buf);

// Header sanity (RIFF/WAVE, PCM, mono, 22.05k, 16-bit).
const ok =
  buf.toString("ascii", 0, 4) === "RIFF" &&
  buf.toString("ascii", 8, 12) === "WAVE" &&
  buf.readUInt32LE(4) === 36 + N * 2 &&
  buf.readUInt16LE(22) === 1 &&
  buf.readUInt32LE(24) === SR &&
  buf.readUInt16LE(34) === 16;
if (!ok) {
  console.error("FAIL: WAV header wrong");
  process.exit(1);
}
console.log(`OK: wrote ${N} samples (${buf.length} bytes) to out/_wav_selftest.wav`);
