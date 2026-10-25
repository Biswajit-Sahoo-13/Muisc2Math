/* Browser-side audio conditioning for the M2M engine.
 *
 * The engine wants mono 16-bit PCM WAV. Rather than trusting the server to
 * decode webm/opus (decoder availability varies by build), the extension
 * decodes whatever it captured or uploaded, resamples to 22.05 kHz mono
 * with OfflineAudioContext, and encodes a plain WAV - the exact format
 * data/output/_fixtures uses.
 */

// Encode a mono Float32Array at `sampleRate` into a 16-bit PCM WAV
// ArrayBuffer. Pure function - node-testable (extension/selftest.cjs).
function encodeWav(samples, sampleRate) {
  const n = samples.length;
  const buffer = new ArrayBuffer(44 + n * 2);
  const view = new DataView(buffer);
  const ascii = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + n * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);   // PCM chunk size
  view.setUint16(20, 1, true);    // PCM format
  view.setUint16(22, 1, true);    // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);    // block align
  view.setUint16(34, 16, true);   // bits per sample
  ascii(36, "data");
  view.setUint32(40, n * 2, true);
  let off = 44;
  for (let i = 0; i < n; i++, off += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buffer;
}

// Decode arbitrary audio/video bytes into a mono `targetRate` Float32Array.
// A 1-channel OfflineAudioContext downmixes stereo (L+R)/2 while resampling,
// in one pass. Crops to `maxSeconds` before resampling to bound the work.
async function decodeToMono(blob, targetRate, maxSeconds) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const decoded = await ctx.decodeAudioData(await blob.arrayBuffer());
  await ctx.close();
  const cap = Math.min(
    decoded.length,
    Math.max(1, Math.floor(maxSeconds * decoded.sampleRate))
  );
  const frames = Math.max(
    1,
    Math.ceil((cap / decoded.sampleRate) * targetRate)
  );
  const off = new OfflineAudioContext(1, frames, targetRate);
  const src = off.createBufferSource();
  // Feed only the capped span: build a cropped view of the decoded buffer.
  if (cap < decoded.length) {
    const crop = off.createBuffer(
      decoded.numberOfChannels, cap, decoded.sampleRate
    );
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      crop.copyToChannel(
        decoded.getChannelData(ch).subarray(0, cap), ch
      );
    }
    src.buffer = crop;
  } else {
    src.buffer = decoded;
  }
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0);
}

if (typeof module !== "undefined") {
  module.exports = { encodeWav, decodeToMono };
}
