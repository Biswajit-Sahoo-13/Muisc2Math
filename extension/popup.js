/* Music2Math popup: upload or capture audio -> local engine -> equations TXT.
 *
 * Everything talks to the local engine server (src/res_server.py,
 * `python main.py serve`). The popup must stay open while a job runs -
 * closing it cancels the fetch. Recording uses chrome.tabCapture, which
 * needs the button click as its user gesture and captures exactly what
 * the tab is playing.
 */
"use strict";

const DEFAULT_URL = "http://127.0.0.1:8765";
const TARGET_RATE = 22050;
const MAX_UPLOAD_S = 60;

const PRESETS = {
  fast: { iterations: 2, noise: 0, lane: 0, parity: "ls", polish: 0, mrstft: 0 },
  standard: { iterations: 6, noise: 1, lane: 1, parity: "rms", polish: 0, mrstft: 0 },
  full: { iterations: 8, noise: 1, lane: 1, parity: "rms", polish: 1, mrstft: 1 },
};

const $ = (id) => document.getElementById(id);
const state = {
  busy: false, recorder: null, countdown: null, abort: null, result: null,
  discard: false, cancelled: false, stopTimer: null,
};

/* ---------- settings ---------- */

async function loadSettings() {
  const got = await chrome.storage.local.get(["serverUrl", "preset", "duration", "stemsMode"]);
  $("server-url").value = got.serverUrl || DEFAULT_URL;
  $("preset").value = got.preset || "standard";
  $("duration").value = got.duration || 15;
  $("stems").value = got.stemsMode || "none";
}
async function saveSettings() {
  await chrome.storage.local.set({
    serverUrl: $("server-url").value.trim() || DEFAULT_URL,
    preset: $("preset").value,
    duration: clampDuration($("duration").value),
    stemsMode: $("stems").value,
  });
}
const serverUrl = () => ($("server-url").value.trim() || DEFAULT_URL).replace(/\/+$/, "");
const clampDuration = (v) => Math.max(2, Math.min(60, parseInt(v, 10) || 15));

/* ---------- server health ---------- */

async function checkHealth() {
  const dot = $("server-dot"), note = $("server-note");
  const url = serverUrl();
  if (!/^https?:\/\//i.test(url)) {
    dot.className = "dot bad";
    note.textContent = "URL must start with http:// or https://";
    return;
  }
  try {
    const r = await fetch(`${url}/api/health`, { signal: AbortSignal.timeout(4000) });
    const j = await r.json();
    if (j && j.ok) {
      dot.className = "dot ok";
      note.textContent = "engine online";
    } else throw new Error("bad reply");
  } catch {
    dot.className = "dot bad";
    note.textContent = "offline — run: python main.py serve";
  }
}

/* ---------- progress UI ---------- */

function setBusy(title) {
  state.busy = true;
  $("progress").classList.remove("hidden");
  $("results").classList.add("hidden");
  $("progress-title").textContent = title;
  const t0 = Date.now();
  const stemsOn = $("stems") && $("stems").value !== "none";
  const details = stemsOn
    ? [
        "keep this popup open",
        "Demucs separation is the slow part…",
        "four stems, one loop each…",
        "stems jobs can take ~25 minutes",
      ]
    : [
        "keep this popup open",
        "lifting coordinates…",
        "hypothesis tournament…",
        "referee is listening…",
        "long clips take a few minutes",
      ];
  clearInterval(state.countdown);
  state.countdown = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    $("progress-timer").textContent = `${s} s`;
    $("progress-detail").textContent = details[Math.min(
      details.length - 1, Math.floor(s / 20)
    )];
  }, 1000);
  $("cancel-btn").classList.remove("hidden");
}
function clearBusy() {
  state.busy = false;
  clearInterval(state.countdown);
  state.countdown = null;
  $("progress").classList.add("hidden");
  $("cancel-btn").classList.add("hidden");
}

/* ---------- engine round trip ---------- */

function buildQuery(name, presetKey, duration) {
  const p = PRESETS[presetKey] || PRESETS.standard;
  const q = new URLSearchParams({
    name, duration: String(duration), iterations: String(p.iterations),
    noise: String(p.noise), lane: String(p.lane), parity: p.parity,
    polish: String(p.polish), mrstft: String(p.mrstft),
    joint: presetKey === "full" ? "1" : "0",
    stems: $("stems").value || "none",
    audio: "1",
  });
  return q.toString();
}

async function runEngine(wavBuffer, name) {
  const duration = clampDuration($("duration").value);
  const url = `${serverUrl()}/api/recreate?${buildQuery(name, $("preset").value, duration)}`;
  state.abort = new AbortController();
  setBusy("Engine running…");
  try {
    // Intentionally no Content-Type header: a simple CORS request, and the
    // server sniffs the WAV magic anyway.
    const r = await fetch(url, { method: "POST", body: wavBuffer, signal: state.abort.signal });
    const j = await r.json();
    if (r.status === 409) throw new Error("engine busy — another job is running, retry in a moment");
    if (!j || !j.ok) throw new Error((j && j.error) || `HTTP ${r.status}`);
    showResults(j, name);
  } catch (err) {
    if (err.name === "AbortError") { clearBusy(); return; }
    clearBusy();
    alert("Music2Math: " + err.message);
    checkHealth();
    return;
  }
  clearBusy();
}

/* ---------- results ---------- */

function pillar(label, pct) {
  const v = Math.max(0, Math.min(100, pct || 0));
  const div = document.createElement("div");
  div.className = "pillar";
  div.innerHTML = `<span>${label}</span><div class="bar"><div class="fill" style="width:${v}%"></div></div><span>${v.toFixed(0)}%</span>`;
  return div;
}

function showResults(j, name) {
  state.result = { j, name };
  const f = j.fidelity || {};
  $("r-fidelity").textContent = f.fidelity_pct != null ? f.fidelity_pct.toFixed(1) : "–";
  $("r-corr").textContent = j.metrics && j.metrics.correlation != null ? j.metrics.correlation.toFixed(4) : "–";
  $("r-snr").textContent = j.metrics && j.metrics.snr_db != null ? j.metrics.snr_db.toFixed(1) : "–";
  $("r-eqs").textContent = j.equation_count != null ? j.equation_count : "–";
  const pillars = $("pillars");
  pillars.innerHTML = "";
  if (f.tone_pct != null) {
    pillars.append(
      pillar("tone", f.tone_pct),
      pillar("spectrum", f.spectrum_pct),
      pillar("bands", f.band_pct),
      pillar("loudness", f.loudness_pct)
    );
  }
  const kb = (j.equation_bytes / 1024).toFixed(1);
  const srcKb = Math.round(j.duration * TARGET_RATE * 2 / 1024);
  $("r-note").textContent =
    `${j.equation_count} equations · ${kb} KB vs ${srcKb} KB source (${(100 * j.equation_bytes / (j.duration * TARGET_RATE * 2)).toFixed(1)}%) · ${j.elapsed_s}s engine time`;
  if (j.best_wav_b64) {
    $("playback").src = "data:audio/wav;base64," + j.best_wav_b64;
  }
  $("results").classList.remove("hidden");
}

/* ---------- flows ---------- */

async function handleFile(file) {
  // Reset FIRST, on every path, so re-picking the same file always fires
  // `change` (including the early-return paths below).
  $("file-input").value = "";
  if (state.busy || !file) return;
  if (file.size === 0) {
    alert("That file is empty.");
    return;
  }
  state.cancelled = false;
  try {
    setBusy("Decoding " + file.name + "…");
    const mono = await decodeToMono(file, TARGET_RATE, MAX_UPLOAD_S);
    if (state.cancelled) { state.cancelled = false; return; }
    if (!mono || mono.length < TARGET_RATE) {
      throw new Error("no usable audio track (under 1 second)");
    }
    const wav = encodeWav(mono, TARGET_RATE);
    await runEngine(wav, file.name);
  } catch (err) {
    clearBusy();
    alert("Could not decode that file: " + err.message);
  }
}

function startRecording() {
  if (state.busy) return;
  if (!chrome.tabCapture) {
    alert("Tab capture is not available in this browser — use file upload.");
    return;
  }
  // Synchronous guard: without it, a double-click enters the async capture
  // callback twice and the second recorder orphans the first (its stream
  // stays live, its onstop chain is lost).
  state.busy = true;
  state.discard = false;
  state.cancelled = false;
  chrome.tabCapture.capture({ audio: true, video: false }, (stream) => {
    if (!stream) {
      const err = chrome.runtime.lastError
        ? chrome.runtime.lastError.message
        : "no stream (try a normal tab playing audio)";
      clearBusy();
      alert("Capture failed: " + err);
      return;
    }
    const mime = ["audio/webm;codecs=opus", "audio/webm"].find(
      (m) => window.MediaRecorder && MediaRecorder.isTypeSupported(m)
    );
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const seconds = clampDuration($("duration").value);
    rec.onstop = async () => {
      // This recording is over: its stop timer must never fire again
      // (a stale timer closing over state.recorder could stop a LATER
      // recording partway through).
      if (state.stopTimer) { clearTimeout(state.stopTimer); state.stopTimer = null; }
      stream.getTracks().forEach((t) => t.stop());
      try { chrome.tabCapture.release(); } catch { /* older Chrome */ }
      $("record-btn").textContent = "Record this tab";
      $("record-btn").onclick = startRecording;
      if (state.discard) {
        // The user pressed Cancel while recording: throw the capture away,
        // do NOT send a half song to the engine.
        state.discard = false;
        return;
      }
      const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
      await handleBlob(blob, "tab_capture.webm", seconds);
    };
    rec.start(500);
    state.recorder = rec;
    setBusy(`Recording tab for ${seconds}s — play the song!`);
    $("record-btn").textContent = "Stop early";
    $("record-btn").onclick = () => {
      if (state.recorder && state.recorder.state !== "inactive") {
        state.recorder.stop();
      }
    };
    // The timer closes over THIS recorder (not state.recorder), so it can
    // never stop a recording made after this one.
    state.stopTimer = setTimeout(() => {
      if (rec.state !== "inactive") rec.stop();
    }, seconds * 1000);
  });
}

async function handleBlob(blob, name, seconds) {
  if (!blob || blob.size === 0) {
    clearBusy(); // the busy spinner has been up since recording started
    alert("Nothing was recorded — keep the tab playing audio while recording.");
    return;
  }
  state.cancelled = false;
  try {
    const mono = await decodeToMono(blob, TARGET_RATE, seconds);
    if (state.cancelled) { state.cancelled = false; return; }
    if (!mono || mono.length < TARGET_RATE) {
      throw new Error("capture too short (under 1 second)");
    }
    const wav = encodeWav(mono, TARGET_RATE);
    await runEngine(wav, name);
  } catch (err) {
    clearBusy();
    alert("Could not decode the capture: " + err.message);
  }
}

/* ---------- wire up ---------- */

document.addEventListener("DOMContentLoaded", async () => {
  await loadSettings();
  checkHealth();
  $("server-url").addEventListener("change", async () => { await saveSettings(); checkHealth(); });
  $("preset").addEventListener("change", saveSettings);
  $("duration").addEventListener("change", saveSettings);
  $("stems").addEventListener("change", saveSettings);
  $("file-input").addEventListener("change", (e) => handleFile(e.target.files[0]));
  // onclick (not addEventListener) so the Stop-early reassignment in
  // startRecording is the ONLY live handler - no double-fire.
  $("record-btn").onclick = startRecording;
  $("cancel-btn").addEventListener("click", () => {
    if (state.stopTimer) { clearTimeout(state.stopTimer); state.stopTimer = null; }
    if (state.recorder && state.recorder.state !== "inactive") {
      // Cancel during recording discards the capture (see rec.onstop).
      state.discard = true;
      state.recorder.stop();
    }
    // Cancel during a decode window: nothing is abortable there, so latch
    // it - handleFile/handleBlob check the latch after the decode resolves.
    state.cancelled = true;
    state.abort && state.abort.abort();
    clearBusy();
    $("record-btn").textContent = "Record this tab";
    $("record-btn").onclick = startRecording;
  });
  $("download-btn").addEventListener("click", () => {
    if (!state.result) return;
    const { j, name } = state.result;
    const blob = new Blob([j.equations || ""], { type: "text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (name.replace(/\.[^.]+$/, "") || "equations") + "_equations.txt";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  });
});
