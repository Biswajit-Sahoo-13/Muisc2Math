# The browser-extension roadmap: song → equations, from any tab

**The idea (user, 2026-10-25):** a Chrome/browser extension that captures
or accepts the sound of any video/audio, recreates it as the engine's math
equations, and hands back a downloadable equation sheet (TXT) — Music2Math
as a one-click tool on the open web.

**What exists after Phase 0 (this session):**

- `src/res_server.py` + `python main.py serve` — a stdlib-only local API
  (`GET /api/health`, `POST /api/recreate`) wrapping
  `res_ui.run_res_payload`, bound to 127.0.0.1, one job at a time, 40 MB
  body cap, CORS open (no auth, no secrets, no outbound requests).
- `extension/` — a Manifest V3 popup: upload any audio/video file, or
  record the current tab (tabCapture + MediaRecorder), decode + resample
  to mono 22.05 kHz PCM16 **in the browser** (`wav.js`), POST to the
  engine, show fidelity/correlation/SNR + the four pillars, play the
  recreation, download `*_equations.txt`.
- Verification: `tests/test_res_server.py` (6 tests incl. one full
  engine run over real HTTP), `extension/selftest.cjs` (the JS WAV
  encoder cross-checked **bit-exactly** against Python's `wave` module),
  `tools/e2e_extension_demo.py` (runs a real song through the exact
  extension path end to end).

## Architecture decision (and why)

The engine is NumPy/scipy/librosa. Porting it to run *inside* a browser
(WASM) is a multi-month rewrite and would freeze it away from the Python
mainline. So v1 is a **local loop**: the extension is a thin client, the
engine stays Python on the user's machine, and the contract between them
is one POST of a browser-encoded WAV. This keeps every engine increment
(presets now include the polish refiner and the MR-STFT referee lane)
instantly available to the extension with zero extension changes — the
presets are just query strings.

The division of labor is deliberate: the **browser** does what browsers
are world-class at (decode any codec the page is playing, resample with
OfflineAudioContext, play the result), and the **engine** does what it is
world-class at (equations). The browser-side WAV format was chosen to
match the frozen fixtures exactly (mono 22.05 kHz PCM16) so the extension
upload and the scoreboard evidence are the same kind of bytes.

## Phase 1 — make the loop feel alive (next)

- **Progress streaming.** `run_res_payload` already maintains a progress
  dict (stage, iteration, corr history). Expose it: `/api/job` with a job
  id + polling (simplest) or SSE `/api/progress` (one step up). The popup
  shows the live convergence curve the way the Streamlit app does.
- **Job queue instead of 409.** A one-slot queue with position feedback;
  second request waits rather than bouncing.
- **History.** Keep the last N runs in the popup (chrome.storage), each
  with its TXT, metrics and recreation — a "your captures" list.
- **Options page.** Per-run dials beyond presets: iterations, parity
  mode, stems mode (Demucs when installed), exact-mode residual export
  (the 99.97% codec — see below).

## Phase 2 — capture that survives the popup

- **Offscreen document recording.** v1 requires the popup to stay open
  (fetch + tabCapture die with it). Move recording into an offscreen
  document with `chrome.tabCapture.getMediaStreamId`, so a capture can
  run for a whole song while the user browses on; the service worker
  coordinates.
- **Page-context capture.** A content-script path using
  `mediaElement.captureStream()` on detected `<audio>`/`<video>`
  elements (with a page-action badge when media is found) — this is also
  the Firefox-compatible route.
- **Smart span selection.** Auto-pick the loudest/most representative N
  seconds (the evidence builder's section machinery can advise), instead
  of "first N seconds played".

## Phase 3 — beyond TXT

- **Equation JSON download** alongside TXT (the ledger's partials), so
  the sheet can be re-rendered/edited offline (`m2m synthesize` already
  renders equations from disk).
- **A/B playback** in the popup: original vs recreation vs
  equations+residual (exact mode, `src/exact_codec.py`) — "3.6% of the
  size at ~70% fidelity, or 73% of the size at 100%" as a user-facing
  choice.
- **Share card**: a small image (metrics + pillars + equation count) the
  user can save — the natural social artifact for this project.

## Phase 4 — distribution

- **Chrome Web Store**: the extension is unpacked-only today. Store
  listing needs the demo GIF the README also wants (one recording serves
  both).
- **Firefox port**: file upload works as-is; tab capture maps to
  `browser.tabCapture`/`mediaElement.captureStream` (Phase 2 already
  unblocks this).
- **One-command install** for the server side: `pipx install m2m-res`
  (the package already builds — CI does wheel builds) plus a tray
  shortcut, so non-CLI users can run `m2m serve` by double-click.

## Phase 5 — beyond localhost (deliberately last)

- **Hosted engine** (Streamlit Community Cloud already runs the app; a
  `/api/recreate` twin behind a queue is a small step) for users who
  won't run Python. Costs: CPU minutes per capture, upload size, abuse
  limits. Do this only with the queue + rate limits from Phase 1.
- **WASM engine port** as the far-future "no server at all" answer:
  realistic only for a frozen feature subset (the lifter without the
  bank), and only if the Python engine stabilizes enough to port without
  divergence. Not before the architecture refactor is fully landed.

## Non-goals & standing constraints

- No cloud telemetry, no accounts, no storing captures beyond the local
  `data/` dirs the engine already uses.
- The server never makes outbound requests and never serves arbitrary
  files — responses carry computed artifacts (JSON + base64 WAV), never
  filesystem paths' contents by reference.
- Analyse what you have the right to analyse; the extension README says
  so and the store listing must too.
