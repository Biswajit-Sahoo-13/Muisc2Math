# Music2Math browser extension (v0.1)

Turn any audio the browser can play — or any audio/video file — into a set
of editable math equations, and download the equation sheet as a `.txt`.

The extension is a thin client: all analysis runs on **your machine**, on
the RES engine in this repository. Nothing leaves localhost.

## Install (2 minutes)

1. Run the engine server from the repo root:

   ```
   .venv/Scripts/python.exe main.py serve        # Windows
   python main.py serve                          # Linux/macOS
   ```

   It listens on `http://127.0.0.1:8765` (change the port with
   `--port`).

2. Load the extension:
   - Chrome / Edge / Brave: open `chrome://extensions`, enable
     **Developer mode**, click **Load unpacked**, pick this `extension/`
     folder.
   - The puzzle-piece toolbar now shows the Music2Math icon.

## Use

- **Upload a file** — click the icon, then *Upload audio / video file*.
  The browser decodes it, resamples to mono 22.05 kHz and sends it to the
  engine.
- **Record a tab** — click *Record this tab*, then play the song on that
  tab (YouTube, SoundCloud, Bandcamp, anything). Recording stops after the
  configured number of seconds (or click *Stop early*). **Keep the popup
  open** while recording and while the engine runs — closing it cancels
  the job (that is the v1 limitation; background capture is on the
  roadmap).
- When the engine finishes you get: fidelity %, correlation, SNR, the
  four pillars, the equation count and size — and the **Download
  equations .txt** button. You can also play the recreation right in the
  popup and compare it against the original.

Presets: *Fast* (2 iterations, no extras), *Standard* (6 iterations, noise
layer + spectral lane + loudness parity — the README's full-fidelity
recipe), *Full* (adds the experimental polish refiner and MR-STFT referee
lane).

## Troubleshooting

- **Red dot / "offline"** — the server isn't running. Start it with
  `python main.py serve`; check the URL in the popup settings matches.
- **409 engine busy** — the server runs one job at a time; wait and retry.
- **Capture failed** — tab capture needs the button click (a user
  gesture) and cannot capture other extension pages or `chrome://` pages.
- **Firefox** — v1 targets Chromium; the file-upload path works anywhere,
  tab capture needs the Firefox port (roadmap).

## Privacy & security

- All processing is local; the extension only ever talks to
  `127.0.0.1`/`localhost`.
- The server makes no outbound requests, keeps no secrets, and caps
  uploads at 40 MB.
- Analyse music you have the right to analyse; the equation sheet is for
  your own study/production use.
