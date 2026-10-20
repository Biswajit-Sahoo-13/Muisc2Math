"""Browser-driven RES runs: run the loop on an uploaded song and package
artifacts for a UI render (Streamlit tab, HTTP API, anything else).

The heavy lifting mirrors ``main.py cmd_res`` exactly — build_evidence +
run_loop with the same defaults — so the CLI and the browser run the same
verified pipeline. Every field in the returned payload is measured, nothing
is claimed without a score behind it.
"""

from __future__ import annotations

import io
import re
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Float32 audio -> 16-bit PCM WAV bytes (mono), peak-safe."""
    y = np.asarray(y, dtype=np.float64)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.9
    pcm = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Background jobs (for UIs that must survive script reruns, e.g. Streamlit:
# module-level state in the *script* is re-executed every rerun, but a module
# import is cached per process, so the registry lives here).
# ---------------------------------------------------------------------------

_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

# Scoreboard runs live in their own registry: they are minutes-long batch jobs
# over the frozen fixture set, not a reconstruction of one upload, and the RES
# tab must never see them.
_SB_JOBS: Dict[str, Dict[str, Any]] = {}
_SB_JOBS_LOCK = threading.Lock()


def demucs_status() -> tuple[bool, str]:
    """Report whether real Demucs source separation is actually usable.

    ``separate_stems`` falls back to an HPSS band split when the Demucs CLI
    cannot be found, and that fallback is *silent*: the UI would label a
    harmonic band as "vocals" while the user believes they got real source
    separation. This makes the degradation visible instead.
    """
    try:
        from src.res_evidence import _demucs_command
        cmd = _demucs_command()
    except Exception as e:  # pragma: no cover - defensive
        return False, f"Demucs check failed: {type(e).__name__}: {e}"
    if not cmd:
        return False, (
            "Demucs is not installed, so stem separation falls back to an "
            "HPSS band split. `pip install demucs torch torchaudio` to get real "
            "isolated stems (and the higher per-stem correlations)."
        )
    return True, f"Demucs available: {' '.join(cmd)}"


class StageTracker:
    """Records which stage a run is in, with per-stage timing.

    The evidence phase is ~44% of wall-clock, so a UI that only shows
    "running" for minutes is useless. This gives the UI a live stage, a
    percentage within that stage, and elapsed time per completed stage.
    """

    def __init__(self) -> None:
        self.stages: List[Dict[str, Any]] = []
        self._t0 = time.monotonic()
        self._cur: Optional[Dict[str, Any]] = None

    def _now(self) -> float:
        return time.monotonic()

    def begin(self, name: str, detail: str = "") -> None:
        self.end()
        self._cur = {"name": name, "detail": detail, "pct": 0.0,
                     "t0": self._now(), "t1": None}
        self.stages.append(self._cur)

    def update(self, pct: float, detail: str = "") -> None:
        if self._cur is None:
            return
        self._cur["pct"] = max(0.0, min(1.0, float(pct)))
        if detail:
            self._cur["detail"] = detail

    def end(self, detail: str = "") -> None:
        if self._cur is not None and self._cur["t1"] is None:
            self._cur["t1"] = self._now()
            if detail:
                self._cur["detail"] = detail
        self._cur = None

    def snapshot(self) -> List[Dict[str, Any]]:
        out = []
        for s in self.stages:
            t1 = s["t1"] if s["t1"] is not None else self._now()
            out.append({"name": s["name"], "detail": s["detail"],
                        "pct": s["pct"], "seconds": round(t1 - s["t0"], 2)})
        return out

    def total_seconds(self) -> float:
        return round(self._now() - self._t0, 2)


def _res_worker(job: Dict[str, Any], blob: bytes, name: str, dur: float,
                iters: int, bank: bool, root: Path,
                stem_mode: str = "none",
                enable_noise: bool = False,
                spectral_lane: bool = False) -> None:
    try:
        payload = run_res_payload(
            blob, name, dur, iters, bank, root=root, progress=job,
            stem_mode=stem_mode, enable_noise=enable_noise,
            spectral_lane=spectral_lane,
        )
        job["payload"] = payload
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


def spawn_job(key: str, blob: bytes, name: str, dur: float, iters: int,
              bank: bool, root: Path, stem_mode: str = "none",
              enable_noise: bool = False,
              spectral_lane: bool = False) -> Dict[str, Any]:
    """Start (or return the existing) background RES job for ``key``."""
    with _JOBS_LOCK:
        busy = [
            j for j in _JOBS.values()
            if j.get("state") in ("queued", "running")
        ]
        if busy:
            return busy[0]
        if key in _JOBS and _JOBS[key].get("state") != "error":
            return _JOBS[key]
        _JOBS.clear()
        job: Dict[str, Any] = {
            "state": "queued", "stage": "starting",
            "iter": 0, "total": int(iters), "corr": None,
            "t0": time.monotonic(), "key": key,
            "corr_history": [], "stem_mode": stem_mode,
        }
        _JOBS[key] = job
        threading.Thread(
            target=_res_worker,
            args=(job, blob, name, dur, iters, bank, root, stem_mode,
                  enable_noise, spectral_lane),
            daemon=True,
        ).start()
        return job


def running_jobs() -> list:
    with _JOBS_LOCK:
        return [
            j for j in _JOBS.values()
            if j.get("state") in ("queued", "running")
        ]


def done_job() -> Optional[Dict[str, Any]]:
    with _JOBS_LOCK:
        return next(
            (j for j in _JOBS.values()
             if j.get("state") == "done" and j.get("payload")),
            None,
        )


# ------------------------------------------------------- scoreboard harness
def _sb_worker(job: Dict[str, Any], label: str, fixtures, duration: float,
               iterations: int, root: Path) -> None:
    try:
        from src.scoreboard import run_label

        def _progress(done: int, total: int, name: str, corr) -> None:
            job["done"] = done
            job["total"] = total
            job["fixture"] = name
            job["corr"] = corr
            job["stage"] = f"{done}/{total} {name}"

        rows = run_label(label, fixtures=fixtures, duration=duration,
                         iterations=iterations, root=root, progress=_progress)
        job["rows"] = rows
        job["state"] = "done"
    except Exception as e:
        job["state"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"


def spawn_scoreboard_job(label: str, fixtures=None, duration: float = 10.0,
                         iterations: int = 6,
                         root: Optional[Path] = None) -> Dict[str, Any]:
    """Start (or return the existing) background scoreboard run for ``label``."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    key = f"{label}:{duration}:{iterations}:{sorted(fixtures or [])}"
    with _SB_JOBS_LOCK:
        busy = [j for j in _SB_JOBS.values()
                if j.get("state") in ("queued", "running")]
        if busy:
            return busy[0]
        if key in _SB_JOBS and _SB_JOBS[key].get("state") != "error":
            return _SB_JOBS[key]
        _SB_JOBS.clear()
        job: Dict[str, Any] = {
            "state": "queued", "stage": "starting", "label": label,
            "done": 0, "total": 0, "fixture": None, "corr": None,
            "t0": time.monotonic(), "key": key,
        }
        _SB_JOBS[key] = job
        threading.Thread(
            target=_sb_worker,
            args=(job, label, fixtures, duration, iterations, root),
            daemon=True,
        ).start()
        return job


def running_scoreboard_jobs() -> list:
    with _SB_JOBS_LOCK:
        return [j for j in _SB_JOBS.values()
                if j.get("state") in ("queued", "running")]


def done_scoreboard_job() -> Optional[Dict[str, Any]]:
    with _SB_JOBS_LOCK:
        return next(
            (j for j in _SB_JOBS.values()
             if j.get("state") in ("done", "error")),
            None,
        )


def _section_equation_texts(ledger: Dict[str, Any]) -> List[str]:
    """One equation sheet per section, so the UI can offer a dropdown.

    Renders the same sheet the full model gets, but with a single section, so
    the maths stays identical to what the ledger records.
    """
    from src.res_loop import _equations_text
    out: List[str] = []
    for sec in (ledger.get("sections") or []):
        try:
            out.append(_equations_text({**ledger, "sections": [sec]}))
        except Exception:
            out.append("")
    return out


def run_res_payload(
    upload_bytes: bytes,
    name: str,
    duration: float,
    iterations: int,
    bank: bool = True,
    window_s: float = 2.5,
    sr: int = 22050,
    root: Optional[Path] = None,
    progress: Optional[Dict[str, Any]] = None,
    stem_mode: str = "none",
    enable_noise: bool = False,
    spectral_lane: bool = False,
    k_notes: int = 2,
    k_hits: int = 12,
) -> Dict[str, Any]:
    """Analyze ``upload_bytes`` with the RES loop and collect display artifacts.

    Writes ``data/uploads/<stamp>_<name>`` plus the evidence and RES output
    directories under ``data/output/`` (anchored at ``root``, which defaults
    to the repo root), loads nothing at import time beyond what the loop
    needs, and returns a plain-JSON-able payload dict.

    ``progress`` (optional) is a plain dict updated in place: ``stage`` goes
    evidence -> loop -> packaging, with ``iter``/``total``/``corr`` during the
    loop, so a UI thread can poll it while this runs elsewhere.
    ``corr_history`` accumulates one entry per completed iteration so the UI
    can plot the convergence curve live.

    ``stem_mode`` is ``"none"`` (single mix) or ``"auto"``/``"demucs"`` (real
    source separation via the stems pipeline). Demucs may be unavailable, in
    which case the engine falls back to an HPSS band split; the payload records
    which actually happened so the UI can say so.

    ``enable_noise`` lets the hypothesis bank model the residual with a
    stochastic class (LPC formant noise). A sum of sinusoids cannot express
    breath, air or sibilance, so that content is absent without it.
    """
    from src.analysis import load_audio
    from src.res_evidence import build_evidence
    from src.res_loop import run_loop, _equations_text

    root_path = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_stem = re.sub(r"[^\w\-]+", "_", Path(name).stem)[:40] or "song"
    suffix = Path(name).suffix or ".wav"

    uploads_dir = root_path / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved = uploads_dir / f"{stamp}_{safe_stem}{suffix}"
    saved.write_bytes(upload_bytes)

    out_root = root_path / "data" / "output"
    out_root.mkdir(parents=True, exist_ok=True)
    ev_dir = out_root / f"res_web_ev_{stamp}"
    out_dir = out_root / f"res_web_{stamp}_{safe_stem}"

    tracker = StageTracker()
    if progress is not None:
        progress["stages"] = tracker.snapshot
        progress["stage_tracker"] = tracker

    def _cb(stage: str, pct: float, detail: str) -> None:
        """Progress callback from the engine, mirrored into the job dict."""
        if tracker._cur is None or tracker._cur["name"] != stage:
            tracker.begin(stage, detail)
        else:
            tracker.update(pct, detail)
        if progress is not None:
            progress.update(stage=stage, stage_pct=pct, stage_detail=detail)
            progress["stages"] = tracker.snapshot()

    tracker.begin("prepare", "loading audio, HPSS split, beat detection")

    ledger = build_evidence(
        str(saved), str(ev_dir), sr=sr, max_duration=duration,
        window_s=window_s, threshold=0.01, max_tracks=400,
        progress=_cb,
    )
    tracker.end("evidence complete")
    start_corr = float(ledger["best"]["correlation"])
    if progress is not None:
        progress.update(stage="evidence-done", iter=0, total=int(iterations),
                        corr=start_corr, stage_pct=1.0)
        progress["stages"] = tracker.snapshot()
    start_sections = [
        round(float(s["metrics"]["correlation"]), 4) for s in ledger["sections"]
    ]
    start_partials = sum(int(s["partial_count"]) for s in ledger["sections"])

    def _prog(k: int, total: int, g: Dict[str, Any], accepted: bool) -> None:
        if tracker._cur is None or tracker._cur["name"] != "loop":
            tracker.begin("loop", f"iteration {k}/{total}")
        tracker.update(k / max(1, total), f"iteration {k}/{total}")
        if progress is not None:
            progress.update(stage="loop", iter=int(k), total=int(total),
                            corr=float(g["correlation"]), accepted=bool(accepted),
                            stage_pct=k / max(1, total),
                            stage_detail=f"iteration {k}/{total}")
            hist = progress.setdefault("corr_history", [])
            hist.append({"iteration": int(k),
                         "corr": float(g["correlation"]),
                         "snr_db": float(g.get("snr_db", 0.0)),
                         "lsd_db": float(g.get("log_spectral_distance_db", 0.0)),
                         "seg_snr_db": g.get("seg_snr_db"),
                         "onset_corr": g.get("onset_corr"),
                         "accepted": bool(accepted)})
            progress["stages"] = tracker.snapshot()

    stems_payload: Optional[Dict[str, Any]] = None
    separation_note = "single mix (no stem separation)"
    if stem_mode in ("auto", "demucs"):
        from src.res_loop import run_stems_res_loop
        if progress is not None:
            progress.update(stage="separation", stage_pct=0.0,
                            stage_detail="separating sources (Demucs)")
        tracker.begin("separation", "separating sources (Demucs)")
        multi = run_stems_res_loop(
            str(saved), str(out_dir), stems_mode=stem_mode, sr=sr,
            max_duration=duration, window_s=window_s, iterations=iterations,
            target_corr=0.0, bank=bank, progress_fn=_prog if progress is not None else None,
            enable_noise=enable_noise, spectral_lane=spectral_lane,
            k_notes=k_notes, k_hits=k_hits,
        )
        stems_payload = multi.get("stems") or {}
        try:
            from src.res_evidence import LAST_SEPARATION_BACKEND
            backend = LAST_SEPARATION_BACKEND
        except Exception:
            backend = None
        separation_note = f"stem separation backend: {backend or 'unknown'}"
        if backend == "hpss":
            separation_note += (
                " — Demucs was unavailable or failed, so this is an HPSS band "
                "split, not real source separation"
            )
        ledger = multi
    else:
        ledger = run_loop(
            ledger, str(saved), str(out_dir),
            iterations=iterations, target_corr=0.0, bank=bank,
            progress_fn=_prog if progress is not None else None,
            enable_noise=enable_noise, spectral_lane=spectral_lane,
            k_notes=k_notes, k_hits=k_hits,
        )
    if progress is not None:
        progress.update(stage="packaging", stage_pct=1.0,
                        stage_detail="writing outputs")
    tracker.begin("package", "writing outputs")

    # Original span bytes: native-rate decode, cropped to the analysed window.
    y, native_sr = load_audio(saved, target_sr=None)
    span = int(round(float(ledger["duration"]) * native_sr))
    orig_bytes = _wav_bytes(y[: min(span, y.size)], native_sr)
    best_path = Path(ledger["outputs"]["best_wav"])
    best_bytes = best_path.read_bytes() if best_path.exists() else b""

    # Write the playable files to disk and expose *paths*, not bytes. Passing
    # bytes to st.audio makes Streamlit mint a new media file on every render,
    # so the browser reloads the <audio> element and restarts playback from
    # zero. A stable file path keeps the same URL across re-renders.
    play_dir = out_dir
    play_dir.mkdir(parents=True, exist_ok=True)
    orig_wav = play_dir / "res_original_span.wav"
    orig_wav.write_bytes(orig_bytes)
    best_wav_path = best_path if best_path.exists() else None
    if best_wav_path is not None and not best_wav_path.is_absolute():
        best_wav_path = (root_path / best_wav_path).resolve()
    if best_wav_path is not None and not best_wav_path.exists():
        best_wav_path = None
    if best_wav_path is None:
        best_wav_path = play_dir / "res_best.wav"
        best_wav_path.write_bytes(best_bytes)
    uploaded_wav = Path(saved).resolve()

    accepted = []
    for i, sec in enumerate(ledger["sections"]):
        for rec in (sec.get("hypothesis_bank") or []):
            if not rec.get("accepted"):
                continue
            accepted.append({
                "section": i,
                "class": rec.get("class") or "harmonic_string",
                "f0": rec.get("f0"),
                "t0": rec.get("t0"),
                "partials": rec.get("m_count"),
                "params": rec.get("param_count"),
                "residual_cut": round(
                    float(rec["residual_before"]) - float(rec["residual_after"]), 3
                ),
            })
    trail = [{
        "iteration": it["iteration"],
        "provider": it["provider"],
        "corr": it["global"]["correlation"],
        "snr_db": it["global"]["snr_db"],
        "lsd_db": it["global"]["log_spectral_distance_db"],
        "seg_snr_db": it["global"].get("seg_snr_db"),
        "onset_corr": it["global"].get("onset_corr"),
        "llm_edits": it["llm_edits_applied"],
        "accepted": bool(it["accepted"]),
    } for it in (ledger.get("iterations") or [])]
    if not trail and progress is not None:
        # Stem mode returns a multitrack ledger without an "iterations" list,
        # so fall back to the live history captured during the loop.
        for h in (progress.get("corr_history") or []):
            trail.append({
                "iteration": h.get("iteration"),
                "provider": "stems",
                "corr": h.get("corr"),
                "snr_db": h.get("snr_db"),
                "lsd_db": h.get("lsd_db"),
                "seg_snr_db": h.get("seg_snr_db"),
                "onset_corr": h.get("onset_corr"),
                "llm_edits": 0,
                "accepted": bool(h.get("accepted")),
            })
    sections_tbl = [{
        "section": i,
        "start": float(sec["start"]),
        "end": float(sec["end"]),
        "corr": float(sec["metrics"]["correlation"]),
        "snr_db": float(sec["metrics"]["snr_db"]),
        "partials": int(sec["partial_count"]),
    } for i, sec in enumerate(ledger["sections"])]

    # Residual (error) signal: what the equations did NOT reproduce.
    residual_bytes = b""
    residual_path = None
    if best_bytes and orig_bytes:
        try:
            y_r, _ = load_audio(str(best_wav_path), target_sr=None)
            n_r = min(len(y_r), len(y[: min(span, y.size)]))
            if n_r > 0:
                res = y[:n_r] - np.asarray(y_r[:n_r], dtype=np.float64)
                residual_bytes = _wav_bytes(res, native_sr)
                residual_path = play_dir / "res_residual.wav"
                residual_path.write_bytes(residual_bytes)
                residual_path = str(residual_path.resolve())
        except Exception:
            residual_bytes = b""
            residual_path = None

    return {
        "start_corr": start_corr,
        "start_sections": start_sections,
        "start_partials": start_partials,
        "best": dict(ledger["best"]),
        "trail": trail,
        "sections": sections_tbl,
        "accepted": accepted,
        "equations": _equations_text(ledger),
        "section_equations": _section_equation_texts(ledger),
        "equation_classes": ledger.get("equation_classes") or [],
        "duration": float(ledger["duration"]),
        "orig_bytes": orig_bytes,
        "best_bytes": best_bytes,
        "residual_bytes": residual_bytes,
        # Stable file paths for the players: st.audio(bytes) mints a new media
        # file on every render, which makes the browser restart playback.
        "orig_path": str(orig_wav.resolve()),
        "best_path": str(best_wav_path),
        "residual_path": residual_path,
        "uploaded_path": str(uploaded_wav),
        "ledger_path": ledger["outputs"]["ledger"],
        "best_wav": str(best_path),
        "source_path": str(saved),
        "stems": stems_payload,
        "separation_note": separation_note,
    }
