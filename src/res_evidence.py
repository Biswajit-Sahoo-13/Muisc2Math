"""RES (Reverse Engineer a Song) evidence builder.

The RES loop mirrors REA's method for audio: analysis tools gather evidence,
an agent (or numeric fallback) forms hypotheses about the *equations* that
produce the observed sound, verification renders those equations and scores
them against the original, and the ledger records every try.

This module is the agent-facing intake: it turns a song into a versioned,
JSON-friendly evidence ledger whose per-section partial lists are the
hypothesis substrate. Every number in it was measured, never assumed.

In-session agent usage (the same contract REA-style agents use):

    from src.res_evidence import evidence_prompt, build_evidence
    ledger = build_evidence("song.mp3", out_dir, window_s=2.5)
    prompt = evidence_prompt(ledger)   # compact text; plan edits from it
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import scipy.signal
import librosa

from src.analysis import load_audio
from src.beat_detection import detect_beats
from src.sinusoidal_model import sinusoidal_reconstruction
from src.residual import (
    compute_error_metrics,
    segmental_snr_db,
    onset_corr,
)
from src.res_synthesis import _corr

_EVIDENCE_CACHE: Dict[str, Any] = {}


def _demucs_command() -> Optional[List[str]]:
    """Locate a usable Demucs CLI, preferring the current interpreter.

    ``shutil.which("demucs")`` alone is not enough: when the project runs from a
    virtualenv whose ``Scripts`` directory is not on ``PATH`` (the normal case
    here) it returns ``None`` even though Demucs is installed, and the caller
    silently falls back to HPSS. Try the interpreter's own Scripts directory
    first, then ``python -m demucs``.
    """
    found = shutil.which("demucs")
    if found:
        return [found]
    scripts = Path(sys.executable).parent
    for name in ("demucs.exe", "demucs.cmd", "demucs"):
        cand = scripts / name
        if cand.exists():
            return [str(cand)]
    try:
        import importlib.util
        if importlib.util.find_spec("demucs") is not None:
            return [sys.executable, "-m", "demucs"]
    except Exception:
        pass
    return None


# Records which separation backend actually produced the last stem set.
# "requested" is what the caller asked for; this is what really happened, so a
# UI can tell the user when Demucs silently degraded to an HPSS band split.
LAST_SEPARATION_BACKEND: Optional[str] = None


def separate_stems(
    y: np.ndarray,
    sr: int,
    mode: str = "auto",
    stems_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Tri-Mode Stem Separation with exact partition of unity.

    Mode 1 (Directory):
      If ``stems_dir`` points to a directory containing bass.wav, drums.wav,
      vocals.wav, and other.wav, loads them directly and aligns length to y.

    Mode 2 (Subprocess):
      If ``mode in ('auto', 'demucs')`` and `demucs` is in system PATH,
      invokes it via subprocess and loads separated stems.

    Mode 3 (Native DSP Fallback - Zero-Phase Complementary Partition of Unity):
      1. Zero-phase 4th-order Butterworth low-pass filter at 120 Hz:
         y_bass = scipy.signal.sosfiltfilt(sos_lp, y)
      2. Exact subtraction for remainder:
         y_mid_high = y - y_bass
      3. Librosa HPSS splits y_mid_high into y_harm (vocals/synths) and y_perc (drums):
         y_harm, y_perc = librosa.effects.hpss(y_mid_high)
      4. Exact residual to other:
         y_other = y_mid_high - (y_harm + y_perc)
      Guarantees: y_bass + y_drums + y_vocals + y_other == y to machine precision (< 1e-12).
    """
    global LAST_SEPARATION_BACKEND
    n = len(y)

    # Mode 1: Directory Mode
    target_dir = stems_dir if (stems_dir and Path(stems_dir).is_dir()) else (mode if (mode and Path(mode).is_dir()) else None)
    if target_dir:
        stems_path = Path(target_dir)
        loaded: Dict[str, np.ndarray] = {}
        for stem_name in ("bass", "drums", "vocals", "other"):
            for ext in (".wav", ".flac", ".mp3", ".ogg"):
                candidate = stems_path / f"{stem_name}{ext}"
                if candidate.exists():
                    s_audio, _ = load_audio(str(candidate), target_sr=sr)
                    if len(s_audio) < n:
                        s_audio = np.pad(s_audio, (0, n - len(s_audio)))
                    else:
                        s_audio = s_audio[:n]
                    loaded[stem_name] = s_audio
                    break
        if len(loaded) == 4:
            LAST_SEPARATION_BACKEND = "directory"
            return loaded

    # Mode 2: Demucs Subprocess Mode
    demucs_bin = _demucs_command()
    if (mode == "demucs" or mode == "auto") and demucs_bin and not stems_dir:
        import tempfile
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp_d:
            in_wav = Path(tmp_d) / "input.wav"
            sf.write(str(in_wav), y, sr)
            out_d = Path(tmp_d) / "out"
            # Full four-stem separation. "--two-stems=bass" emits only bass and
            # no_bass, so the four-stem load below could never complete and the
            # caller fell through to the HPSS fallback every single run.
            cmd = demucs_bin + ["-o", str(out_d), str(in_wav)]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, timeout=1800)
                if res.returncode == 0:
                    model_dirs = list(out_d.glob("*"))
                    if model_dirs and model_dirs[0].is_dir():
                        track_d = model_dirs[0] / in_wav.stem
                        if track_d.is_dir():
                            loaded_demucs = {}
                            for s_name in ("bass", "drums", "vocals", "other"):
                                p_stem = track_d / f"{s_name}.wav"
                                if p_stem.exists():
                                    s_arr, _ = load_audio(str(p_stem), target_sr=sr)
                                    if len(s_arr) < n:
                                        s_arr = np.pad(s_arr, (0, n - len(s_arr)))
                                    else:
                                        s_arr = s_arr[:n]
                                    loaded_demucs[s_name] = s_arr
                            if len(loaded_demucs) == 4:
                                LAST_SEPARATION_BACKEND = "demucs"
                                return loaded_demucs
            except Exception:
                pass  # Fallback to Mode 3

    # Mode 3: Native DSP Fallback (Zero-Phase Partition of Unity)
    sos_lp = scipy.signal.butter(4, 120.0, btype="lowpass", fs=sr, output="sos")
    y_bass = scipy.signal.sosfiltfilt(sos_lp, y)
    y_mid_high = y - y_bass

    if n >= 1024:
        try:
            yh, yp = librosa.effects.hpss(y_mid_high)
        except Exception:
            yh = y_mid_high
            yp = np.zeros_like(y_mid_high)
    else:
        yh = y_mid_high
        yp = np.zeros_like(y_mid_high)

    # "other" must be a genuine band, not the HPSS residual. librosa splits with
    # soft masks that sum back to the input, so `y_mid_high - (yh + yp)` is
    # identically zero and the stem is empty in every run. Split the harmonic
    # component instead: vocal formants (200-4000 Hz) versus everything else
    # (synths, keys, guitars, and harmonic air above the vocal band). The
    # partition still closes exactly, because y_vocals + y_other == yh.
    try:
        sos_vocal = scipy.signal.butter(
            4, (200.0, 4000.0), btype="bandpass", fs=sr, output="sos"
        )
        # sosfiltfilt already preserves length; truncating here would silence
        # every sample past the first 4096 and empty the stem entirely.
        y_vocals = scipy.signal.sosfiltfilt(sos_vocal, yh)
        y_other = yh - y_vocals
    except Exception:
        y_vocals = yh
        y_other = np.zeros_like(yh)

    LAST_SEPARATION_BACKEND = "hpss"
    return {
        "bass": y_bass,
        "drums": yp,
        "vocals": y_vocals,
        "other": y_other,
    }


def clear_evidence_cache() -> None:
    """Clear in-memory evidence cache."""
    _EVIDENCE_CACHE.clear()


def _loudest_window(y: np.ndarray, win: int) -> np.ndarray:
    """Return the highest-energy ``win``-sample slice of ``y``.

    The low and mid detectors use long windows for frequency resolution. Reading
    them from offset 0 blinds the detector whenever a track opens with silence
    or a fade-in - a 4096-sample window at 22.05 kHz is only 186 ms, and real
    recordings routinely start with more silence than that. Selecting the
    loudest window instead guarantees the detector sees sustained material
    rather than a digital-silence pad.
    """
    if win <= 0 or len(y) <= win:
        return y
    energy = np.convolve(y ** 2, np.ones(win), mode="valid")
    start = int(np.argmax(energy))
    return y[start:start + win]


def multi_resolution_peaks(
    y: np.ndarray,
    sr: int,
    top_k: int = 24,
    high_quota: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Multi-resolution spectral peak estimation.

    Standard fixed-window STFT forces an undesirable trade-off between time
    and frequency resolution. This function extracts candidates across three
    frequency bands with tailored window lengths:
      - Low band (<250 Hz): Large window (N=4096) for sub-bass/808 frequency resolution.
      - Mid band (250 - 2000 Hz): Medium window (N=2048) for balanced musical body.
      - High band (>2000 Hz): Small window (N=512) for transient/hi-hat time resolution.

    ``high_quota`` reserves slots for the high band so spectral tilt cannot
    starve sibilance and cymbals out of the top-k cut. Defaults to a third of
    ``top_k``, which holds the total peak budget constant.
    """
    n = len(y)
    if n < 16:
        return []

    if high_quota is None:
        high_quota = max(1, int(round(top_k / 3.0)))

    peaks: List[Dict[str, Any]] = []

    # 1. Low Band (< 250 Hz) with large window (N=4096)
    w_low = np.hanning(min(n, 4096))
    seg_low = _loudest_window(y, len(w_low)) * w_low
    fft_low = np.abs(np.fft.rfft(seg_low, n=4096))
    freqs_low = np.fft.rfftfreq(4096, 1.0 / sr)
    mask_low = (freqs_low >= 20.0) & (freqs_low <= 250.0)

    sub_mag = fft_low.copy()
    sub_mag[~mask_low] = 0.0
    for idx in range(1, len(sub_mag) - 1):
        if mask_low[idx] and sub_mag[idx] > sub_mag[idx - 1] and sub_mag[idx] > sub_mag[idx + 1]:
            alpha = float(sub_mag[idx - 1])
            beta = float(sub_mag[idx])
            gamma = float(sub_mag[idx + 1])
            denom = alpha - 2 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            p_freq = float(freqs_low[idx] + delta * (freqs_low[1] - freqs_low[0]))
            p_mag = float(beta - 0.25 * (alpha - gamma) * delta)
            peaks.append({
                "frequency": round(p_freq, 2),
                "magnitude": round(p_mag, 4),
                "band": "low",
                "resolution_hz": round(float(sr) / 4096.0, 2),
            })

    # 2. Mid Band (250 Hz - 2000 Hz) with N=2048
    w_mid = np.hanning(min(n, 2048))
    seg_mid = _loudest_window(y, len(w_mid)) * w_mid
    fft_mid = np.abs(np.fft.rfft(seg_mid, n=2048))
    freqs_mid = np.fft.rfftfreq(2048, 1.0 / sr)
    mask_mid = (freqs_mid > 250.0) & (freqs_mid <= 2000.0)

    mid_mag = fft_mid.copy()
    mid_mag[~mask_mid] = 0.0
    for idx in range(1, len(mid_mag) - 1):
        if mask_mid[idx] and mid_mag[idx] > mid_mag[idx - 1] and mid_mag[idx] > mid_mag[idx + 1]:
            alpha = float(mid_mag[idx - 1])
            beta = float(mid_mag[idx])
            gamma = float(mid_mag[idx + 1])
            denom = alpha - 2 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            p_freq = float(freqs_mid[idx] + delta * (freqs_mid[1] - freqs_mid[0]))
            p_mag = float(beta - 0.25 * (alpha - gamma) * delta)
            peaks.append({
                "frequency": round(p_freq, 2),
                "magnitude": round(p_mag, 4),
                "band": "mid",
                "resolution_hz": round(float(sr) / 2048.0, 2),
            })

    # 3. High Band (> 2000 Hz) with short windows (N=512) across hops to capture transients
    n_high = 512
    hop_high = 256
    w_high = np.hanning(n_high)
    high_peaks_set: List[Dict[str, Any]] = []
    for start in range(0, max(1, n - n_high + 1), hop_high):
        chunk = y[start:start + n_high]
        if len(chunk) < n_high:
            chunk = np.pad(chunk, (0, n_high - len(chunk)))
        fft_h = np.abs(np.fft.rfft(chunk * w_high))
        freqs_h = np.fft.rfftfreq(n_high, 1.0 / sr)
        mask_h = freqs_h > 2000.0
        h_mag = fft_h.copy()
        h_mag[~mask_h] = 0.0
        for idx in range(1, len(h_mag) - 1):
            if mask_h[idx] and h_mag[idx] > h_mag[idx - 1] and h_mag[idx] > h_mag[idx + 1]:
                p_freq = float(freqs_h[idx])
                p_mag = float(h_mag[idx])
                high_peaks_set.append({
                    "frequency": round(p_freq, 2),
                    "magnitude": round(p_mag, 4),
                    "band": "high",
                    "t_offset_s": round(float(start) / sr, 4),
                    "resolution_hz": round(float(sr) / 512.0, 2),
                })

    high_peaks_set.sort(key=lambda p: p["magnitude"], reverse=True)
    dedup_high: List[Dict[str, Any]] = []
    for hp in high_peaks_set:
        if not any(abs(hp["frequency"] - d["frequency"]) < 50.0 and abs(hp.get("t_offset_s", 0) - d.get("t_offset_s", 0)) < 0.02 for d in dedup_high):
            dedup_high.append(hp)
            if len(dedup_high) >= 10:
                break

    # Band-reserved selection. A single global magnitude sort silently deletes
    # every high-frequency peak: music carries roughly -12 dB/octave of spectral
    # tilt, so a bass fundamental outranks a 6 kHz sibilance component no matter
    # how present the sibilance is. The dedicated >2 kHz detector above is
    # therefore worthless unless its survivors are reserved a quota before the
    # loud-band competition. Remaining slots are still filled by magnitude, so
    # low/mid behaviour is unchanged.
    high_band = sorted(dedup_high, key=lambda p: p["magnitude"], reverse=True)
    reserved = high_band[:max(0, min(int(high_quota), len(high_band), top_k))]
    rest = [p for p in peaks if p.get("band") != "high"]
    rest.sort(key=lambda p: p["magnitude"], reverse=True)
    chosen = rest[: max(0, top_k - len(reserved))] + reserved
    chosen.sort(key=lambda p: p["magnitude"], reverse=True)
    return chosen


def get_cached_section_evidence(
    section: Dict[str, Any],
    sec_y: np.ndarray,
    sr: int,
) -> Dict[str, Any]:
    """Retrieve or compute cached spectral evidence for a section."""
    start_s = int(section.get("start_sample", round(section.get("start", 0.0) * sr)))
    cache_key = f"{start_s}_{len(sec_y)}_{sr}"
    if cache_key in _EVIDENCE_CACHE:
        return _EVIDENCE_CACHE[cache_key]

    n = len(sec_y)
    if n < 8:
        ev = {
            "fft_mag": np.zeros(1),
            "freq_bins": np.zeros(1),
            "energy": 0.0,
            "multi_resolution_peaks": [],
        }
        _EVIDENCE_CACHE[cache_key] = ev
        return ev

    w = np.hanning(n)
    mag = np.abs(np.fft.rfft(sec_y * w))
    bins = np.fft.rfftfreq(n, 1.0 / sr)
    energy = float(np.sum(sec_y ** 2))
    ev = {
        "fft_mag": mag,
        "freq_bins": bins,
        "energy": energy,
        "multi_resolution_peaks": multi_resolution_peaks(sec_y, sr),
    }
    _EVIDENCE_CACHE[cache_key] = ev
    return ev


def _rephase_to_section(
    p: Dict[str, Any], section_start: float
) -> Dict[str, Any]:
    """Re-reference a partial's phase from its analysis window to a section.

    The SMS engine projects every partial against its *analysis window's* local
    time axis, so the phase it returns is window-relative. The RES renderer
    draws every partial on the *section's* local axis (the equation sheet
    documents it: ``x = t - t_s``). Those two origins agree only when the
    section boundaries happen to coincide with the analysis windows.

    When they do not, every partial is rotated by ``2*pi*f*(origin difference)``
    - hundreds of full cycles for a mid-range note - so the reconstruction is
    uncorrelated with the reference even though each partial is individually
    correct. Beat-quantised sections move the origins, which is why
    beat-quantised material collapsed while fixed-window material did not.

    The origin is known on both sides here, so the phase is re-referenced once,
    at build time, and the renderer stays a plain section-local renderer.
    """
    origin = p.get("phase_origin")
    if origin is None:
        return p
    q = dict(p)
    f = float(p.get("frequency", 0.0))
    delta = float(origin) - float(section_start)
    if p.get("model") == "glide" and p.get("trajectory"):
        # A glide's phase is integral + 2*pi*(f_ref - f_guess)*t, so the
        # origin shift rides on the (f_ref - f_guess) term, not on f.
        anchor = float(p.get("phase_ref_hz", 0.0))
        shift = 2.0 * np.pi * (f - anchor) * delta
    else:
        shift = 2.0 * np.pi * f * delta
    q["phase"] = round(float(p.get("phase", 0.0)) + shift, 6)
    return q


def _provenance(
    input_audio_path: Any,
    sr: int,
    window_s: float,
    max_duration: float,
) -> Dict[str, Any]:
    """The citation for the numbers in this ledger.

    A correlation without the fixture it was measured on is not a result,
    it is an anecdote. This block records which bytes were analysed (the
    SHA-256 of the source file), the analysis parameters, and the engine
    revision, so any figure quoted from the ledger can be reproduced
    against the same input.
    """
    import hashlib
    import subprocess

    prov: Dict[str, Any] = {
        "sha256": None,
        "bytes": None,
        "sr": sr,
        "window_s": window_s,
        "max_duration": max_duration,
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        p = Path(str(input_audio_path))
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            prov["sha256"] = h.hexdigest()
            prov["bytes"] = p.stat().st_size
    except Exception:
        pass
    try:
        def _git(*args: str) -> str:
            return subprocess.run(
                ["git", *args], cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()

        prov["git_commit"] = _git("rev-parse", "HEAD") or None
        prov["git_dirty"] = bool(_git("status", "--porcelain"))
    except Exception:
        pass
    return prov


def build_evidence(
    input_audio_path: Union[str, Path, np.ndarray],
    output_dir: str,
    sr: int = 22050,
    max_duration: float = 20.0,
    window_s: float = 2.5,
    n_fft: int = 2048,
    hop_length: int = 256,
    threshold: float = 0.01,
    max_tracks: int = 400,
    min_residual_ratio: float = 1e-6,
    seed: int = 0,
    use_hpss: bool = True,
    beat_quantized: bool = True,
      glide_tolerance_hz: float = 6.0,
      boundaries: Optional[List[int]] = None,
      progress: Optional[Callable[[str, float, str], None]] = None,
  ) -> Dict[str, Any]:
    """Analyse the song and emit the initial evidence ledger.

    Sections are SMS analysis windows. Each carries the fitted partial
    equations that belong to it plus its measured fidelity metrics against the
    original, so an agent can see exactly which sections are under-described.
    """
    import librosa

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(input_audio_path, np.ndarray):
        y = np.asarray(input_audio_path, dtype=np.float64).copy()
        if max_duration > 0.0:
            y = y[: int(max_duration * sr)]
    else:
        y, _ = load_audio(input_audio_path, target_sr=sr)
        y = y[: int(max_duration * sr)]
    n = len(y)

    hpss_applied = False
    percussive_ratio = 0.0
    y_harm = y
    if use_hpss and n >= 4 * hop_length:
        try:
            yh, yp = librosa.effects.hpss(y)
            e_tot = float(np.sum(y ** 2))
            if e_tot > 1e-12:
                percussive_ratio = float(np.sum(yp ** 2)) / e_tot
            y_harm = yh
            hpss_applied = True
        except Exception:
            y_harm = y
            hpss_applied = False

    if progress is not None:
        progress("evidence:sinusoids", 0.0, "fitting sinusoids")
    result = sinusoidal_reconstruction(
        y_harm,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        threshold=threshold,
        max_tracks=max_tracks,
        add_noise_layer=False,
        min_residual_ratio=min_residual_ratio,
        seed=seed,
        window_s=window_s,
        glide_tolerance_hz=glide_tolerance_hz,
        progress=progress,
    )
    partials = result["fitted_parameters"]

    try:
        beats = detect_beats(y, sr=sr)
        tempo = round(float(beats.get("tempo", 120.0)), 1)
        beat_times = np.asarray(beats.get("beat_times", []), dtype=np.float64)
        beat_conf = float(beats.get("confidence", 0.0))
    except Exception:
        tempo = 0.0
        beat_times = np.array([], dtype=np.float64)
        beat_conf = 0.0

    # Determine section boundaries (beat-quantized vs fixed window)
    min_span_samples = 4 * hop_length
    total_dur = n / float(sr)
    is_beat_quantized = False
    if boundaries is not None:
        sec_bounds = [int(b) for b in boundaries if 0 <= b <= n]
        if not sec_bounds or sec_bounds[0] != 0:
            sec_bounds.insert(0, 0)
        if sec_bounds[-1] != n:
            sec_bounds.append(n)
        boundaries = sec_bounds
    elif beat_quantized and beat_conf >= 0.15 and len(beat_times) >= 2 and total_dur > window_s:
        boundaries = [0]
        cur_t = window_s
        last_sample = 0
        while cur_t < total_dur - (window_s * 0.4):
            # Find closest beat within tolerance [cur_t - 0.45*window_s, cur_t + 0.45*window_s]
            in_range = beat_times[
                (beat_times >= cur_t - 0.45 * window_s) &
                (beat_times <= cur_t + 0.45 * window_s)
            ]
            if len(in_range) > 0:
                best_bt = float(in_range[np.argmin(np.abs(in_range - cur_t))])
                sample_idx = int(round(best_bt * sr))
                if sample_idx - last_sample >= min_span_samples and (n - sample_idx) >= min_span_samples:
                    boundaries.append(sample_idx)
                    last_sample = sample_idx
                    is_beat_quantized = True
            cur_t += window_s
        boundaries.append(n)
    else:
        boundaries = [0]
        win = max(int(round(window_s * sr)), min_span_samples)
        for w_start in range(win, n, win):
            if n - w_start >= min_span_samples:
                boundaries.append(w_start)
        boundaries.append(n)

    sections: List[Dict[str, Any]] = []
    n_sections = max(1, len(boundaries) - 1)
    for i in range(len(boundaries) - 1):
        if progress is not None:
            progress("evidence:sections", i / n_sections,
                     f"scoring section {i + 1}/{n_sections}")
        w_start = boundaries[i]
        w_end = boundaries[i + 1]
        if w_end - w_start < min_span_samples:
            continue
        lo, hi = w_start / sr, w_end / sr
        # A partial belongs to every section it SOUNDS in, not only the one
        # where its envelope starts. Assigning by envelope_t0 alone truncated
        # a partial at the first section boundary it crossed, so every later
        # section lost the model for the tone that was filling it: a section
        # covering 2.74-5.25 s received no model at all for the 196 Hz tone
        # that ran through it, because the partial modelling that tone was
        # fitted in the 2.5 s analysis window and started before the section
        # did. The ledger then reported that section as 0.96 correlated while
        # its partials actually rendered to 0.40.
        sec_partials = []
        for p in partials:
            p_start = float(p.get("start_time", p.get("envelope_t0", 0.0)))
            p_end = float(p.get("end_time", p.get("envelope_t0", 0.0)))
            if p_end <= lo or p_start >= hi:
                continue
            sec_partials.append(_rephase_to_section(p, lo))
        sec_y = y[w_start:w_end]
        sec_r = result["sinusoid_audio"][w_start:w_end]
        metrics = compute_error_metrics(sec_y, sec_r, sr=sr)
        sections.append({
            "start": round(lo, 3),
            "end": round(hi, 3),
            "start_sample": int(w_start),
            "end_sample": int(w_end),
            "partial_count": len(sec_partials),
            "partials": sec_partials,
            "hypothesis_bank": [],
            "metrics": {
                "correlation": round(_corr(sec_y, sec_r), 6),
                "snr_db": round(float(metrics["snr_db"]), 3),
                "log_spectral_distance_db": round(
                    float(metrics["log_spectral_distance_db"]), 3
                ),
                "seg_snr_db": round(segmental_snr_db(sec_y, sec_r, sr), 3),
                "onset_corr": round(onset_corr(sec_y, sec_r, sr), 6),
            },
        })

    global_corr = _corr(y, result["sinusoid_audio"])
    recon = result["sinusoid_audio"]
    g_m = compute_error_metrics(y, recon, sr=sr)
    ledger: Dict[str, Any] = {
        "schema": "res/1.1",
        "source": str(input_audio_path),
        "provenance": _provenance(input_audio_path, sr, window_s, max_duration),
        "sample_rate": sr,
        "duration": round(n / sr, 3),
        "n_samples": int(n),
        "window_s": window_s,
        "hpss_applied": hpss_applied,
        "percussive_ratio": round(percussive_ratio, 4),
        "beat_quantized": is_beat_quantized,
        "stft_half_window_s": round((n_fft / 2.0) / sr, 6),
        "equation_form": "A*exp(-lambda*(t - t0))*cos(2*pi*f*t - phi)",
        "equation_classes": ["damped_cosine", "harmonic_string"],
        "tempo_bpm": tempo,
        "iterations": [],
        "best": {
            "correlation": round(global_corr, 6),
            "snr_db": round(float(result["snr_db"]), 3),
            "log_spectral_distance_db": round(
                float(g_m["log_spectral_distance_db"]), 3
            ),
            "seg_snr_db": round(segmental_snr_db(y, recon, sr), 3),
            "onset_corr": round(onset_corr(y, recon, sr), 6),
        },
        "sections": sections,
    }

    path = out_dir / "res_evidence_initial.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    ledger["evidence_path"] = str(path)
    return ledger


def evidence_prompt(ledger: Dict[str, Any], max_partials_per_section: int = 6) -> str:
    """Compact, LLM-readable summary of the evidence state.

    Deliberately short: the agent must reason about section metrics and the
    equation forms, not wade through every sample. Sections are ranked by how
    under-described they are, and the weakest partials are listed first.
    """
    lines = [
        f"song={Path(ledger['source']).name} sr={ledger['sample_rate']} "
        f"duration={ledger['duration']}s window={ledger['window_s']}s "
        f"tempo={ledger['tempo_bpm']}bpm",
        f"equation form: {ledger['equation_form']}",
        f"best global: corr={ledger['best']['correlation']} "
        f"snr={ledger['best']['snr_db']}dB "
        f"iterations={len(ledger['iterations'])}",
        "",
    ]
    ranked = sorted(
        range(len(ledger["sections"])),
        key=lambda i: ledger["sections"][i]["metrics"]["correlation"],
    )
    for idx in ranked:
        s = ledger["sections"][idx]
        m = s["metrics"]
        accepted_b = [
            r for r in (s.get("hypothesis_bank") or [])
            if r.get("accepted") and r.get("f0") is not None
        ]
        accepted_hits = [
            r for r in (s.get("hypothesis_bank") or [])
            if r.get("accepted") and r.get("class") == "percussive_bundle"
        ]
        lines.append(
            f"section {idx} [{s['start']:.2f}-{s['end']:.2f}s] "
            f"corr={m['correlation']:.3f} snr={m['snr_db']:.1f}dB "
            f"seg_snr={m.get('seg_snr_db', 0.0):.1f}dB "
            f"onset_corr={m.get('onset_corr', 0.0):.3f} "
            f"lsd={m['log_spectral_distance_db']:.1f}dB "
            f"partials={s['partial_count']}"
            + (
                f" notes=[" + ", ".join(
                    f"HL f0={float(r['f0']):.2f} B={float(r['B']):.1e}"
                    for r in accepted_b
                ) + "]"
                if accepted_b else ""
            )
            + (
                f" hits=[" + ", ".join(
                    f"t0={float(r['t0']):.3f}"
                    for r in accepted_hits
                ) + "]"
                if accepted_hits else ""
            )
        )
        for pid, p in sorted(
            enumerate(s["partials"]),
            key=lambda q: float(q[1].get("amplitude", 0.0)),
            reverse=True,
        )[:max_partials_per_section]:
            # The id is the index into the section's own partial list, which
            # is what apply_edits resolves. The list below is sorted by
            # amplitude and truncated, so without the id a reader cannot
            # tell which partial a line refers to.
            lines.append(
                f"   part_{pid:03d}: "
                f"{float(p.get('amplitude', 0.0)):.4f} "
                f"exp(-{float(p.get('envelope_decay', 0.0)):.3f}*(t-{float(p.get('envelope_t0', 0.0)):.3f})) "
                f"cos(2pi*{float(p.get('frequency', 0.0)):.3f}*t - "
                f"{float(p.get('phase', 0.0)):.3f}) "
                f"[{p.get('model')}]"
            )
    return "\n".join(lines)
