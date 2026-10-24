"""Tone-to-equation recreation: identify tones, pick formulas, render Math-DAW.

Pipeline:
1. Detect tempo/beats/onsets (``src.beat_detection``).
2. Identify tones with time spans (``src.toneprint.identify_tones``).
3. Fit the best ``FORMULA_BANK`` family per tone (``fit_best_formula_for_segment``).
4. Translate each winning formula into a Math-DAW track equation + quantized
   beat-step pattern, add a percussive onset track, render, and evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from src.analysis import load_audio
from src.beat_detection import detect_beats
from src.toneprint import identify_tones
from src.formula_bank import (
    build_equation_string,
    fit_best_formula_for_segment,
)
from src.math_track_engine import Project, Track, MathTrackEngine
from src.math_track_visualize import plot_project_timeline, save_timeline_player
from src.error_correction import correct_reconstruction, plot_ab_comparison
from src.sinusoidal_model import sinusoidal_reconstruction
from src.residual import compute_error_metrics


def _formula_to_track_equation(match: Dict[str, Any]) -> str:
    """Translate a fitted formula-bank match into a Math-DAW track equation."""
    A = float(np.clip(match.get("A", 0.5), 0.05, 1.0))
    lam = float(np.clip(match.get("decay_lambda", 1.0), 0.0, 12.0))
    p1 = float(match.get("timbre_param", 1.0))
    f0 = float(match.get("f0", 440.0))
    return build_equation_string(
        match.get("formula_type", "pure_sine"),
        f0=f0,
        A=A,
        decay_lambda=lam,
        timbre_param=p1,
        include_gate=True,
        carrier="ph",
    )


def _quantize_hit_to_pattern(
    start: float, end: float, midi: int, step_times: np.ndarray, beat: float
) -> List[Optional[int]]:
    pattern: List[Optional[int]] = []
    for s in step_times:
        step_end = s + beat
        active = (s < end) and (step_end > start)
        pattern.append(midi if active else None)
    return pattern


def _sms_equations_text(result: Dict[str, Any], sr: int) -> str:
    """Render the SMS model as plain-text math equations (no audio).

    Stationary partials get an exact closed form - amplitude, decay rate,
    frequency and phase were all fitted against the reference, so the numbers
    reproduce what the synthesizer rendered (fidelity metrics in the header).
    Glide partials keep the integral form, with their trajectory in the JSON.
    """
    out: List[str] = []
    out.append("# SMS equation model")
    out.append("# s(t) = sum_k A_k(t) * cos(theta_k(t)),  theta_k(t) = int 2*pi*f_k(t) dt")
    out.append(f"# sr = {sr},  SNR = {result['snr_db']:.2f} dB,  "
               f"correlation = {result['correlation']:.4f},  "
               f"tracks = {result['track_count']} "
               f"(of {result['candidate_tracks']} candidates)")
    out.append("")
    n_const = 0
    for i, p in enumerate(result["fitted_parameters"]):
        tag = f"part_{i:03d}"
        if p.get("model") == "glide":
            traj = p.get("trajectory", {})
            n_f = len(traj.get("f_hz", []))
            out.append(
                f"{tag}: GLIDE  theta(t) = int 2*pi*f(t) dt + c*t, "
                f"c = 2*pi*({p.get('frequency', 0.0):.3f} - mean(f)), "
                f"A(t) from trajectory  [t = {p.get('start_time', 0.0):.3f}"
                f"..{p.get('end_time', 0.0):.3f} s, {n_f} frames]"
            )
        else:
            n_const += 1
            amp = float(p.get("amplitude", 0.0))
            dec = float(p.get("envelope_decay", 0.0))
            t0 = float(p.get("envelope_t0", 0.0))
            f0 = float(p.get("frequency", 0.0))
            ph = float(p.get("phase", 0.0))
            t_start = float(p.get("start_time", 0.0))
            t_end = float(p.get("end_time", 0.0))
            out.append(
                f"{tag}: {amp:.4f} * exp(-{dec:.4f}*(t - {t0:.4f}))"
                f" * cos(2*pi*{f0:.4f}*t - {ph:.4f})"
                f"   for t = {t_start:.3f}..{t_end:.3f} s"
            )
    if not result["fitted_parameters"]:
        out.append("(no sinusoidal partials survived residual-threshold pruning)")
    return "\n".join(out)


def recreate_song_from_sinusoids(
    input_audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    max_duration: float = 20.0,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 256,
    threshold: float = 0.01,
    max_tracks: int = 600,
    add_noise_layer: bool = True,
    equations_only: bool = False,
    min_residual_ratio: float = 1e-4,
    window_s: float = 0.0,
) -> Dict[str, Any]:
    """Faithful song -> equations -> song reconstruction via sinusoidal modeling.

    This is the high-fidelity path. Unlike the tone-based recreation, which keeps
    only a median frequency and one phase per note, this retains the full
    frequency and amplitude trajectories of every sinusoidal partial and rebuilds
    phase by integration, so stationary content is reconstructed essentially
    exactly. Achieved SNR and correlation are returned so fidelity is measured,
    never assumed.

    With ``equations_only`` the WAV/plot generation is skipped and the output is
    just the equations: a ``*_sms_equations.txt`` with closed-form equations per
    partial plus a ``*_sms_model.json`` with the full parameter set.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(str(input_audio_path)).stem.replace(" ", "_")

    y, _ = load_audio(input_audio_path, target_sr=sr)
    y = y[: int(max_duration * sr)]

    result = sinusoidal_reconstruction(
        y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        threshold=threshold,
        max_tracks=max_tracks,
        add_noise_layer=add_noise_layer and not equations_only,
        min_residual_ratio=min_residual_ratio,
        window_s=window_s,
    )

    model_path = out_path / f"{stem}_sms_model.json"
    payload = {
        "model": "sinusoidal-modeling-synthesis",
        "sample_rate": sr,
        "equation": "s(t) = sum_k A_k(t) * sin(theta_k(t)),  "
                    "theta_k(t) = int 2*pi*f_k(t) dt",
        "snr_db": result["snr_db"],
        "correlation": result["correlation"],
        "track_count": result["track_count"],
        "candidate_tracks": result["candidate_tracks"],
        "pruned_tracks": result["pruned_tracks"],
        "noise_layer": result["noise_layer"],
        "tracks": result["model"],
        "fitted_parameters": result["fitted_parameters"],
    }

    if equations_only:
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        txt_path = out_path / f"{stem}_sms_equations.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(_sms_equations_text(result, sr) + "\n")
        return {
            "status": "success",
            "equations_only": True,
            "duration": round(len(y) / sr, 3),
            "sample_rate": sr,
            "snr_db": result["snr_db"],
            "correlation": result["correlation"],
            "track_count": result["track_count"],
            "candidate_tracks": result["candidate_tracks"],
            "pruned_tracks": result["pruned_tracks"],
            "fitted_parameters": result["fitted_parameters"],
            "equations_txt": str(txt_path),
            "model_json": str(model_path),
        }

    recon = result["audio"]

    wav_path = out_path / f"{stem}_sms.wav"
    sf.write(str(wav_path), np.clip(recon, -1.0, 1.0), sr)

    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_path = out_path / f"{stem}_sms_comparison.png"
    # There is no before/after correction pair on this path: the SMS pass is
    # single-stage. Fabricating a "before" of zeros would print a fake
    # improvement on the plot, so the measured numbers go in the title instead
    # and `metrics_before`/`metrics_after` stay unset.
    plot_ab_comparison(
        y,
        recon,
        recon,
        sr=sr,
        out_path=plot_path,
        title=(f"{stem} - sinusoidal reconstruction "
               f"(SNR {result['snr_db']:.1f} dB, corr {result['correlation']:.4f})"),
    )

    return {
        "status": "success",
        "duration": round(len(y) / sr, 3),
        "sample_rate": sr,
        "snr_db": result["snr_db"],
        "correlation": result["correlation"],
        "track_count": result["track_count"],
        "candidate_tracks": result["candidate_tracks"],
        "pruned_tracks": result["pruned_tracks"],
        "noise_layer": result["noise_layer"].get("construction"),
        "remake_wav": str(wav_path),
        "model_json": str(model_path),
        "comparison_plot": str(plot_path),
    }


def recreate_song_from_tones(
    input_audio_path: Union[str, Path],
    output_dir: Union[str, Path],
    max_duration: float = 16.0,
    sr: int = 22050,
    segment: float = 0.25,
    max_tone_tracks: int = 8,
    refine: bool = True,
    refine_rounds: int = 3,
    refine_terms: int = 4,
) -> Dict[str, Any]:
    """Identify tones in a song and recreate it as Math-DAW equation tracks.

    When ``refine`` is enabled the rendered recreation is passed through
    closed-loop error correction (lag alignment, residual correction layers,
    least-squares level refit) so the result tracks the original far more
    closely than the raw tone-only recreation.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    stem = Path(str(input_audio_path)).stem.replace(" ", "_")

    y, _ = load_audio(input_audio_path, target_sr=sr)
    if len(y) > int(max_duration * sr):
        y = y[: int(max_duration * sr)]
    duration = float(len(y) / sr)

    beats = detect_beats(y, sr=sr)
    tempo = float(beats["tempo"])
    bpm = tempo if 60.0 <= tempo <= 200.0 else 120.0
    beat = 60.0 / bpm
    n_steps = max(4, int(round(duration / beat)))
    step_times = np.arange(n_steps) * beat

    tone_hits = identify_tones(y, sr=sr, segment=segment)
    tone_hits = sorted(tone_hits, key=lambda h: h.duration, reverse=True)[:max_tone_tracks]

    tracks: List[Track] = []
    tone_records: List[Dict[str, Any]] = []
    for i, hit in enumerate(tone_hits):
        s0 = max(0, int((hit.start_time - 0.05) * sr))
        s1 = min(len(y), int((hit.end_time + 0.05) * sr))
        fit_end = min(s1, s0 + int(1.5 * sr))
        seg = y[s0:fit_end] if fit_end - s0 >= 64 else y[s0:s1]
        match = fit_best_formula_for_segment(seg, sr, float(hit.pitch_hz))
        if match is None:
            match = {
                "formula_type": "pure_sine", "A": 0.5,
                "decay_lambda": 1.0, "timbre_param": 1.0, "spectral_mse": 1.0,
            }
        equation = _formula_to_track_equation(match)
        pattern = _quantize_hit_to_pattern(hit.start_time, hit.end_time, hit.pitch_midi, step_times, beat)
        tracks.append(Track(
            name=f"Tone_{hit.pitch_midi}_{i}",
            equation=equation,
            pattern=pattern,
            gain=0.9,
            pan=0.0,
        ))
        tone_records.append({
            "pitch_midi": hit.pitch_midi,
            "pitch_hz": round(hit.pitch_hz, 2),
            "start_time": round(hit.start_time, 3),
            "end_time": round(hit.end_time, 3),
            "formula": match["formula_type"],
            "A": round(float(np.clip(match.get("A", 0.5), 0.05, 1.0)), 3),
            "decay_lambda": round(float(np.clip(match.get("decay_lambda", 1.0), 0.0, 12.0)), 3),
            "timbre_param": round(float(match.get("timbre_param", 1.0)), 3),
            "equation": equation,
        })

    # Percussive onset track from detected onsets quantized to the beat grid.
    onset_steps = set()
    for ot in np.asarray(beats["onset_times"]).ravel().tolist():
        idx = int(np.clip(round(float(ot) / beat), 0, n_steps - 1))
        onset_steps.add(idx)
    if len(onset_steps) >= 2:
        hat_pattern = [42 if s in onset_steps else None for s in range(n_steps)]
        tracks.append(Track(
            name="Perc_Onsets",
            equation="g * exp(-22.0 * u) * tanh(4.0 * rnd * sin(2.0 * pi * 8500.0 * t))",
            pattern=hat_pattern,
            gain=0.4,
            pan=0.2,
        ))

    project = Project(
        name=f"{stem} (Tone-Equation Recreation)",
        bpm=bpm,
        duration=duration,
        sr=sr,
        tracks=tracks,
        master_gain=0.95,
        seed=7,
    )
    engine = MathTrackEngine(sr=sr)
    rendered = engine.render_project(project)
    y_synth = np.asarray(rendered["mix"], dtype=np.float64)

    wav_path = out_path / f"{stem}_tone_remake.wav"
    sf.write(str(wav_path), y_synth, sr)
    project_path = out_path / f"{stem}_tone_project.json"
    project.save(project_path)
    tones_path = out_path / f"{stem}_tones.json"
    with open(tones_path, "w", encoding="utf-8") as f:
        json.dump(tone_records, f, indent=2)
    bytebeat_code = tone_records_to_bytebeat(
        tone_records, bpm, beat, onset_steps, n_steps
    )
    bytebeat_path = out_path / f"{stem}_bytebeat.txt"
    with open(bytebeat_path, "w", encoding="utf-8") as f:
        f.write("// Paste into dollchan bytebeat in floatbeat mode\n")
        f.write(bytebeat_code + "\n")

    min_len = min(len(y), len(y_synth))
    metrics = compute_error_metrics(y[:min_len], y_synth[:min_len], sr=sr)

    correction_report: Optional[Dict[str, Any]] = None
    y_corrected = y_synth
    if refine:
        n_beats = min(len(step_times), 96)
        seg_indices = [
            (int(step_times[i] * sr), int((step_times[i] + beat) * sr))
            for i in range(n_beats)
        ]
        correction_report = correct_reconstruction(
            y,
            y_synth,
            sr=sr,
            rounds=int(refine_rounds),
            terms_per_round=int(refine_terms),
            segments=seg_indices,
        )
        y_corrected = correction_report.pop("corrected_audio")
        residual_audio = correction_report.pop("residual_audio")
        level_gain = float(correction_report.get("initial_level_gain", 1.0) or 1.0)
        # Save the raw recreation at matched level so the A/B comparison is fair
        # instead of showing a loud/quiet mismatch.
        y_synth = y_corrected * (1.0 / level_gain) if level_gain > 1e-6 else y_synth
    else:
        residual_audio = y[:min_len] - y_synth[:min_len]

    if len(y_corrected) < min_len:
        y_corrected = np.pad(y_corrected, (0, min_len - len(y_corrected)))
    corrected_metrics = compute_error_metrics(
        y[:min_len], y_corrected[:min_len], sr=sr
    )

    corrected_wav_path = out_path / f"{stem}_tone_corrected.wav"
    sf.write(str(corrected_wav_path), y_corrected[:min_len], sr)
    residual_wav_path = out_path / f"{stem}_error.wav"
    sf.write(str(residual_wav_path), residual_audio[:min_len], sr)

    plot_path = out_path / f"{stem}_tone_comparison.png"
    ab_plot_path = out_path / f"{stem}_ab_comparison.png"
    plot_ab_comparison(
        y,
        y_synth,
        y_corrected,
        sr=sr,
        out_path=ab_plot_path,
        title=f"{stem} - original vs recreation vs corrected",
        metrics_before=correction_report["metrics_before"] if correction_report else None,
        metrics_after=correction_report["metrics_after"] if correction_report else None,
    )
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    view = min(len(y), sr * 4)
    t_axis = np.arange(view) / sr
    axes[0].plot(t_axis, y[:view], lw=0.6, label="Original")
    axes[0].plot(t_axis, y_synth[:view], lw=0.6, alpha=0.7, label="Tone equations")
    axes[0].set_title(f"Tone-Equation Remake (BPM={bpm:.1f}, tones={len(tone_records)})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_axis, (y[:view] - y_synth[:view]), lw=0.6, color="#e74c3c")
    axes[1].set_title("Residual (first 4s)")
    axes[1].grid(True, alpha=0.3)
    fig.savefig(str(plot_path), dpi=120)
    plt.close(fig)

    timeline_path = out_path / f"{stem}_timeline.png"
    plot_project_timeline(project, timeline_path)
    player_path = out_path / f"{stem}_timeline_player.html"
    save_timeline_player(project, y_synth, sr, player_path)

    return {
        "status": "success",
        "bpm": round(bpm, 2),
        "duration": round(duration, 2),
        "tone_count": len(tone_records),
        "track_count": len(tracks),
        "tracks": [t.name for t in tracks],
        "tones": tone_records,
        "refined": bool(refine),
        "correction": correction_report,
        "correlation": round(float(np.corrcoef(y[:min_len], y_synth[:min_len])[0, 1]), 4),
        "spectral_mse": round(float(metrics["spectral_mse"]), 6),
        "corrected_correlation": round(float(np.corrcoef(y[:min_len], y_corrected[:min_len])[0, 1]), 4),
        "corrected_spectral_mse": round(float(corrected_metrics["spectral_mse"]), 8),
        "project_json": str(project_path),
        "tones_json": str(tones_path),
        "bytebeat_txt": str(bytebeat_path),
        "remake_wav": str(wav_path),
        "corrected_wav": str(corrected_wav_path),
        "error_wav": str(residual_wav_path),
        "comparison_plot": str(plot_path),
        "ab_comparison_plot": str(ab_plot_path),
        "timeline_plot": str(timeline_path),
        "timeline_player": str(player_path),
    }


def tone_records_to_bytebeat(
    tones: List[Dict[str, Any]],
    bpm: float,
    beat: float,
    onset_steps: Optional[set] = None,
    n_steps: int = 0,
    sr_out: int = 8000,
) -> str:
    """Convert derived tone records into a dollchan floatbeat function.

    Each tone becomes one gated equation layer (like the video's ``math``,
    ``math #2`` tracks); detected onsets become a noise-hat layer. Paste the
    result into a bytebeat player in floatbeat mode.
    """
    def js_timbre(rec: Dict[str, Any]) -> str:
        kind = rec.get("formula", "pure_sine")
        A = rec.get("A", 0.5)
        lam = rec.get("decay_lambda", 1.0)
        p1 = rec.get("timbre_param", 1.0)
        f = rec.get("pitch_hz", 440.0)
        env = f"{A}*Math.exp(-{lam}*D)"
        if kind == "harmonic_power_saw":
            terms = "+".join(
                f"{1.0 / (h ** max(0.3, p1)):.3f}*Math.sin(6.2832*{h}*{f}*D)"
                for h in range(1, 5)
            )
            return f"{env}*({terms})"
        if kind == "soft_square_tanh":
            k = max(0.4, p1 * 3.0)
            return f"{env}*Math.tanh({k:.3f}*Math.sin(6.2832*{f}*D))"
        if kind == "fm_synthesis_bell":
            mod_index = p1 * 2.5
            return (
                f"{env}*Math.sin(6.2832*{f}*D+{mod_index:.3f}"
                f"*Math.exp(-2*{lam}*D)*Math.sin(12.5664*{f}*D))"
            )
        if kind == "damped_harmonic_series":
            terms = "+".join(
                f"{1.0 / (h ** max(0.4, p1)):.3f}*Math.sin(6.2832*{h * np.sqrt(1.0 + 0.0004 * h * h):.4f}*{f}*D)"
                for h in range(1, 5)
            )
            return f"{env}*({terms})"
        if kind in ("arctan_cot_saw", "rich_sawtooth_arctan"):
            return (
                f"(2*{A}/3.1416)*Math.exp(-{lam}*D)*Math.atan(Math.cos(3.1416*{f}*D)"
                f"/(Math.sin(3.1416*{f}*D)+1e-9))"
            )
        return f"{env}*Math.sin(6.2832*{f}*D)"

    lines = [f"S={sr_out},T=t/S"]
    layers: List[str] = []
    for rec in tones:
        s, e = rec["start_time"], rec["end_time"]
        layers.append(f"((T>={s}&&T<{e})?(D=T-{s},{js_timbre(rec)}):0)")
    if onset_steps and n_steps > 0:
        grid = ",".join("1" if s in onset_steps else "0" for s in range(n_steps))
        layers.append(
            f"(HP=[{grid}],HP[((T/{beat:.4f})|0)%{n_steps}]"
            f"*((t*15923%997)/997-.5)*Math.exp(-(T%{beat:.4f})*22)*0.5)"
        )
    # Normalize by peak simultaneous layers, not total layers.
    events: List[Tuple[float, int]] = []
    for rec in tones:
        events.append((rec["start_time"], 1))
        events.append((rec["end_time"], -1))
    concurrent = peak = 0
    for _, delta in sorted(events):
        concurrent = max(0, concurrent + delta)
        peak = max(peak, concurrent)
    norm = max(1, peak + (1 if (onset_steps and n_steps > 0) else 0))
    lines.append(f"({'+'.join(layers)})/{norm}")
    return ",\n".join(lines)


__all__ = ["recreate_song_from_tones", "tone_records_to_bytebeat"]
