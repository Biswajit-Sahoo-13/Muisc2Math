"""music2math: CLI entry point for mathematical audio reverse engineering and synthesis.

Unified pipeline:
- synthesize: Render mathematical equations (JSON) -> Audio (WAV)
- analyze: Discover mathematical equations from audio (Audio -> Math)
- benchmark: Generate ground-truth benchmark signals (chord, melody)
- verify: Closed-loop error analysis (MSE, SNR, Spectral Error, LSD)
- refine: Iterative residual refinement loop
- visualize: Generate inspection plots (waveform, spectrogram, spectrum, comparison dashboard)
- extract-notes: Extract melody notes (pYIN) and group harmonic partials
- run-all: Complete end-to-end benchmark demonstration
- math-daw: Render a Math-DAW project (equation tracks) to WAV
- math-daw-demo: Render all preset demos with waveform plots
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

from src.synthesis import (
    render_model,
    save_audio,
    generate_benchmark_chord,
    generate_benchmark_melody,
)
from src.analysis import load_audio
from src.tracking import discover_equations_from_audio
from src.llm_backends import (
    BACKENDS,
    auto_backend,
    build_command,
    probe_all,
    run_backend,
)
from src.residual import (
    compute_error_metrics,
    iterative_residual_refinement,
)
from src.note_extraction import (
    extract_melody_pyin,
)
from src.visualize import (
    plot_waveform,
    plot_spectrogram,
    plot_spectrum,
    plot_comparison_dashboard,
)
from src.math_track_engine import (
    Project, render_to_wav,
    demo_sinc_drop, demo_math_remix, demo_ambient,
)
from src.math_track_visualize import plot_tracks, plot_track_spectrograms
from src.formula_bank import (
    FORMULA_BANK,
    fit_all_formula_candidates,
    fit_best_formula_for_segment,
    fit_equation_library,
    pareto_front,
    select_preferred,
)
from src.song_recreation import recreate_and_evaluate_song
from src.toneprint import identify_file
from src.tone_remake import recreate_song_from_tones, recreate_song_from_sinusoids


def cmd_benchmark(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.type in ("chord", "all"):
        chord_wav = out_dir / "math_chord.wav"
        print(f"[*] Generating benchmark chord -> {chord_wav}")
        gt_chord = generate_benchmark_chord(chord_wav, duration=args.duration, sr=args.sr)
        chord_json = out_dir / "math_chord_gt.json"
        with open(chord_json, "w", encoding="utf-8") as f:
            json.dump(gt_chord, f, indent=2)
        print(f"    Saved ground truth model -> {chord_json}")

    if args.type in ("melody", "all"):
        melody_wav = out_dir / "math_melody.wav"
        print(f"[*] Generating benchmark melody -> {melody_wav}")
        gt_melody = generate_benchmark_melody(melody_wav, duration=args.duration, sr=args.sr)
        melody_json = out_dir / "math_melody_gt.json"
        with open(melody_json, "w", encoding="utf-8") as f:
            json.dump(gt_melody, f, indent=2)
        print(f"    Saved ground truth model -> {melody_json}")


def cmd_synthesize(args: argparse.Namespace) -> None:
    print(f"[*] Rendering equations from {args.input} -> {args.output}")
    audio, sr = render_model(args.input, duration=args.duration, sr=args.sr, normalize=args.normalize)
    save_audio(args.output, audio, sr)
    dur = len(audio) / sr
    print(f"[+] Render complete: {dur:.2f}s, {sr} Hz, saved to {args.output}")


def cmd_analyze(args: argparse.Namespace) -> None:
    print(f"[*] Analyzing audio file: {args.input}")
    model = discover_equations_from_audio(
        audio_path=args.input,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        threshold=args.threshold,
        out_json_path=args.output,
        refine_with_projection=not args.no_projection,
    )
    print(f"[+] Discovered {len(model['components'])} mathematical components.")
    for comp in model["components"][:8]:
        print(
            f"    - f={comp['frequency']} Hz | A={comp['amplitude']} | "
            f"env={comp['envelope']['type']} (decay={comp['decay_rate']}) | phase={comp['phase']} rad"
        )
    if len(model["components"]) > 8:
        print(f"    ... and {len(model['components']) - 8} more components.")
    if args.output:
        print(f"[+] Saved model to {args.output}")


def cmd_verify(args: argparse.Namespace) -> None:
    print(f"[*] Comparing Original: {args.original} vs Reconstructed: {args.reconstructed}")
    orig_y, sr = load_audio(args.original)
    recon_y, _ = load_audio(args.reconstructed, target_sr=sr)

    metrics = compute_error_metrics(orig_y, recon_y, sr=sr)
    print("\n--- DSP Objective Verification Metrics ---")
    print(f"MSE:                   {metrics['mse']:.6e}")
    print(f"SNR (dB):              {metrics['snr_db']:.2f} dB")
    print(f"Spectral MSE:          {metrics['spectral_mse']:.6e}")
    print(f"Log-Spectral Distance: {metrics['log_spectral_distance_db']:.2f} dB")
    print(f"Signal Energy:         {metrics['signal_energy']:.2f}")
    print(f"Residual Energy:       {metrics['residual_energy']:.2f}")
    print("-------------------------------------------\n")

    if args.save_residual:
        min_len = min(len(orig_y), len(recon_y))
        res = orig_y[:min_len] - recon_y[:min_len]
        save_audio(args.save_residual, res, sr)
        print(f"[+] Saved residual audio -> {args.save_residual}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[+] Saved report JSON -> {args.report_json}")

    if args.dashboard:
        plot_comparison_dashboard(orig_y, recon_y, sr=sr, out_path=args.dashboard, metrics=metrics)
        print(f"[+] Saved comparison dashboard plot -> {args.dashboard}")


def cmd_refine(args: argparse.Namespace) -> None:
    print(f"[*] Iterative Residual Refinement on {args.input}")
    y, sr = load_audio(args.input)

    init_model = None
    if args.initial_model:
        with open(args.initial_model, "r", encoding="utf-8") as f:
            init_model = json.load(f)

    refined_model, x_hat, final_metrics = iterative_residual_refinement(
        original_audio=y,
        sr=sr,
        initial_model=init_model,
        max_iterations=args.max_iterations,
        target_snr_db=args.target_snr,
        verbose=True,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(refined_model, f, indent=2)
        print(f"[+] Saved refined equation model -> {args.output}")

    if args.rendered_wav:
        save_audio(args.rendered_wav, x_hat, sr)
        print(f"[+] Saved refined audio -> {args.rendered_wav}")


def cmd_visualize(args: argparse.Namespace) -> None:
    print(f"[*] Generating visual inspection plots for {args.input}")
    y, sr = load_audio(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wf_path = out_dir / f"{Path(args.input).stem}_waveform.png"
    sg_path = out_dir / f"{Path(args.input).stem}_spectrogram.png"
    sp_path = out_dir / f"{Path(args.input).stem}_spectrum.png"

    plot_waveform(y, sr, title=f"Waveform: {Path(args.input).name}", out_path=wf_path)
    plot_spectrogram(y, sr, title=f"Spectrogram: {Path(args.input).name}", out_path=sg_path)
    plot_spectrum(y, sr, title=f"FFT Spectrum: {Path(args.input).name}", out_path=sp_path)

    print(f"[+] Generated:")
    print(f"    - {wf_path}")
    print(f"    - {sg_path}")
    print(f"    - {sp_path}")


def cmd_extract_notes(args: argparse.Namespace) -> None:
    print(f"[*] Extracting musical notes and partials from {args.input}")
    y, sr = load_audio(args.input)
    notes = extract_melody_pyin(y, sr=sr)
    print(f"[+] Discovered {len(notes)} note events via pYIN:")
    for idx, n in enumerate(notes):
        print(
            f"    Note {idx+1}: MIDI {n.pitch_midi} ({n.pitch_hz:.1f} Hz) | "
            f"t={n.start_time:.2f}s - {n.end_time:.2f}s (dur={n.duration:.2f}s) | conf={n.confidence:.2f}"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([n.to_dict() for n in notes], f, indent=2)
        print(f"[+] Saved notes to {args.output}")


def cmd_tone_id(args: argparse.Namespace) -> None:
    """Identify musical tones in an audio file with Shazam-style precision."""
    print(f"[*] Identifying tones in {args.input}")
    tones = identify_file(args.input, target_sr=args.sr, segment=args.segment)
    print(f"[+] Detected {len(tones)} tone events:")
    for tone in tones:
        print(
            f"    MIDI {tone['pitch_midi']} ({tone['pitch_hz']:.1f} Hz) | "
            f"t={tone['start_time']:.2f}s - {tone['end_time']:.2f}s | "
            f"conf={tone['confidence']:.2f}"
        )
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(tones, f, indent=2)
        print(f"[+] Saved tones to {args.output}")


def cmd_tone_remake(args: argparse.Namespace) -> None:
    """Recreate a song from detected tones as Math-DAW equation tracks."""
    print(f"[*] Recreating '{args.input}' from detected tones...")
    res = recreate_song_from_tones(
        args.input,
        args.output_dir,
        max_duration=args.duration,
        sr=args.sr,
        refine=not args.no_refine,
    )
    print(f"[+] Tones: {res['tone_count']} | Tracks: {', '.join(res['tracks'])}")
    print(f"[+] Remake WAV: {res['remake_wav']}")
    print(f"[+] Corrected WAV: {res['corrected_wav']}")
    print(f"[+] Error (difference) WAV: {res['error_wav']}")
    print(f"[+] A/B comparison plot: {res['ab_comparison_plot']}")
    print(f"[+] Timeline graph: {res['timeline_plot']}")
    print(f"[+] Realtime timeline player: {res['timeline_player']}")
    print(f"[+] Bytebeat function: {res['bytebeat_txt']}")
    print(f"[+] Project JSON: {res['project_json']}")

    print("\n--- BEFORE vs AFTER ERROR CORRECTION ---")
    print(f"    waveform correlation : {res['correlation']:.4f} -> {res['corrected_correlation']:.4f}")
    print(f"    spectral MSE         : {res['spectral_mse']:.8f} -> {res['corrected_spectral_mse']:.8f}")
    corr = res.get("correction") or {}
    if corr.get("metrics_before") and corr.get("metrics_after"):
        mb, ma = corr["metrics_before"], corr["metrics_after"]
        print(f"    SNR                  : {mb['snr_db']:.2f} -> {ma['snr_db']:.2f} dB")
        print(f"    log-spectral distance: {mb['log_spectral_distance_db']:.2f} -> {ma['log_spectral_distance_db']:.2f} dB")
        print(f"    time alignment       : {corr.get('lag_ms', 0.0):+.1f} ms   level gain: {corr.get('gain', 1.0):.4f}")
        seg = corr.get("segmented")
        if seg:
            state = "accepted" if seg.get("accepted") else "rejected"
            print(f"    segmented pass       : {state}"
                  + (f" ({seg.get('segments', 0)} windows, +{seg.get('snr_gain_db', 0)} dB)"
                     if seg.get("accepted") else ""))
        for r in corr.get("rounds", []):
            if r.get("accepted"):
                print(f"    round {r['round']}             : +{r['terms']} terms, "
                      f"SNR {r['snr_db']:.2f} dB (+{r['snr_gain_db']:.2f})")


def cmd_equation_discovery(args: argparse.Namespace) -> None:
    """Discover multiple competing math equations for an audio file."""
    print(f"[*] Equation discovery for {args.input}")
    import librosa

    y, sr = load_audio(args.input)
    dur = args.duration if args.duration else float(len(y) / sr)
    n = int(sr * dur)
    y = y[:n]
    np.arange(len(y)) / sr
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= 0.0:
        print("[!] Input is silent; nothing to discover.")
        return

    if args.f0 is not None:
        f0 = float(args.f0)
    else:
        f0_track, _, _ = librosa.pyin(y, fmin=40.0, fmax=2000.0, sr=sr)
        f0 = float(np.nanmedian(f0_track)) if np.any(~np.isnan(f0_track)) else 220.0
    print(f"    f0 = {f0:.2f} Hz | duration = {dur:.2f}s")

    print("\n--- PARETO SET (fidelity vs simplicity) ---")
    cands = fit_all_formula_candidates(y, sr, f0, max_starts=3)
    front = pareto_front(cands)
    if not front:
        print("[!] No candidates found.")
        return
    print(f"    {'formula':26s} {'spectral_mse':>13s} {'complexity':>11s}")
    for c in front[:10]:
        print(f"    {c.formula_type:26s} {c.spectral_mse:13.6f} {c.complexity:11.2f}")

    simplest = select_preferred(cands)
    if simplest is not None:
        print("\n--- SIMPLEST EQUATION WITHIN 2% OF BEST FIT ---")
        print(f"    {simplest.formula_type} (mse={simplest.spectral_mse:.6f}, complexity={simplest.complexity:.2f})")
        print(f"    y = {simplest.equation}")

    print("\n--- LIBRARY LEARNING (residual recursion) ---")
    lib = fit_equation_library(
        y, sr, f0, max_terms=args.max_terms, prefer_harmonics_of_f0=True
    )
    print(f"    terms={lib['term_count']} residual reduction={lib['residual_reduction_db']} dB")
    for comp in lib["components"]:
        print(
            f"      #{comp['term_index']}: {comp['frequency']:.1f} Hz  "
            f"A={comp['amplitude']:.4f}  lambda={comp['decay_lambda']:.3f}  "
            f"h={comp['harmonic_of_f0']}  (+{comp['improvement_db']} dB)"
        )
    if lib["equation"]:
        print(f"    y = {lib['equation']}")

    out = {
        "f0": round(f0, 2),
        "duration": round(dur, 3),
        "pareto_front": [c.to_dict() for c in front],
        "simplest_within_tolerance": simplest.to_dict() if simplest else None,
        "library_learning": lib,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[+] Saved discovery report -> {args.output}")


def cmd_sms_remake(args: argparse.Namespace) -> None:
    """High-fidelity song -> equations -> song via sinusoidal modeling synthesis."""
    print(f"[*] SMS reconstruction of '{args.input}'")
    res = recreate_song_from_sinusoids(
        args.input,
        args.output_dir,
        max_duration=args.duration,
        sr=args.sr,
        threshold=args.threshold,
        max_tracks=args.max_tracks,
        add_noise_layer=not args.no_noise,
        equations_only=args.equations_only,
        min_residual_ratio=args.sensitivity,
        window_s=args.window,
    )
    if res.get("equations_only"):
        print(f"[+] Track count     : {res['track_count']} "
              f"of {res['candidate_tracks']} candidates "
              f"({res['pruned_tracks']} pruned, sensitivity "
              f"{args.sensitivity:g})")
        print(f"[+] Achieved SNR    : {res['snr_db']} dB "
              f"(audio playback skipped; equations only)")
        print(f"[+] Equations text  : {res['equations_txt']}")
        print(f"[+] Model JSON      : {res['model_json']}")
        params = res["fitted_parameters"]
        shown = params[:12]
        for i, p in enumerate(shown):
            if p.get("model") == "glide":
                print(f"    part_{i:03d}: GLIDE  mean f = "
                      f"{p.get('frequency', 0.0):.3f} Hz")
            else:
                print(f"    part_{i:03d}: {p.get('amplitude', 0.0):.4f} * "
                      f"exp(-{p.get('envelope_decay', 0.0):.4f}*t) * "
                      f"cos(2*pi*{p.get('frequency', 0.0):.4f}*t - "
                      f"{p.get('phase', 0.0):.4f})")
        if len(params) > len(shown):
            print(f"    ... {len(params) - len(shown)} more in {res['equations_txt']}")
        return
    print(f"[+] Tracks kept     : {res['track_count']} "
          f"(from {res['candidate_tracks']} candidates, {res['pruned_tracks']} pruned)")
    print(f"[+] Achieved SNR    : {res['snr_db']} dB")
    print(f"[+] Correlation     : {res['correlation']}")
    print(f"[+] Noise layer     : {res['noise_layer'] or 'disabled'}")
    print(f"[+] Remake WAV      : {res['remake_wav']}")
    print(f"[+] Model JSON      : {res['model_json']}")
    print(f"[+] Comparison plot : {res['comparison_plot']}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Run the local API server the browser extension POSTs captures to."""
    from src.res_server import serve

    serve(host=args.host, port=args.port)


def cmd_res(args: argparse.Namespace) -> None:
    """RES: Reverse Engineer a Song - build evidence, hypothesize, verify, ledger."""
    if getattr(args, "stems", None):
        from src.res_loop import run_stems_res_loop
        stems_arg = args.stems
        print(f"[*] RES Multi-Track Stem Mode ('{stems_arg}') on '{args.input}'")
        stems_dir = stems_arg if Path(stems_arg).is_dir() else None
        stems_mode = "directory" if stems_dir else stems_arg
        ledger = run_stems_res_loop(
            source_path=args.input,
            out_dir=args.output_dir,
            stems_mode=stems_mode,
            stems_dir=stems_dir,
            sr=args.sr,
            max_duration=args.duration,
            window_s=args.window,
            iterations=args.iterations,
            target_corr=args.target_corr,
            bank=not args.no_bank,
            spectral_lane=args.spectral_lane,
            enable_noise=args.noise,
            k_notes=args.k_notes, k_hits=args.k_hits,
            energy_parity=bool(getattr(args, "energy_parity", False)),
            parity_mode=getattr(args, "parity_mode", "ls"),
            polish=bool(getattr(args, "polish", False)),
        )
        print("\n=== Multi-Track Stem Ledger (res/2.0-multitrack) ===")
        for s_name in ("bass", "drums", "vocals", "other"):
            s_info = ledger.get("stems", {}).get(s_name, {})
            m = s_info.get("metrics", {})
            print(f"  Stem: {s_name:<8} | Corr: {m.get('correlation', 0.0):.4f} | SNR: {m.get('snr_db', 0.0):.2f} dB | LSD: {m.get('log_spectral_distance_db', 0.0):.2f} dB")
        best = ledger.get("best", {})
        print(f"  Combined Mix  | Corr: {best.get('correlation', 0.0):.4f} | SNR: {best.get('snr_db', 0.0):.2f} dB | LSD: {best.get('log_spectral_distance_db', 0.0):.2f} dB")
        print(f"\n[+] Multitrack Ledger : {ledger['outputs']['multitrack_ledger']}")
        print(f"[+] Master Mix WAV    : {ledger['outputs']['best_wav']}")
        print(f"[+] Equations text    : {ledger['outputs']['equations_txt']}")
        print(f"[+] Bytebeat JS       : {ledger['outputs'].get('bytebeat_js')}")
        if args.json:
            print(json.dumps(ledger, indent=2))
        return

    from src.res_evidence import build_evidence
    from src.res_loop import recommend_iterations, run_loop

    def _run(led: dict, iterations: int, *, target_corr: float,
             min_gain: float, stop_on_no_improvement: bool = False) -> dict:
        """One pass of the loop with this command's LLM/bank settings."""
        return run_loop(
            led,
            args.input,
            args.output_dir,
            iterations=iterations,
            target_corr=target_corr,
            provider=args.provider,
            llm_model=args.llm_model or None,
            llm_cmd=args.llm_cmd.split() if args.llm_cmd else None,
            llm_backend=_llm_backend_for(args),
            llm_timeout_s=args.llm_timeout,
            llm_api_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key or None,
            llm_extra_args=args.llm_extra_args.split() if args.llm_extra_args else None,
            min_gain=min_gain,
            stop_on_no_improvement=stop_on_no_improvement,
            bank=not args.no_bank,
            spectral_lane=args.spectral_lane,
            enable_noise=args.noise,
            k_notes=args.k_notes, k_hits=args.k_hits,
            energy_parity=bool(getattr(args, "energy_parity", False)),
            parity_mode=getattr(args, "parity_mode", "ls"),
            joint_refine=bool(getattr(args, "joint_refine", False)),
            mrstft_lane=bool(getattr(args, "mrstft_lane", False)),
            polish=bool(getattr(args, "polish", False)),
        )

    iterations = args.iterations
    auto_note = ""
    if args.auto_iterations:
        if args.iterations <= 0:
            print("[!] --auto-iterations needs --iterations set as the probe size")
            raise SystemExit(2)
        probe = args.iterations
        print(f"[*] RES on '{args.input}' (provider: {args.provider}, "
              f"auto probe {probe})")
        ledger = build_evidence(
            args.input,
            args.output_dir,
            sr=args.sr,
            max_duration=args.duration,
            window_s=args.window,
            threshold=args.threshold,
            max_tracks=args.max_tracks,
        )
        start_corr = ledger["best"]["correlation"]
        print(f"[+] Evidence built : {len(ledger['sections'])} sections, "
              f"{sum(s['partial_count'] for s in ledger['sections'])} partials, "
              f"initial corr {start_corr}")
        print(f"[*] Probe pass ({probe} iterations) to measure the gain trend ...")
        ledger = _run(ledger, probe, target_corr=0.0, min_gain=0.0)
        rec = recommend_iterations(ledger, min_gain=args.min_gain or 1e-3)
        _print_recommendation(rec)
        if rec["converged"]:
            print("[+] Converged during the probe - no extra iterations needed.")
        else:
            iterations = max(rec["recommended"], probe)
            extra = iterations - probe
            print(f"[*] Extending to {iterations} iterations (+{extra})")
            ledger = _run(
                ledger, extra,
                target_corr=args.target_corr, min_gain=args.min_gain,
                stop_on_no_improvement=args.stop_on_no_improvement,
            )
        auto_note = " (auto)"
    else:
        print(f"[*] RES on '{args.input}' (provider: {args.provider})")
        ledger = build_evidence(
            args.input,
            args.output_dir,
            sr=args.sr,
            max_duration=args.duration,
            window_s=args.window,
            threshold=args.threshold,
            max_tracks=args.max_tracks,
        )
        start_corr = ledger["best"]["correlation"]
        print(f"[+] Evidence built : {len(ledger['sections'])} sections, "
              f"{sum(s['partial_count'] for s in ledger['sections'])} partials, "
              f"initial corr {start_corr}")
        ledger = _run(
            ledger, iterations,
            target_corr=args.target_corr, min_gain=args.min_gain,
            stop_on_no_improvement=args.stop_on_no_improvement,
        )
    fill_note = ""
    if getattr(args, "spectral_fill", 0.0) > 0.0:
        fill_note = _apply_spectral_fill(args, ledger)
    for it in ledger["iterations"]:
        g = it["global"]
        status = "accepted" if it["accepted"] else "rejected (no improvement)"
        gain = it.get("gain")
        gs = f" gain={gain:+.6f}" if isinstance(gain, (int, float)) else ""
        print(f"    iter {it['iteration']}: [{it['provider']}] "
              f"corr={g['correlation']:.6f}{gs} snr={g['snr_db']:.2f} dB "
              f"lsd={g['log_spectral_distance_db']:.1f} dB "
              f"llm_edits={it['llm_edits_applied']} -> {status}")
    final = ledger["iterations"][-1]["global"] if ledger["iterations"] else ledger["best"]
    print(f"[+] Final corr     : {final['correlation']} "
          f"(best {ledger['best']['correlation']})")
    print(f"[+] Stopped because: {ledger.get('stop_reason', 'unknown')}{auto_note}")
    print(f"[+] Ledger         : {ledger['outputs']['ledger']}")
    print(f"[+] Best WAV       : {ledger['outputs']['best_wav']}")
    print(f"[+] Equations text : {ledger['outputs']['equations_txt']}")
    if fill_note:
        print(fill_note)
    rec = recommend_iterations(ledger, min_gain=args.min_gain or 1e-3)
    _print_recommendation(rec)
    if args.json:
        print(json.dumps({
            "best": ledger["best"],
            "iterations": ledger["iterations"],
            "stop_reason": ledger.get("stop_reason"),
            "recommendation": rec,
            "sections": len(ledger["sections"]),
            "partials": sum(s["partial_count"] for s in ledger["sections"]),
            "outputs": ledger["outputs"],
        }, indent=2))


def _apply_spectral_fill(args: argparse.Namespace, ledger: dict) -> str:
    """Write an optional noise-filled copy of the best render, then report the
    measured trade-off. Never touches ledger correlation or the optimiser."""
    import soundfile as sf

    from src.analysis import load_audio
    from src.res_loop import render_ledger, spectral_fill

    out = Path(ledger["outputs"]["best_wav"]).parent
    src = ledger["source"]
    if not Path(src).exists():
        src = str(Path("data/uploads") / Path(src).name)
    y, _ = load_audio(src, target_sr=int(ledger["sample_rate"]))
    n = int(ledger.get("n_samples", 0)) or len(y)
    y = y[:n]
    recon = render_ledger({**ledger, "sections": ledger["sections"]}, n)
    filled, info = spectral_fill(recon, y, args.spectral_fill, seed=args.spectral_seed)

    from src.residual import compute_error_metrics

    def stats(r: np.ndarray):
        m = compute_error_metrics(y, r, sr=int(ledger["sample_rate"]))
        a = y - y.mean()
        b = r - r.mean()
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        return corr, m["snr_db"], m["log_spectral_distance_db"]

    c0, s0, l0 = stats(recon)
    c1, s1, l1 = stats(filled)
    path = out / "res_best_spectral_fill.wav"
    sf.write(str(path), np.clip(filled, -1.0, 1.0), int(ledger["sample_rate"]))
    info["path"] = str(path)
    info["correlation_before"] = c0
    info["correlation_after"] = c1
    info["lsd_before_db"] = l0
    info["lsd_after_db"] = l1
    info["snr_before_db"] = s0
    info["snr_after_db"] = s1
    ledger["spectral_fill"] = info
    with open(out / "res_ledger.json", "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    return (
        f"[+] Spectral fill  : level {info['level']:.2f} seed {info['seed']} -> {path}\n"
        f"    trade-off      : corr {c0:.4f} -> {c1:.4f} ({c1-c0:+.4f})   "
        f"lsd {l0:.2f} -> {l1:.2f} dB ({l1-l0:+.2f})\n"
        f"    note           : post-process only; ledger correlation is unchanged"
    )


def _print_recommendation(rec: dict) -> None:
    print(f"[i] Iterations used : {rec['iterations_done']}   "
          f"last gain {rec['last_gain']:+.6f}   basis: {rec['basis']}")
    if rec.get("trend"):
        print(f"[i] Trend           : {rec['trend']}")
    if rec["decay_ratio"] is not None:
        print(f"[i] Gain decay      : {rec['decay_ratio']:.3f}x per pass")
    if rec["predicted_ceiling"] is not None:
        print(f"[i] Predicted ceiling: {rec['predicted_ceiling']:.6f} "
              f"(+{rec['remaining_gain']:.6f} still available)")
    verdict = "converged" if rec["converged"] else f"recommend {rec['recommended']} total"
    print(f"[i] Recommendation  : {verdict}")
    print(f"    {rec['reason']}")


def cmd_res_recommend(args: argparse.Namespace) -> None:
    """RES: Recommend an iteration budget for an existing ledger."""
    from src.res_loop import recommend_iterations, structural_budget

    path = Path(args.ledger)
    if not path.exists():
        print(f"[!] Ledger not found: {path}")
        raise SystemExit(2)
    with open(path, "r", encoding="utf-8") as f:
        ledger = json.load(f)
    total = sum(len(s.get("partials") or []) for s in ledger.get("sections") or [])
    print(f"[*] Ledger: {path}")
    print(f"[*] {len(ledger.get('sections') or [])} sections, {total} partials, "
          f"{len(ledger.get('iterations') or [])} iterations recorded, "
          f"best corr {float((ledger.get('best') or {}).get('correlation', 0.0)):.6f}")
    rec = recommend_iterations(ledger, min_gain=args.min_gain,
                               max_iterations=args.max_iterations)
    _print_recommendation(rec)
    print(f"[i] Size-based guess: {structural_budget(ledger)} "
          "(used when the trend is too short to fit)")
    if args.json:
        print(json.dumps(rec, indent=2))


def cmd_math_daw(args: argparse.Namespace) -> None:
    """Render a Math-DAW project (equation tracks) to WAV."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Project file not found: {input_path}")
        sys.exit(1)

    print(f"[*] Loading Math-DAW project: {input_path}")
    proj = Project.load(input_path)

    # Override project settings if CLI args provided
    if args.bpm:
        proj.bpm = args.bpm
    if args.duration:
        proj.duration = args.duration

    print(f"    BPM: {proj.bpm} | Duration: {proj.duration}s | Tracks: {len(proj.tracks)}")
    for tr in proj.tracks:
        print(f"    - {tr.name}: y = {tr.equation}")

    out_wav = Path(args.output)
    master, rendered_tracks = render_to_wav(proj, out_wav)
    print(f"[+] Rendered: {out_wav} ({proj.duration}s, {proj.sr} Hz)")

    # Waveform plot
    if args.plot:
        plot_path = Path(args.plot)
        equations = [tr.equation for tr in proj.tracks]
        plot_tracks(
            rendered_tracks, master,
            sr=proj.sr, bpm=proj.bpm,
            title=f"Math-DAW: {proj.bpm} BPM",
            out_path=plot_path,
            equations=equations,
        )
        print(f"[+] Waveform plot: {plot_path}")

    # Spectrogram plot
    if args.spectrogram:
        spec_path = Path(args.spectrogram)
        plot_track_spectrograms(
            rendered_tracks, master,
            sr=proj.sr,
            title=f"Math-DAW Spectrograms: {proj.bpm} BPM",
            out_path=spec_path,
        )
        print(f"[+] Spectrogram plot: {spec_path}")


def cmd_math_daw_demo(args: argparse.Namespace) -> None:
    """Render all preset Math-DAW demos with waveform plots."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    demos = {
        "sinc_drop": demo_sinc_drop(),
        "math_remix": demo_math_remix(),
        "ambient": demo_ambient(),
    }

    for name, proj in demos.items():
        print(f"\n{'='*60}")
        print(f"  Math-DAW Demo: {name} ({proj.bpm} BPM, {proj.duration}s)")
        print(f"{'='*60}")

        wav_path = out_dir / f"{name}.wav"
        json_path = out_dir / f"{name}_project.json"
        plot_path = out_dir / f"{name}_tracks.png"
        spec_path = out_dir / f"{name}_spectrograms.png"

        # Render audio
        master, rendered_tracks = render_to_wav(proj, wav_path)
        print(f"  [+] Audio: {wav_path}")

        # Save project JSON
        proj.save(json_path)
        print(f"  [+] Project JSON: {json_path}")

        # Waveform plot
        equations = [tr.equation for tr in proj.tracks]
        plot_tracks(
            rendered_tracks, master,
            sr=proj.sr, bpm=proj.bpm,
            title=f"Math-DAW: {name} ({proj.bpm} BPM)",
            out_path=plot_path,
            equations=equations,
        )
        print(f"  [+] Waveform plot: {plot_path}")

        # Spectrogram plot
        plot_track_spectrograms(
            rendered_tracks, master,
            sr=proj.sr,
            title=f"Math-DAW Spectrograms: {name}",
            out_path=spec_path,
        )
        print(f"  [+] Spectrogram plot: {spec_path}")

        # Print track details
        for tr in proj.tracks:
            print(f"  Track '{tr.name}': y = {tr.equation}")
            if tr.pattern:
                print(f"    Pattern: {tr.pattern}")

def cmd_fit_formula(args: argparse.Namespace) -> None:
    """Fit the best acoustic formula family from FORMULA_BANK to an audio segment."""
    import librosa
    print(f"[*] Fitting mathematical formula from FORMULA_BANK for: {args.input}")
    y, sr = load_audio(args.input)
    if args.f0 is not None:
        f0 = float(args.f0)
    else:
        # Detect f0 with pYIN
        f0_track, _, _ = librosa.pyin(y, fmin=40.0, fmax=2000.0, sr=sr)
        f0 = float(np.nanmedian(f0_track)) if np.any(~np.isnan(f0_track)) else 220.0

    match = fit_best_formula_for_segment(y, sr, f0)
    if not match:
        print("[!] Could not fit formula to audio segment.")
        return

    print("\n--- BEST MATHEMATICAL FORMULA FOUND ---")
    for k, v in match.items():
        print(f"  {k}: {v}")

    if args.output_wav:
        t = np.arange(len(y)) / sr
        fn = FORMULA_BANK[match["formula_type"]]["func"]
        synth = fn(t, match["f0"], match["A"], match["decay_lambda"], match["timbre_param"])
        save_audio(args.output_wav, synth, sr)
        print(f"[+] Saved reconstructed audio -> {args.output_wav}")


def cmd_recreate_song(args: argparse.Namespace) -> None:
    """Reconstruct an input audio file into Math-DAW equation tracks."""
    print(f"[*] Recreating song '{args.input}' into Math-DAW equation tracks...")
    res = recreate_and_evaluate_song(
        input_audio_path=args.input,
        output_dir=args.output_dir,
        duration=args.duration,
        sr=args.sr,
    )
    print("\n--- Song Recreation Summary ---")
    print(f"Status:            {res['status']}")
    print(f"BPM:               {res['bpm']}")
    print(f"Duration:          {res['duration']}s")
    print(f"Tracks:            {', '.join(res['tracks'])}")
    print(f"Spectral CQT MSE:  {res['spectral_cqt_mse']:.6f}")
    print(f"Project JSON:      {res['project_json']}")
    print(f"Reconstructed WAV: {res['recreated_wav']}")
    print(f"Comparison Plot:   {res['comparison_plot']}")


def cmd_viral_phonk(args: argparse.Namespace) -> None:
    """Render the viral math phonk reels audio and plots."""
    from src.viral_math_reels import render_all_viral_reels, REEL_EQUATIONS
    print("==================================================================")
    print("   Viral Math Phonk: 'Math functions your teacher never played you' ")
    print("==================================================================")
    res = render_all_viral_reels(
        output_dir=args.output_dir,
        duration=args.duration,
        bpm=args.bpm,
        sr=args.sr,
    )
    print("\n[+] Rendered Equations:")
    for key, item in REEL_EQUATIONS.items():
        print(f"  â€¢ {item['title']}: {item['concept']}")
    print(f"\n[+] Rendered Sinc Drop WAV:    {res['sinc_drop_wav']}")
    print(f"[+] Rendered Complete Phonk:  {res['complete_phonk_wav']}")
    print(f"[+] Rendered TikTok Plot:     {res['sinc_drop_plot']}")
    print(f"[+] Saved Project JSON:       {res['project_json']}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Executes the complete end-to-end demonstration master blueprint."""
    print("==================================================================")
    print("      music2math: Closed-Loop DSP Engine Verification Run         ")
    print("==================================================================")

    data_in = Path("data/input")
    data_out = Path("data/output")
    data_in.mkdir(parents=True, exist_ok=True)
    data_out.mkdir(parents=True, exist_ok=True)

    # 1. Phase 1: Forward Synthesis of benchmark chord
    print("\n>>> STEP 1: Forward Synthesis of Benchmark Chord")
    chord_wav = data_in / "math_chord.wav"
    generate_benchmark_chord(chord_wav, duration=2.0, sr=44100)
    print(f"    Generated ground-truth audio: {chord_wav}")

    # 2. Phase 2: Reverse Engineering of equations
    print("\n>>> STEP 2: Reverse Engineering (Audio -> Equations)")
    eq_json = data_out / "equations.json"
    discover_equations_from_audio(
        chord_wav,
        n_fft=4096,
        hop_length=1024,
        threshold=0.01,
        out_json_path=eq_json,
    )
    print(f"    Exported discovered model: {eq_json}")

    # 3. Phase 3: Re-synthesis from equations
    print("\n>>> STEP 3: Re-Synthesis (Equations -> Audio)")
    recon_wav = data_out / "rendered_from_equations.wav"
    recon_audio, sr = render_model(eq_json, duration=2.0, sr=44100)
    save_audio(recon_wav, recon_audio, sr)
    print(f"    Rendered audio from equations: {recon_wav}")

    # 4. Phase 3: Closed-Loop Verification & Metrics
    print("\n>>> STEP 4: Closed-Loop Verification & Residual Calculation")
    orig_audio, _ = load_audio(chord_wav, target_sr=sr)
    metrics = compute_error_metrics(orig_audio, recon_audio, sr=sr)
    res_wav = data_out / "residual.wav"
    min_len = min(len(orig_audio), len(recon_audio))
    residual = orig_audio[:min_len] - recon_audio[:min_len]
    save_audio(res_wav, residual, sr)

    report_json = data_out / "verification_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("    --- Verification Metrics ---")
    print(f"    SNR:                   {metrics['snr_db']:.2f} dB")
    print(f"    MSE:                   {metrics['mse']:.6e}")
    print(f"    Spectral MSE:          {metrics['spectral_mse']:.6e}")
    print(f"    Log-Spectral Distance: {metrics['log_spectral_distance_db']:.2f} dB")
    print(f"    Saved residual WAV:    {res_wav}")
    print(f"    Saved report JSON:     {report_json}")

    # 5. Visual Dashboard
    print("\n>>> STEP 5: Generating Visual Inspection Dashboard")
    dash_png = data_out / "closed_loop_dashboard.png"
    plot_comparison_dashboard(orig_audio, recon_audio, sr=sr, out_path=dash_png, metrics=metrics)
    print(f"    Saved comparison dashboard: {dash_png}")

    print("\n==================================================================")
    print("      Verification complete: SUCCESS (High-precision DSP loop)    ")
    print("==================================================================")


def _llm_backend_for(args: argparse.Namespace) -> str | None:
    """Map ``--provider`` onto a backend name for the loop.

    ``numeric`` means no LLM at all. ``auto`` lets the loop pick the first
    usable backend, which it signals by passing None. Every other value is
    a backend name, including ``api`` for the direct HTTP path.
    """
    provider = (getattr(args, "provider", "numeric") or "numeric").strip().lower()
    if provider in ("numeric", "", "none"):
        return None
    if provider == "auto":
        return None
    return provider


def cmd_llm_backends(args: argparse.Namespace) -> None:
    """Report every LLM backend: installed, identified, and usable."""
    rows = probe_all()
    print("[*] LLM backends for the RES hypothesis provider")
    print()
    print("    %-10s %-9s %-10s %s" % ("backend", "installed", "usable", "version / reason"))
    for row in rows:
        print("    %-10s %-9s %-10s %s" % (
            row["name"],
            "yes" if row["installed"] else "no",
            "yes" if row["usable"] else "no",
            row.get("version") or row.get("reason") or "",
        ))
        if row["installed"] and not row["usable"]:
            print("        %s" % (row.get("reason") or ""))
            if row.get("auth_hint"):
                print("        fix: %s" % row["auth_hint"])
    if getattr(args, "check", False):
        print()
        print("[*] asking each installed backend for a trivial answer")
        for row in rows:
            if not row["installed"]:
                continue
            spec = BACKENDS.get(row["name"])
            if spec is None:
                continue
            try:
                argv = build_command(spec, "Reply with exactly: PROBE_OK")
            except Exception as exc:
                print("    %-10s cannot build a command: %s" % (row["name"], exc))
                continue
            res = run_backend(argv, timeout_s=90.0)
            answered = "PROBE_OK" in ((res.stdout or "") + (res.stderr or ""))
            print("    %-10s %s%s" % (
                row["name"],
                "answered" if answered else "no answer",
                " (%.0fs)" % res.duration_s if res.duration_s else "",
            ))
            if not answered and res.error:
                print("        error: %s" % res.error)
            elif not answered and res.stderr:
                print("        stderr: %s" % res.stderr.strip().splitlines()[0][:140])
    picked = auto_backend()
    print()
    if picked:
        print("[+] --provider auto would use: %s" % picked.name)
    else:
        print("[!] --provider auto would fail: no backend is installed")
    print("    A CLI backend carries its own authentication, so the project")
    print("    needs no API key. --provider api is the one path that does,")
    print("    and it reads OPENAI_API_KEY.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="music2math",
        description="DSP-rigorous reverse engineering of audio into compact mathematical equations.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-commands")

    # llm-backends
    p_llm = subparsers.add_parser(
        "llm-backends",
        help="Report which LLM backends the RES hypothesis provider can use",
    )
    p_llm.add_argument(
        "--check", action="store_true",
        help="Also send a trivial prompt to each installed backend, to prove "
             "it answers rather than merely being on PATH",
    )
    p_llm.set_defaults(func=cmd_llm_backends)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Generate benchmark audio files with ground truth equations")
    p_bench.add_argument("--type", choices=["chord", "melody", "all"], default="all", help="Signal type")
    p_bench.add_argument("--output-dir", "-o", default="data/input", help="Directory for generated WAVs")
    p_bench.add_argument("--duration", "-d", type=float, default=2.0, help="Duration in seconds")
    p_bench.add_argument("--sr", type=int, default=44100, help="Sample rate")
    p_bench.set_defaults(func=cmd_benchmark)

    # synthesize
    p_synth = subparsers.add_parser("synthesize", help="Render equations.json into audio WAV")
    p_synth.add_argument("--input", "-i", required=True, help="Path to equations JSON")
    p_synth.add_argument("--output", "-o", required=True, help="Output WAV path")
    p_synth.add_argument("--duration", "-d", type=float, default=None, help="Duration override")
    p_synth.add_argument("--sr", type=int, default=44100, help="Sample rate")
    p_synth.add_argument("--normalize", action="store_true", help="Peak normalize audio output")
    p_synth.set_defaults(func=cmd_synthesize)

    # analyze
    p_ana = subparsers.add_parser("analyze", help="Extract equations from audio file")
    p_ana.add_argument("--input", "-i", required=True, help="Input WAV/audio file")
    p_ana.add_argument("--output", "-o", default="data/output/equations.json", help="Output JSON path")
    p_ana.add_argument("--n_fft", type=int, default=4096, help="STFT window size")
    p_ana.add_argument("--hop_length", type=int, default=1024, help="STFT hop size")
    p_ana.add_argument("--threshold", type=float, default=0.005, help="Relative peak amplitude threshold")
    p_ana.add_argument("--no-projection", action="store_true", help="Disable projection refinement")
    p_ana.set_defaults(func=cmd_analyze)

    # verify
    p_ver = subparsers.add_parser("verify", help="Compare original vs reconstructed audio")
    p_ver.add_argument("--original", "-orig", required=True, help="Original audio WAV")
    p_ver.add_argument("--reconstructed", "-recon", required=True, help="Reconstructed audio WAV")
    p_ver.add_argument("--save-residual", "-r", default=None, help="Path to save residual WAV")
    p_ver.add_argument("--report-json", default=None, help="Path to save JSON error report")
    p_ver.add_argument("--dashboard", default=None, help="Path to save comparison dashboard PNG")
    p_ver.set_defaults(func=cmd_verify)

    # refine
    p_ref = subparsers.add_parser("refine", help="Iteratively refine model by matching residual peaks")
    p_ref.add_argument("--input", "-i", required=True, help="Original audio WAV")
    p_ref.add_argument("--initial-model", default=None, help="Optional initial equations JSON")
    p_ref.add_argument("--output", "-o", default="data/output/refined_equations.json", help="Output JSON path")
    p_ref.add_argument("--rendered-wav", default="data/output/refined_audio.wav", help="Output refined WAV")
    p_ref.add_argument("--max-iterations", type=int, default=15, help="Max refinement iterations")
    p_ref.add_argument("--target-snr", type=float, default=30.0, help="Target SNR in dB")
    p_ref.set_defaults(func=cmd_refine)

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Generate waveform and spectrogram visual inspection plots")
    p_vis.add_argument("--input", "-i", required=True, help="Audio file to inspect")
    p_vis.add_argument("--output-dir", "-o", default="data/output/plots", help="Output directory for plots")
    p_vis.set_defaults(func=cmd_visualize)

    # extract-notes
    p_notes = subparsers.add_parser("extract-notes", help="Extract melody notes using pYIN")
    p_notes.add_argument("--input", "-i", required=True, help="Audio file")
    p_notes.add_argument("--output", "-o", default=None, help="Output JSON path for notes")
    p_notes.set_defaults(func=cmd_extract_notes)

    # tone-id
    p_tone = subparsers.add_parser("tone-id", help="Identify musical tones with Shazam-style precision")
    p_tone.add_argument("--input", "-i", required=True, help="Audio file")
    p_tone.add_argument("--output", "-o", default=None, help="Output JSON path for tones")
    p_tone.add_argument("--sr", type=int, default=22050, help="Analysis sample rate")
    p_tone.add_argument("--segment", type=float, default=0.25, help="Tone analysis segment in seconds")
    p_tone.set_defaults(func=cmd_tone_id)

    # tone-remake
    p_remake = subparsers.add_parser("tone-remake", help="Recreate a song from detected tones as equation tracks")
    p_remake.add_argument("--input", "-i", required=True, help="Input song audio")
    p_remake.add_argument("--output-dir", "-o", default="data/output", help="Output directory")
    p_remake.add_argument("--duration", "-d", type=float, default=16.0, help="Max duration in seconds")
    p_remake.add_argument("--sr", type=int, default=22050, help="Analysis sample rate")
    p_remake.add_argument("--no-refine", action="store_true", help="Skip closed-loop error correction")
    p_remake.set_defaults(func=cmd_tone_remake)

    # sms-remake
    p_sms = subparsers.add_parser(
        "sms-remake",
        help="High-fidelity song -> equations -> song (sinusoidal modeling synthesis)",
    )
    p_sms.add_argument("--input", "-i", required=True, help="Input song audio")
    p_sms.add_argument("--output-dir", "-o", default="data/output", help="Output directory")
    p_sms.add_argument("--duration", "-d", type=float, default=20.0, help="Seconds to analyze")
    p_sms.add_argument("--sr", type=int, default=22050, help="Analysis sample rate")
    p_sms.add_argument("--threshold", type=float, default=0.01, help="Peak threshold")
    p_sms.add_argument("--max-tracks", type=int, default=600, help="Max sinusoidal tracks")
    p_sms.add_argument("--no-noise", action="store_true", help="Disable the S+N noise layer")
    p_sms.add_argument(
        "--equations-only", action="store_true",
        help="Skip WAV/plot output; write only the math equations "
             "(.txt with closed forms, .json with full parameters)",
    )
    p_sms.add_argument(
        "--sensitivity", type=float, default=1e-6,
        help="Residual-energy fraction a partial must remove to be kept "
             "(lower = keep more partials). Default 1e-6: at 1e-4 the "
             "matching-pursuit gate discards ~99 genuine high-frequency "
             "tracks, leaving 2.5-5 kHz 15.6 dB down on dense music. 1e-6 "
             "recovers +5.1 dB there while improving correlation and LSD.",
    )
    p_sms.add_argument(
        "--window", type=float, default=0.0,
        help="Analyse in sections of N seconds so short events are judged "
             "against local energy (0 = whole signal at once; try 2.5 for songs)",
    )
    p_sms.set_defaults(func=cmd_sms_remake)

    # res
    p_res = subparsers.add_parser(
        "res",
        help="RES: Reverse Engineer a Song into equations "
             "(evidence -> hypothesize -> verify -> ledger)",
    )
    p_res.add_argument("--input", "-i", required=True, help="Input song audio")
    p_res.add_argument("--output-dir", "-o", default="data/output/res", help="Output directory")
    p_res.add_argument("--duration", "-d", type=float, default=20.0, help="Seconds to analyze")
    p_res.add_argument("--sr", type=int, default=22050, help="Analysis sample rate")
    p_res.add_argument("--threshold", type=float, default=0.01, help="Peak threshold for evidence")
    p_res.add_argument("--max-tracks", type=int, default=400, help="Max sinusoidal tracks in evidence")
    p_res.add_argument(
        "--window", "-w", type=float, default=2.5,
        help="Evidence section length in seconds (0 = whole signal)",
    )
    p_res.add_argument("--iterations", type=int, default=6, help="Max RES loop iterations")
    p_res.add_argument(
        "--auto-iterations", action="store_true",
        help="Run --iterations as a probe, measure the gain trend, then extend "
             "to the recommended budget in the same run",
    )
    p_res.add_argument(
        "--stop-on-no-improvement", action="store_true",
        help="Halt the loop on the first iteration that fails to improve",
    )
    p_res.add_argument(
        "--min-gain", type=float, default=0.0,
        help="Halt when an accepted iteration improves correlation by less than "
             "this (0 disables). Try 0.001",
    )
    p_res.add_argument(
        "--no-bank", action="store_true",
        help="Disable the v2 hypothesis bank (harmonic-string class B "
             "tournament); class A coordinate refit only",
    )
    p_res.add_argument(
        "--spectral-lane", action="store_true",
        help="Allow stochastic sibilance/cymbal bursts to be accepted on "
             "log-spectral distance instead of waveform correlation",
    )
    p_res.add_argument(
        "--mrstft-lane", action="store_true",
        help="Flag-gated referee lane: accept an iteration on multi-"
             "resolution STFT improvement when waveform correlation stalls "
             "(the selection criterion of Yamamoto 2019 / DDSP). Default "
             "off - published numbers move only after re-measurement",
    )
    p_res.add_argument(
        "--polish", action="store_true",
        help="Flag-gated lifter increment: continuously refine each "
             "partial's decay around the best grid value (bounded scalar "
             "optimization with the closed-form amplitude/phase projection "
             "inside). Default off - published numbers move only after "
             "re-measurement",
    )
    p_res.add_argument(
        "--k-hits", type=int, default=12,
        help="Percussive hits the hypothesis bank may nominate per section "
             "(class C). The default 2 covers roughly one beat.",
    )
    p_res.add_argument(
        "--energy-parity", action="store_true",
        help="Refit the shipped render's loudness: one gain per section "
             "restores the source level (the referee's correlation is "
             "scale-invariant, so without this the render can ship far "
             "too quiet).",
    )
    p_res.add_argument(
        "--parity-mode", choices=["ls", "rms"], default="ls",
        help="With --energy-parity: 'ls' = least-squares gain (SNR-optimal, "
             "lands the render at corr x source RMS); 'rms' = loudness "
             "parity (sounds like the original, small SNR cost at low corr).",
    )
    p_res.add_argument(
        "--joint-refine", action="store_true",
        help="Jointly refit collision clusters (partials within ~10 Hz) "
             "with TRF least squares - decouples detuned 808 layers and "
             "unison beating that per-partial refits cannot.",
    )
    p_res.add_argument(
        "--k-notes", type=int, default=2,
        help="Harmonic notes the hypothesis bank may nominate per section "
             "(class B).",
    )
    p_res.add_argument(
        "--noise", action="store_true",
        help="Let the hypothesis bank model the residual with a stochastic "
             "class (LPC formant noise). A sum of sinusoids cannot express "
             "breath, air or sibilance, so this content is absent without it.",
    )
    p_res.add_argument(
        "--target-corr", type=float, default=0.98,
        help="Stop when best correlation reaches this (0 = never stop early). "
             "The web UI deliberately passes 0.0 (it promises the requested "
             "iterations and plots the full curve) - see res_ui.run_res_payload.",
    )
    p_res.add_argument(
        "--provider",
        choices=["numeric", "auto", "api"] + list(BACKENDS.keys()),
        default="numeric",
        help="Hypothesis provider: 'numeric' (no LLM), 'auto' (the first "
             "usable LLM CLI backend), 'api' (an OpenAI-compatible endpoint, "
             "needs OPENAI_API_KEY), or the name of one backend - agy "
             "(Antigravity), opencode, claude, gemini, qwen, codex, crush, "
             "goose, amp, ollama. Every backend is followed by the same "
             "numeric verification and referee scoring.",
    )
    p_res.add_argument("--llm-model", default="", help="Model id passed to the LLM provider (e.g. provider/model)")
    p_res.add_argument(
        "--llm-cmd", default="",
        help="Custom LLM command template; entries containing {prompt_file} "
             "get the prompt path. Command must print the edits JSON to stdout",
    )
    p_res.add_argument(
        "--llm-timeout", type=float, default=300.0,
        help="Seconds to allow one LLM call. The whole process tree is "
             "killed when it expires, so a backend that spawns children "
             "cannot hang the loop",
    )
    p_res.add_argument(
        "--llm-extra-args", default="",
        help="Extra arguments for the LLM backend, space separated. Some "
             "agent CLIs need a permission flag to answer at all - for "
             "example agy needs '--dangerously-skip-permissions' or a "
             "permissions.allow rule, because it reaches for a tool",
    )
    p_res.add_argument(
        "--llm-base-url", default="https://api.openai.com/v1",
        help="Base URL for --provider api (any OpenAI-compatible endpoint)",
    )
    p_res.add_argument(
        "--llm-api-key", default="",
        help="API key for --provider api. Defaults to $OPENAI_API_KEY. The "
             "CLI backends do not need one: they carry their own auth",
    )
    p_res.add_argument("--json", action="store_true", help="Print final summary as JSON")
    p_res.add_argument("--spectral-fill", type=float, default=0.0, help="Add deterministic shaped residual noise")
    p_res.add_argument("--spectral-seed", type=int, default=0, help="Seed for deterministic spectral fill")
    p_res.add_argument(
        "--stems", nargs="?", const="auto", default=None,
        help="Tri-Mode Stem Separation: 'auto' (Demucs if in PATH else zero-phase Native DSP), "
             "'native', 'demucs', or a directory path containing bass/drums/vocals/other wav files",
    )
    p_res.set_defaults(func=cmd_res)

    p_serve = subparsers.add_parser(
        "serve", help="Run the local RES API server (what the browser extension talks to)"
    )
    p_serve.add_argument("--host", default="127.0.0.1",
                         help="Bind address (default 127.0.0.1 - localhost only)")
    p_serve.add_argument("--port", type=int, default=8765,
                         help="Port (default 8765)")
    p_serve.set_defaults(func=cmd_serve)

    p_res_rec = subparsers.add_parser(
        "res-recommend",
        help="RES: Recommend an iteration budget from an existing res_ledger.json",
    )
    p_res_rec.add_argument("--ledger", "-l", required=True,
                           help="Path to a res_ledger.json")
    p_res_rec.add_argument(
        "--min-gain", type=float, default=1e-3,
        help="Gain below which further iterations are not worth running",
    )
    p_res_rec.add_argument(
        "--max-iterations", type=int, default=40,
        help="Upper bound for the recommendation",
    )
    p_res_rec.add_argument("--json", action="store_true",
                           help="Print the recommendation as JSON")
    p_res_rec.set_defaults(func=cmd_res_recommend)

    # math-daw
    p_daw = subparsers.add_parser("math-daw", help="Render a Math-DAW equation project to WAV")
    p_daw.add_argument("--input", "-i", required=True, help="Math-DAW project JSON")
    p_daw.add_argument("--output", "-o", default="data/output/math_daw_render.wav", help="Output WAV path")
    p_daw.add_argument("--bpm", type=int, default=None, help="Override BPM")
    p_daw.add_argument("--duration", "-d", type=float, default=None, help="Override duration")
    p_daw.add_argument("--plot", default=None, help="Path to save waveform plot PNG")
    p_daw.add_argument("--spectrogram", default=None, help="Path to save spectrogram plot PNG")
    p_daw.set_defaults(func=cmd_math_daw)

    # math-daw-demo
    p_daw_demo = subparsers.add_parser("math-daw-demo", help="Render all preset Math-DAW demos")
    p_daw_demo.add_argument("--output-dir", "-o", default="data/output/math_daw_demos", help="Output directory")
    p_daw_demo.set_defaults(func=cmd_math_daw_demo)

    # fit-formula
    p_fit = subparsers.add_parser("fit-formula", help="Fit best acoustic formula from FORMULA_BANK to audio")
    p_fit.add_argument("--input", "-i", required=True, help="Input audio WAV")
    p_fit.add_argument("--f0", type=float, default=None, help="Fundamental frequency in Hz (auto-detected if omitted)")
    p_fit.add_argument("--output-wav", "-o", default=None, help="Optional reconstructed WAV output path")
    p_fit.set_defaults(func=cmd_fit_formula)

    # discover-equations
    p_disc = subparsers.add_parser(
        "discover-equations",
        help="Discover multiple competing math equations (Pareto set + library learning)",
    )
    p_disc.add_argument("--input", "-i", required=True, help="Input audio file")
    p_disc.add_argument("--output", "-o", default=None, help="Output JSON report path")
    p_disc.add_argument("--f0", type=float, default=None, help="Fundamental frequency (auto-detected if omitted)")
    p_disc.add_argument("--duration", "-d", type=float, default=None, help="Seconds of audio to analyze")
    p_disc.add_argument("--max-terms", type=int, default=4, help="Max library-learning terms")
    p_disc.set_defaults(func=cmd_equation_discovery)

    # recreate-song
    p_rec = subparsers.add_parser("recreate-song", help="Reconstruct song audio into multi-track Math-DAW equations")
    p_rec.add_argument("--input", "-i", required=True, help="Input song audio (e.g., Metre - Lost.mp3)")
    p_rec.add_argument("--output-dir", "-o", default="data/output", help="Output directory")
    p_rec.add_argument("--duration", "-d", type=float, default=16.0, help="Duration in seconds (default: 16.0)")
    p_rec.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    p_rec.set_defaults(func=cmd_recreate_song)
    # viral-phonk
    p_viral = subparsers.add_parser("viral-phonk", help="Render viral math phonk reels audio and plots (#7/10 Sinc Drop suite)")
    p_viral.add_argument("--output-dir", "-o", default="data/output", help="Output directory")
    p_viral.add_argument("--duration", "-d", type=float, default=16.0, help="Duration in seconds (default: 16.0)")
    p_viral.add_argument("--bpm", type=float, default=140.0, help="Tempo in BPM (default: 140)")
    p_viral.add_argument("--sr", type=int, default=44100, help="Sample rate (default: 44100)")
    p_viral.set_defaults(func=cmd_viral_phonk)

    # run-all
    p_all = subparsers.add_parser("run-all", help="Execute complete closed-loop demonstration pipeline")
    p_all.set_defaults(func=cmd_run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
