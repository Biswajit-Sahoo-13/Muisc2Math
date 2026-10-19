"""Sinusoidal tracking and log-linear envelope regression engine.

Converts frame-by-frame spectral peaks into coherent continuous tracks,
then fits mathematical equations (exponential decay, constant, linear) to each track.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.analysis import compute_normalized_stft, find_parabolic_peaks


class SinusoidalTrack:
    """Represents a continuous frequency partial tracked across STFT time frames."""

    def __init__(self, track_id: int, initial_frame: int, peak: Dict[str, float], time: float):
        self.track_id = track_id
        self.frame_indices: List[int] = [initial_frame]
        self.times: List[float] = [time]
        self.frequencies: List[float] = [peak["frequency"]]
        self.amplitudes: List[float] = [peak["amplitude"]]
        self.phases: List[float] = [peak["phase"]]
        self.closed: bool = False
        self.inactive_count: int = 0

    def update(self, frame_idx: int, peak: Dict[str, float], time: float) -> None:
        self.frame_indices.append(frame_idx)
        self.times.append(time)
        self.frequencies.append(peak["frequency"])
        self.amplitudes.append(peak["amplitude"])
        self.phases.append(peak["phase"])
        self.inactive_count = 0

    @property
    def mean_frequency(self) -> float:
        return float(np.mean(self.frequencies))

    @property
    def peak_amplitude(self) -> float:
        return float(np.max(self.amplitudes))

    @property
    def duration(self) -> float:
        return float(self.times[-1] - self.times[0]) if len(self.times) > 1 else 0.0


def track_sinusoids(
    stft_res: Dict[str, Any],
    threshold: float = 0.005,
    max_freq_delta_hz: float = 25.0,
    max_freq_delta_rel: float = 0.04,
    min_track_duration_frames: int = 3,
    max_inactive_frames: int = 2,
    max_peaks_per_frame: int = 35,
) -> List[SinusoidalTrack]:
    """Tracks spectral peaks across STFT frames using nearest-frequency association.

    Matches peaks to active tracks within max_freq_delta.
    """
    raw_mags = stft_res["raw_mags"]
    phases = stft_res["phases"]
    times = stft_res["time_frames"]
    sr = stft_res["sr"]
    n_fft = stft_res["n_fft"]
    coherent_gain = stft_res["coherent_gain"]
    n_frames = raw_mags.shape[1]

    active_tracks: List[SinusoidalTrack] = []
    completed_tracks: List[SinusoidalTrack] = []
    next_track_id = 0

    for m in range(n_frames):
        frame_time = float(times[m])
        frame_mags = raw_mags[:, m]
        frame_phases = phases[:, m]

        peaks = find_parabolic_peaks(
            raw_mags_frame=frame_mags,
            coherent_gain=coherent_gain,
            sr=sr,
            n_fft=n_fft,
            phase_frame=frame_phases,
            threshold=threshold,
            max_peaks=max_peaks_per_frame,
        )

        matched_tracks = set()
        matched_peaks = set()

        # Match active tracks to candidate peaks greedily by minimum frequency distance
        candidates = []
        for t_idx, track in enumerate(active_tracks):
            last_freq = track.frequencies[-1]
            for p_idx, peak in enumerate(peaks):
                f_peak = peak["frequency"]
                delta_f = abs(f_peak - last_freq)
                allowed_delta = max(max_freq_delta_hz, max_freq_delta_rel * last_freq)
                if delta_f <= allowed_delta:
                    candidates.append((delta_f, t_idx, p_idx))

        # Sort candidates by smallest frequency difference
        candidates.sort(key=lambda item: item[0])

        for delta_f, t_idx, p_idx in candidates:
            if t_idx in matched_tracks or p_idx in matched_peaks:
                continue
            active_tracks[t_idx].update(m, peaks[p_idx], frame_time)
            matched_tracks.add(t_idx)
            matched_peaks.add(p_idx)

        # Handle unmatched active tracks
        unmatched_active = []
        for t_idx, track in enumerate(active_tracks):
            if t_idx not in matched_tracks:
                track.inactive_count += 1
                if track.inactive_count > max_inactive_frames:
                    track.closed = True
                    completed_tracks.append(track)
                else:
                    unmatched_active.append(track)
            else:
                unmatched_active.append(track)
        active_tracks = unmatched_active

        # Handle unmatched peaks: spawn new tracks
        for p_idx, peak in enumerate(peaks):
            if p_idx not in matched_peaks:
                new_track = SinusoidalTrack(next_track_id, m, peak, frame_time)
                next_track_id += 1
                active_tracks.append(new_track)

    # Close remaining active tracks
    completed_tracks.extend(active_tracks)

    # Filter out short transient/noise tracks
    valid_tracks = [t for t in completed_tracks if len(t.frame_indices) >= min_track_duration_frames]
    # Sort descending by peak amplitude
    valid_tracks.sort(key=lambda t: t.peak_amplitude, reverse=True)
    return valid_tracks


def fit_log_linear_envelope(
    times: np.ndarray,
    amplitudes: np.ndarray,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """Fits an exponential decay equation A(t) = A_0 * exp(-lambda * t) in log-space.

    Formula:
        ln(A(t)) = ln(A_0) - lambda * t
    Uses np.polyfit(t - t[0], ln(A), 1).
    """
    if len(times) < 2:
        a0 = float(amplitudes[0]) if len(amplitudes) > 0 else 0.0
        return {
            "type": "constant",
            "a0": a0,
            "decay_rate": 0.0,
            "r_squared": 1.0,
        }

    rel_times = times - times[0]
    safe_amps = np.maximum(amplitudes, eps)
    log_amps = np.log(safe_amps)

    # Linear regression in log space: log(A) = slope * t + intercept
    poly, residuals, _, _, _ = np.polyfit(rel_times, log_amps, deg=1, full=True)
    slope = float(poly[0])
    intercept = float(poly[1])

    decay_rate = -slope
    a0 = float(np.exp(intercept))

    # Calculate coefficient of determination R^2
    fitted_log_amps = slope * rel_times + intercept
    ss_tot = np.sum((log_amps - np.mean(log_amps)) ** 2)
    ss_res = np.sum((log_amps - fitted_log_amps) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + eps)))

    # Classification logic
    if decay_rate > 0.05 and r2 >= 0.70:
        env_type = "exponential"
    elif abs(decay_rate) <= 0.05:
        env_type = "constant"
        a0 = float(np.median(amplitudes))
        decay_rate = 0.0
    else:
        # # ponytail: fall back to exponential fit with bounded decay rate; upgrade to ADSR segmenter if needed
        decay_rate = max(0.0, decay_rate)
        env_type = "exponential"

    return {
        "type": env_type,
        "a0": a0,
        "decay_rate": decay_rate,
        "r_squared": r2,
    }


def tracks_to_equations(
    tracks: List[SinusoidalTrack],
    sample_rate: int = 44100,
    total_duration: float = 2.0,
    min_energy_ratio: float = 0.001,
    original_audio: Optional[np.ndarray] = None,
    refine_with_projection: bool = True,
) -> Dict[str, Any]:
    """Converts a collection of SinusoidalTracks into an exportable JSON equations model.

    Optionally refines amplitudes and phases using simultaneous least-squares projection
    on the original audio, eliminating phase cancellation artifacts.
    """
    components = []
    if not tracks:
        return {
            "model_name": "discovered_equations",
            "sample_rate": sample_rate,
            "duration": total_duration,
            "components": [],
        }

    max_amp = max(t.peak_amplitude for t in tracks)
    cutoff = max_amp * min_energy_ratio

    valid_candidates = []
    for tr in tracks:
        if tr.peak_amplitude < cutoff:
            continue
        t_arr = np.array(tr.times, dtype=np.float64)
        a_arr = np.array(tr.amplitudes, dtype=np.float64)
        env_fit = fit_log_linear_envelope(t_arr, a_arr)

        start_t = float(tr.times[0])
        track_dur = float(tr.times[-1] - tr.times[0])

        # Snap near-zero onsets to 0.0
        if start_t < 0.06:
            start_t = 0.0
        if total_duration - (start_t + track_dur) < 0.06:
            track_dur = total_duration - start_t

        # Use median frequency across frames to avoid STFT boundary padding transients
        med_freq = float(np.median(tr.frequencies))

        valid_candidates.append({
            "track": tr,
            "mean_freq": med_freq,
            "init_phase": float(tr.phases[0]),
            "start_time": start_t,
            "duration": track_dur,
            "env_fit": env_fit,
        })

    # Simultaneous projection & fine-tuning refinement if original audio is provided.
    # Performance note: every basis is zero outside its component's active interval.
    # Rows outside those intervals therefore cannot change the least-squares
    # solution, so tuning and projection are restricted to active segments.
    # Components with disjoint active intervals are also solved in separate
    # overlap groups, preserving the simultaneous solution while avoiding one
    # dense full-signal design matrix.
    if refine_with_projection and original_audio is not None and valid_candidates:
        from scipy.optimize import minimize_scalar

        n_samples = len(original_audio)
        t_vec = np.arange(n_samples, dtype=np.float64) / sample_rate

        for cand in valid_candidates:
            start_idx = int(np.searchsorted(t_vec, cand["start_time"], side="left"))
            end_idx = int(np.searchsorted(t_vec, cand["start_time"] + cand["duration"], side="left"))
            start_idx = max(0, min(start_idx, n_samples))
            end_idx = max(0, min(end_idx, n_samples))
            if end_idx <= start_idx:
                # Degenerate timing: preserve legacy full-signal behavior.
                start_idx, end_idx = 0, n_samples
            cand["start_idx"] = start_idx
            cand["end_idx"] = end_idx

        # Sub-Hz frequency fine tuning on each component's active segment.
        for cand in valid_candidates:
            f0 = cand["mean_freq"]
            st = cand["start_time"]
            dec = cand["env_fit"]["decay_rate"]
            start_idx = cand["start_idx"]
            end_idx = cand["end_idx"]

            seg_t = t_vec[start_idx:end_idx]
            seg_y = original_audio[start_idx:end_idx]
            t_rel_seg = np.maximum(0.0, seg_t - st)
            env_seg = np.exp(-dec * t_rel_seg)

            def score_freq(test_f: float) -> float:
                omega = 2.0 * np.pi * test_f * t_rel_seg
                s = env_seg * np.sin(omega)
                c = env_seg * np.cos(omega)
                g00 = float(np.dot(s, s))
                g01 = float(np.dot(s, c))
                g11 = float(np.dot(c, c))
                b0 = float(np.dot(s, seg_y))
                b1 = float(np.dot(c, seg_y))
                det = g00 * g11 - g01 * g01
                if det > 1e-12 * max(1.0, g00 * g11):
                    inv_det = 1.0 / det
                    c0 = (g11 * b0 - g01 * b1) * inv_det
                    c1 = (g00 * b1 - g01 * b0) * inv_det
                    # For least squares, projected energy equals c^T b.
                    return -float(c0 * b0 + c1 * b1)
                mat = np.column_stack((s, c))
                coefs, _, _, _ = np.linalg.lstsq(mat, seg_y, rcond=None)
                pred = mat @ coefs
                return -float(np.sum(pred ** 2))

            # Optimize frequency within +/- 1.5 Hz around parabolic peak median
            opt_res = minimize_scalar(score_freq, bounds=(f0 - 1.5, f0 + 1.5), method="bounded")
            if opt_res.success:
                cand["mean_freq"] = float(opt_res.x)

        # Group components by overlapping active intervals. Disjoint groups have
        # block-diagonal normal equations and can be solved independently.
        order = sorted(range(len(valid_candidates)), key=lambda i: valid_candidates[i]["start_idx"])
        groups: List[List[int]] = []
        group_end = -1
        for idx in order:
            start_idx = valid_candidates[idx]["start_idx"]
            end_idx = valid_candidates[idx]["end_idx"]
            if not groups or start_idx < group_end:
                if not groups:
                    groups.append([])
                groups[-1].append(idx)
                group_end = max(group_end, end_idx) if group_end >= 0 else end_idx
            else:
                groups.append([idx])
                group_end = end_idx

        for group in groups:
            group_start = min(valid_candidates[i]["start_idx"] for i in group)
            group_end = max(valid_candidates[i]["end_idx"] for i in group)
            group_len = group_end - group_start
            if group_len <= 0:
                continue

            group_y = original_audio[group_start:group_end]
            A_group = np.zeros((group_len, 2 * len(group)), dtype=np.float64)
            for col_i, cand_idx in enumerate(group):
                cand = valid_candidates[cand_idx]
                f = cand["mean_freq"]
                st = cand["start_time"]
                dec = cand["env_fit"]["decay_rate"]
                start_idx = cand["start_idx"]
                end_idx = cand["end_idx"]

                seg_t = t_vec[start_idx:end_idx]
                t_rel_seg = np.maximum(0.0, seg_t - st)
                env_seg = np.exp(-dec * t_rel_seg)
                omega = 2.0 * np.pi * f * t_rel_seg
                offset = start_idx - group_start
                A_group[offset:offset + (end_idx - start_idx), 2 * col_i] = env_seg * np.sin(omega)
                A_group[offset:offset + (end_idx - start_idx), 2 * col_i + 1] = env_seg * np.cos(omega)

            coefs, _, _, _ = np.linalg.lstsq(A_group, group_y, rcond=None)
            for col_i, cand_idx in enumerate(group):
                c_sin = coefs[2 * col_i]
                c_cos = coefs[2 * col_i + 1]
                refined_amp = float(np.sqrt(c_sin ** 2 + c_cos ** 2))
                refined_phi = float(np.arctan2(c_cos, c_sin))

                valid_candidates[cand_idx]["env_fit"]["a0"] = refined_amp
                valid_candidates[cand_idx]["init_phase"] = refined_phi

            # Joint frequency refinement: re-estimate each partial against the
            # residual left by its neighbours, then re-solve the group and keep
            # the result only if the group residual actually drops. The guard is
            # what makes this safe to run unconditionally.
            group_cands = [valid_candidates[i] for i in group]
            refined_f = _deflated_freq_refine(
                group_cands, group_y, group_start, sample_rate) \
                if JOINT_FREQ_REFINE else {}
            if refined_f:
                before = float(np.sum((group_y - A_group @ coefs) ** 2))
                for col_i, cand_idx in enumerate(group):
                    if col_i in refined_f:
                        valid_candidates[cand_idx]["mean_freq"] = refined_f[col_i]
                A_new = np.zeros((group_len, 2 * len(group)), dtype=np.float64)
                for col_i, cand_idx in enumerate(group):
                    cand = valid_candidates[cand_idx]
                    f = cand["mean_freq"]
                    st = cand["start_time"]
                    dec = cand["env_fit"]["decay_rate"]
                    start_idx = cand["start_idx"]
                    end_idx = cand["end_idx"]
                    seg_t = t_vec[start_idx:end_idx]
                    t_rel_seg = np.maximum(0.0, seg_t - st)
                    env_seg = np.exp(-dec * t_rel_seg)
                    omega = 2.0 * np.pi * f * t_rel_seg
                    offset = start_idx - group_start
                    A_new[offset:offset + (end_idx - start_idx), 2 * col_i] = env_seg * np.sin(omega)
                    A_new[offset:offset + (end_idx - start_idx), 2 * col_i + 1] = env_seg * np.cos(omega)
                coefs_new, _, _, _ = np.linalg.lstsq(A_new, group_y, rcond=None)
                after = float(np.sum((group_y - A_new @ coefs_new) ** 2))
                if after < before:
                    for col_i, cand_idx in enumerate(group):
                        c_sin = coefs_new[2 * col_i]
                        c_cos = coefs_new[2 * col_i + 1]
                        valid_candidates[cand_idx]["env_fit"]["a0"] = float(
                            np.sqrt(c_sin ** 2 + c_cos ** 2))
                        valid_candidates[cand_idx]["init_phase"] = float(
                            np.arctan2(c_cos, c_sin))
                else:
                    for col_i, cand_idx in enumerate(group):
                        valid_candidates[cand_idx]["mean_freq"] = float(
                            group_cands[col_i]["mean_freq"])

    for idx, cand in enumerate(valid_candidates):
        env_fit = cand["env_fit"]
        comp = {
            "id": f"component_{idx}",
            # Frequency needs more than 3 decimals: rounding to 0.001 Hz injects
            # a phase error that accumulates over the note (2*pi*0.001*t) and
            # shows up as a large error at the attack, where the signal is near
            # zero and any phase error is a large relative error.
            "frequency": round(cand["mean_freq"], 6),
            "amplitude": round(env_fit["a0"], 4),
            "phase": round(cand["init_phase"], 6),
            "start_time": round(cand["start_time"], 4),
            "duration": round(cand["duration"], 4),
            "decay_rate": round(env_fit["decay_rate"], 4),
            "envelope": {
                "type": env_fit["type"],
                "a0": round(env_fit["a0"], 4),
                "decay_rate": round(env_fit["decay_rate"], 4),
                "r_squared": round(env_fit.get("r_squared", 1.0), 4),
            },
        }
        components.append(comp)

    return {
        "model_name": "discovered_equations",
        "sample_rate": sample_rate,
        "duration": total_duration,
        "component_count": len(components),
        "components": components,
    }


# Joint frequency refinement (deflation-based, see docs/math_model.md 8.2).
# Toggle for A/B measurement; on by default.
JOINT_FREQ_REFINE = True


def _deflated_freq_refine(
      cands: List[Dict[str, Any]],
      group_y: np.ndarray,
      group_start: int,
      sr: int,
      width_hz: float = 0.05,
      n_eval: int = 13,
) -> Dict[int, float]:
    """Re-estimate each candidate's frequency against the deflated residual.

    The per-candidate fit inside :func:`discover_equations_from_audio` scores a
    partial against the *raw* audio, so overlapping neighbours bias its
    frequency. Here each candidate is fitted against the residual left by the
    **other** candidates in the group -- the same deflation principle as the
    clustered joint refinement in ``docs/math_model.md`` 8.2, applied to the
    discovery path.

    Large groups are left alone: 8.2 deliberately keeps large sets at their
    greedy optimum because joint optimisation of many partials is
    ill-conditioned, and the same reasoning applies here.

    Returns ``{candidate_index: refined_frequency}``; empty when the group is
    too large to refine safely.
    """
    n = len(cands)
    if n < 2 or n > 12:
        return {}

    def contribution(j: int, f: float, a0: float, ph: float) -> np.ndarray:
        """Candidate j's fitted contribution over the whole group span."""
        c = cands[j]
        out = np.zeros(len(group_y))
        s, e = int(c["start_idx"]), int(c["end_idx"])
        lo, hi = max(s, group_start), min(e, group_start + len(group_y))
        if hi <= lo:
            return out
        t_rel = np.maximum(0.0, np.arange(lo, hi) / float(sr) - c["start_time"])
        env = np.exp(-c["env_fit"]["decay_rate"] * t_rel)
        out[lo - group_start:hi - group_start] = (
            a0 * env * np.sin(2.0 * np.pi * f * t_rel + ph))
        return out

    refined: Dict[int, float] = {}
    for i, c in enumerate(cands):
        s, e = int(c["start_idx"]), int(c["end_idx"])
        lo, hi = max(s, group_start), min(e, group_start + len(group_y))
        if hi - lo < 32:
            continue
        others = np.zeros(hi - lo)
        for j in range(n):
            if j == i:
                continue
            oj = cands[j]
            cont = contribution(j, float(oj["mean_freq"]),
                                float(oj["env_fit"]["a0"]),
                                float(oj.get("init_phase", 0.0)))
            others += cont[lo - group_start:hi - group_start]
        target = group_y[lo - group_start:hi - group_start] - others

        t_rel = np.maximum(0.0, np.arange(lo, hi) / float(sr) - c["start_time"])
        env = np.exp(-c["env_fit"]["decay_rate"] * t_rel)
        f0 = float(c["mean_freq"])

        def resid_at(test_f: float) -> float:
            mat = np.column_stack((env * np.sin(2.0 * np.pi * test_f * t_rel),
                                   env * np.cos(2.0 * np.pi * test_f * t_rel)))
            try:
                coefs, *_ = np.linalg.lstsq(mat, target, rcond=None)
            except np.linalg.LinAlgError:
                return float("inf")
            err = target - mat @ coefs
            return float(np.sum(err ** 2))

        # A fixed grid is far too coarse: the frequency errors worth fixing
        # are ~1e-3 Hz, so the search must resolve well below that.
        from scipy.optimize import minimize_scalar
        opt = minimize_scalar(resid_at, bounds=(f0 - width_hz, f0 + width_hz),
                              method="bounded", options={"xatol": 1e-4})
        best_f = float(opt.x) if opt.success else f0
        if best_f != f0:
            refined[i] = best_f
    return refined


def discover_equations_from_audio(
    audio_path: Union[str, Path],
    n_fft: int = 4096,
    hop_length: int = 1024,
    threshold: float = 0.005,
    out_json_path: Optional[Union[str, Path]] = None,
    refine_with_projection: bool = True,
) -> Dict[str, Any]:
    """High-level pipeline: Audio -> STFT -> Normalized Peaks -> Tracking -> Equation Model."""
    from src.analysis import load_audio

    y, sr = load_audio(audio_path)
    dur = float(len(y) / sr)

    stft_res = compute_normalized_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    tracks = track_sinusoids(stft_res, threshold=threshold)
    model = tracks_to_equations(
        tracks,
        sample_rate=sr,
        total_duration=dur,
        original_audio=y if refine_with_projection else None,
        refine_with_projection=refine_with_projection,
    )

    if out_json_path:
        out_p = Path(out_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)

    return model
