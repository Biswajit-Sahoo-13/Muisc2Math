"""Sinusoidal Modeling Synthesis (SMS): phase-accurate audio <-> equations.

Following the classical SMS formulation (Quatieri & McAulay; Serra & Smith;
Peeters & Rodet), a tonal signal is modelled as a sum of sinusoidal tracks

    s(t) = sum_k  A_k(t) * sin( theta_k(t) ),   theta_k(t) = int 2*pi*f_k(t) dt

The model is a hybrid, decided per track by observed frequency spread:

* **Stationary partial** (spread within a few Hz): one refined frequency fitted
  by residual minimisation, amplitude and phase by least-squares projection.
  Interpolating the noisy per-frame frequency instead integrates its noise into
  a phase random walk, which is strictly worse (measured: 44.7 dB versus
  10.2 dB on a stationary 440 Hz partial).
* **Genuine glide** (vibrato, glissando): the per-frame frequency trajectory is
  integrated into a piecewise-linear phase, anchored per track. This is the
  classical SMS construction and the only one that follows a moving partial.

A Sines+Noise residual layer is included so that the non-tonal part of the
signal (which sinusoids model poorly) is represented as filtered noise, giving
the full S+N model rather than a sines-only approximation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import os

import numpy as np
import librosa

from src.tracking import SinusoidalTrack, track_sinusoids


def _interp_track(
    frame_times: np.ndarray,
    values: np.ndarray,
    n_samples: int,
    sr: int,
) -> np.ndarray:
    """Linearly interpolate per-frame parameters onto the sample grid."""
    if frame_times.size == 0:
        return np.zeros(n_samples, dtype=np.float64)
    if frame_times.size == 1:
        return np.full(n_samples, float(values[0]), dtype=np.float64)
    t_samples = np.arange(n_samples, dtype=np.float64) / sr
    return np.interp(t_samples, frame_times, values).astype(np.float64)


def _smooth_envelope(values: np.ndarray, width: int = 5) -> np.ndarray:
    """Median-smooth a per-frame amplitude trajectory.

    STFT peak amplitudes wobble with window overlap and boundary effects. That
    wobble is pure estimation noise, but resynthesis reproduces it verbatim as
    amplitude modulation, which is the dominant residual error once frequency is
    refined. A short median filter removes outliers without shifting the decay.
    """
    if values.size < width or width < 3:
        return values
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=-1)


def _analytic_envelope(
    times: np.ndarray,
    amps: np.ndarray,
    skip_frames: int = 2,
) -> Tuple[np.ndarray, float, float, float]:
    """Fit a closed-form envelope ``A0 * exp(-lambda * (t - t0))`` on interior frames.

    Per-frame STFT peak amplitudes are systematically attenuated in the first and
    last frames (window overlap and reflection padding). Copying those values into
    the resynthesis reproduces the artifact as real gain error, and it is the
    dominant residual once frequency and phase are correct.

    Returns ``(envelope_values, a0, decay, t0_anchor)``. ``t0_anchor`` is the
    absolute time the ``a0`` value refers to, needed for closed-form equations.
    The amplitudes are absolute, so the downstream projection must *not*
    renormalise them.
    """
    if times.size == 0:
        return amps, 0.0, 0.0, 0.0
    interior = (
        slice(skip_frames, times.size - skip_frames)
        if times.size > 2 * skip_frames + 2
        else slice(None)
    )
    ti = times[interior]
    anchor = float(ti[0]) if ti.size else float(times[0])
    ai = np.maximum(amps[interior], 1e-12)
    if ti.size < 4:
        a0 = float(np.median(amps))
        return np.full_like(amps, a0), a0, 0.0, anchor

    t_rel = ti - ti[0]
    try:
        slope, intercept = np.polyfit(t_rel, np.log(ai), 1)
    except Exception:
        a0 = float(np.median(amps))
        return np.full_like(amps, a0), a0, 0.0, anchor
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        a0 = float(np.median(amps))
        return np.full_like(amps, a0), a0, 0.0, anchor

    decay = max(0.0, -float(slope))
    a0 = float(np.exp(intercept))
    return a0 * np.exp(-decay * (times - ti[0])), a0, decay, anchor


def _prepare_frames(track: SinusoidalTrack) -> Tuple[np.ndarray, ...]:
    times = np.asarray(track.times, dtype=np.float64)
    freqs = np.asarray(track.frequencies, dtype=np.float64)
    amps = np.asarray(track.amplitudes, dtype=np.float64)
    phases = np.asarray(track.phases, dtype=np.float64)
    med_f = float(np.median(freqs)) if freqs.size else 0.0
    if freqs.size:
        # Phase 6: Allow wide pitch glides/vibrato (up to 35% of fundamental or 25 Hz)
        # instead of the legacy 2.0 Hz clamp which destroyed 808 pitch glides
        keep = np.abs(freqs - med_f) <= max(25.0, 0.35 * med_f)
        if int(np.count_nonzero(keep)) >= 2:
            times, freqs = times[keep], freqs[keep]
            amps, phases = amps[keep], phases[keep]
    amps, a0, decay, anchor = _analytic_envelope(times, amps)
    return times, freqs, amps, phases, a0, decay, anchor


def _trajectories(
    times: np.ndarray,
    freqs: np.ndarray,
    amps: np.ndarray,
    n_samples: int,
    sr: int,
    freq_scale: float = 1.0,
    normalize_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (amplitude, integral, frequency) trajectories on the sample grid."""
    f_traj = _interp_track(times, _smooth_envelope(freqs * freq_scale, 9), n_samples, sr)
    amp_env = _smooth_envelope(amps)
    if normalize_amp:
        peak = float(np.max(amp_env))
        if peak > 1e-12:
            amp_env = amp_env / peak
    a_traj = _interp_track(times, amp_env, n_samples, sr)
    integral_frames = np.zeros(times.size, dtype=np.float64)
    if times.size > 1:
        f_smooth = _smooth_envelope(freqs * freq_scale, 9)
        integral_frames[1:] = np.cumsum(
            2.0 * np.pi * 0.5 * (f_smooth[:-1] + f_smooth[1:]) * np.diff(times)
        )
    integral = _interp_track(times, integral_frames, n_samples, sr)
    return a_traj, integral, f_traj


def synthesize_track(
    track: SinusoidalTrack,
    n_samples: int,
    sr: int,
) -> np.ndarray:
    """Render one track with an integrated phase trajectory (phase-agnostic).

    The absolute phase offset is not meaningful on its own: the measured STFT
    phase carries an unknown constant window offset. Callers that have the
    reference signal should use :func:`refine_track`, which recovers the offset
    (and the amplitude) by projection instead of assuming a convention.
    """
    times, freqs, amps, _phases, _a0, _decay, _anchor = _prepare_frames(track)
    if times.size == 0:
        return np.zeros(n_samples, dtype=np.float64)
    a_traj, integral, _ = _trajectories(times, freqs, amps, n_samples, sr)
    return a_traj * np.cos(integral)


def refine_track(
    track: SinusoidalTrack,
    reference: np.ndarray,
    sr: int,
    freq_tolerance_hz: float = 2.0,
    glide_tolerance_hz: float = 6.0,
    refine_frequency: bool = True,
    stft_half_window_s: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit one track's frequency and complex gain against the reference signal.

    Two corrections happen here, and both are necessary for a faithful round
    trip:

    * **Frequency.** Spectral peak estimates carry a small bias (order 0.1 Hz on
      a 440 Hz partial here). Because phase is the *integral* of frequency, that
      bias accumulates into tens of radians over a few seconds and rotates the
      whole track away from the original.
    * **Complex gain.** The measured phase has an unknown constant offset, so
      amplitude and phase are recovered by a two-column least-squares projection
      onto the track's cosine and sine components, which is exactly optimal.

    ``stft_half_window_s`` (n_fft / 2 / sr) marks how far back an STFT frame
    reaches: the frame that detected the peak integrates that far before the
    frame time, so content there is measured, not invented. The track support
    extends one half-window backwards accordingly; 0 disables the extension.
    """
    times, freqs, amps, phases, env_a0, env_decay, env_t0 = _prepare_frames(track)
    if times.size == 0:
        return np.zeros(len(reference)), {
            "frequency": 0.0,
            "amplitude": 0.0,
            "phase": 0.0,
            "envelope_a0": 0.0,
            "envelope_decay": 0.0,
            "envelope_t0": 0.0,
            "projection_gain": 0.0,
            "model": "constant",
            "freq_spread_hz": 0.0,
            "start_time": 0.0,
            "end_time": 0.0,
        }

    n = len(reference)
    t0s = float(times[0])
    t1s = float(times[-1])
    pre_start = (
        max(0.0, t0s - stft_half_window_s) if stft_half_window_s > 0.0 else t0s
    )
    start = max(0, int(pre_start * sr))
    end = min(n, int(times[-1] * sr) + int(0.25 * sr))
    if end - start < 64:
        end = min(n, start + int(0.25 * sr))

    f_guess = float(np.median(freqs))
    # A track whose peak frequencies really move (vibrato, glissando, portamento)
    # must keep its trajectory; a stationary one must not, or its estimation noise
    # is integrated into phase drift. Decide by observed spread.
    # Robust spread: a few corrupted boundary frames can fake a wide range on an
    # otherwise stationary partial, so measure dispersion around the median
    # instead of using max-min.
    if freqs.size >= 4:
        spread = (
            float(np.max(freqs) - np.min(freqs))
            if f_guess <= 150.0
            else float(np.percentile(freqs, 95) - np.percentile(freqs, 5))
        )
    elif freqs.size:
        spread = 0.0
    else:
        spread = 0.0
    use_trajectory = spread > abs(glide_tolerance_hz)

    t_axis: Optional[np.ndarray] = None
    amp_traj: Optional[np.ndarray] = None
    integral: Optional[np.ndarray] = None
    support_mask: Optional[np.ndarray] = None
    if use_trajectory:
        a_traj, integral, _ = _trajectories(
            times, freqs, amps, n, sr, freq_scale=1.0, normalize_amp=False
        )
    else:
        # The amplitude envelope does not depend on frequency, so compute it
        # once; the frequency search below would otherwise rebuild both
        # trajectories on every scalar-minimiser evaluation (~30 per track).
        amp_traj = _interp_track(times, amps, n, sr)
        t_axis = np.arange(n, dtype=np.float64) / sr

    # A partial does not exist for ever. The raw envelope is clamped by
    # np.interp outside the tracked frames, so the rendered track rings at its
    # boundary amplitude over the entire signal - invented energy (before the
    # track starts, and for ever after) that poisons the residual test: a
    # short but real fragment loses more energy outside its window than it
    # removes inside it, and is pruned even though it genuinely explains part
    # of the signal. Mask the envelope so the rendered track lives only where
    # the STFT measurement and the projection window support it: content one
    # half-window before the first frame (that reach is what the detecting
    # window measured), constant through the projection window, and past the
    # last frame a fitted decay extends a little before fading out.
    grid = np.arange(n, dtype=np.float64) / sr
    if stft_half_window_s > 0.0:
        pre_start = max(0.0, t0s - stft_half_window_s)
        if (t0s - stft_half_window_s) < 0.0:
            # First frame of the signal: the detecting window touches the
            # signal start, so the partial demonstrably exists at full
            # strength at t=0. A rising ramp would only attenuate real signal.
            attack_term = np.where(grid >= 0.0, 1.0, 0.0)
        else:
            attack_term = np.clip(
                (grid - pre_start) / max(t0s - pre_start, 1e-9), 0.0, 1.0
            )
    else:
        pre_start = t0s
        attack_len = min(0.02, 0.5 * max(t0s, 1e-6))
        attack_term = (
            np.clip((grid - pre_start) / attack_len, 0.0, 1.0)
            if attack_len > 1e-9
            else np.where(grid >= pre_start, 1.0, 0.0)
        )

    if env_decay > 0.0 and env_a0 > 0.0:
        hold_s = min(1.5, float(np.log(100.0)) / env_decay)
    else:
        hold_s = 0.25
    support_end = t1s + hold_s
    release_term = np.clip((grid - support_end) / 0.25, 0.0, 1.0)
    support_mask = attack_term * (1.0 - release_term)

    def project(freq: float) -> Tuple[float, float, float, np.ndarray]:
        if use_trajectory:
            theta = integral + 2.0 * np.pi * (freq - f_guess) * (np.arange(n) / sr)
            c = a_traj * np.cos(theta)
            s = a_traj * np.sin(theta)
        else:
            ang = 2.0 * np.pi * freq * t_axis
            c = amp_traj * np.cos(ang)
            s = amp_traj * np.sin(ang)
        c = c * support_mask
        s = s * support_mask
        mat = np.column_stack((c[start:end], s[start:end]))
        coefs, *_ = np.linalg.lstsq(mat, reference[start:end], rcond=None)
        sig = np.zeros(n, dtype=np.float64)
        sig += coefs[0] * c + coefs[1] * s
        energy = float(np.sum((reference[start:end] - mat @ coefs) ** 2))
        return energy, float(coefs[0]), float(coefs[1]), sig

    best_f = f_guess
    if refine_frequency:
        from scipy.optimize import minimize_scalar

        lo = f_guess - abs(freq_tolerance_hz)
        hi = f_guess + abs(freq_tolerance_hz)
        if lo > 10.0 and hi < sr / 2.0 - 1.0:

            def objective(f: float) -> float:
                return project(f)[0]

            try:
                res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
                if res.success:
                    best_f = float(res.x)
            except Exception:
                best_f = f_guess

    _, c_coef, s_coef, sig = project(best_f)
    coeff = float(np.hypot(c_coef, s_coef))
    phase = float(np.arctan2(s_coef, c_coef))
    params = {
        "frequency": round(best_f, 4),
        # Physical amplitude = envelope a0 scaled by the fitted coefficient.
        "amplitude": round(float(env_a0) * coeff, 6),
        "projection_gain": round(coeff, 6),
        "envelope_a0": round(float(env_a0), 6),
        "envelope_decay": round(float(env_decay), 6),
        # Anchor time of the analytic envelope: needed for exact closed-form
        # equations  A(t) = a0 * exp(-decay * (t - envelope_t0)).
        "envelope_t0": round(float(env_t0), 6),
        "phase": round(phase, 6),
        "model": "glide" if use_trajectory else "constant",
        "freq_spread_hz": round(spread, 3),
        "start_time": round(float(times[0]), 6),
        "end_time": round(float(times[-1]), 6),
    }
    if use_trajectory:
        params["trajectory"] = {
            "t": [round(float(v), 6) for v in times],
            "f_hz": [round(float(v), 4) for v in freqs],
            "a": [round(float(v), 6) for v in amps],
        }
    return sig, params


def synthesize_tracks(
    tracks: Sequence[SinusoidalTrack],
    n_samples: int,
    sr: int,
) -> np.ndarray:
    """Sum all sinusoidal tracks into one phase-continuous signal."""
    out = np.zeros(n_samples, dtype=np.float64)
    for track in tracks:
        out += synthesize_track(track, n_samples, sr)
    return out


def model_tracks(
    tracks: Sequence[SinusoidalTrack],
    n_samples: int,
    sr: int,
) -> List[Dict[str, Any]]:
    """Describe each track as a maths equation with its full trajectories."""
    out: List[Dict[str, Any]] = []
    for idx, track in enumerate(tracks):
        times = np.asarray(track.times, dtype=np.float64)
        freqs = np.asarray(track.frequencies, dtype=np.float64)
        amps = np.asarray(track.amplitudes, dtype=np.float64)
        phases = np.asarray(track.phases, dtype=np.float64)
        if times.size == 0:
            continue
        out.append({
            "id": f"sinusoid_{idx}",
            "expression": "A(t) * sin(theta(t)),  theta(t) = int 2*pi*f(t) dt",
            "start_time": round(float(times[0]), 6),
            "end_time": round(float(times[-1]), 6),
            "mean_frequency_hz": round(float(np.median(freqs)), 3),
            "peak_amplitude": round(float(np.max(amps)), 5),
            "initial_phase": round(float(phases[0]), 5),
            "frames": int(times.size),
            "f_hz": [round(float(v), 4) for v in freqs],
            "a": [round(float(v), 6) for v in amps],
            "phi": [round(float(v), 5) for v in phases],
            "frame_times": [round(float(v), 6) for v in times],
        })
    return out


def noise_residual_layer(
    original: np.ndarray,
    sinusoids: np.ndarray,
    sr: int,
    n_bands: int = 6,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sines+Noise stochastic layer: noise shaped to the residual spectrum.

    Plain white noise at the residual's RMS energy does not actually model the
    residual - it only matches its power, so it neither sounds like nor
    correlates with what is left. The residual's magnitude spectrum is therefore
    measured, smoothed across time, and imposed on a noise generator, which is
    the standard S+N construction: the noise then carries the correct *timbre*
    in each band instead of a flat one.
    """
    residual = original - sinusoids
    if residual.size == 0:
        return np.zeros(0, dtype=np.float64), {"enabled": False}

    n_fft = 1024
    hop = max(1, n_fft // 4)
    if residual.size < n_fft:
        return np.zeros_like(residual), {"enabled": False}

    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    if residual_rms <= 1e-12:
        # Sinusoids already explain everything; adding noise would only
        # corrupt the result. A silent residual makes the spectrum-shaping
        # ratios below 0/0 = NaN, so bail out before that.
        return np.zeros_like(residual), {"enabled": False, "reason": "silent residual"}

    mag = np.abs(librosa.stft(residual, n_fft=n_fft, hop_length=hop))
    if mag.size == 0:
        return np.zeros_like(residual), {"enabled": False}

    # Smooth in time: the noise layer should be stationary per band, not flicker.
    kernel = np.hanning(9)
    kernel /= np.sum(kernel)
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), 1, mag
    )

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(residual.size)
    target_spec = np.abs(librosa.stft(white, n_fft=n_fft, hop_length=hop))
    if float(np.mean(target_spec)) <= 1e-12:
        return np.zeros_like(residual), {"enabled": False, "reason": "degenerate white noise"}

    # In STFT domain: replace the white noise's magnitude with the target's,
    # keep its random phase, then invert.
    ratio = (smoothed / (target_spec + 1e-12)).mean(axis=1, keepdims=True)
    shaped_stft = librosa.stft(white, n_fft=n_fft, hop_length=hop)
    shaped = shaped_stft * (smoothed / (np.abs(shaped_stft) + 1e-12)) / ratio
    noise = librosa.istft(shaped, hop_length=hop, length=residual.size)
    if noise.shape[0] < residual.size:
        noise = np.pad(noise, (0, residual.size - noise.shape[0]))

    residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms > 1e-12:
        noise *= residual_rms / noise_rms

    return noise, {
        "enabled": True,
        "construction": "spectrum-shaped noise (S+N)",
        "n_fft": int(n_fft),
        "hop_length": int(hop),
        "band_rms": [
            round(float(np.sqrt(np.mean(smoothed[i] ** 2))), 6)
            for i in range(smoothed.shape[0])
        ],
    }


HIGH_BAND_HZ = 2000.0
HIGH_TRACK_FRACTION = 1.0 / 3.0


def select_tracks_banded(tracks: List[SinusoidalTrack], max_tracks: int) -> List[SinusoidalTrack]:
    """Truncate to ``max_tracks`` with slots reserved for the high band.

    A plain amplitude sort plus ``[:max_tracks]`` structurally deletes the top
    end of the spectrum. Music carries roughly -12 dB/octave of spectral tilt,
    so a 60 Hz fundamental outranks a 6 kHz cymbal component no matter how
    present the cymbal is, and every slot is spent below the tilt line. That is
    what leaves 2.5-5 kHz ~13 dB down on a dense master.

    When the tracker returns no more tracks than the budget the sort is returned
    untouched, so short/quiet material is completely unaffected.
    """
    by_amp = sorted(tracks, key=lambda tr: tr.peak_amplitude, reverse=True)
    if max_tracks <= 0 or len(by_amp) <= max_tracks:
        return by_amp
    n_high = max(1, int(round(max_tracks * HIGH_TRACK_FRACTION)))
    high = [t for t in by_amp if t.mean_frequency > HIGH_BAND_HZ]
    rest = [t for t in by_amp if t.mean_frequency <= HIGH_BAND_HZ]
    reserved = high[:n_high]
    # Fill the remaining slots from every leftover (including high-band tracks
    # beyond the reservation), still ordered by amplitude. Filling only from
    # `rest` would let the reservation cap the result when the population is
    # itself mostly high-band.
    reserved_ids = {id(t) for t in reserved}
    leftover = [t for t in by_amp if id(t) not in reserved_ids]
    chosen = reserved + leftover[: max(0, max_tracks - len(reserved))]
    chosen.sort(key=lambda tr: tr.peak_amplitude, reverse=True)
    return chosen


def _reconstruct_window_task(args: Tuple[Any, ...]) -> Dict[str, Any]:
    """Picklable per-window reconstruction worker.

    Windows are disjoint slices of the source, so the caller pastes each result
    into its own region and ``executor.map`` preserves window order -- the
    parallel result is bit-for-bit identical to the serial loop.
    """
    (chunk, sr, n_fft, hop_length, threshold, max_tracks, min_track_frames,
     refine_frequency, min_residual_ratio, seed, glide_tolerance_hz) = args
    return sinusoidal_reconstruction(
        chunk,
        sr,
        n_fft=n_fft,
        hop_length=hop_length,
        threshold=threshold,
        max_tracks=max_tracks,
        min_track_frames=min_track_frames,
        add_noise_layer=False,
        refine_frequency=refine_frequency,
        min_residual_ratio=min_residual_ratio,
        seed=seed,
        window_s=0.0,
        glide_tolerance_hz=glide_tolerance_hz,
        workers=1,
    )


def _reconstruction_windowed(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    threshold: float,
    max_tracks: int,
    min_track_frames: int,
    refine_frequency: bool,
    min_residual_ratio: float,
    seed: int,
    window_s: float,
    add_noise_layer: bool,
    glide_tolerance_hz: float = 6.0,
    workers: Optional[int] = None,
    progress: Optional[Callable[[str, float, str], None]] = None,
) -> Dict[str, Any]:
    """Run the reconstruction section-by-section.

    The OMP acceptance test works in *energy fractions* of the signal being
    analysed. Over a whole song, a real but short partial contributes almost
    nothing to the total energy, so every fragment fails the test even though
    it genuinely explains its own few milliseconds. Sectioning gives each event
    a local denominator: with a 2.5 s window, a 0.1 s hit competes against 2.5 s
    of energy, not 20 s, and survives. Section boundaries slightly truncate
    partials spanning them, which is priced in by the OMP test.
    """
    n_samples = len(y)
    win = max(int(round(window_s * sr)), 4 * hop_length)
    sin_audio = np.zeros(n_samples, dtype=np.float64)
    fitted: List[Dict[str, Any]] = []
    accepted_tracks: List[SinusoidalTrack] = []
    candidate_count = 0
    pruned_total = 0

    starts: List[int] = []
    tasks: List[Tuple[Any, ...]] = []
    for w_start in range(0, n_samples, win):
        chunk = y[w_start:w_start + win]
        if len(chunk) < 4 * hop_length:
            break
        starts.append(w_start)
        tasks.append((
            chunk, sr, n_fft, hop_length, threshold, max_tracks,
            min_track_frames, refine_frequency, min_residual_ratio,
            seed, glide_tolerance_hz,
        ))

    n_workers = (
        workers if workers is not None
        else min(8, os.cpu_count() or 1, max(1, len(tasks)))
    )
    subs: List[Dict[str, Any]] = []
    n_tasks = max(1, len(tasks))
    if n_workers > 1 and len(tasks) > 1:
        try:
            from src.res_loop import _get_pool
            # Stream the map so progress is reported as windows complete
            # instead of appearing all at once at the end.
            for wi, sub in enumerate(_get_pool(n_workers).map(
                    _reconstruct_window_task, tasks)):
                subs.append(sub)
                if progress is not None:
                    progress("evidence:windows", (wi + 1) / n_tasks,
                             f"analysis window {wi + 1}/{n_tasks}")
        except Exception:
            subs = []
    if not subs:
        subs = []
        for wi, t in enumerate(tasks):
            subs.append(_reconstruct_window_task(t))
            if progress is not None:
                progress("evidence:windows", (wi + 1) / n_tasks,
                         f"analysis window {wi + 1}/{n_tasks}")

    for w_start, sub in zip(starts, subs):
        off = w_start / sr
        for tr in sub["tracks"]:
            tr.times = [float(v) + off for v in tr.times]
        for p in sub["fitted_parameters"]:
            for key in ("envelope_t0", "start_time", "end_time"):
                p[key] = round(float(p.get(key, 0.0)) + off, 6)
            if "trajectory" in p:
                p["trajectory"]["t"] = [
                    round(float(v) + off, 6) for v in p["trajectory"]["t"]
                ]
        sin_audio[w_start:w_start + len(sub["sinusoid_audio"])] = sub["sinusoid_audio"]
        fitted.extend(sub["fitted_parameters"])
        accepted_tracks.extend(sub["tracks"])
        candidate_count += sub["candidate_tracks"]
        pruned_total += sub["pruned_tracks"]

    noise_audio = np.zeros(n_samples, dtype=np.float64)
    noise_meta: Dict[str, Any] = {"enabled": False}
    if add_noise_layer:
        noise_audio, noise_meta = noise_residual_layer(y, sin_audio, sr=sr, seed=seed)

    recon = sin_audio + noise_audio
    ref_e = float(np.sum(y ** 2))
    err_e = float(np.sum((y - recon) ** 2))
    snr_db = 10.0 * np.log10((ref_e + 1e-20) / (err_e + 1e-20))
    std_y = float(np.std(y))
    std_r = float(np.std(recon))
    corr = (
        float(np.corrcoef(y, recon)[0, 1])
        if std_y > 1e-12 and std_r > 1e-12
        else 0.0
    )

    return {
        "audio": recon,
        "sinusoid_audio": sin_audio,
        "noise_audio": noise_audio,
        "tracks": accepted_tracks,
        "model": model_tracks(accepted_tracks, n_samples, sr),
        "fitted_parameters": fitted,
        "track_count": len(fitted),
        "candidate_tracks": candidate_count,
        "pruned_tracks": pruned_total,
        "snr_db": round(snr_db, 3),
        "correlation": round(corr, 6),
        "noise_layer": noise_meta,
        "sr": int(sr),
        "n_samples": int(n_samples),
    }


def sinusoidal_reconstruction(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 256,
    threshold: float = 0.01,
    max_tracks: int = 400,
    min_track_frames: int = 3,
    add_noise_layer: bool = True,
    refine_frequency: bool = True,
    min_residual_ratio: float = 1e-6,
    seed: int = 0,
    window_s: float = 0.0,
    glide_tolerance_hz: float = 6.0,
    workers: Optional[int] = None,
    progress: Optional[Callable[[str, float, str], None]] = None,
) -> Dict[str, Any]:
    """Full analysis/synthesis round trip using the SMS model.

    Returns the reconstructed signal together with the sinusoidal tracks and the
    achieved SNR, so the result can be verified rather than assumed.

    ``window_s`` > 0 analyses section-by-section (see
    :func:`_reconstruction_windowed`); 0 analyses the whole signal at once.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples = len(y)
    if n_samples == 0:
        return {
            "audio": np.zeros(0),
            "sinusoid_audio": np.zeros(0),
            "noise_audio": np.zeros(0),
            "tracks": [],
            "model": [],
            "fitted_parameters": [],
            "track_count": 0,
            "candidate_tracks": 0,
            "pruned_tracks": 0,
            "snr_db": 0.0,
            "correlation": 0.0,
            "noise_layer": {"enabled": False},
            "sr": int(sr),
            "n_samples": 0,
        }

    if window_s and window_s > 0.0:
        return _reconstruction_windowed(
            y, sr, n_fft=n_fft, hop_length=hop_length, threshold=threshold,
            max_tracks=max_tracks, min_track_frames=min_track_frames,
            refine_frequency=refine_frequency,
            min_residual_ratio=min_residual_ratio, seed=seed,
            window_s=window_s, add_noise_layer=add_noise_layer,
            glide_tolerance_hz=glide_tolerance_hz,
            workers=workers, progress=progress,
        )

    from src.analysis import compute_normalized_stft

    stft_res = compute_normalized_stft(y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    tracks = track_sinusoids(
        stft_res,
        threshold=threshold,
        min_track_duration_frames=min_track_frames,
    )
    tracks = select_tracks_banded(tracks, max_tracks)

    sin_audio = np.zeros(n_samples, dtype=np.float64)
    fitted: List[Dict[str, Any]] = []
    pruned = 0

    # Accept tracks greedily by residual reduction rather than by a fixed
    # threshold. A fixed floor cannot work across material: on dense music most
    # genuine partials are individually faint (4% of total amplitude here), so
    # thresholding discards real content and leaves a huge residual.
    #
    # Order by candidate strength, then keep a track only if removing the
    # projection from the residual measurably reduces its energy. This is
    # orthogonal matching pursuit over the sinusoidal dictionary, so the
    # coefficients stay jointly optimal and no track can be added that the
    # already-accepted tracks already explain.
    residual = y.copy()
    min_energy_drop = float(min_residual_ratio) * float(np.sum(y ** 2))

    accepted: List[Tuple[SinusoidalTrack, Dict[str, Any], np.ndarray, float]] = []
    for track in tracks:
        sig, params = refine_track(
            track, residual, sr, refine_frequency=refine_frequency,
            stft_half_window_s=(n_fft / 2.0) / sr,
            glide_tolerance_hz=glide_tolerance_hz,
        )
        energy_before = float(np.sum(residual ** 2))
        residual = residual - sig
        energy_after = float(np.sum(residual ** 2))
        if energy_after > energy_before - min_energy_drop:
            residual = residual + sig
            pruned += 1
            continue
        accepted.append((track, params, sig, energy_after))

    accepted.sort(key=lambda item: item[3])
    accepted_tracks: List[SinusoidalTrack] = []
    for track, params, sig, _ in accepted:
        sin_audio += sig
        fitted.append(params)
        accepted_tracks.append(track)

    noise_audio = np.zeros(n_samples, dtype=np.float64)
    noise_meta: Dict[str, Any] = {"enabled": False}
    if add_noise_layer:
        noise_audio, noise_meta = noise_residual_layer(
            y, sin_audio, sr=sr, seed=seed
        )

    recon = sin_audio + noise_audio

    ref_e = float(np.sum(y ** 2))
    err_e = float(np.sum((y - recon) ** 2))
    snr_db = 10.0 * np.log10((ref_e + 1e-20) / (err_e + 1e-20))
    std_y = float(np.std(y))
    std_r = float(np.std(recon))
    corr = (
        float(np.corrcoef(y, recon)[0, 1])
        if std_y > 1e-12 and std_r > 1e-12
        else 0.0
    )

    return {
        "audio": recon,
        "sinusoid_audio": sin_audio,
        "noise_audio": noise_audio,
        "tracks": accepted_tracks,
        # The model must describe what synthesis actually uses: candidates OMP
        # pruned carry no fitted parameters, so listing them would overstate it.
        "model": model_tracks(accepted_tracks, n_samples, sr),
        "fitted_parameters": fitted,
        "track_count": len(fitted),
        "candidate_tracks": len(tracks),
        "pruned_tracks": pruned,
        "snr_db": round(snr_db, 3),
        "correlation": round(corr, 6),
        "noise_layer": noise_meta,
        "sr": int(sr),
        "n_samples": int(n_samples),
    }


__all__ = [
    "model_tracks",
    "noise_residual_layer",
    "refine_track",
    "sinusoidal_reconstruction",
    "synthesize_track",
    "synthesize_tracks",
]