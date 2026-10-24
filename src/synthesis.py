"""Phase-integrated Forward Synthesis Engine (Math -> Audio).

Adheres to DSP principles:
1. True instantaneous phase integration: theta(t) = phi_0 + 2*pi * integral(f(tau) dtau)
   Discrete implementation: np.cumsum(2 * np.pi * f / sr) avoids pitch-widening artifacts.
2. Exact envelope modeling: exponential decay, ADSR, linear, and constant.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import soundfile as sf


def compute_phase_integration(
    f_trajectory: np.ndarray,
    sr: int,
    initial_phase: float = 0.0,
) -> np.ndarray:
    """Computes instantaneous phase via trapezoidal cumulative integration.

    Formula:
        theta[n] = phi_0 + 2*pi * cumsum(f) / sr
    Using trapezoidal rule preserves sub-sample accuracy for dynamic frequency bends.
    """
    if len(f_trajectory) == 0:
        return np.array([], dtype=np.float64)
    # Trapezoidal cumulative sum for accurate phase integration
    dt = 1.0 / sr
    f_mid = 0.5 * (f_trajectory[:-1] + f_trajectory[1:])
    theta = np.empty(len(f_trajectory), dtype=np.float64)
    theta[0] = initial_phase
    theta[1:] = initial_phase + 2.0 * np.pi * np.cumsum(f_mid) * dt
    return theta


def compute_envelope(
    env_cfg: Dict[str, Any],
    t: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Computes time-domain amplitude envelope A(t).

    Supported types:
    - 'exponential': A_0 * exp(-lambda * t)
    - 'constant': A_0
    - 'linear': A_0 + slope * t
    - 'adsr': standard Attack, Decay, Sustain, Release
    """
    env_type = env_cfg.get("type", "constant").lower()
    a0 = float(env_cfg.get("a0", env_cfg.get("amplitude", 1.0)))

    if env_type == "exponential":
        decay_rate = float(env_cfg.get("decay_rate", env_cfg.get("lambda", 0.0)))
        return a0 * np.exp(-decay_rate * t)

    elif env_type == "constant":
        return np.full_like(t, a0)

    elif env_type == "linear":
        decay_rate = float(env_cfg.get("decay_rate", 0.0))
        return np.maximum(0.0, a0 - decay_rate * t)

    elif env_type == "adsr":
        # Attack, Decay, Sustain, Release parameters
        # times in seconds, sustain is relative fraction (0.0 - 1.0) of a0
        att = float(env_cfg.get("attack", 0.05))
        dec = float(env_cfg.get("decay", 0.1))
        sus = float(env_cfg.get("sustain", 0.7)) * a0
        rel = float(env_cfg.get("release", 0.1))

        env = np.zeros_like(t)
        sus_end = max(att + dec, duration - rel)

        # Attack: 0 -> a0
        mask_att = t < att
        if np.any(mask_att) and att > 0:
            env[mask_att] = a0 * (t[mask_att] / att)

        # Decay: a0 -> sus
        mask_dec = (t >= att) & (t < att + dec)
        if np.any(mask_dec) and dec > 0:
            env[mask_dec] = a0 - (a0 - sus) * ((t[mask_dec] - att) / dec)

        # Sustain
        mask_sus = (t >= att + dec) & (t < sus_end)
        if np.any(mask_sus):
            env[mask_sus] = sus

        # Release: sus -> 0
        mask_rel = t >= sus_end
        if np.any(mask_rel) and rel > 0:
            env[mask_rel] = np.maximum(0.0, sus * (1.0 - (t[mask_rel] - sus_end) / rel))

        return env

    elif env_type == "sampled":
        # # ponytail: 1D linear interpolation for arbitrary envelope curves; upgrade to cubic spline if smooth derivatives needed
        points = env_cfg.get("points", [(0.0, a0), (duration, 0.0)])
        t_pts, a_pts = zip(*points)
        return np.interp(t, t_pts, a_pts)

    else:
        return np.full_like(t, a0)


def render_component(
    comp: Dict[str, Any],
    duration: float,
    sr: int = 44100,
) -> np.ndarray:
    """Renders a single mathematical component into audio samples.

    Component format:
    {
        "frequency": 440.0, # or trajectory config
        "amplitude": 0.5,   # base amplitude
        "phase": 0.0,       # initial phase in radians
        "envelope": {"type": "exponential", "decay_rate": 1.2, "a0": 0.5},
        "vibrato": {"frequency": 5.0, "depth": 3.0} # optional
    }
    """
    n_samples = int(np.round(duration * sr))
    t = np.arange(n_samples, dtype=np.float64) / sr

    # 1. Frequency trajectory
    base_f = float(comp.get("frequency", 440.0))
    f_trajectory = np.full(n_samples, base_f, dtype=np.float64)

    # Optional vibrato
    if "vibrato" in comp:
        vib = comp["vibrato"]
        v_f = float(vib.get("frequency", 5.0))
        v_d = float(vib.get("depth", 2.0))
        f_trajectory += v_d * np.sin(2.0 * np.pi * v_f * t)

    # Optional pitch bend / chirp
    if "pitch_bend" in comp:
        bend = comp["pitch_bend"]
        end_f = float(bend.get("target_frequency", base_f))
        f_trajectory += (end_f - base_f) * (t / duration)

    # 2. Phase integration
    initial_phase = float(comp.get("phase", 0.0))
    phase = compute_phase_integration(f_trajectory, sr, initial_phase)

    # 3. Envelope
    env_cfg = comp.get("envelope", {})
    if not env_cfg and "amplitude" in comp:
        decay = float(comp.get("decay_rate", comp.get("lambda", 0.0)))
        if decay > 0:
            env_cfg = {"type": "exponential", "a0": comp["amplitude"], "decay_rate": decay}
        else:
            env_cfg = {"type": "constant", "a0": comp["amplitude"]}

    amp = compute_envelope(env_cfg, t, duration)

    # 4. Synthesize carrier: A(t) * sin(theta(t))
    # # ponytail: sine oscillator only; harmonics are modeled as distinct sinusoidal partials
    return (amp * np.sin(phase)).astype(np.float64)


def render_model(
    model: Union[Dict[str, Any], str, Path],
    duration: Optional[float] = None,
    sr: int = 44100,
    normalize: bool = False,
) -> Tuple[np.ndarray, int]:
    """Renders a multi-component mathematical audio model.

    Returns:
        (audio_array, sample_rate)
    """
    if isinstance(model, (str, Path)):
        with open(model, "r", encoding="utf-8") as f:
            model_dict = json.load(f)
    else:
        model_dict = model

    sr = int(model_dict.get("sample_rate", sr))
    model_dur = float(model_dict.get("duration", 2.0))
    dur = duration if duration is not None else model_dur
    n_samples = int(np.round(dur * sr))

    components = model_dict.get("components", [])
    output = np.zeros(n_samples, dtype=np.float64)

    for comp in components:
        # Check start_time and component-specific duration
        start_time = float(comp.get("start_time", 0.0))
        comp_dur = float(comp.get("duration", dur - start_time))

        if start_time >= dur or comp_dur <= 0:
            continue

        comp_dur = min(comp_dur, dur - start_time)
        sig = render_component(comp, comp_dur, sr)

        start_idx = int(np.round(start_time * sr))
        end_idx = start_idx + len(sig)

        if end_idx > n_samples:
            sig = sig[: n_samples - start_idx]
            end_idx = n_samples

        output[start_idx:end_idx] += sig

    if normalize and np.max(np.abs(output)) > 0:
        output = output / (np.max(np.abs(output)) + 1e-12) * 0.95

    return output, sr


def save_audio(filename: Union[str, Path], audio: np.ndarray, sr: int = 44100,
               subtype: str = "PCM_24") -> int:
    """Save a float numpy audio array as a WAV file.

    Returns the number of samples that had to be clipped to fit the target
    format. Silently clipping is how the synthetic benchmark fixture used to
    lose 95 samples and make the engine look 3 dB worse than it is, so the
    count is now observable instead of silent.

    ``subtype="FLOAT"`` (32-bit float) preserves samples outside [-1, 1]
    exactly; integer subtypes cannot represent them.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float64)
    if subtype.upper() == "FLOAT":
        sf.write(str(path), audio, sr, subtype="FLOAT")
        return 0
    clipped = np.clip(audio, -1.0, 1.0)
    n_clipped = int(np.count_nonzero(np.abs(audio) > 1.0))
    sf.write(str(path), clipped, sr, subtype=subtype)
    return n_clipped


def generate_benchmark_chord(
      out_path: Union[str, Path],
      duration: float = 2.0,
      sr: int = 44100,
      target_peak: float = 0.95,
) -> Dict[str, Any]:
    """Step 1.3: Generates math_chord.wav with known, ground-truth mathematical parameters.

    Triad: A4 (440.0 Hz), C#5 (554.365 Hz), E5 (659.255 Hz)
    Known exponential decays and initial amplitudes for closed-loop reverse engineering verification.

    The three partials add constructively to a peak of 1.14, which overran
    full scale and was hard-clipped by the writer. The fixture is a *mathematical
    ground truth*, so it must not be distorted: the amplitudes are scaled by a
    single factor so the rendered sum peaks at ``target_peak`` with headroom,
    and the model returned reflects the scaled amplitudes. A closed-loop test
    against a clipped file measures the writer, not the fitter.
    """
    model = {
        "model_name": "math_chord_benchmark",
        "sample_rate": sr,
        "duration": duration,
        "components": [
            {
                "id": "partial_1",
                "frequency": 440.0,
                "amplitude": 0.60,
                "phase": 0.0,
                "decay_rate": 1.10,
                "envelope": {"type": "exponential", "a0": 0.60, "decay_rate": 1.10},
            },
            {
                "id": "partial_2",
                "frequency": 554.365,
                "amplitude": 0.35,
                "phase": np.pi / 4,
                "decay_rate": 1.45,
                "envelope": {"type": "exponential", "a0": 0.35, "decay_rate": 1.45},
            },
            {
                "id": "partial_3",
                "frequency": 659.255,
                "amplitude": 0.25,
                "phase": np.pi / 2,
                "decay_rate": 1.80,
                "envelope": {"type": "exponential", "a0": 0.25, "decay_rate": 1.80},
            },
        ],
    }
    audio, _ = render_model(model, duration=duration, sr=sr)
    # Scale so the rendered sum peaks just under full scale, then rewrite the
    # model's amplitudes to match. Relative amplitudes (and therefore the
    # fitter's task) are unchanged.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        scale = float(target_peak) / peak
        for comp in model["components"]:
            comp["amplitude"] = float(comp["amplitude"]) * scale
            comp["envelope"]["a0"] = float(comp["envelope"]["a0"]) * scale
        audio = audio * scale
    n_clipped = save_audio(out_path, audio, sr)
    if n_clipped:
        raise RuntimeError(
            f"benchmark fixture clipped {n_clipped} samples: the ground truth "
            f"must be exact, so this is a bug, not a tolerance."
        )
    return model


def generate_benchmark_melody(
    out_path: Union[str, Path],
    duration: float = 3.0,
    sr: int = 44100,
) -> Dict[str, Any]:
    """Generates math_melody.wav featuring pitch vibrato and sequential notes to verify phase integration."""
    model = {
        "model_name": "math_melody_benchmark",
        "sample_rate": sr,
        "duration": duration,
        "components": [
            {
                "id": "note_1",
                "start_time": 0.0,
                "duration": 1.0,
                "frequency": 440.0,
                "amplitude": 0.5,
                "phase": 0.0,
                "vibrato": {"frequency": 5.0, "depth": 4.0},
                "envelope": {"type": "adsr", "attack": 0.05, "decay": 0.1, "sustain": 0.8, "release": 0.15},
            },
            {
                "id": "note_2",
                "start_time": 1.0,
                "duration": 1.0,
                "frequency": 523.25,  # C5
                "amplitude": 0.5,
                "phase": 0.0,
                "vibrato": {"frequency": 6.0, "depth": 5.0},
                "envelope": {"type": "adsr", "attack": 0.05, "decay": 0.1, "sustain": 0.8, "release": 0.15},
            },
            {
                "id": "note_3",
                "start_time": 2.0,
                "duration": 1.0,
                "frequency": 659.25,  # E5
                "amplitude": 0.5,
                "phase": 0.0,
                "pitch_bend": {"target_frequency": 698.46},  # Bend to F5
                "envelope": {"type": "adsr", "attack": 0.05, "decay": 0.1, "sustain": 0.8, "release": 0.15},
            },
        ],
    }
    audio, _ = render_model(model, duration=duration, sr=sr)
    save_audio(out_path, audio, sr)
    return model
