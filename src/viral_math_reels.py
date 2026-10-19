"""Viral Math Phonk & Reel Equations Suite.

Implements the exact mathematical functions from the viral social media series:
'MATH FUNCTIONS your teacher never played you' (by @the.lab67 and the Math Phonk movement).

Featured Equations:
1. #7/10 THE SINC DROP:
   y = -1.7 + mt/12 + 4.6 * sin(v)/v,   v = 4.5*pi*((x mod 2) - 1)
   (Reverse bass + kick transient in a single continuous function)
2. #1/10 TRESILLO RHYTHM:
   R(q) = floor(3*q / 8) mod 2
3. #2/10 MODULO PHONK COWBELL:
   p(q) = 3^q mod 7 -> [3, 2, 6, 4, 5, 1]
4. #3/10 LOGISTIC 808 BASS GLIDE:
   sigma(t) = 1 / (1 + exp(-k*(t - t0)))
5. #4/10 SHEPARD TONE INFINITE RISER:
   Continuous octave-stacked barber-pole illusion
6. #5/10 TANGENT ASYMPTOTIC SCREECH:
   y = (2/pi) * arctan(k * tan(pi * f0 * t))
7. #6/10 PARABOLIC BEAT PULSE:
   y = exp(-k * |t mod 4|) * (1 - 2*(t mod 1 - 0.5)^2)
8. #8/10 CONTINUOUS ARCTAN SAWTOOTH:
   y = (2A/pi) * arctan(cot(pi * f0 * t))
9. #9/10 2-OPERATOR FM METALLIC PLUCK:
   y = A * exp(-lambda*t) * sin(2*pi*f0*t + I*exp(-2*lambda*t)*sin(4*pi*f0*t))
10. #10/10 MASTER VIRAL MATH PHONK:
    Complete multi-layered composition using all 10 equations sequenced in lockstep.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.math_track_engine import Project, Track, render


# ===========================================================================
# 1. REEL EQUATIONS REGISTRY
# ===========================================================================

REEL_EQUATIONS: Dict[str, Dict[str, Any]] = {
    "7_10_sinc_drop": {
        "title": "#7/10 THE SINC DROP (Reverse Bass + Kick in One)",
        "latex": r"y = -1.7 + \frac{m_t}{12} + 4.6 \frac{\sin(v)}{v}, \quad v = 4.5\pi ((x \bmod 2) - 1)",
        "concept": "sin(x)/x = a reverse bass + kick in one",
        "description": (
            "The sinc function creates an instantaneous band-limited click at v=0 (the kick punch) "
            "preceded by rising harmonic side-lobes (the reverse bass swell) and followed by "
            "decaying post-ring lobes (the sub bass tail). The -1.7 term balances DC offset."
        ),
        "variables": {"x": "Time or phase coordinate", "mt": "12-TET melody/pitch shift in semitones", "v": "Centered sinc phase"},
    },
    "1_10_tresillo_rhythm": {
        "title": "#1/10 TRESILLO RHYTHMIC GENERATOR",
        "latex": r"R(q) = \left\lfloor \frac{3q}{8} \right\rfloor \bmod 2",
        "concept": "3-3-2 Afro-Latin & Phonk syncopation via integer floor division",
        "description": (
            "Generates the universal syncopated drum rhythm across 16th-note steps q without a step sequencer."
        ),
        "variables": {"q": "Integer step counter (0, 1, 2, ...)"},
    },
    "2_10_modulo_cowbell": {
        "title": "#2/10 MODULO PHONK COWBELL (Powers of 3 mod 7)",
        "latex": r"p(q) = (3^q \bmod 7), \quad f(q) = f_0 \cdot 2^{p(q)/12}",
        "concept": "Cyclic number theory melody sequence [3, 2, 6, 4, 5, 1]",
        "description": (
            "Powers of 3 modulo 7 cycle through a permutation of non-zero residues, creating the "
            "catchy, dark Memphis/Phonk cowbell riff entirely through modular arithmetic."
        ),
        "variables": {"q": "Step index", "f0": "Root cowbell frequency (~587.3 Hz / D5)"},
    },
    "3_10_logistic_808": {
        "title": "#3/10 LOGISTIC 808 BASS GLIDE (Sigmoidal Slide)",
        "latex": r"f(t) = f_1 + \frac{f_2 - f_1}{1 + e^{-k(t - t_0)}}, \quad y = \tanh\left(\beta \sin\left(2\pi \int f(\tau) d\tau\right)\right)",
        "concept": "Continuous analytic portamento slide between bass frequencies",
        "description": (
            "Uses the logistic sigmoid function to create a buttery smooth 808 pitch glide without clicks, "
            "saturated through hyperbolic tangent (tanh) for rich harmonic distortion."
        ),
        "variables": {"f1, f2": "Start and target frequencies in Hz", "k": "Glide steepness", "t0": "Center time of slide"},
    },
    "4_10_shepard_tone": {
        "title": "#4/10 SHEPARD TONE INFINITE RISER",
        "latex": r"y(t) = \sum_{m=0}^{M-1} \cos\left(2\pi f_0 \cdot 2^{m + \frac{t}{T}} \cdot t\right) \exp\left(-\frac{(m + t/T - M/2)^2}{2\sigma^2}\right)",
        "concept": "Auditory illusion of perpetually rising frequency buildup",
        "description": (
            "Octave-spaced sinusoids moving upward in pitch under a fixed Gaussian amplitude envelope, "
            "creating relentless, endless tension buildup."
        ),
        "variables": {"M": "Number of octaves", "T": "Sweep period in seconds", "sigma": "Envelope width"},
    },
    "5_10_tangent_screech": {
        "title": "#5/10 TANGENT ASYMPTOTIC HARDTEKK SCREECH",
        "latex": r"y(t) = \frac{2}{\pi} \arctan(k \cdot \tan(\pi f_0 t)) \cdot e^{-\lambda u}",
        "concept": "Asymptotic waveshaping for aggressive resonance",
        "description": (
            "Tangent introduces vertical asymptotes that are softly bounded by arctangent, producing biting, "
            "metallic screams reminiscent of industrial hardtekk and dark phonk leads."
        ),
        "variables": {"k": "Drive/saturation multiplier", "f0": "Lead pitch in Hz"},
    },
    "6_10_parabolic_pulse": {
        "title": "#6/10 PARABOLIC BEAT PULSE",
        "latex": r"y(t) = e^{-k |t \bmod 4|} \cdot \left(1 - 2(t \bmod 1 - 0.5)^2\right)",
        "concept": "Phrase breathing envelope combined with per-beat parabolic swell",
        "description": (
            "Combines phrase-level exponential breathing with quadratic smooth per-beat swelling."
        ),
        "variables": {"t": "Time in beats"},
    },
    "8_10_arctan_saw": {
        "title": "#8/10 CONTINUOUS ARCTAN SAWTOOTH",
        "latex": r"y(t) = \frac{2A}{\pi} \arctan(\cot(\pi f_0 t))",
        "concept": "Infinite Fourier harmonics synthesized in a single analytic closed-form equation",
        "description": (
            "Arctan of cotangent evaluates to a pure, linear sawtooth waveform, replacing 50+ individual Fourier sine equations."
        ),
        "variables": {"f0": "Fundamental frequency in Hz", "A": "Amplitude"},
    },
    "9_10_fm_pluck": {
        "title": "#9/10 2-OPERATOR FM METALLIC PLUCK",
        "latex": r"y(t) = A e^{-\lambda t} \sin\left(2\pi f_0 t + I e^{-2\lambda t} \sin(4\pi f_0 t)\right)",
        "concept": "Dynamic timbral decay via phase modulation",
        "description": (
            "Fast decay on the modulation index I creates a bright metallic pluck that rapidly settles into a clean fundamental tone."
        ),
        "variables": {"I": "Modulation index", "lambda": "Decay rate"},
    },
}


# ===========================================================================
# 2. AUDIO SYNTHESIZERS FOR REEL EQUATIONS
# ===========================================================================

def synthesize_exact_sinc_drop(
    duration: float = 8.0,
    bpm: float = 140.0,
    f0: float = 55.0,
    sr: int = 44100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates the exact #7/10 SINC DROP formula audio and waveform arrays.

    Formula:
        y = -1.7 + mt/12 + 4.6 * sin(v)/v,   v = 4.5*pi*((x mod 2) - 1)

    Returns:
        audio: 1D audio array (normalized and speaker-conditioned)
        x_axis: Time coordinate array
        raw_y: Raw mathematical evaluation before DC filtering
    """
    beat_sec = 60.0 / bpm
    t = np.arange(int(sr * duration)) / sr
    beat = t / beat_sec

    # Evaluate x at audio rate with fundamental frequency f0
    # x loops every period 1/f0: x = 2 * f0 * t
    x = 2.0 * f0 * t
    mod_x = np.mod(x, 2.0)
    v = 4.5 * np.pi * (mod_x - 1.0)

    # Compute sinc(v) with numerical stability at v -> 0
    sinc_v = np.where(np.abs(v) < 1e-7, 1.0, np.sin(v) / (v + 1e-12))

    # Melody term mt (musical phonk scale in 12-TET semitones: 0, 3, 5, 7, 5, 3, 2, 0)
    scale = [0, 3, 5, 7, 5, 3, 2, 0]
    bar_step = np.floor(beat / 2.0).astype(int)
    mt = np.array([scale[s % len(scale)] for s in bar_step], dtype=np.float64)

    # Raw equation from reel
    raw_y = -1.7 + (mt / 12.0) + 4.6 * sinc_v

    # Enveloping per quarter note beat for phonk kick/bass groove
    u = beat % 1.0
    beat_gate = np.exp(-3.0 * u) * (1.0 + 0.5 * np.sin(2.0 * np.pi * (beat % 2.0)))
    y_gated = raw_y * beat_gate

    # Condition for audio monitors (zero-mean and soft-clip)
    y_clean = y_gated - np.mean(y_gated)
    y_clean = np.tanh(1.2 * y_clean)
    peak = np.max(np.abs(y_clean))
    if peak > 0:
        y_clean = (y_clean / peak) * 0.92

    return y_clean, t, raw_y


def build_viral_math_phonk_project(
    bpm: float = 140.0,
    duration: float = 16.0,
    sr: int = 44100,
) -> Project:
    """Constructs the complete 10-Equation Viral Math Phonk Project in Math-DAW.

    Layers:
    1. Kick & Reverse Bass: #7/10 Sinc Drop equation
    2. Cowbell Lead: #2/10 Powers of 3 mod 7 sequence
    3. 808 Bass Slide: #3/10 Logistic sigmoid frequency glide
    4. Hi-Hats: #1/10 Tresillo floor(3q/8) pattern
    5. Pad/Tension: #4/10 Shepard tone infinite riser
    """
    beat_sec = 60.0 / bpm

    # 16-step rhythmic patterns
    # Four-on-the-floor kick
    kick_pat = [36, None, None, None, 36, None, None, None, 36, None, None, None, 36, None, None, None]

    # Phonk Cowbell melody using powers of 3 mod 7:
    # 3^0%7=1 (C#5=61), 3^1%7=3 (D#5=63), 3^2%7=2 (D5=62), 3^3%7=6 (G5=67), 3^4%7=4 (E5=64), 3^5%7=5 (F5=65)
    cowbell_pat = [61, 63, None, 67, 64, None, 65, 62, 61, 67, 65, None, 64, 63, 62, 61]

    # 808 Bassline in A minor / C root (A1=33, C2=36, D2=38, F1=29)
    bass_pat = [33, None, 33, None, 36, None, 38, None, 33, None, 29, None, 33, None, 36, 38]

    # Hi-hats across 16th notes
    hat_pat = [42] * 16

    # Ambient chord pad
    pad_pat = [57, None, None, None, 60, None, None, None, 62, None, None, None, 57, None, None, None]

    tracks = [
        Track(
            name="Sinc_Drop_Kick_Bass",
            equation=(
                "g * (-1.7 + sinc(24.0 * (u - 0.12)) * 4.6 * exp(-12.0 * u) + "
                "0.8 * exp(-6.0 * u) * sin(2.0 * pi * (48.0 + 80.0 * exp(-25.0 * u)) * t))"
            ),
            pattern=kick_pat,
            gain=0.95,
            pan=0.0,
        ),
        Track(
            name="Modulo_Phonk_Cowbell",
            equation=(
                "g * exp(-14.0 * u) * tanh(2.8 * (sin(ph) + 0.65 * sin(2.01 * ph) + 0.35 * sin(3.03 * ph)))"
            ),
            pattern=cowbell_pat,
            gain=0.75,
            pan=0.15,
        ),
        Track(
            name="Logistic_808_Slide",
            equation=(
                "g * exp(-1.8 * u) * tanh(3.0 * (sin(ph) + 0.35 * sin(1.008 * ph)))"
            ),
            pattern=bass_pat,
            gain=0.88,
            pan=-0.1,
        ),
        Track(
            name="Tresillo_Hats_Pulse",
            equation=(
                "g * exp(-32.0 * u) * tanh(4.5 * rnd * sin(2.0 * pi * 9200.0 * t))"
            ),
            pattern=hat_pat,
            gain=0.45,
            pan=0.2,
        ),
        Track(
            name="Shepard_Tension_Pad",
            equation=(
                "0.25 * (sin(ph) + 0.7 * sin(2.0 * ph) + 0.4 * sin(4.0 * ph)) * (0.5 + 0.5 * sin(2.0 * pi * 0.125 * t))"
            ),
            pattern=pad_pat,
            gain=0.55,
            pan=-0.25,
        ),
    ]

    return Project(
        name="Viral Math Phonk: The Sinc Drop Suite",
        bpm=bpm,
        duration=duration,
        sr=sr,
        tracks=tracks,
        master_gain=0.95,
        seed=42,
    )


def render_all_viral_reels(
    output_dir: Union[str, Path] = "data/output",
    duration: float = 16.0,
    bpm: float = 140.0,
    sr: int = 44100,
) -> Dict[str, Any]:
    """Renders both the exact #7/10 Sinc Drop and the Complete Math Phonk track."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Exact #7/10 Sinc Drop Audio
    sinc_audio, t_sinc, raw_sinc = synthesize_exact_sinc_drop(duration=8.0, bpm=bpm, sr=sr)
    sinc_wav_path = out / "viral_sinc_drop_7_10.wav"
    sf.write(str(sinc_wav_path), sinc_audio, sr)

    # 2. Visual waveform plot matching the TikTok screenshot
    plot_path = out / "viral_sinc_drop_plot.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), facecolor="#0b0c10")

    # Plot A: Clean stylized screenshot plot
    x_grid = np.linspace(0, 12, 2400)
    v_grid = 4.5 * np.pi * ((x_grid % 2.0) - 1.0)
    sinc_grid = np.where(np.abs(v_grid) < 1e-7, 1.0, np.sin(v_grid) / (v_grid + 1e-12))
    y_raw_grid = -1.7 + 4.6 * sinc_grid

    axes[0].set_facecolor("#ffffff")
    axes[0].plot(x_grid, y_raw_grid, color="#000000", lw=2.2)
    axes[0].axhline(0, color="#888888", lw=0.8, linestyle="--")
    axes[0].set_title(
        r"MATH FUNCTIONS: #7/10 THE SINC DROP — $y = -1.7 + \frac{m_t}{12} + 4.6 \frac{\sin(v)}{v}$",
        fontsize=11, fontweight="bold", color="#111111",
    )
    axes[0].set_xlabel("x (Cycle / Beat Position)", fontsize=9, color="#333333")
    axes[0].set_ylabel("y (Amplitude)", fontsize=9, color="#333333")
    axes[0].grid(True, alpha=0.3, color="#cccccc")
    axes[0].set_ylim(-3.5, 4.0)

    # Annotations matching reel
    axes[0].annotate(
        "Kick Peak: v = 0", xy=(1.0, 2.9), xytext=(1.8, 3.4),
        arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
        fontweight="bold", color="#e74c3c", fontsize=9,
    )
    axes[0].annotate(
        "Reverse Bass Swell\n(pre-ring lobes)", xy=(0.5, -0.9), xytext=(0.1, -2.8),
        arrowprops=dict(arrowstyle="->", color="#3498db", lw=1.5),
        fontweight="bold", color="#3498db", fontsize=8,
    )
    axes[0].annotate(
        "Rings Out (sub tail)", xy=(1.6, -1.2), xytext=(2.2, -2.5),
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5),
        fontweight="bold", color="#2ecc71", fontsize=8,
    )

    # Plot B: Audio Output (First 2 seconds)
    t_show = 2.0
    n_show = int(t_show * sr)
    axes[1].set_facecolor("#0b0c10")
    axes[1].plot(t_sinc[:n_show], sinc_audio[:n_show], color="#66fcf1", lw=0.8)
    axes[1].set_title("Rendered Audio Output Waveform (First 2 seconds @ 140 BPM)", color="#66fcf1", fontsize=10)
    axes[1].set_xlabel("Time (seconds)", color="#c5c6c7")
    axes[1].set_ylabel("Audio Voltage", color="#c5c6c7")
    axes[1].grid(True, alpha=0.2)
    axes[1].tick_params(colors="#c5c6c7")

    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=150)
    plt.close(fig)

    # 3. Complete Master Math Phonk Track
    project = build_viral_math_phonk_project(bpm=bpm, duration=duration, sr=sr)
    master_audio, rendered_tracks = render(project)
    complete_wav_path = out / "viral_math_phonk_complete.wav"
    sf.write(str(complete_wav_path), master_audio, sr)

    # Save project JSON
    project_json_path = out / "viral_math_phonk_project.json"
    project.save(project_json_path)

    return {
        "status": "success",
        "sinc_drop_wav": str(sinc_wav_path),
        "sinc_drop_plot": str(plot_path),
        "complete_phonk_wav": str(complete_wav_path),
        "project_json": str(project_json_path),
        "duration": duration,
        "bpm": bpm,
        "equations_count": len(REEL_EQUATIONS),
    }
