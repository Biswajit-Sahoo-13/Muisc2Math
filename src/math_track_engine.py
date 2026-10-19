"""Multi-track equation-based audio engine (Math-DAW core).

This is the heart of the formula-based Digital Audio Workstation.
Each track is defined by:
- A mathematical equation string (the "shape" of the sound)
- A MIDI note pattern (the "melody/rhythm")
- A gain knob

The engine provides pre-computed time variables to each equation:
    t    — absolute time in seconds
    u    — phase inside current beat step (0.0 → 1.0)
    s    — step index (0, 1, 2, …)
    n    — current note frequency in Hz (from pattern)
    g    — gate (1.0 when a note is playing, 0.0 on rest)
    ph   — phase-continuous oscillator phase via cumsum (no clicks on pitch changes)
    rnd  — seeded white noise array (reproducible)
    beat — duration of one beat in seconds

The four primitives that build all viral equation music:
    Loop     → (t mod T)            — rhythm, sequencing, bars
    Envelope → exp(-k·u), parabola  — plucks, fades, thumps
    Carrier  → sin(ph), sinc, rnd   — pitch, kick, hats
    Mix      → sum, tanh            — layers, distortion, glue
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.safe_eval import safe_eval, SafeEvalError, SAFE_FUNCTIONS, SAFE_CONSTANTS

SR = 44100


def midi2freq(midi_note: int) -> float:
    """Convert a MIDI note number to frequency in Hz.

    Formula: f = 440 · 2^((m-69)/12)
    """
    return 440.0 * 2.0 ** ((midi_note - 69) / 12.0)


def freq2midi(freq: float) -> int:
    """Convert a frequency in Hz to the nearest MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * np.log2(freq / 440.0)))


@dataclass
class Track:
    """A single equation track in the Math-DAW project."""
    name: str
    equation: str
    pattern: Optional[List[Optional[int]]] = None  # MIDI notes; None = rest
    gain: float = 1.0
    pan: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "equation": self.equation,
            "pattern": self.pattern,
            "gain": self.gain,
            "pan": self.pan,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Track":
        return cls(
            name=d["name"],
            equation=d["equation"],
            pattern=d.get("pattern"),
            gain=d.get("gain", 1.0),
            pan=d.get("pan", 0.0),
        )


@dataclass
class Project:
    """A Math-DAW project: multi-track equation-based composition.

    Parameters
    ----------
    bpm : float
        Beats per minute (tempo).
    duration : float
        Total duration in seconds.
    sr : int
        Sample rate (default 44100).
    seed : int
        Random seed for reproducible noise.
    tracks : list of Track
        Tracks comprising the project.
    name : str
        Name/title of the project.
    master_gain : float
        Overall master volume multiplier.
    """
    bpm: float = 140.0
    duration: float = 6.0
    sr: int = SR
    seed: int = 7
    tracks: List[Track] = field(default_factory=list)
    name: str = "Untitled"
    master_gain: float = 1.0

    def add(
        self,
        name: str,
        equation: str,
        pattern: Optional[List[Optional[int]]] = None,
        gain: float = 1.0,
        pan: float = 0.0,
    ) -> None:
        """Add an equation track to the project."""
        self.tracks.append(Track(name=name, equation=equation, pattern=pattern, gain=gain, pan=pan))

    def time_vars(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute the universal time variables for this project.

        Returns
        -------
        t : ndarray
            Absolute time in seconds.
        u : ndarray
            Phase inside current beat step, 0.0 → 1.0 (resets each beat).
        s : ndarray
            Step index (integer, which beat we're on).
        beat : float
            Duration of one beat in seconds.
        """
        n_samples = int(self.sr * self.duration)
        t = np.arange(n_samples, dtype=np.float64) / self.sr
        beat = 60.0 / self.bpm
        s = np.floor(t / beat)
        u = t / beat - s
        return t, u, s, beat

    def pattern_arrays(
        self,
        s: np.ndarray,
        pattern: Optional[List[Optional[int]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert a MIDI pattern list to frequency and gate arrays.

        Parameters
        ----------
        s : ndarray
            Step index array.
        pattern : list of int|None
            MIDI note numbers. None means rest (gate=0).

        Returns
        -------
        freq : ndarray
            Instantaneous frequency in Hz per sample.
        gate : ndarray
            Gate signal: 1.0 when note is active, 0.0 on rest.
        """
        if not pattern:
            # No pattern: constant 440Hz, always gated on
            return np.full_like(s, 440.0), np.ones_like(s)

        freq_per_step = np.array(
            [midi2freq(m) if m is not None else 0.0 for m in pattern],
            dtype=np.float64,
        )
        gate_per_step = np.array(
            [0.0 if m is None else 1.0 for m in pattern],
            dtype=np.float64,
        )
        idx = s.astype(int) % len(pattern)
        return freq_per_step[idx], gate_per_step[idx]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the project to a JSON-compatible dict."""
        return {
            "format": "math-daw-project",
            "version": "1.0",
            "name": self.name,
            "bpm": self.bpm,
            "duration": self.duration,
            "sr": self.sr,
            "seed": self.seed,
            "master_gain": self.master_gain,
            "tracks": [tr.to_dict() for tr in self.tracks],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Project":
        """Deserialize a project from a dict."""
        proj = cls(
            name=d.get("name", "Untitled"),
            bpm=d.get("bpm", 140.0),
            duration=d.get("duration", 6.0),
            sr=d.get("sr", SR),
            seed=d.get("seed", 7),
            master_gain=d.get("master_gain", 1.0),
        )
        for tr_dict in d.get("tracks", []):
            proj.tracks.append(Track.from_dict(tr_dict))
        return proj

    def save(self, path: Union[str, Path]) -> None:
        """Save the project to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Project":
        """Load a project from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def render(
    proj: Project,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Render a multi-track Math-DAW project to audio.

    Processing chain per track:
        1. Compute time variables (t, u, s, beat)
        2. Compute pattern arrays (n=frequency, g=gate)
        3. Build phase-continuous oscillator (ph = cumsum(2π·n/sr))
        4. Generate seeded noise (rnd)
        5. Evaluate the user's equation string safely
        6. Multiply by track gain

    Master bus:
        1. Sum all tracks
        2. Remove DC offset (subtract mean)
        3. Soft-clip via tanh(1.4 · x) for glue/saturation
        4. Peak-normalize to 0.9

    Parameters
    ----------
    proj : Project
        The project to render.

    Returns
    -------
    master : ndarray
        The final mixed master audio array.
    rendered_tracks : list of (name, ndarray)
        Per-track rendered audio for visualization.
    """
    t, u, s, beat = proj.time_vars()
    rng = np.random.default_rng(proj.seed)
    rnd = rng.standard_normal(t.shape)

    master = np.zeros_like(t)
    rendered_tracks: List[Tuple[str, np.ndarray]] = []

    for track in proj.tracks:
        n, g = proj.pattern_arrays(s, track.pattern)

        # Phase-continuous oscillator: cumsum integrates instantaneous frequency
        # This prevents clicks/pops when pitch changes between steps
        ph = np.cumsum(2.0 * np.pi * n / proj.sr)

        # Build the namespace of variables available to the equation
        variables = {
            "t": t,
            "u": u,
            "s": s,
            "n": n,
            "g": g,
            "ph": ph,
            "rnd": rnd,
            "beat": beat,
        }

        try:
            y = safe_eval(track.equation, variables)
        except SafeEvalError as exc:
            print(f"[WARNING] Track '{track.name}' equation error: {exc}")
            y = np.zeros_like(t)

        # Ensure result is an array of the correct length
        if np.isscalar(y):
            y = np.full_like(t, float(y))
        elif len(y) != len(t):
            y = np.broadcast_to(y, t.shape).copy()

        y = y.astype(np.float64) * track.gain
        rendered_tracks.append((track.name, y))
        master += y

    # === Master bus processing ===

    # 1. Kill DC offset (the -1.6 / -1.7 trap from viral equations)
    master -= np.mean(master)

    # 2. Soft-clip via tanh for saturation/glue (prevents harsh digital clipping)
    master = np.tanh(1.4 * master)

    # 3. Apply master gain
    master = master * float(proj.master_gain)

    # 4. Peak-normalize to 0.9 (-0.9 dB headroom)
    peak = np.max(np.abs(master))
    if peak > 0:
        master *= 0.9 / peak

    return master, rendered_tracks


class MathTrackEngine:
    """Convenience wrapper engine for rendering Math-DAW projects."""

    def __init__(self, sr: int = SR):
        self.sr = sr

    def render_project(self, project: Project) -> Dict[str, Any]:
        """Render a project and return a dictionary with mix and track outputs."""
        master, tracks = render(project)
        return {
            "mix": master,
            "master": master,
            "tracks": tracks,
            "sr": project.sr,
        }


def render_to_wav(
    proj: Project,
    output_path: Union[str, Path],
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Render a project and save the master to a WAV file.

    Parameters
    ----------
    proj : Project
        The project to render.
    output_path : str or Path
        Output WAV file path.

    Returns
    -------
    master : ndarray
        The rendered master audio.
    rendered_tracks : list of (name, ndarray)
        Per-track audio.
    """
    import soundfile as sf

    master, rendered_tracks = render(proj)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(p), master, proj.sr, subtype="PCM_24")
    return master, rendered_tracks


def render_stems_to_wav(
    proj: Project,
    output_dir: Union[str, Path],
) -> List[Path]:
    """Render each track as a separate WAV stem file.

    Parameters
    ----------
    proj : Project
        The project to render.
    output_dir : str or Path
        Directory to write stem WAV files.

    Returns
    -------
    paths : list of Path
        Paths to the created stem files.
    """
    import soundfile as sf

    _, rendered_tracks = render(proj)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, audio in rendered_tracks:
        # Normalize individual stem
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * 0.9 / peak
        stem_path = out / f"{name}.wav"
        sf.write(str(stem_path), audio, proj.sr, subtype="PCM_24")
        paths.append(stem_path)
    return paths


# ---------------------------------------------------------------------------
# Demo / preset projects
# ---------------------------------------------------------------------------

def demo_sinc_drop() -> Project:
    """Create the demo 'Sinc Drop' phonk beat — the TikTok viral equation sound.

    Four-track layered composition:
    - Kick:  sinc pulse (band-limited impulse = the perfect math kick drum)
    - Bass:  sine oscillator with exponential decay
    - Lead:  FM synthesis (sine + vibrato via phase modulation) with decay
    - Hats:  noise burst with fast exponential decay, gated by pattern

    Each track's sound is defined entirely by a mathematical equation.
    """
    p = Project(bpm=140, duration=6.0)
    p.add(
        "kick",
        "sinc(26*(u-0.08))",
        pattern=[36, 36, 36, 36],
        gain=1.0,
    )
    p.add(
        "bass",
        "0.55*sin(ph)*exp(-2.5*u)",
        pattern=[33, 33, 31, 31, 36, 36, 29, 29],
        gain=1.0,
    )
    p.add(
        "lead",
        "0.22*sin(ph + 0.4*sin(2*pi*5.5*t))*exp(-4*u)",
        pattern=[69, None, 72, None, 76, None, 72, 67],
        gain=1.0,
    )
    p.add(
        "hats",
        "0.12*rnd*exp(-35*u)",
        pattern=[None, 72, None, 72, None, 72, None, 76],
        gain=1.0,
    )
    return p


def demo_math_remix() -> Project:
    """Create a 'Math Remix' style beat using parabolic gate pulses.

    Uses the phrase-level fade + smooth per-beat gate from the
    second viral equation family: exp(-k|mod|) × (1 - 2(mod-1)²)
    """
    p = Project(bpm=130, duration=8.0)
    p.add(
        "pulse_kick",
        "0.8 * exp(-0.5 * abs(mod(t, 4.0))) * (1.0 - 2.0 * (mod(t, 1.0) - 0.5)**2)",
        pattern=[36, 36, 36, 36],
        gain=1.0,
    )
    p.add(
        "sub_bass",
        "0.5 * sin(ph) * exp(-3.0 * u) * g",
        pattern=[33, None, 33, None, 31, None, 31, None],
        gain=1.0,
    )
    p.add(
        "arp",
        "0.18 * sin(ph + 0.3 * sin(2*pi*7*t)) * exp(-6*u) * g",
        pattern=[60, 64, 67, 72, 60, 64, 67, 76],
        gain=0.8,
    )
    p.add(
        "reverse_stab",
        "0.3 * sinc(40*(u-0.5)) * g",
        pattern=[None, 60, None, None, None, 64, None, None],
        gain=0.7,
    )
    return p


def demo_ambient() -> Project:
    """Create a gentle ambient pad using slow sine sweeps and filtered noise."""
    p = Project(bpm=80, duration=10.0, seed=42)
    p.add(
        "pad",
        "0.3 * sin(ph) * (0.5 + 0.5*sin(2*pi*0.1*t))",
        pattern=[60, 60, 64, 64, 67, 67, 72, 72],
        gain=1.0,
    )
    p.add(
        "shimmer",
        "0.08 * sin(3*ph + sin(2*pi*0.3*t)) * exp(-1.5*u)",
        pattern=[72, 76, 79, 84, 72, 76, 79, 84],
        gain=0.6,
    )
    p.add(
        "texture",
        "0.04 * rnd * exp(-8*u) * g",
        pattern=[None, None, 60, None, None, None, 64, None],
        gain=0.5,
    )
    return p


def multitrack_ledger_to_math_tracks(ledger: Dict[str, Any]) -> List[Track]:
    """Convert a res/2.0-multitrack ledger into independent Track objects for Math-DAW.

    Each stem (e.g. 'bass', 'drums', 'vocals', 'other') becomes an independent,
    solo-able track with its mathematical equation synthesized from the ledger's
    dominant partials and envelopes.
    """
    tracks: List[Track] = []
    stems = ledger.get("stems", {})
    if not stems and "sections" in ledger:
        # Fallback: group partials in top-level sections by 'stem' tag
        stems_dict: Dict[str, Any] = {}
        for sec in ledger.get("sections", []):
            for p in sec.get("partials", []):
                s_tag = p.get("stem", "master")
                stems_dict.setdefault(s_tag, {"sections": [{"partials": []}], "metrics": {}})
                stems_dict[s_tag]["sections"][0]["partials"].append(p)
        stems = stems_dict

    for stem_name, stem_info in stems.items():
        sections = stem_info.get("sections", [])
        all_partials: List[Dict[str, Any]] = []
        for sec in sections:
            all_partials.extend(sec.get("partials", []))

        # Build closed-form equation representation for the track
        terms: List[str] = []
        for p in all_partials[:6]:
            model = p.get("model", "damped")
            if model == "glide":
                f_mean = float(p.get("frequency", 55.0))
                g = float(p.get("projection_gain", 1.0))
                terms.append(f"{g:.3f} * sin(2*pi*({f_mean:.1f} - 10*(1-exp(-3*u)))*t)")
            elif model == "filtered_noise":
                amp = float(p.get("amplitude", 0.05))
                terms.append(f"{amp:.3f} * rnd * exp(-8*u)")
            else:
                amp = float(p.get("amplitude", 0.1))
                freq = float(p.get("frequency", 100.0))
                dec = float(p.get("envelope_decay", 1.0))
                terms.append(f"{amp:.3f} * exp(-{dec:.2f}*u) * cos(2*pi*{freq:.1f}*t)")

        if terms:
            eq = " + ".join(terms[:3])
            if len(terms) > 3:
                eq += f" + ... [{len(all_partials)} partials]"
        else:
            eq = "0.0 * sin(2*pi*t)"

        tracks.append(
            Track(
                name=stem_name,
                equation=eq,
                gain=1.0,
                pan=0.0,
            )
        )
    return tracks


# ---------------------------------------------------------------------------
# CLI entry point (standalone)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=== Math-DAW: Multi-Track Equation Engine ===")
    print()

    # Render the Sinc Drop demo
    p = demo_sinc_drop()
    master, tracks = render_to_wav(p, "data/output/sinc_drop.wav")
    print(f"[+] Rendered sinc_drop.wav ({p.duration}s, {p.bpm} BPM, {len(p.tracks)} tracks)")

    # Save project JSON
    p.save("data/output/sinc_drop_project.json")
    print(f"[+] Saved project file: sinc_drop_project.json")

    # Plot per-track + master waveforms
    fig, axes = plt.subplots(
        len(tracks) + 1, 1,
        figsize=(12, 2.5 * (len(tracks) + 1)),
        sharex=True,
    )
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for idx, (ax, (name, y_track)) in enumerate(zip(axes[:-1], tracks)):
        t_axis = np.arange(len(y_track)) / p.sr
        ax.plot(t_axis, y_track, lw=0.5, color=colors[idx % len(colors)])
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    t_master = np.arange(len(master)) / p.sr
    axes[-1].plot(t_master, master, lw=0.5, color="#2c3e50")
    axes[-1].set_ylabel("master", fontsize=10, fontweight="bold")
    axes[-1].set_xlabel("Time (seconds)")
    axes[-1].set_ylim(-1.05, 1.05)
    axes[-1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Math-DAW: Sinc Drop ({p.bpm} BPM)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig("data/output/sinc_drop_tracks.png", dpi=150)
    print(f"[+] Saved waveform plot: sinc_drop_tracks.png")
    plt.close()

    # Also render the Math Remix demo
    p2 = demo_math_remix()
    render_to_wav(p2, "data/output/math_remix.wav")
    p2.save("data/output/math_remix_project.json")
    print(f"[+] Rendered math_remix.wav ({p2.duration}s, {p2.bpm} BPM, {len(p2.tracks)} tracks)")

    print()
    print("[*] Done. Open the WAV files in Audacity or any media player to listen.")
