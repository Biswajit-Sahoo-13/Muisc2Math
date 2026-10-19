"""Multi-Formula Bank, Dictionary-Based Sparse Modeling, and Formula Search Engine.

Implements the 3-Tier Formula Architecture:
1. Tier 1: Built-in & Manually Extensible Formula Bank (Dictionary Matching).
   Includes: Pure Sine, Harmonic Power Saw, Soft Square (tanh), FM Synthesis Bell,
   Damped Harmonic Series (inharmonic piano/guitar), and Arctan/Cot Sawtooth.
2. Tier 2: Automated Formula Invention via Symbolic Regression.
   Discovers new formulas combining mathematical operators (sin, cos, exp, tanh, +, *)
   when signals deviate from standard acoustic primitives.
3. Tier 3: Extensible Custom Formula Loader & Scraper Hooks.
   Enables dynamic loading of synthesis recipes from external Python modules or JSON.
"""

from __future__ import annotations
import ast
import importlib.util
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import librosa
from scipy.optimize import minimize


# ===========================================================================
# 1. TIER 1: BUILT-IN FORMULA BANK (DICTIONARY MATCHING)
# ===========================================================================

FORMULA_BANK: Dict[str, Dict[str, Any]] = {
    "pure_sine": {
        "latex": "A * e^(-lambda * t) * sin(2*pi * f0 * t)",
        "description": "Pure Sine Wave (Flutes, Sub-bass, Clean Tones)",
        "bounds": (0.5, 2.0),
        "default_p": 1.0,
        "func": lambda t, f0, A, lam, p1: A * np.exp(-lam * t) * np.sin(2.0 * np.pi * f0 * t),
    },
    "harmonic_power_saw": {
        "latex": "A * e^(-lambda * t) * sum_{h=1}^H [ (1/h^p) * sin(2*pi * h * f0 * t) ]",
        "description": "Additive Harmonic Sawtooth (Strings, Brass, Lead Synths)",
        "bounds": (0.3, 3.5),
        "default_p": 1.0,
        "func": lambda t, f0, A, lam, p1: A * np.exp(-lam * t) * sum(
            (1.0 / (h ** max(0.3, p1))) * np.sin(2.0 * np.pi * h * f0 * t)
            for h in range(1, 9)
        ),
    },
    "soft_square_tanh": {
        "latex": "A * e^(-lambda * t) * tanh(k * sin(2*pi * f0 * t))",
        "description": "Soft-Clipped Square/Pulse Wave (Clarinets, Retro Chiptune, Hollow Synths)",
        "bounds": (0.2, 4.0),
        "default_p": 1.2,
        "func": lambda t, f0, A, lam, p1: A * np.exp(-lam * t) * np.tanh(
            max(0.4, p1 * 3.0) * np.sin(2.0 * np.pi * f0 * t)
        ),
    },
    "fm_synthesis_bell": {
        "latex": "A * e^(-lambda * t) * sin(2*pi * f0 * t + I * e^(-2*lambda*t) * sin(4*pi * f0 * t))",
        "description": "2-Operator Frequency Modulation (Bells, Electric Pianos, Plucks)",
        "bounds": (0.2, 4.0),
        "default_p": 1.2,
        "func": lambda t, f0, A, lam, p1: A * np.exp(-lam * t) * np.sin(
            2.0 * np.pi * f0 * t + (p1 * 2.5) * np.exp(-2.0 * max(0.1, lam) * t) * np.sin(4.0 * np.pi * f0 * t)
        ),
    },
    "damped_harmonic_series": {
        "latex": "A * e^(-lambda * t) * sum_{h=1}^H [ (1/h^p) * sin(2*pi * h * f0 * sqrt(1 + B*h^2) * t) ]",
        "description": "Inharmonic Damped Series (Acoustic Piano, Guitar, Plucked Strings)",
        "bounds": (0.4, 3.0),
        "default_p": 1.1,
        "func": lambda t, f0, A, lam, p1: A * np.exp(-lam * t) * sum(
            (1.0 / (h ** max(0.4, p1))) * np.sin(2.0 * np.pi * h * f0 * np.sqrt(1.0 + 0.0004 * (h ** 2)) * t)
            for h in range(1, 9)
        ),
    },
    "arctan_cot_saw": {
        "latex": "(2A / pi) * e^(-lambda * t) * arctan(cot(pi * f0 * t))",
        "description": "Analytic Continuous Sawtooth Wave",
        "bounds": (0.5, 2.0),
        "default_p": 1.0,
        "func": lambda t, f0, A, lam, p1: (2.0 * A / np.pi) * np.exp(-lam * t) * np.arctan(
            np.cos(np.pi * f0 * t) / (np.sin(np.pi * f0 * t) + 1e-9)
        ),
    },
}

# Provide rich_sawtooth_arctan alias
FORMULA_BANK["rich_sawtooth_arctan"] = FORMULA_BANK["arctan_cot_saw"]


def register_formula(
    name: str,
    latex: str,
    func: Callable[[np.ndarray, float, float, float, float], np.ndarray],
    description: str = "",
    bounds: Tuple[float, float] = (0.2, 4.0),
    default_p: float = 1.0,
) -> None:
    """Dynamically register a new mathematical formula into FORMULA_BANK."""
    FORMULA_BANK[name] = {
        "latex": latex,
        "description": description or name,
        "bounds": bounds,
        "default_p": default_p,
        "func": func,
    }


def get_formula_bank() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the current formula bank."""
    return dict(FORMULA_BANK)


# ---------------------------------------------------------------------------
# EQUATION CANDIDATES, COMPLEXITY SCORING, EQUATION CODE GENERATION
# ---------------------------------------------------------------------------

SAWTOOTH_ALIASES = ("arctan_cot_saw", "rich_sawtooth_arctan")

_BINARY_COST = {
    ast.Add: 1.0,
    ast.Sub: 1.0,
    ast.Mult: 1.5,
    ast.Div: 1.5,
    ast.Mod: 1.5,
    ast.FloorDiv: 1.5,
    ast.Pow: 3.0,
}

_UNARY_COST = {ast.USub: 0.2, ast.UAdd: 0.2, ast.Not: 0.5}

_CALL_COST = {
    "sin": 1.0,
    "cos": 1.0,
    "sqrt": 1.0,
    "abs": 0.8,
    "tanh": 1.5,
    "exp": 1.5,
    "log": 1.5,
    "arctan": 1.5,
    "atan": 1.5,
    "sinc": 1.5,
    "pow": 2.0,
    "mod": 1.5,
}

_UNPARSEABLE_COMPLEXITY = 1000.0


@dataclass
class EquationCandidate:
    """One fitted acoustic formula plus its fidelity and simplicity scores."""
    formula_type: str
    equation: str
    latex: str
    f0: float
    A: float
    decay_lambda: float
    timbre_param: float
    spectral_mse: float
    complexity: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {
            "formula_type": data["formula_type"],
            "description": FORMULA_BANK.get(self.formula_type, {}).get("description", self.formula_type),
            "equation_template": data["latex"],
            "equation": data["equation"],
            "f0": round(float(data["f0"]), 2),
            "A": round(float(data["A"]), 4),
            "decay_lambda": round(float(data["decay_lambda"]), 3),
            "timbre_param": round(float(data["timbre_param"]), 3),
            "spectral_mse": round(float(data["spectral_mse"]), 6),
            "complexity": round(float(data["complexity"]), 3),
        }


def _fmt(value: float, sig: int = 4) -> str:
    """Format a float compactly for embedding into equation code."""
    return f"{float(value):.{sig}g}"


def equation_complexity(equation: str) -> float:
    """Score how complex an equation string is (lower is simpler).

    Walks the Python AST and accumulates a weighted node cost, so an
    equation that reaches the same fidelity with fewer operations wins.
    """
    try:
        tree = ast.parse(str(equation), mode="eval")
    except (SyntaxError, ValueError):
        return _UNPARSEABLE_COMPLEXITY

    total = 0.0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            total += _BINARY_COST.get(type(node.op), 1.5)
        elif isinstance(node, ast.UnaryOp):
            total += _UNARY_COST.get(type(node.op), 0.2)
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            total += _CALL_COST.get(str(name), 1.5)
        elif isinstance(node, ast.Constant):
            total += 0.5
        elif isinstance(node, ast.Name):
            total += 0.3
    return float(total)


def build_equation_string(
    formula_type: str,
    f0: float,
    A: float,
    decay_lambda: float,
    timbre_param: float,
    include_gate: bool = False,
    carrier: str = "t",
) -> str:
    """Emit a safe_eval-compatible equation string for a fitted formula.

    Parameters
    ----------
    carrier : {"t", "ph"}
        ``"t"`` builds an absolute-time expression that ignores note patterns,
        which suits decomposition of a fixed recording. ``"ph"`` builds an
        expression driven by the phase-continuous oscillator, which is what
        Math-DAW pattern tracks need so pitch follows the MIDI notes.
    include_gate : bool
        When True the envelope is beat-relative (``u``/``beat``) and prefixed by
        the gate ``g``, matching Math-DAW pitched tracks.
    """
    kind = str(formula_type)
    f0 = float(f0)
    A = float(A)
    lam = max(0.0, float(decay_lambda))
    p1 = float(timbre_param)
    use_phase = str(carrier).lower() in ("ph", "phase")

    if include_gate:
        env = f"exp(-{_fmt(lam)}*u*beat)"
        prefix = "g * "
    else:
        env = f"exp(-{_fmt(lam)}*t)"
        prefix = ""

    if use_phase:
        base = "ph"
        harmonic = lambda h: f"sin({_fmt(float(h))}*ph)"
        double = "sin(2*ph)"
        saw_arg = "(ph/2)"
    else:
        base = f"2*pi*{_fmt(f0)}*t"
        harmonic = lambda h: f"sin({_fmt(2 * h)}*pi*{_fmt(f0)}*t)"
        double = f"sin(4*pi*{_fmt(f0)}*t)"
        saw_arg = f"(pi*{_fmt(f0)}*t)"

    if kind == "harmonic_power_saw":
        terms = " + ".join(
            f"{1.0 / (h ** max(0.3, p1)):.3f}*{harmonic(h)}" for h in range(1, 7)
        )
        return f"{prefix}{_fmt(A)}*{env}*({terms})"

    if kind == "soft_square_tanh":
        k = max(0.4, p1 * 3.0)
        return f"{prefix}{_fmt(A)}*{env}*tanh({_fmt(k)}*{base})"

    if kind == "fm_synthesis_bell":
        mod_index = p1 * 2.5
        mod_env = (
            f"exp(-2*{_fmt(max(0.1, lam))}*u*beat)"
            if include_gate
            else f"exp(-2*{_fmt(max(0.1, lam))}*t)"
        )
        return (
            f"{prefix}{_fmt(A)}*{env}*sin({base} + {_fmt(mod_index)}"
            f"*{mod_env}*{double})"
        )

    if kind == "damped_harmonic_series":
        if use_phase:
            terms = " + ".join(
                f"{1.0 / (h ** max(0.4, p1)):.3f}"
                f"*sin({_fmt(h * float(np.sqrt(1.0 + 0.0004 * h * h)))}*ph)"
                for h in range(1, 7)
            )
        else:
            terms = " + ".join(
                f"{1.0 / (h ** max(0.4, p1)):.3f}"
                f"*sin({_fmt(2 * h * np.sqrt(1.0 + 0.0004 * h * h))}*pi*{_fmt(f0)}*t)"
                for h in range(1, 7)
            )
        return f"{prefix}{_fmt(A)}*{env}*({terms})"

    if kind in SAWTOOTH_ALIASES:
        return (
            f"{prefix}(2*{_fmt(A)}/pi)*{env}"
            f"*arctan(cos({saw_arg})/(sin({saw_arg}) + 1e-9))"
        )

    return f"{prefix}{_fmt(A)}*{env}*sin({base})"


# ===========================================================================
# 2. SPECTROGRAM MATCHER, MULTI-SOLUTION BEAM SEARCH & LIBRARY LEARNING
# ===========================================================================

def _normalized_spectrum(y_seg: np.ndarray) -> Optional[np.ndarray]:
    """Max-normalized magnitude spectrum used as the fitting target."""
    windowed = y_seg * np.hanning(len(y_seg))
    spec = np.abs(np.fft.rfft(windowed))
    peak = float(np.max(spec))
    if peak <= 1e-12:
        return None
    return spec / peak


def _estimate_initial_lambda(y_seg: np.ndarray, sr: int) -> float:
    """Estimate a decay rate from the RMS envelope via log-linear regression."""
    frame_len = min(1024, max(128, len(y_seg) // 2))
    hop_len = max(32, frame_len // 4)
    try:
        rms = librosa.feature.rms(y=y_seg, frame_length=frame_len, hop_length=hop_len)[0]
    except Exception:
        return 1.0
    if len(rms) < 2 or float(np.max(rms)) <= 1e-6:
        return 1.0
    t_rms = np.linspace(0.0, len(y_seg) / sr, len(rms))
    try:
        slope, _ = np.polyfit(t_rms, np.log(rms + 1e-6), 1)
    except Exception:
        return 1.0
    if not np.isfinite(slope):
        return 1.0
    return float(np.clip(-slope, 0.0, 15.0))


def _optimize_formula(
    item: Dict[str, Any],
    t: np.ndarray,
    target_spec: np.ndarray,
    f0: float,
    A: float,
    x0: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Optimize (decay_lambda, timbre_param) for one formula family.

    Amplitude is a pure global scale and cancels under max-normalisation of
    the spectrum, so it is held fixed and only these two shape parameters
    are searched.
    """
    p1_bounds = tuple(item.get("bounds", (0.2, 4.0)))
    lam_lo, lam_hi = 0.0, 15.0
    bounds = [(lam_lo, lam_hi), (float(p1_bounds[0]), float(p1_bounds[1]))]

    window = np.hanning(len(t))

    def loss(params: np.ndarray) -> float:
        lam = float(np.clip(params[0], lam_lo, lam_hi))
        p1 = float(params[1])
        synth = item["func"](t, f0, A, lam, p1)
        spec = np.abs(np.fft.rfft(synth * window))
        peak = float(np.max(spec))
        if peak <= 1e-12:
            return 1.0
        return float(np.mean((target_spec - spec / peak) ** 2))

    x0_clipped = (
        float(np.clip(x0[0], lam_lo, lam_hi)),
        float(np.clip(x0[1], bounds[1][0], bounds[1][1])),
    )
    try:
        res = minimize(loss, x0=list(x0_clipped), bounds=bounds, method="L-BFGS-B")
        return float(res.fun), float(np.clip(res.x[0], lam_lo, lam_hi)), float(res.x[1])
    except Exception:
        return loss(np.array(x0_clipped)), x0_clipped[0], x0_clipped[1]


def fit_all_formula_candidates(
    y_seg: np.ndarray,
    sr: int,
    f0: float,
    candidates: Optional[List[str]] = None,
    max_starts: int = 3,
) -> List[EquationCandidate]:
    """Beam search across FORMULA_BANK, returning every distinct fitted solution.

    Unlike a single argmin winner this keeps the whole Pareto set, because the
    synthesis problem is ill-posed: several parameter sets can explain the same
    spectrum, so the choice between them should be a complexity decision.
    """
    if len(y_seg) < 64 or f0 <= 10.0 or np.isnan(f0):
        return []

    target_spec = _normalized_spectrum(y_seg)
    if target_spec is None:
        return []

    t = np.arange(len(y_seg)) / sr
    lam_init = _estimate_initial_lambda(y_seg, sr)
    A = float(np.max(np.abs(y_seg)))
    if A <= 1e-6:
        A = 0.5

    test_keys = candidates if candidates else list(FORMULA_BANK.keys())
    results: List[EquationCandidate] = []

    for name in test_keys:
        if name not in FORMULA_BANK:
            continue
        item = FORMULA_BANK[name]
        p1_bounds = tuple(item.get("bounds", (0.2, 4.0)))
        lo, hi = float(p1_bounds[0]), float(p1_bounds[1])
        default_p = float(np.clip(float(item.get("default_p", 1.0)), lo, hi))

        n_starts = max(1, int(max_starts))
        if n_starts == 1:
            p1_seeds = [default_p]
            lam_seeds = [lam_init]
        else:
            p1_seeds = sorted(
                {default_p}
                | {float(lo + (hi - lo) * frac) for frac in np.linspace(0.0, 1.0, n_starts)}
            )
            lam_seeds = sorted({
                lam_init,
                max(0.2, lam_init * 0.4),
                min(14.0, lam_init * 2.0 + 0.5),
            })

        seen: set = set()
        for p1_seed in p1_seeds:
            for lam_seed in lam_seeds:
                key = (round(p1_seed, 4), round(lam_seed, 4))
                if key in seen:
                    continue
                seen.add(key)

                err, lam, p1 = _optimize_formula(
                    item, t, target_spec, f0, A, (lam_seed, p1_seed)
                )
                equation = build_equation_string(name, f0, A, lam, p1)
                results.append(EquationCandidate(
                    formula_type=name,
                    equation=equation,
                    latex=item["latex"],
                    f0=float(f0),
                    A=float(A),
                    decay_lambda=float(lam),
                    timbre_param=float(p1),
                    spectral_mse=float(err),
                    complexity=equation_complexity(equation),
                ))

    results.sort(key=lambda c: (c.spectral_mse, c.complexity))
    return results


def pareto_front(candidates: Sequence[EquationCandidate]) -> List[EquationCandidate]:
    """Keep only non-dominated candidates on (spectral_mse, complexity)."""
    front: List[EquationCandidate] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            no_worse = (
                other.spectral_mse <= cand.spectral_mse
                and other.complexity <= cand.complexity
            )
            strictly_better = (
                other.spectral_mse < cand.spectral_mse
                or other.complexity < cand.complexity
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(cand)
    front.sort(key=lambda c: (c.spectral_mse, c.complexity))
    return front


def select_preferred(
    candidates: Sequence[EquationCandidate],
    error_tolerance: float = 0.02,
) -> Optional[EquationCandidate]:
    """Pick the simplest equation that stays within tolerance of the best fit.

    Complexity is only allowed to break near-ties in fidelity, so this never
    trades away a meaningful amount of accuracy for brevity.
    """
    if not candidates:
        return None
    best_err = min(c.spectral_mse for c in candidates)
    cutoff = best_err * (1.0 + float(error_tolerance)) + 1e-12
    viable = [c for c in candidates if c.spectral_mse <= cutoff]
    return min(viable, key=lambda c: (c.complexity, c.spectral_mse))


def fit_best_formula_for_segment(
    y_seg: np.ndarray,
    sr: int,
    f0: float,
    candidates: Optional[List[str]] = None,
    complexity_aware: bool = False,
) -> Optional[Dict[str, Any]]:
    """Tests formulas in FORMULA_BANK and selects the best spectral match.

    Parameters
    ----------
    y_seg : ndarray
        Real audio segment.
    sr : int
        Sample rate in Hz.
    f0 : float
        Fundamental frequency detected by CQT or pYIN.
    candidates : list of str, optional
        Restrict the search to these formula names.
    complexity_aware : bool
        When True, return the simplest equation that stays within 2% of the
        best achievable spectral error instead of the raw argmin winner.

    Returns
    -------
    dict or None
        Winning formula, equation string, optimized parameters, spectral MSE
        and a complexity score.
    """
    if len(y_seg) < 64 or f0 <= 10.0 or np.isnan(f0):
        return None

    t = np.arange(len(y_seg)) / sr
    if len(t) == 0:
        return None

    target_spec = _normalized_spectrum(y_seg)
    if target_spec is None:
        return None

    lam_init = _estimate_initial_lambda(y_seg, sr)

    A = float(np.max(np.abs(y_seg)))
    if A <= 1e-6:
        A = 0.5

    found = fit_all_formula_candidates(
        y_seg, sr, f0, candidates=candidates, max_starts=1
    )
    if not found:
        return None

    winner = select_preferred(found) if complexity_aware else found[0]
    return winner.to_dict()


def _refine_partial(
    residual: np.ndarray,
    t: np.ndarray,
    freq: float,
    amp: float,
    lam: float,
) -> Tuple[float, float, float, float]:
    """Refine (freq, amplitude, decay) of one partial against the residual.

    Spectral peak picking alone is not accurate enough for iterative
    subtraction: a fraction of a hertz of frequency error accumulates phase
    drift over the window and cancels badly. This minimizes the actual
    time-domain residual instead, so each subtraction really removes energy.
    """
    from scipy.optimize import minimize

    base = np.asarray(residual, dtype=np.float64)
    lam_hi = min(15.0, max(1.0, lam * 1.6))
    bounds = [
        (max(1.0, freq - 2.0), freq + 2.0),
        (max(1e-6, amp * 0.6), max(1e-5, amp * 1.6)),
        (0.0, lam_hi),
    ]
    x0 = [freq, max(1e-6, amp), float(np.clip(lam, 0.0, lam_hi))]

    def loss(params: np.ndarray) -> float:
        f, a, l = float(params[0]), float(params[1]), float(max(0.0, params[2]))
        synth = a * np.exp(-l * t) * np.sin(2.0 * np.pi * f * t)
        diff = base - synth
        return float(np.dot(diff, diff))

    try:
        res = minimize(loss, x0=x0, bounds=bounds, method="L-BFGS-B")
        if not res.success:
            return loss(np.array(x0)), x0[0], x0[1], x0[2]
        return float(res.fun), float(res.x[0]), float(res.x[1]), float(max(0.0, res.x[2]))
    except Exception:
        return loss(np.array(x0)), x0[0], x0[1], x0[2]


def _partial_envelope(
    raw_mags: np.ndarray,
    phases: np.ndarray,
    times: np.ndarray,
    coherent_gain: float,
    sr: int,
    n_fft: int,
    target_freq: float,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track the amplitude envelope of the partial nearest ``target_freq``."""
    from src.analysis import find_parabolic_peaks

    amps: List[float] = []
    times_out: List[float] = []
    for frame in range(raw_mags.shape[1]):
        frame_peaks = find_parabolic_peaks(
            raw_mags[:, frame],
            coherent_gain,
            sr,
            n_fft,
            phase_frame=phases[:, frame],
            threshold=0.002,
            max_peaks=10,
        )
        best = None
        best_err = tol
        for peak in frame_peaks:
            err = abs(peak["frequency"] - target_freq)
            if err <= best_err:
                best_err = err
                best = peak
        if best is not None and best["amplitude"] > 0.0:
            amps.append(float(best["amplitude"]))
            times_out.append(float(times[frame]))
    return np.asarray(times_out), np.asarray(amps)


def _decay_from_envelope(times: np.ndarray, amps: np.ndarray) -> Tuple[float, float]:
    """Log-linear decay rate and initial amplitude from a partial envelope."""
    if len(times) < 3:
        if len(amps) == 0:
            return 1.0, 0.0
        return 0.0, float(np.max(amps))
    rel = times - times[0]
    log_amps = np.log(np.maximum(amps, 1e-12))
    try:
        slope, intercept = np.polyfit(rel, log_amps, 1)
    except Exception:
        return 1.0, float(np.max(amps))
    if not np.isfinite(slope):
        return 1.0, float(np.max(amps))
    lam = float(np.clip(-slope, 0.0, 15.0))
    a0 = float(np.clip(np.exp(intercept), 0.0, float(np.max(amps)) + 1e-12))
    return lam, a0


def fit_equation_library(
    y_seg: np.ndarray,
    sr: int,
    f0: float,
    max_terms: int = 4,
    min_improvement_db: float = 0.5,
    n_candidates: int = 5,
    prefer_harmonics_of_f0: bool = False,
) -> Dict[str, Any]:
    """Library-learning equation discovery on the residual.

    InceptionSR-style recursion: fit one partial equation, freeze it, subtract
    it, then re-fit on what is left. Each round shortlists the strongest
    separated spectral peaks and keeps whichever subtraction removes the most
    energy, so a quieter partial is not shadowed by leftovers from a louder
    one. Each term's decay rate is fitted from that partial's own amplitude
    envelope, then (frequency, amplitude, decay) are refined against the
    time-domain residual. The result is a readable sum of terms rather than an
    opaque expression tree.
    """
    if len(y_seg) < 64:
        return {
            "equation": "",
            "components": [],
            "term_count": 0,
            "residual_reduction_db": 0.0,
        }

    from src.analysis import compute_normalized_stft, find_parabolic_peaks

    y_seg = np.asarray(y_seg, dtype=np.float64)
    residual = y_seg.copy()
    t_abs = np.arange(len(y_seg)) / sr
    initial_energy = float(np.mean(residual ** 2))
    if initial_energy <= 1e-14:
        return {
            "equation": "",
            "components": [],
            "term_count": 0,
            "residual_reduction_db": 0.0,
        }

    components: List[Dict[str, Any]] = []
    terms: List[str] = []
    reference_amp = 0.0

    for _ in range(max(1, int(max_terms))):
        energy_before = float(np.mean(residual ** 2))
        if energy_before <= 1e-14:
            break

        stft_res = compute_normalized_stft(
            residual, sr=sr, n_fft=2048, hop_length=256
        )
        raw_mags = stft_res["raw_mags"]
        phases = stft_res["phases"]
        times = stft_res["time_frames"]
        coherent_gain = stft_res["coherent_gain"]
        n_fft = stft_res["n_fft"]

        frame_peaks: List[Dict[str, float]] = []
        for frame in range(raw_mags.shape[1]):
            frame_peaks.extend(find_parabolic_peaks(
                raw_mags[:, frame],
                coherent_gain,
                sr,
                n_fft,
                phase_frame=phases[:, frame],
                threshold=0.002,
                max_peaks=8,
            ))
        frame_peaks = [
            p for p in frame_peaks
            if 10.0 < p["frequency"] < sr / 2.0 and p["amplitude"] > 0.0
        ]
        if not frame_peaks:
            break

        def _rank(peak: Dict[str, float]) -> float:
            score = peak["amplitude"]
            if prefer_harmonics_of_f0 and f0 > 10.0:
                ratio = peak["frequency"] / f0
                nearest = round(ratio)
                if 1 <= nearest <= 16 and abs(ratio - nearest) / nearest < 0.05:
                    score *= 1.5
            return score

        frame_peaks.sort(key=_rank, reverse=True)

        shortlist: List[float] = []
        for peak in frame_peaks:
            sep = max(10.0, 0.02 * peak["frequency"])
            if all(abs(peak["frequency"] - f) > sep for f in shortlist):
                shortlist.append(peak["frequency"])
            if len(shortlist) >= int(n_candidates):
                break
        if not shortlist:
            break

        best_result: Optional[Tuple[float, float, float, float, float]] = None
        for cand_freq in shortlist:
            tol = max(4.0, 0.012 * cand_freq)
            env_times, env_amps = _partial_envelope(
                raw_mags, phases, times, coherent_gain, sr, n_fft, cand_freq, tol
            )
            if env_amps.size == 0:
                continue
            env_lam, env_amp = _decay_from_envelope(env_times, env_amps)
            if not np.isfinite(env_amp) or env_amp <= 0.0:
                continue
            _, rf, ra, rl = _refine_partial(residual, t_abs, cand_freq, env_amp, env_lam)
            if not (np.isfinite(rf) and np.isfinite(ra) and np.isfinite(rl)):
                continue
            if ra <= 0.0:
                continue
            synth = ra * np.exp(-rl * t_abs) * np.sin(2.0 * np.pi * rf * t_abs)
            gain = float(np.dot(residual, synth))
            after = float(np.mean((residual - synth) ** 2))
            improvement = 10.0 * np.log10((energy_before + 1e-20) / (after + 1e-20))
            key = (improvement, rf, ra, rl, gain)
            if best_result is None or improvement > best_result[0]:
                best_result = key

        if best_result is None:
            break
        improvement_db, freq, amp, lam, _gain = best_result
        if amp <= 0.0:
            break

        if reference_amp <= 0.0:
            reference_amp = float(amp)
        elif amp < 1e-3 * reference_amp:
            break

        term = (
            f"{_fmt(amp)}*exp(-{_fmt(lam)}*t)"
            f"*sin(2*pi*{_fmt(freq)}*t)"
        )
        synthesized = amp * np.exp(-lam * t_abs) * np.sin(2.0 * np.pi * freq * t_abs)
        next_residual = residual - synthesized

        if improvement_db < float(min_improvement_db):
            break

        terms.append(term)
        components.append({
            "term_index": len(components),
            "frequency": round(freq, 3),
            "amplitude": round(amp, 5),
            "decay_lambda": round(lam, 4),
            "harmonic_of_f0": (
                round(freq / f0, 3) if f0 and f0 > 10.0 else None
            ),
            "improvement_db": round(float(improvement_db), 3),
        })
        residual = next_residual

    final_energy = float(np.mean(residual ** 2))
    reduction_db = 10.0 * np.log10((initial_energy + 1e-20) / (final_energy + 1e-20))

    return {
        "equation": " + ".join(terms),
        "components": components,
        "term_count": len(terms),
        "residual_reduction_db": round(float(reduction_db), 3),
    }


# ===========================================================================
# 3. TIER 2: AUTOMATED FORMULA INVENTION (SYMBOLIC REGRESSION)
# ===========================================================================

class SymbolicFormulaRegressor:
    """Lightweight evolutionary symbolic regression engine for audio wave synthesis.

    Combines mathematical operators (sin, cos, exp, tanh, +, *) into candidate equations
    to discover novel acoustic shapes when dictionary formulas are insufficient.
    """

    OPERATORS = ["sin", "cos", "tanh", "exp_decay"]

    def __init__(self, population_size: int = 16, generations: int = 6):
        self.pop_size = population_size
        self.generations = generations

    def fit(self, y_seg: np.ndarray, sr: int, f0: float) -> Dict[str, Any]:
        """Evolves a composite waveform equation matching the target audio segment."""
        t = np.arange(len(y_seg)) / sr
        target_spec = np.abs(np.fft.rfft(y_seg * np.hanning(len(y_seg))))
        target_spec /= (np.max(target_spec) + 1e-12)
        A = float(np.max(np.abs(y_seg)))

        # Candidate equation forms: A * exp(-lam*t) * ( w1*op1(2pi*f0*t) + w2*op2(4pi*f0*t) )
        best_genome = None
        best_loss = float("inf")

        # Initial random population of genomes: (op1, op2, k1, k2, lam)
        for _ in range(self.pop_size):
            op1 = np.random.choice(["sin", "tanh"])
            op2 = np.random.choice(["sin", "cos"])
            k1 = float(np.random.uniform(0.5, 2.5))
            k2 = float(np.random.uniform(0.1, 1.2))
            lam = float(np.random.uniform(0.5, 4.0))

            def eval_candidate(o1, o2, p1, p2, l_decay):
                w1 = np.sin(2 * np.pi * f0 * t) if o1 == "sin" else np.tanh(p1 * np.sin(2 * np.pi * f0 * t))
                w2 = np.sin(4 * np.pi * f0 * t) if o2 == "sin" else np.cos(4 * np.pi * f0 * t)
                wave = A * np.exp(-l_decay * t) * (w1 + p2 * w2)
                spec = np.abs(np.fft.rfft(wave * np.hanning(len(wave))))
                spec /= (np.max(spec) + 1e-12)
                return float(np.mean((target_spec - spec) ** 2))

            loss = eval_candidate(op1, op2, k1, k2, lam)
            if loss < best_loss:
                best_loss = loss
                best_genome = (op1, op2, k1, k2, lam)

        op1, op2, k1, k2, lam = best_genome
        latex = f"{A:.3f} * exp(-{lam:.2f}*t) * ({op1}(2*pi*{f0:.1f}*t) + {k2:.2f}*{op2}(4*pi*{f0:.1f}*t))"
        return {
            "formula_type": "symbolic_regression",
            "equation_template": latex,
            "f0": round(float(f0), 2),
            "A": round(float(A), 4),
            "decay_lambda": round(float(lam), 3),
            "timbre_param": round(float(k1), 3),
            "spectral_mse": round(float(best_loss), 6),
        }


# ===========================================================================
# 4. TIER 3: DYNAMIC CUSTOM FORMULA LOADER
# ===========================================================================

def load_custom_formulas(file_path: Union[str, Path]) -> int:
    """Loads and registers custom formulas from an external Python module or JSON file.

    Parameters:
        file_path: Path to .py or .json formula definition file

    Returns:
        Number of newly registered formulas
    """
    path = Path(file_path)
    if not path.is_file():
        return 0

    registered_count = 0

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, item in data.items():
            latex = item.get("latex", "")
            desc = item.get("description", "")
            # Safe AST lambda generation from formula text
            from src.safe_eval import safe_eval
            func = lambda t, f0, A, lam, p1, eq=latex: safe_eval(
                eq,
                {"t": t, "n": f0, "g": 1.0, "ph": 2.0 * np.pi * f0 * t, "p1": p1, "A": A, "lam": lam, "u": 0.5, "s": 0, "rnd": np.zeros_like(t), "beat": 0.5}
            )
            register_formula(name, latex, func, desc)
            registered_count += 1

    elif path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("custom_formulas", str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "CUSTOM_FORMULAS"):
                for name, item in mod.CUSTOM_FORMULAS.items():
                    register_formula(
                        name,
                        item.get("latex", ""),
                        item["func"],
                        item.get("description", ""),
                        item.get("bounds", (0.2, 4.0)),
                        item.get("default_p", 1.0),
                    )
                    registered_count += 1

    return registered_count
