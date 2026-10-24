"""Joint frequency refinement in the discovery path.

Each partial's frequency is re-estimated against the residual left by its
neighbours (deflation) rather than against the raw audio, then the group is
re-solved and the result kept only if the group residual drops.
"""
import numpy as np

from src.tracking import _deflated_freq_refine


def _cand(freq, a0, phase, decay, start_s, end_s, sr=44100, start=0):
    return {
        "mean_freq": freq,
        "start_time": start_s,
        "start_idx": int(start_s * sr),
        "end_idx": int(end_s * sr),
        "init_phase": phase,
        "env_fit": {"a0": a0, "decay_rate": decay},
    }


def _render(cands, sr=44100, dur=2.0):
    n = int(dur * sr)
    y = np.zeros(n)
    for c in cands:
        s, e = c["start_idx"], c["end_idx"]
        t = np.maximum(0.0, np.arange(s, e) / float(sr) - c["start_time"])
        env = np.exp(-c["env_fit"]["decay_rate"] * t)
        y[s:e] += (c["env_fit"]["a0"] * env
                   * np.sin(2 * np.pi * c["mean_freq"] * t + c["init_phase"]))
    return y


def test_returns_empty_for_a_single_candidate():
    c = [_cand(440.0, 0.5, 0.0, 1.1, 0.0, 2.0)]
    y = _render(c)
    assert _deflated_freq_refine(c, y, 0, 44100) == {}


def test_returns_empty_for_oversized_groups():
    """8.2 keeps large sets at their greedy optimum; so do we."""
    c = [_cand(440.0 + i, 0.1, 0.0, 1.0, 0.0, 2.0) for i in range(13)]
    y = _render(c)
    assert _deflated_freq_refine(c, y, 0, 44100) == {}


def test_refinement_recovers_a_perturbed_frequency():
    """A partial whose stored frequency is slightly off must be pulled back."""
    sr = 44100
    true_f = 440.0
    stored = [_cand(true_f + 0.08, 0.5, 0.0, 1.1, 0.0, 2.0),
              _cand(554.365, 0.3, 0.4, 1.45, 0.0, 2.0)]
    # the AUDIO is at the true frequency; only the stored value is wrong
    truth = [_cand(true_f, 0.5, 0.0, 1.1, 0.0, 2.0),
             _cand(554.365, 0.3, 0.4, 1.45, 0.0, 2.0)]
    y = _render(truth)
    out = _deflated_freq_refine(stored, y, 0, sr)
    assert 0 in out, "the perturbed candidate must be refined"
    assert abs(out[0] - true_f) < 0.06, (
        f"expected ~{true_f}, got {out[0]}"
    )


def test_refinement_is_a_no_op_when_frequencies_are_exact():
    """An exact fit must not move meaningfully.

    The bounded optimiser lands a couple of microhertz from the minimum on a
    flat objective, so the tolerance is set far below the ~1e-3 Hz errors this
    refinement exists to remove.
    """
    sr = 44100
    c = [_cand(440.0, 0.5, 0.0, 1.1, 0.0, 2.0),
         _cand(554.365, 0.3, 0.4, 1.45, 0.0, 2.0)]
    y = _render(c)
    out = _deflated_freq_refine(c, y, 0, sr)
    for i, f in out.items():
        assert abs(f - c[i]["mean_freq"]) < 1e-4, "an exact fit must not move"


def test_discovery_frequency_error_does_not_increase():
    """End-to-end: the discovered frequencies must be at least as good."""
    import tempfile
    from pathlib import Path
    from src.synthesis import generate_benchmark_chord
    from src.tracking import discover_equations_from_audio

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "chord.wav"
        model = generate_benchmark_chord(p, duration=2.0, sr=44100)
        disc = discover_equations_from_audio(
            p, n_fft=4096, hop_length=1024, threshold=0.01,
            out_json_path=str(Path(td) / "e.json"))
        worst = 0.0
        for c, g in zip(disc["components"], model["components"]):
            worst = max(worst, abs(c["frequency"] - g["frequency"]))
        assert worst < 0.002, (
            f"worst frequency error {worst:.6f} Hz; the refinement should keep "
            f"it under 2 mHz"
        )
