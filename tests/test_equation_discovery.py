"""Tests for multi-solution equation discovery.

Covers:
1. Equation complexity scoring (simpler expressions must score lower).
2. Equation code generation staying inside the safe_eval whitelist.
3. Beam search returning the Pareto set rather than a single winner.
4. Complexity-aware selection never sacrificing real fidelity.
5. Library learning recovering multiple partials with per-partial decay.
6. Degenerate inputs (silence, short segments, invalid f0).
"""
import unittest

import numpy as np

from src.formula_bank import (
    FORMULA_BANK,
    EquationCandidate,
    build_equation_string,
    equation_complexity,
    fit_all_formula_candidates,
    fit_best_formula_for_segment,
    fit_equation_library,
    pareto_front,
    select_preferred,
)
from src.safe_eval import safe_eval

SR = 22050


def _vars(t, f0=220.0):
    return {
        "t": t,
        "u": t,
        "s": t,
        "n": f0,
        "g": np.ones_like(t),
        "ph": 2.0 * np.pi * f0 * t,
        "rnd": np.zeros_like(t),
        "beat": 0.5,
    }


class TestEquationComplexity(unittest.TestCase):
    def test_simpler_scores_lower(self):
        simple = equation_complexity("0.5*sin(2*pi*220*t)")
        complex_eq = equation_complexity(
            "0.5*exp(-2*t)*(1.0*sin(2*pi*220*t)+0.5*sin(4*pi*220*t)"
            "+0.33*sin(6*pi*220*t)+0.25*sin(8*pi*220*t))"
        )
        self.assertLess(simple, complex_eq)

    def test_unparseable_is_maximally_complex(self):
        self.assertGreater(equation_complexity("sin("), 900.0)
        self.assertGreater(equation_complexity("__import__('os')"), 0.0)

    def test_bank_equations_are_safe_and_finite(self):
        t = np.arange(int(SR * 0.5)) / SR
        for include_gate in (False, True):
            for carrier in ("t", "ph"):
                for name in FORMULA_BANK:
                    eq = build_equation_string(
                        name, 220.0, 0.6, 2.0, 1.2,
                        include_gate=include_gate, carrier=carrier,
                    )
                    y = safe_eval(eq, _vars(t))
                    self.assertIsInstance(y, np.ndarray, name)
                    self.assertEqual(len(y), len(t), name)
                    self.assertTrue(np.all(np.isfinite(y)), name)
                    self.assertLess(equation_complexity(eq), 900.0, name)

    def test_phase_carrier_ignores_absolute_frequency(self):
        eq_t = build_equation_string("pure_sine", 220.0, 0.6, 2.0, 1.0)
        eq_ph = build_equation_string("pure_sine", 220.0, 0.6, 2.0, 1.0, carrier="ph")
        self.assertIn("220", eq_t)
        self.assertNotIn("220", eq_ph)
        self.assertTrue(eq_ph.endswith("sin(ph)"))

    def test_phase_carrier_follows_pattern_pitch(self):
        t = np.arange(int(SR * 0.5)) / SR
        eq = build_equation_string(
            "pure_sine", 220.0, 0.6, 0.0, 1.0, include_gate=True, carrier="ph"
        )
        ph = 2.0 * np.pi * 440.0 * t
        y = safe_eval(eq, _vars(t) | {"ph": ph})
        ref = 0.6 * np.sin(ph)
        self.assertLess(float(np.max(np.abs(y - ref))), 1e-6)

    def test_gate_is_respected(self):
        t = np.arange(int(SR * 0.5)) / SR
        eq = build_equation_string("pure_sine", 220.0, 0.6, 2.0, 1.0, include_gate=True)
        vars_on = _vars(t)
        vars_off = _vars(t)
        vars_off["g"] = np.zeros_like(t)
        self.assertGreater(float(np.max(np.abs(safe_eval(eq, vars_on)))), 0.01)
        self.assertAlmostEqual(float(np.max(np.abs(safe_eval(eq, vars_off)))), 0.0, places=12)

    def test_unknown_formula_falls_back_to_sine(self):
        eq = build_equation_string("does_not_exist", 440.0, 0.5, 1.0, 1.0)
        self.assertIn("sin(2*pi*440*t)", eq)


class TestBeamSearchAndPareto(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(int(SR * 0.5)) / SR
        self.target = 0.6 * np.exp(-2.0 * self.t) * np.tanh(
            2.5 * np.sin(2.0 * np.pi * 330.0 * self.t)
        )

    def test_beam_search_returns_multiple_distinct_solutions(self):
        cands = fit_all_formula_candidates(self.target, SR, 330.0, max_starts=3)
        self.assertGreater(len(cands), 5)
        self.assertGreater(len({c.formula_type for c in cands}), 1)

    def test_candidates_sorted_and_scored(self):
        cands = fit_all_formula_candidates(self.target, SR, 330.0, max_starts=3)
        errs = [c.spectral_mse for c in cands]
        self.assertEqual(errs, sorted(errs))
        for c in cands:
            self.assertGreaterEqual(c.complexity, 0.0)
            self.assertLess(c.complexity, 900.0)
            self.assertTrue(np.isfinite(c.spectral_mse))

    def test_pareto_front_is_non_dominated(self):
        cands = fit_all_formula_candidates(self.target, SR, 330.0, max_starts=3)
        front = pareto_front(cands)
        self.assertTrue(front)
        self.assertLessEqual(len(front), len(cands))
        for a in front:
            for b in front:
                if a is b:
                    continue
                dominated = (
                    b.spectral_mse <= a.spectral_mse
                    and b.complexity <= a.complexity
                    and (b.spectral_mse < a.spectral_mse or b.complexity < a.complexity)
                )
                self.assertFalse(dominated)

    def test_complexity_aware_never_loses_fidelity(self):
        cands = fit_all_formula_candidates(self.target, SR, 330.0, max_starts=3)
        best_err = min(c.spectral_mse for c in cands)
        argmin = cands[0]
        picked = select_preferred(cands)
        self.assertIsNotNone(picked)
        self.assertLessEqual(picked.spectral_mse, best_err * 1.02 + 1e-12)
        self.assertLessEqual(picked.complexity, argmin.complexity + 1e-9)

    def test_complexity_aware_fit_flag(self):
        default = fit_best_formula_for_segment(self.target, SR, 330.0)
        aware = fit_best_formula_for_segment(self.target, SR, 330.0, complexity_aware=True)
        self.assertIsNotNone(default)
        self.assertIsNotNone(aware)
        self.assertLessEqual(aware["complexity"], default["complexity"] + 1e-9)
        self.assertAlmostEqual(aware["f0"], 330.0, places=1)
        self.assertLess(aware["spectral_mse"], 0.05)

    def test_pareto_front_of_empty_input(self):
        self.assertEqual(pareto_front([]), [])
        self.assertIsNone(select_preferred([]))

    def test_degenerate_inputs(self):
        self.assertEqual(fit_all_formula_candidates(np.zeros(4096), SR, 220.0), [])
        self.assertEqual(fit_all_formula_candidates(np.zeros(10), SR, 220.0), [])
        self.assertIsNone(fit_best_formula_for_segment(self.target, SR, float("nan")))
        self.assertIsNone(fit_best_formula_for_segment(self.target, SR, 5.0))

    def test_candidate_subset_restriction(self):
        cands = fit_all_formula_candidates(
            self.target, SR, 330.0, candidates=["pure_sine", "soft_square_tanh"]
        )
        self.assertTrue(cands)
        self.assertLessEqual({c.formula_type for c in cands}, {"pure_sine", "soft_square_tanh"})

    def test_candidate_to_dict_shape(self):
        cand = EquationCandidate(
            formula_type="pure_sine",
            equation="0.5*sin(2*pi*220*t)",
            latex="A*sin",
            f0=220.0,
            A=0.5,
            decay_lambda=1.0,
            timbre_param=1.0,
            spectral_mse=0.01,
            complexity=15.2,
        )
        d = cand.to_dict()
        for key in (
            "formula_type", "description", "equation_template", "equation",
            "f0", "A", "decay_lambda", "timbre_param", "spectral_mse", "complexity",
        ):
            self.assertIn(key, d)


class TestLibraryLearning(unittest.TestCase):
    def test_recovers_two_partials_with_per_partial_decay(self):
        sr = SR
        t = np.arange(int(sr * 1.0)) / sr
        f1, a1, l1 = 220.0, 0.5, 1.5
        f2, a2, l2 = 660.0, 0.25, 4.0
        mix = (
            a1 * np.exp(-l1 * t) * np.sin(2 * np.pi * f1 * t)
            + a2 * np.exp(-l2 * t) * np.sin(2 * np.pi * f2 * t)
        )
        lib = fit_equation_library(mix, sr, f0=f1, max_terms=4)
        self.assertGreaterEqual(lib["term_count"], 2)
        self.assertGreater(lib["residual_reduction_db"], 8.0)

        found = [c["frequency"] for c in lib["components"]]
        self.assertTrue(any(abs(f - f1) < 5.0 for f in found), found)
        self.assertTrue(any(abs(f - f2) < 8.0 for f in found), found)

        decays = {round(c["frequency"]): c["decay_lambda"] for c in lib["components"]}
        for freq, truth in ((f1, l1), (f2, l2)):
            match = min(decays, key=lambda f: abs(f - freq))
            self.assertAlmostEqual(decays[match], truth, delta=0.6,
                                   msg=f"decay for {freq} Hz")

        rendered = safe_eval(lib["equation"], _vars(t, f1))
        self.assertTrue(np.all(np.isfinite(rendered)))
        snr = 10.0 * np.log10(
            np.sum(mix ** 2) / (np.sum((mix - rendered) ** 2) + 1e-20)
        )
        self.assertGreater(snr, 10.0)

    def test_equation_is_plain_sum_of_terms(self):
        sr = SR
        t = np.arange(int(sr * 0.6)) / sr
        y = (
            0.5 * np.exp(-2.0 * t) * np.sin(2 * np.pi * 330.0 * t)
            + 0.3 * np.exp(-4.0 * t) * np.sin(2 * np.pi * 660.0 * t)
        )
        lib = fit_equation_library(y, sr, f0=330.0, max_terms=3)
        self.assertGreaterEqual(lib["term_count"], 2)
        self.assertIn(" + ", lib["equation"])
        for part in lib["equation"].split(" + "):
            self.assertEqual(part.count("sin("), 1)
            self.assertIn("exp(-", part)

    def test_pure_tone_yields_single_term(self):
        sr = SR
        t = np.arange(int(sr * 0.6)) / sr
        y = 0.5 * np.exp(-2.0 * t) * np.sin(2 * np.pi * 440.0 * t)
        lib = fit_equation_library(y, sr, f0=440.0, max_terms=4)
        self.assertEqual(lib["term_count"], 1)
        self.assertAlmostEqual(lib["components"][0]["frequency"], 440.0, delta=3.0)

    def test_harmonic_ratio_recorded(self):
        sr = SR
        t = np.arange(int(sr * 0.8)) / sr
        y = 0.5 * np.exp(-1.0 * t) * np.sin(2 * np.pi * 200.0 * t)
        lib = fit_equation_library(y, sr, f0=200.0, max_terms=2)
        self.assertAlmostEqual(lib["components"][0]["harmonic_of_f0"], 1.0, delta=0.02)

    def test_max_terms_respected(self):
        sr = SR
        t = np.arange(int(sr * 1.0)) / sr
        y = (
            0.4 * np.exp(-1.0 * t) * np.sin(2 * np.pi * 200.0 * t)
            + 0.3 * np.exp(-1.5 * t) * np.sin(2 * np.pi * 400.0 * t)
            + 0.2 * np.exp(-2.0 * t) * np.sin(2 * np.pi * 800.0 * t)
        )
        lib = fit_equation_library(y, sr, f0=200.0, max_terms=2)
        self.assertLessEqual(lib["term_count"], 2)

    def test_recovers_all_three_chord_partials(self):
        """Three-partial chord: every ground-truth frequency must be found."""
        import tempfile
        from pathlib import Path

        from src.analysis import load_audio
        from src.synthesis import generate_benchmark_chord

        with tempfile.TemporaryDirectory() as tmp:
              wav = Path(tmp) / "chord.wav"
              model = generate_benchmark_chord(wav, duration=2.0, sr=44100)
              y, sr = load_audio(wav, target_sr=44100)
        y = y[: int(2.0 * sr)]

        lib = fit_equation_library(y, sr, f0=440.0, max_terms=4, n_candidates=6)
        self.assertGreaterEqual(lib["term_count"], 3)
        found = [c["frequency"] for c in lib["components"]]
        for truth in (440.0, 554.365, 659.255):
              self.assertTrue(
                  any(abs(f - truth) < 4.0 for f in found),
                  f"missing {truth} Hz in {found}",
              )
        first = min(lib["components"], key=lambda c: abs(c["frequency"] - 440.0))
        # The fixture is scaled so the rendered sum peaks under full scale, so
        # the expected amplitude comes from the ground-truth model, not a
        # hard-coded constant.
        truth_amp = model["components"][0]["amplitude"]
        self.assertAlmostEqual(first["frequency"], 440.0, delta=1.0)
        self.assertAlmostEqual(first["amplitude"], truth_amp, delta=0.05)
        self.assertAlmostEqual(first["decay_lambda"], 1.10, delta=0.10)
        self.assertGreater(lib["residual_reduction_db"], 8.0)

    def test_degenerate_inputs(self):
        empty = fit_equation_library(np.zeros(10), SR, 220.0)
        self.assertEqual(empty["term_count"], 0)
        self.assertEqual(empty["equation"], "")
        silence = fit_equation_library(np.zeros(4096), SR, 220.0)
        self.assertEqual(silence["term_count"], 0)
        self.assertEqual(silence["residual_reduction_db"], 0.0)

    def test_never_fits_below_noise_floor(self):
        sr = SR
        rng = np.random.default_rng(0)
        noise = 0.02 * rng.standard_normal(int(sr * 1.0))
        lib = fit_equation_library(noise, sr, f0=220.0, max_terms=4)
        self.assertLessEqual(lib["term_count"], 1)


if __name__ == "__main__":
    unittest.main()
