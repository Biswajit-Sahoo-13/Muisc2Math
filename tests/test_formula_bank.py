"""Unit tests for Tier 1-3 Formula Bank and Spectral Optimization."""

import unittest
import numpy as np
from src.formula_bank import (
    FORMULA_BANK,
    register_formula,
    fit_best_formula_for_segment,
    load_custom_formulas,
)


class TestFormulaBank(unittest.TestCase):
    """Test suite for FORMULA_BANK and parameter optimization."""

    def test_default_formula_families_exist(self):
        """All required formula families must be present in the bank."""
        expected = [
            "pure_sine",
            "harmonic_power_saw",
            "soft_square_tanh",
            "fm_synthesis_bell",
            "damped_harmonic_series",
            "rich_sawtooth_arctan",
        ]
        for name in expected:
            self.assertIn(name, FORMULA_BANK, f"Missing formula family: {name}")

    def test_formula_execution_and_bounds(self):
        """All formulas must produce real, finite, bounded outputs."""
        sr = 22050
        dur = 0.5
        t = np.arange(int(sr * dur)) / sr
        f0 = 220.0
        A = 0.8
        lam = 2.0
        p1 = 1.2

        for name, entry in FORMULA_BANK.items():
            func = entry["func"]
            out = func(t, f0, A, lam, p1)
            self.assertIsInstance(out, np.ndarray, f"{name} did not return ndarray")
            self.assertEqual(len(out), len(t), f"{name} returned unexpected length")
            self.assertFalse(np.any(np.isnan(out)), f"{name} returned NaN")
            self.assertFalse(np.any(np.isinf(out)), f"{name} returned Inf")
            # Should be bounded
            self.assertLessEqual(np.max(np.abs(out)), 5.0, f"{name} exceeded amplitude bound")

    def test_register_formula(self):
        """Registering a new custom formula dynamically."""
        custom_name = "test_custom_pulse"
        register_formula(
            name=custom_name,
            func=lambda t, f0, A, lam, p1: A * np.exp(-lam * t) * np.sin(2 * np.pi * f0 * t) ** 3,
            latex="A * e^(-λt) * sin(2π f0 t)^3",
            description="Cubed sine pulse",
        )
        self.assertIn(custom_name, FORMULA_BANK)
        t = np.linspace(0, 0.1, 100)
        res = FORMULA_BANK[custom_name]["func"](t, 220.0, 1.0, 1.0, 1.0)
        self.assertEqual(len(res), 100)

    def test_fit_best_formula_for_segment(self):
        """fit_best_formula_for_segment should identify the correct formula family."""
        sr = 22050
        dur = 0.5
        t = np.arange(int(sr * dur)) / sr
        # Synthetic soft-square target
        target = 0.6 * np.exp(-2.0 * t) * np.tanh(2.5 * np.sin(2 * np.pi * 330.0 * t))

        match = fit_best_formula_for_segment(target, sr=sr, f0=330.0)
        self.assertIsNotNone(match)
        self.assertIn("formula_type", match)
        self.assertIn("spectral_mse", match)
        self.assertLess(match["spectral_mse"], 0.05)
        self.assertAlmostEqual(match["f0"], 330.0, places=1)


if __name__ == "__main__":
    unittest.main()
