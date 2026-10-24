"""Architecture tests for RES module dependencies and import cycles."""

import importlib
import sys
import unittest


class TestArchitectureImports(unittest.TestCase):
    def test_clean_import_order_bank_first(self):
        """Importing res_bank before res_loop must succeed without circular import errors."""
        for mod in list(sys.modules):
            if mod.startswith("src.res_"):
                sys.modules.pop(mod, None)
        bank = importlib.import_module("src.res_bank")
        loop = importlib.import_module("src.res_loop")
        self.assertTrue(hasattr(bank, "tournament_section"))
        self.assertTrue(hasattr(loop, "run_loop"))

    def test_clean_import_order_loop_first(self):
        """Importing res_loop before res_bank must succeed without circular import errors."""
        for mod in list(sys.modules):
            if mod.startswith("src.res_"):
                sys.modules.pop(mod, None)
        loop = importlib.import_module("src.res_loop")
        bank = importlib.import_module("src.res_bank")
        self.assertTrue(hasattr(loop, "run_loop"))
        self.assertTrue(hasattr(bank, "tournament_section"))

    def test_synthesis_primitives_accessible(self):
        """Shared synthesis primitives are available from res_synthesis and re-exported by res_loop."""
        from src.res_synthesis import _corr, _render_partial
        from src.res_loop import _corr as loop_corr, _render_partial as loop_render
        self.assertEqual(_corr, loop_corr)
        self.assertEqual(_render_partial, loop_render)


if __name__ == "__main__":
    unittest.main()
