"""Tests for the LLM backend registry and the safe runner.

The runner is the interesting part: a CLI that spawns children can hold the
output pipe open after it is killed, which makes ``subprocess.run(timeout=)``
block forever instead of timing out. These tests pin the bounded behaviour.
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from src.llm_backends import (
    BACKENDS,
    Backend,
    build_command,
    find_backend,
    probe_backend,
    run_backend,
)
from src.res_loop import apply_edits, apply_edits_detailed


class TestRegistry(unittest.TestCase):
    def test_the_backends_the_project_needs_are_present(self):
        for name in ("agy", "opencode", "claude", "gemini", "qwen", "codex"):
            self.assertIn(name, BACKENDS)
            self.assertIsNotNone(find_backend(name))

    def test_each_backend_builds_a_command_with_its_own_convention(self):
        # agy takes --print, opencode takes a subcommand, gemini takes -p,
        # and qwen takes the prompt positionally because -p is deprecated
        # there. These are different shapes and must not be flattened.
        shapes = {}
        for name in ("agy", "opencode", "gemini", "qwen"):
            spec = BACKENDS[name]
            shapes[name] = spec.build("PROMPT", "MODEL")
        self.assertIn("--print", shapes["agy"])
        self.assertIn("run", shapes["opencode"])
        self.assertIn("-p", shapes["gemini"])
        self.assertEqual(shapes["qwen"][0], "PROMPT")
        for name, argv in shapes.items():
            self.assertIn("PROMPT", argv, name)
            self.assertIn("MODEL", argv, name)

    def test_a_missing_binary_is_a_clear_error(self):
        spec = Backend(name="nope", binary="definitely-not-installed-xyz",
                       build=lambda p, m: [p], identity="nope")
        with self.assertRaises(FileNotFoundError) as ctx:
            build_command(spec, "x")
        self.assertIn("not installed", str(ctx.exception))

    def test_a_prompt_that_cannot_fit_is_refused_not_truncated(self):
        # A silently shortened prompt makes the model answer a different
        # question, which is worse than an error.
        spec = Backend(name="x", binary="python", build=lambda p, m: [p],
                       identity="x")
        huge = "p" * 400000
        with self.assertRaises(ValueError) as ctx:
            build_command(spec, huge)
        self.assertIn("will not fit", str(ctx.exception))


class TestRunner(unittest.TestCase):
    def test_captures_stdout(self):
        res = run_backend([sys.executable, "-c", "print('hello')"], timeout_s=30)
        self.assertTrue(res.ok)
        self.assertIn("hello", res.stdout)

    def test_a_non_zero_exit_is_reported_not_raised(self):
        res = run_backend([sys.executable, "-c", "import sys; sys.exit(3)"],
                           timeout_s=30)
        self.assertFalse(res.ok)
        self.assertEqual(res.exit_code, 3)

    def test_a_missing_executable_is_reported(self):
        res = run_backend(["definitely-not-installed-xyz-123"], timeout_s=10)
        self.assertFalse(res.ok)
        self.assertIn("not found", res.error)

    def test_a_child_that_holds_the_pipe_does_not_hang_the_caller(self):
        # The bug this pins: a CLI that spawns a child keeps the stdout pipe
        # open, so subprocess.run(timeout=) blocks in communicate() forever.
        script = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "time.sleep(120)\n"
        )
        started = time.time()
        res = run_backend([sys.executable, "-c", script], timeout_s=5)
        elapsed = time.time() - started
        self.assertTrue(res.timed_out)
        self.assertLess(elapsed, 60, "the runner did not honour its timeout")

    def test_stdin_is_delivered(self):
        res = run_backend(
            [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
            stdin_text="from-stdin",
            timeout_s=30,
        )
        self.assertTrue(res.ok)
        self.assertIn("from-stdin", res.stdout)


class TestEditReceipts(unittest.TestCase):
    def _ledger(self):
        return {"sections": [{"partials": [
            {"amplitude": 1.0, "phase": 0.0, "envelope_decay": 5.0, "frequency": 100.0},
            {"amplitude": 1.0, "phase": 0.0, "envelope_decay": 5.0, "frequency": 200.0},
        ]}]}

    def test_the_old_return_value_is_unchanged(self):
        ledger = self._ledger()
        edits = [
            {"section": 0, "id": "part_000", "op": "set_decay", "value": 2.5},
            {"section": 0, "id": "part_001", "op": "drop", "value": 0.0},
            {"section": 0, "id": "part_001", "op": "refit_amplitude", "value": 0.5},
        ]
        self.assertEqual(apply_edits(ledger, edits), 3)

    def test_refused_edits_are_reported_with_a_reason(self):
        ledger = self._ledger()
        edits = [
            {"section": 0, "id": "part_000", "op": "set_decay", "value": 2.5},
            {"section": 9, "id": "part_000", "op": "set_decay", "value": 1.0},
            {"section": 0, "id": "part_999", "op": "set_decay", "value": 1.0},
            {"section": 0, "id": "part_000", "op": "nonsense", "value": 1.0},
            {"section": 0, "id": "part_000", "op": "set_decay", "value": "not-a-number"},
        ]
        applied, refused = apply_edits_detailed(ledger, edits)
        self.assertEqual(applied, 1)
        self.assertEqual(len(refused), 4)
        reasons = {r["reason"] for r in refused}
        self.assertIn("no such section", reasons)
        self.assertIn("no such partial", reasons)
        self.assertTrue(any("unknown op" in r for r in reasons))
        self.assertTrue(any("not a number" in r for r in reasons))

    def test_a_refused_edit_does_not_change_the_ledger(self):
        ledger = self._ledger()
        before = dict(ledger["sections"][0]["partials"][0])
        apply_edits_detailed(ledger, [
            {"section": 0, "id": "part_000", "op": "nonsense", "value": 9.0},
        ])
        self.assertEqual(ledger["sections"][0]["partials"][0], before)


class TestProbe(unittest.TestCase):
    def test_a_name_collision_is_caught(self):
        # A binary with the right name but the wrong tool must not be
        # reported as usable.
        with tempfile.TemporaryDirectory() as tmp:
            if os.name == "nt":
                fake = Path(tmp) / "faketool.bat"
                fake.write_text("@echo 9.9.9\n", encoding="utf-8")
            else:
                fake = Path(tmp) / "faketool"
                fake.write_text("#!/bin/sh\necho 9.9.9\n", encoding="utf-8")
                fake.chmod(0o755)
            old = os.environ.get("PATH", "")
            os.environ["PATH"] = tmp + os.pathsep + old
            try:
                spec = Backend(name="faketool", binary="faketool",
                               build=lambda p, m: [p], identity="faketool")
                info = probe_backend(spec, timeout_s=15)
                self.assertTrue(info["installed"])
                self.assertFalse(info["usable"])
                self.assertIn("does not look like", info["reason"])
            finally:
                os.environ["PATH"] = old


if __name__ == "__main__":
    import time
    unittest.main()
