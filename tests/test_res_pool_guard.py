"""Repo invariant: a runnable script that spawns processes must guard __main__.

On Windows ``ProcessPoolExecutor`` uses ``spawn``, so every worker re-imports
the parent script as ``__main__``. Without the guard the worker re-runs the
whole script and spawns its own pool -- a fork bomb. That cost three hours
twice before it was made a test.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Library modules legitimately have no guard; only runnable scripts do.
SKIP_DIRS = {"src", "tests", "site", "tools", "venv", ".git", "node_modules",
             "data", "output", "__pycache__"}


def _candidate_scripts():
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # Hidden/tooling trees (".venv", ".git", ".pytest_cache", ...) hold
        # third-party sources, not repo scripts. A repo-local .venv is the
        # modern default, so skipping every dotted directory keeps this
        # invariant from failing on site-packages instead of on our code.
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


@pytest.mark.parametrize("script", sorted(_candidate_scripts()),
                         ids=lambda p: p.name)
def test_process_spawning_scripts_are_guarded(script: Path):
    src = script.read_text(encoding="utf-8", errors="replace")
    if "ProcessPoolExecutor" not in src:
        pytest.skip("no process pool")
    assert "if __name__" in src, (
        f"{script.name} constructs a ProcessPoolExecutor but has no "
        f"if __name__ == '__main__' guard; Windows spawn will fork-bomb."
    )


def test_guard_actually_wraps_the_entry_point():
    """A guard that never returns/executes is no guard at all."""
    offenders = []
    for script in _candidate_scripts():
        src = script.read_text(encoding="utf-8", errors="replace")
        if "ProcessPoolExecutor" not in src or "if __name__" not in src:
            continue
        try:
            tree = ast.parse(src, str(script))
        except SyntaxError:
            offenders.append(f"{script.name}: does not parse")
            continue
        guard_calls = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Compare)
            and any(getattr(v, "value", "") == "__name__" for v in n.test.left)
            and any(call for stmt in n.body for call in ast.walk(stmt)
                    if isinstance(call, ast.Call))
        ]
        if not guard_calls:
            offenders.append(f"{script.name}: __main__ guard contains no call")
    assert not offenders, "malformed main guards: " + "; ".join(offenders)


def test_cached_pools_are_released_at_exit():
    """A cached pool holds live worker processes past the last test."""
    from src import res_loop
    src = Path(res_loop.__file__).read_text(encoding="utf-8")
    assert "atexit" in src, "res_loop must register atexit pool shutdown"
    assert "_POOL_CACHE" in src, "pool cache should be introspectable"