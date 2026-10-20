"""Structural guards for app.py.

The `key=` regression is the reason these exist: `st.audio()` has no `key`
parameter in Streamlit 1.66, and the failure only fires when a file is
actually uploaded -- so the old results-only tests never executed that branch
and the app shipped broken. These checks are static and always run.
"""
import ast
import inspect
from pathlib import Path

import pytest
import streamlit as st

APP = Path(__file__).resolve().parents[1] / "app.py"


def _tree():
    return ast.parse(APP.read_text(encoding="utf-8"), str(APP))


def test_st_audio_has_no_key_parameter():
    """The bug: st.audio() rejects key= at runtime."""
    params = inspect.signature(st.audio).parameters
    assert "key" not in params, (
        "st.audio gained a key parameter; the app no longer needs the "
        "workaround"
    )


def test_no_st_audio_call_passes_key():
    """Static guard for the whole app, including unreachable branches."""
    offenders = []
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "audio"):
            for kw in node.keywords:
                if kw.arg == "key":
                    offenders.append(f"line {node.lineno}")
    assert not offenders, (
        "st.audio() called with key= (unsupported): " + ", ".join(offenders)
    )


PRODUCT_TABS = ["RES: Song -> Equations -> Replay", "Benchmarks: sample tests"]


def test_app_exposes_only_the_main_tabs():
    """The app is the product surface, not a toolbox.

    Two tabs: the reverse-engineering surface, and the sample-test harness.
    The results panel nests its own Original/Recreated/Residual set.
    """
    labels = []
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "tabs"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "st"
                and node.args
                and isinstance(node.args[0], ast.List)):
            labels.append([e.value for e in node.args[0].elts])
    assert labels, "no st.tabs() call found"
    assert labels[0] == PRODUCT_TABS, labels[0]
    for extra in labels[1:]:
        assert extra == ["Original", "Recreated", "Residual"], extra


def test_no_other_tab_blocks_exist():
    """Only the product tabs may be entered with a `with tab_x:` block."""
    found = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.With):
            for item in node.items:
                if isinstance(item.context_expr, ast.Name) and \
                        item.context_expr.id.startswith("tab_"):
                    found.append(item.context_expr.id)
    assert set(found) <= {"tab_res", "tab_bench"}, f"extra tab blocks: {set(found)}"


def test_no_math_daw_sidebar_remains():
    """The Math-DAW sidebar (BPM / presets / JSON import) was scoped to a
    removed tab; it must not reappear as a global sidebar."""
    for node in ast.walk(_tree()):
        if (isinstance(node, ast.With)
                and any(isinstance(i.context_expr, ast.Call)
                        and isinstance(i.context_expr.func, ast.Attribute)
                        and i.context_expr.func.attr == "sidebar"
                        for i in node.items)):
            pytest.fail("a global st.sidebar block is back in app.py")


def test_upload_path_is_reachable():
    """The upload branch must exist (it is what broke)."""
    src = APP.read_text(encoding="utf-8")
    assert "up.getvalue()" in src, "the full-song upload player is missing"
    assert 'key="res_upload"' in src
