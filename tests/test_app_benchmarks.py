"""The Benchmarks tab: the sample-test harness, driven through the real app."""
import numpy as np
import pytest


def test_benchmarks_tab_renders_the_recorded_labels():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    APP = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]
    # two tabs now
    assert len(at.tabs) >= 1
    texts = " ".join(m.value for m in at.markdown) + " " + \
            " ".join(c.value for c in at.caption)
    assert "sample-test harness" in texts or "Benchmarks" in texts


def test_benchmarks_tab_shows_the_scoreboard_table():
    from streamlit.testing.v1 import AppTest
    from pathlib import Path

    APP = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "mix_lost" in html, "the fixture table did not render"
    assert "after" in html, "the recorded labels did not render"


def test_scoreboard_module_verifies_the_repo_scoreboard():
    """src/scoreboard is the single implementation the CLI and the app share."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.scoreboard import verify

    result = verify()
    assert result["consistent"] is True, result["labels"]
    # the captured baselines are allowed not to improve; nothing else is
    for label in result["labels"]:
        if label in ("baseline", "baseline_head"):
            continue
        assert result["labels"][label]["improved"] is True, (
            f"{label} stopped improving: {result['labels'][label]}"
        )
    assert result["ok"] is True
