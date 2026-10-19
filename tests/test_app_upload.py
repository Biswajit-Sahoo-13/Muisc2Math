"""Exercise the upload path in the real app.

The `st.audio(key=...)` bug survived the whole session because every existing
test rendered the *results* panel without ever uploading a file, so the upload
branch never executed. This test drives the uploader so that branch is
actually run.
"""
import wave
from pathlib import Path

import numpy as np
import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"
SR = 22050


def _wav_bytes(seconds=1.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    y = (0.4 * np.sin(2 * np.pi * 440 * t) * np.exp(-0.8 * t)).astype(np.float32)
    import io
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def test_app_runs_with_a_file_uploaded():
    """The uploader branch must execute without raising."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]

    ups = at.file_uploader
    assert ups, "no file_uploader found in the app"
    try:
        ups[0].upload("probe.wav", _wav_bytes(), "audio/wav")
    except Exception as e:  # pragma: no cover - depends on streamlit version
        pytest.skip(f"AppTest cannot drive file_uploader here: {e}")
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]


def test_upload_player_is_rendered():
    """After an upload the app must expose the full-song audio element."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=300)
    at.run()
    ups = at.file_uploader
    if not ups:
        pytest.skip("no file_uploader")
    try:
        ups[0].upload("probe.wav", _wav_bytes(), "audio/wav")
    except Exception:
        pytest.skip("AppTest cannot drive file_uploader here")
    at.run()
    assert at.exception == [], [str(e.value) for e in at.exception]
    # st.audio renders into the element tree; assert the branch ran by
    # checking the caption text that only exists in the upload branch
    texts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert any("Your upload, in full" in t for t in texts), (
        "the upload branch did not render"
    )
