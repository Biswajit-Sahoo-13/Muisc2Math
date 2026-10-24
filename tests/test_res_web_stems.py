"""Guards for the web-UI stem-separation plumbing."""
import numpy as np
import soundfile as sf


def test_demucs_status_reports_availability():
    from src.res_ui import demucs_status
    ok, msg = demucs_status()
    assert isinstance(ok, bool)
    assert msg and isinstance(msg, str)


def test_separation_backend_is_recorded():
    """The UI must be able to tell real separation from an HPSS fallback."""
    from src.res_evidence import separate_stems
    import src.res_evidence as re
    sr = 22050
    t = np.arange(sr * 2) / sr
    y = (0.4 * np.sin(2 * np.pi * 440 * t) * np.exp(-0.8 * t)).astype(np.float64)
    separate_stems(y, sr=sr, mode="auto")
    assert re.LAST_SEPARATION_BACKEND in (
        "demucs", "hpss", "directory"), (
        f"unexpected backend {re.LAST_SEPARATION_BACKEND!r}"
    )


def test_hpss_fallback_is_labelled_hpss():
    """With Demucs disabled the backend must be reported as hpss, not 'auto'."""
    import src.res_evidence as re
    sr = 22050
    t = np.arange(sr * 2) / sr
    y = (0.4 * np.sin(2 * np.pi * 440 * t) * np.exp(-0.8 * t)).astype(np.float64)
    re.separate_stems(y, sr=sr, mode="none")
    assert re.LAST_SEPARATION_BACKEND == "hpss"


def test_payload_carries_residual_and_stem_note():
    import tempfile
    from pathlib import Path
    from src.res_ui import run_res_payload

    sr = 22050
    t = np.arange(int(3.0 * sr)) / sr
    y = (0.4 * np.sin(2 * np.pi * 440 * t) * np.exp(-0.8 * t)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "p.wav"
        sf.write(str(p), y, sr)
        blob = p.read_bytes()

    root = Path(__file__).resolve().parents[1]
    payload = run_res_payload(blob, "p.wav", 3.0, 1, bank=False, root=root,
                              stem_mode="none")
    assert payload["orig_bytes"], "original audio missing from payload"
    assert payload["best_bytes"], "reconstruction missing from payload"
    assert payload["residual_bytes"], "residual missing from payload"
    assert "separation_note" in payload
    assert payload.get("stems") in (None, {}), \
        "single-mix mode must not report stems"
