"""The partial schema is a contract, not a bag (see GLOSSARY.md).

These tests run the REAL constructors - the analyser's refine_track, the
bank's hypothesis classes, the noise-floor fitter, the exciter - on cheap
synthetic signals and assert their output validates against
src/partial_schema.py. A new model class that mints an unregistered key
fails here before it can silently drift from its consumers.
"""

from __future__ import annotations

import numpy as np

from src.partial_schema import KNOWN_KEYS, MODELS, validate_partial
from src.res_bank import (
    filtered_noise_hypothesis,
    lpc_noise_hypothesis,
    sibilance_burst_hypothesis,
)
from src.res_synthesis import fit_noise_floor_partial
from src.sinusoidal_model import sinusoidal_reconstruction

SR = 22050


def _damped_sinusoid(freq=440.0, decay=1.5, seconds=1.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return np.exp(-decay * t) * np.cos(2.0 * np.pi * freq * t)


def test_registry_covers_every_writer_module():
    # The registry groups must name the keys the four writer modules mint.
    assert {"envelope_a0", "projection_gain", "phase_ref_hz"} <= KNOWN_KEYS
    assert {"seed", "band_centers", "band_mags", "lpc_coeffs"} <= KNOWN_KEYS
    assert {"unit_step", "note_group", "m", "trajectory"} <= KNOWN_KEYS
    assert set(MODELS) == {
        "constant", "damped", "glide", "filtered_noise", "lpc_noise",
    }


def test_analyser_partials_validate():
    # The full public analyser path: audio -> STFT -> tracks -> fitted
    # partials (windowed reconstruction is what production runs use).
    y = _damped_sinusoid()
    result = sinusoidal_reconstruction(y, SR, window_s=1.0)
    fitted = result["fitted_parameters"]
    assert fitted, "the analyser should fit at least one partial"
    for p in fitted:
        problems = validate_partial(p)
        assert problems == [], problems


def test_noise_channel_partials_validate():
    rng = np.random.default_rng(7)
    residual = 0.2 * rng.standard_normal(SR)
    variant = filtered_noise_hypothesis(residual, SR, start_sample=0)
    for p in variant.partials:
        problems = validate_partial(p)
        assert problems == [], problems


def test_lpc_noise_partials_validate():
    rng = np.random.default_rng(11)
    residual = 0.1 * rng.standard_normal(SR)
    variant = lpc_noise_hypothesis(residual, SR, start_sample=0)
    for p in variant.partials:
        problems = validate_partial(p)
        assert problems == [], problems


def test_sibilance_partials_validate():
    rng = np.random.default_rng(3)
    # Band-limited high-frequency noise so the hf_ratio gate opens.
    n = 2 * SR
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    spec[freqs < 3500.0] = 0.0
    residual = np.fft.irfft(spec, n=n)
    variant = sibilance_burst_hypothesis(residual, SR, start_sample=0)
    for p in variant.partials:
        problems = validate_partial(p)
        assert problems == [], problems


def test_noise_floor_partial_validates():
    rng = np.random.default_rng(5)
    residual = 0.05 * rng.standard_normal(2 * SR)
    p = fit_noise_floor_partial(residual, SR, 0.0, 2.0)
    assert p is not None
    problems = validate_partial(p)
    assert problems == [], problems


def _valid_damped():
    return {
        "model": "damped",
        "frequency": 220.0,
        "amplitude": 0.5,
        "phase": 0.1,
        "envelope_decay": 2.0,
        "envelope_t0": 0.0,
        "start_time": 0.0,
        "end_time": 1.0,
        "m": None,
        "note_group": 0.0,
        "unit_step": True,
    }


def test_typo_and_contract_violations_are_caught():
    # The classic silent failure: a misspelled amplitude key.
    p = _valid_damped()
    del p["amplitude"]
    p["amplitde"] = 0.5
    problems = validate_partial(p)
    assert any("missing key 'amplitude'" in s for s in problems)
    assert any("unregistered key 'amplitde'" in s for s in problems)

    # Time order the support mask relies on.
    p = _valid_damped()
    p["start_time"], p["end_time"] = 1.0, 0.5
    assert any("after 'end_time'" in s for s in validate_partial(p))

    # A glide must carry a consistent trajectory.
    p = _valid_damped()
    p["model"] = "glide"
    assert any("trajectory" in s for s in validate_partial(p))
    p["trajectory"] = {"t": [0.0, 1.0], "f_hz": [440.0], "a": [1.0, 1.0]}
    assert any("lengths differ" in s for s in validate_partial(p))

    # filtered_noise without a seed breaks the reproducibility contract.
    p = _valid_damped()
    p["model"] = "filtered_noise"
    assert any("'seed'" in s for s in validate_partial(p))

    # Unknown model.
    p = _valid_damped()
    p["model"] = "sawtooth"
    assert any("unknown model" in s for s in validate_partial(p))

    # The legacy alias is tolerated (docs call the damped atom damped_cosine).
    p = _valid_damped()
    p["model"] = "damped_cosine"
    assert validate_partial(p) == []


def test_validate_partial_is_pure():
    p = _valid_damped()
    validate_partial(p)
    assert set(p) == {
        "model", "frequency", "amplitude", "phase", "envelope_decay",
        "envelope_t0", "start_time", "end_time", "m", "note_group",
        "unit_step",
    }
