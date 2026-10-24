"""Guards for the loop fast paths: pool reuse, worker default, recon assembly."""
import numpy as np

import src.res_loop as rl

SR = 22050


def _ledger(n_sections=3, sec_s=1.0, n_partials=4):
    sections = []
    for i in range(n_sections):
        start = i * sec_s
        partials = [{
            "model": "constant",
            "amplitude": 0.2,
            "frequency": 220.0 * (i + 1),
            "phase": 0.1 * i,
            "envelope_decay": 1.5,
            "envelope_t0": start,
            "start_time": start + 0.05,
            "end_time": start + sec_s - 0.15,
        } for _ in range(n_partials)]
        sections.append({
            "section": i,
            "start": start,
            "end": start + sec_s,
            "start_sample": int(start * SR),
            "partials": partials,
            "lag_samples": 0,
        })
    n = int(n_sections * sec_s * SR)
    return {
        "sample_rate": SR,
        "n_samples": n,
        "stft_half_window_s": 0.25,
        "sections": sections,
        "best": {},
        "iterations": [],
    }


def _audio(n_sections=3, sec_s=1.0):
    n = int(n_sections * sec_s * SR)
    rng = np.random.default_rng(3)
    return (rng.standard_normal(n) * 0.05).astype(np.float32)


def test_assembled_recon_matches_render_ledger():
    """Pasting worker mixes must be identical to re-rendering the ledger."""
    led = _ledger()
    y = _audio()
    n = led["n_samples"]
    st_hw = led["stft_half_window_s"]

    recon = np.zeros(n)
    for sec in led["sections"]:
        w_start = sec["start_sample"]
        w_end = min(n, int(round(sec["end"] * SR)))
        y_sec = y[w_start:w_end]
        updated, _cls, mix = rl._process_section_task(
            (sec, y_sec, SR, w_start, st_hw, False, False))
        assert mix is not None
        end = min(n, w_start + len(mix))
        recon[w_start:end] = mix[: end - w_start]

    reference = rl.render_ledger(led, n)
    assert np.max(np.abs(recon - reference)) < 1e-12


def test_skipped_section_falls_back_to_full_render():
    """A converged section has no mix, so assembly must be abandoned."""
    led = _ledger()
    led["sections"][1]["converged"] = True
    _sec, _cls, mix = rl._process_section_task(
        (led["sections"][1], np.zeros(SR), SR, led["sections"][1]["start_sample"],
         0.25, False, False))
    assert mix is None


def test_pool_is_cached_per_width():
    rl.shutdown_worker_pools()
    try:
        a = rl._get_pool(2)
        b = rl._get_pool(2)
        assert a is b, "same width must reuse the same pool"
        c = rl._get_pool(3)
        assert c is not a, "different width must get its own pool"
    finally:
        rl.shutdown_worker_pools()
    assert not rl._POOL_CACHE


def test_default_worker_count_is_capped_at_eight():
    """16 workers on 24 sections thrashes memory bandwidth; 8 is the sweet spot."""
    import inspect
    src = inspect.getsource(rl.run_loop)
    assert "min(8, os.cpu_count() or 1" in src, "default must cap at 8"
    assert "ProcessPoolExecutor(max_workers=n_workers)" not in src, \
        "the pool must not be constructed inside the iteration loop"


def test_run_loop_end_to_end_matches_serial():
    """Parallel and serial passes must produce the same ledger."""
    import copy
    import tempfile
    a_led = _ledger()
    b_led = copy.deepcopy(_ledger())
    y = _audio()
    with tempfile.TemporaryDirectory() as d:
        try:
            rl.run_loop(a_led, y, d + "/par", iterations=2, target_corr=0.0,
                        workers=4, bank=False)
        finally:
            rl.shutdown_worker_pools()
        rl.run_loop(b_led, y, d + "/ser", iterations=2, target_corr=0.0,
                    workers=1, bank=False)
    pa = [round(s["metrics"]["correlation"], 9) for s in a_led["sections"]]
    pb = [round(s["metrics"]["correlation"], 9) for s in b_led["sections"]]
    assert pa == pb
    assert round(a_led["best"]["correlation"], 6) == round(
        b_led["best"]["correlation"], 6)