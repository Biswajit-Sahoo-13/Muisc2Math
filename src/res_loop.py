"""RES (Reverse Engineer a Song) loop: hypothesize -> render -> score.

The loop over one evidence ledger: each iteration gathers hypotheses (numeric
coordinate refit, or an LLM via a headless command such as ``opencode run``),
renders the equation state to audio, scores it against the original, and only
then updates the ledger. Like REA's finding ledger, every accepted change is
recorded with its measured before/after metrics, and every number comes from
measurement - nothing is claimed without a score backing it.

Edit schema (stable contract for any LLM provider) - apply_edits():

    {"section": int, "id": "part_###" | null,
     "op": "refit_amplitude" | "refit_phase" | "set_decay"
           | "set_frequency_offset" | "drop",
     "value": float}

In-session agent usage (drive it directly like an REA agent would):

    from src.res_evidence import build_evidence, evidence_prompt
    from src.res_loop import run_loop, apply_edits
    ledger = build_evidence("song.mp3", "out")
    prompt = evidence_prompt(ledger)          # read it, plan edits
    ledger = run_loop(ledger, "song.mp3", "out", iterations=4)
    print(ledger["best"], ledger["iterations"])

Antigravity cannot be called headlessly from a script; drive RES in-session
there and supply edits through ``apply_edits`` - the loop verifies them the
same way it verifies any other hypothesis.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import copy
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
from scipy.optimize import least_squares

from src.sinusoidal_model import _main_lobe_refine

from src.res_synthesis import (
  _corr,
  _glide_traj,
  _support_mask,
  _render_partial,
  _score,
  apply_energy_parity,
  fit_noise_floor_partial,
  )
from src.res_bank import tournament_section
from src.residual import segmental_snr_db
from src.error_correction import (
    apply_lag,
)
from src.llm_backends import (
    auto_backend,
    build_command,
    find_backend,
    run_api,
    run_backend,
)

_PROVIDER_NOTE_ANTIGRAVITY = (
    "Antigravity is callable headlessly: 'agy --print <prompt>' runs a "
    "single prompt and prints the answer, so the RES loop can use it as a "
    "proposer like any other CLI backend. The referee still scores every "
    "iteration and the loop still rolls back what does not help."
)


def apply_ola_crossfade(
    blocks: Sequence[np.ndarray],
    crossfade_samples: int = 256,
    sr: int = 22050,
) -> np.ndarray:
    """Concatenate sequential audio blocks with smooth Overlap-Add Hann crossfade.

    Uses a raised-cosine (Hann) crossfade over ``crossfade_samples`` between
    consecutive blocks, guaranteeing C1 continuity and eliminating step
    discontinuities, clicks, and pops across all section boundaries.
    """
    if not blocks:
        return np.array([], dtype=np.float64)
    if len(blocks) == 1:
        return np.asarray(blocks[0], dtype=np.float64)

    xf = max(1, crossfade_samples)
    for b in blocks:
        if len(b) < 2 * xf:
            xf = min(xf, len(b) // 2)
    if xf < 2:
        return np.concatenate(blocks)

    out_parts = [np.asarray(blocks[0], dtype=np.float64)]

    for idx in range(1, len(blocks)):
        prev = out_parts[-1]
        curr = np.asarray(blocks[idx], dtype=np.float64)
        curr_xf = min(xf, len(prev) // 2, len(curr) // 2)
        if curr_xf < 2:
            out_parts.append(curr)
            continue

        overlap_prev = prev[-curr_xf:]
        overlap_curr = curr[:curr_xf]
        t_cur = np.linspace(0.0, np.pi / 2.0, curr_xf)
        blended = overlap_prev * (np.cos(t_cur) ** 2) + overlap_curr * (np.sin(t_cur) ** 2)

        prev_trimmed = prev[:-curr_xf]
        out_parts[-1] = np.concatenate([prev_trimmed, blended])
        out_parts.append(curr[curr_xf:])

    return np.concatenate(out_parts)


def render_ledger(
    ledger: Dict[str, Any],
    n_samples: int,
    stft_half_window_s: Optional[float] = None,
    ola: bool = False,
    crossfade_s: float = 0.015,
) -> np.ndarray:
    """Render the full equation state over the whole song.

    Sections are rendered on their own time origin (stored phases are
    section-referenced) and pasted at their exact start sample. If ``ola`` is
    True, boundaries between consecutive sections are smoothed via Hann
    crossfades.
    """
    if stft_half_window_s is None:
        stft_half_window_s = float(ledger.get("stft_half_window_s", 0.0))
    sr = int(ledger["sample_rate"])
    out = np.zeros(n_samples)
    xf_len = int(round(crossfade_s * sr))

    for sec in ledger["sections"]:
        w_start = int(sec.get("start_sample", round(sec["start"] * sr)))
        # Same clip as the loop's per-section slice, so a section renders
        # exactly as long as the audio it was fitted against.
        sec_len = min(
            n_samples, int(sec.get("end_sample", round(sec["end"] * sr)))
        ) - w_start
        if sec_len <= 0:
            continue
        r_sec = np.zeros(sec_len)
        for p in sec["partials"]:
            r_sec += _render_partial(
                p, sec_len, sr,
                start_sample=w_start, stft_half_window_s=stft_half_window_s,
            )
        lag = int(sec.get("lag_samples", 0))
        if lag != 0:
            r_sec = apply_lag(r_sec, lag)
        n_sec = len(r_sec)
        end = min(n_samples, w_start + n_sec)
        actual_len = end - w_start
        if actual_len <= 0:
            continue

        r_slice = r_sec[:actual_len].copy()
        if ola and w_start > 0 and xf_len > 1 and actual_len > 2 * xf_len:
            curr_xf = min(xf_len, w_start, actual_len // 4)
            if curr_xf > 1:
                t = np.linspace(0.0, np.pi / 2.0, curr_xf)
                fade_in = np.sin(t) ** 2
                fade_out = np.cos(t) ** 2
                # Smooth the seam transition
                prev_val = out[w_start - 1]
                curr_val = r_slice[0]
                if abs(curr_val - prev_val) > 1e-4:
                    r_slice[:curr_xf] = r_slice[:curr_xf] * fade_in + prev_val * fade_out

        out[w_start:end] += r_slice
    return out


def apply_edits(ledger: Dict[str, Any], edits: Sequence[Dict[str, Any]]) -> int:
    """Apply agent edits to the ledger sections. Returns how many were applied.

    An edit targets ``section`` + ``id`` where the id is the section-local
    ``part_###`` label of the partial. Unknown ids and malformed values are
    skipped silently - the caller's edit count is the honest receipt.
    """
    applied, _refused = apply_edits_detailed(ledger, edits)
    return applied


def apply_edits_detailed(
    ledger: Dict[str, Any], edits: Sequence[Dict[str, Any]]
) -> Tuple[int, List[Dict[str, Any]]]:
    """Apply agent edits and report what was refused as well as applied.

    The refused list is the honest half of the receipt: a model that
    proposed twelve edits of which nine named parts that do not exist
    looks the same as a model that proposed three good ones if only the
    applied count is recorded.
    """
    n_applied = 0
    refused: List[Dict[str, Any]] = []
    for e in edits:
        try:
            sec = ledger["sections"][int(e.get("section", -1))]
        except (IndexError, TypeError, ValueError, KeyError):
            refused.append({"edit": e, "reason": "no such section"})
            continue
        pid = str(e.get("id", ""))
        idx = -1
        if pid.startswith("part_"):
            try:
                idx = int(pid[5:])
            except ValueError:
                idx = -1
        if not (0 <= idx < len(sec["partials"])):
            refused.append({"edit": e, "reason": "no such partial"})
            continue
        target = sec["partials"][idx]
        op = e.get("op", "")
        try:
            value = float(e.get("value", 0.0))
        except (TypeError, ValueError):
            refused.append({"edit": e, "reason": "value is not a number"})
            continue
        if op == "refit_amplitude":
            target["amplitude"] = round(max(value, 0.0), 6)
        elif op == "refit_phase":
            target["phase"] = round(value, 6)
        elif op == "set_decay":
            target["envelope_decay"] = round(max(value, 0.0), 6)
        elif op == "set_frequency_offset":
            target["frequency"] = round(float(target.get("frequency", 0.0)) + value, 6)
        elif op == "drop":
            target["amplitude"] = 0.0
        else:
            refused.append({"edit": e, "reason": "unknown op %r" % (op,)})
            continue
        n_applied += 1
    return n_applied, refused


def _section_update(
    y_sec: np.ndarray,
    partials: List[Dict[str, Any]],
    sr: int,
    start_sample: int = 0,
    stft_half_window_s: float = 0.0,
) -> int:
    """Greedy coordinate lifter over the equation set, one section at a time.

    For each partial, candidate edits are proposed: an (amplitude, phase)
    re-projection at each candidate decay of a small grid, a frequency
    refinement, and a drop. Because nearby partials are not orthogonal, a
    locally "optimal" re-projection can drain energy from a near-duplicate and
    wreck the sum - and a least-squares refit can reduce residual energy while
    lowering Pearson corr, so every candidate is accepted only when the
    *section correlation* (the referee's own metric, VERIFY) strictly
    improves. The section score can therefore never get worse, and each
    iteration is verified against the original anyway.
    Returns the number of accepted edits.
    """
    n = len(y_sec)
    if n < 8 or not partials:
        return 0
    t = (np.arange(n, dtype=np.float64) + start_sample) / sr
    t_fit = np.arange(n, dtype=np.float64) / sr
    accepted = 0

    # Incremental residual: render partials once, maintain mix delta
    rendered_partials = {
        id(q): _render_partial(q, n, sr, start_sample=start_sample, stft_half_window_s=stft_half_window_s)
        for q in partials
    }
    cur_mix = np.sum(list(rendered_partials.values()), axis=0) if rendered_partials else np.zeros(n)

    for p in list(partials):
        amp_old = float(p.get("amplitude", 0.0))
        if amp_old <= 0.0 or p.get("model") == "filtered_noise":
            continue

        own = rendered_partials.get(id(p))
        if own is None:
            continue
        others = cur_mix - own
        target = y_sec - others
        cur_full = cur_mix
        cur_corr = _corr(y_sec, cur_full)
        cur_seg_snr = segmental_snr_db(y_sec, cur_full, sr)

        if p.get("model") == "glide":
            traj = p.get("trajectory")
            if not traj:
                continue
            tt = np.asarray(traj.get("t", []), dtype=np.float64)
            ff = np.asarray(traj.get("f_hz", []), dtype=np.float64)
            aa = np.asarray(traj.get("a", []), dtype=np.float64)
            if tt.size == 0 or ff.size != tt.size or aa.size != tt.size:
                continue
            a_env, integral = _glide_traj(tt, ff, aa, t)
            freq_ref = float(p.get("frequency", 0.0))
            st_p = p.get("start_time", None)
            et_p = p.get("end_time", None)
            decay = float(p.get("envelope_decay", 0.0))
            if st_p is not None and et_p is not None:
                m = _support_mask(t, float(st_p), float(et_p), decay, 1.0, stft_half_window_s)
                a_env_m = a_env * m
            else:
                a_env_m = a_env

            best_glide = None
            best_glide_corr = cur_corr
            # Same anchor frequency the renderer uses, or the refit optimises
            # a phase reference the renderer never reproduces.
            ref_hz = p.get("phase_ref_hz")
            glide_anchor = (
                float(ref_hz) if ref_hz is not None else float(np.median(ff))
            )
            for df in (-2.0, -1.0, 0.0, 1.0, 2.0):
                theta = integral + 2.0 * np.pi * (freq_ref + df - glide_anchor) * t_fit
                c_sig = a_env_m * np.cos(theta)
                s_sig = a_env_m * np.sin(theta)
                base = float(np.sum(c_sig ** 2) + np.sum(s_sig ** 2))
                if base <= 1e-12:
                    continue
                cc = float(np.dot(c_sig, target)) / base
                ss = float(np.dot(s_sig, target)) / base
                a = float(np.hypot(cc, ss))
                ph = float(np.arctan2(ss, cc))
                if np.isfinite(a) and np.isfinite(ph) and a > 0.0:
                    cand_sig = a * a_env_m * np.cos(theta - ph)
                    fresh = others + cand_sig
                    c_corr = _corr(y_sec, fresh)
                    c_seg_snr = segmental_snr_db(y_sec, fresh, sr)
                    if c_corr > best_glide_corr + 1e-9 and c_seg_snr >= cur_seg_snr - 0.25:
                        best_glide_corr = c_corr
                        best_glide = ("refit", df, a, ph)

            drop_corr = _corr(y_sec, others)
            drop_seg_snr = segmental_snr_db(y_sec, others, sr)
            if drop_corr > best_glide_corr + 1e-9 and drop_seg_snr >= cur_seg_snr - 0.25:
                best_glide = ("drop", 0.0, 0.0, 0.0)

            if best_glide is not None:
                if best_glide[0] == "refit":
                    _, df, a, ph = best_glide
                    p["frequency"] = round(freq_ref + df, 6)
                    p["phase"] = round(ph, 6)
                    # The renderer multiplies the trajectory envelope by
                    # projection_gain, so the refitted gain has to live there.
                    # It used to be written to `amplitude` and projection_gain
                    # was reset to 1.0, which silently threw the fitted gain
                    # away and left the glide at its raw envelope scale.
                    p["projection_gain"] = round(a, 6)
                    peak_env = float(np.max(np.abs(a_env))) if a_env.size else 0.0
                    p["amplitude"] = round(a * peak_env, 6)
                    new_own = _render_partial(p, n, sr, start_sample=start_sample, stft_half_window_s=stft_half_window_s)
                    rendered_partials[id(p)] = new_own
                    cur_mix = others + new_own
                else:
                    partials.remove(p)
                    del rendered_partials[id(p)]
                    cur_mix = others
                accepted += 1
            continue

        t0 = float(p.get("envelope_t0", 0.0))
        f0 = float(p.get("frequency", 0.0))
        lam0 = float(p.get("envelope_decay", 0.0))

        st_p = p.get("start_time", None)
        et_p = p.get("end_time", None)
        # The envelope and the support mask do not depend on the frequency, so
        # they are computed once per decay and reused across the frequency
        # search. The frequency search evaluates the projection ~20 times per
        # partial, and the support mask is a full-length comparison, so
        # rebuilding it every time dominated the loop.
        _env_cache: Dict[float, np.ndarray] = {}

        def basis(freq: float, decay: float) -> Tuple[np.ndarray, np.ndarray]:
            key = round(float(decay), 12)
            em = _env_cache.get(key)
            if em is None:
                env = np.exp(-decay * np.maximum(t - t0, 0.0))
                if st_p is not None and et_p is not None:
                    env = env * _support_mask(
                        t, float(st_p), float(et_p), decay, 1.0,
                        stft_half_window_s,
                    )
                em = env
                _env_cache[key] = em
            ang = 2.0 * np.pi * freq * t_fit
            return em * np.cos(ang), em * np.sin(ang)

        def project(freq: float, decay: float) -> Tuple[float, float, float]:
            c, s = basis(freq, decay)
            coefs, *_ = np.linalg.lstsq(np.column_stack((c, s)), target, rcond=None)
            a = float(np.hypot(coefs[0], coefs[1]))
            ph = float(np.arctan2(coefs[1], coefs[0]))
            err = float(np.sum((target - (coefs[0] * c + coefs[1] * s)) ** 2))
            return a, ph, err

        def cand_corr(cand: Tuple[float, ...]) -> float:
            c, s = basis(cand[1], cand[2])
            fresh = others + cand[3] * (c * np.cos(cand[4]) + s * np.sin(cand[4]))
            return _corr(y_sec, fresh)

        def cand_seg_snr(cand: Tuple[float, ...]) -> float:
            c, s = basis(cand[1], cand[2])
            fresh = others + cand[3] * (c * np.cos(cand[4]) + s * np.sin(cand[4]))
            return segmental_snr_db(y_sec, fresh, sr)

        a_bound = 10.0 * float(np.std(y_sec)) + 1e-12
        candidates: List[Tuple[float, float, float, float, float]] = []
        # A zero-decay partial must not be locked out of decay fitting. The
        # old grid evaluated `lam0 * k` for five multipliers, which is five
        # IDENTICAL projections when lam0 == 0 (five wasted lstsq calls), and
        # the frequency refinement below was gated on `lam0 > 0.0` - so a
        # zero-decay partial could never have its frequency or its decay
        # touched, ever. 27 of the 36 stationary partials in the mix_lost
        # ledger are zero-decay.
        if lam0 > 0.0:
            decay_grid = [lam0 * k for k in (0.5, 0.8, 1.0, 1.25, 2.0)]
        else:
            decay_grid = [0.0, 0.5, 1.5, 4.0]
        for lam_c in decay_grid:
            a, ph, err = project(f0, lam_c)
            if not (np.isfinite(a) and np.isfinite(ph) and np.isfinite(err)):
                continue
            if a <= a_bound:
                candidates.append((err, f0, lam_c, a, ph))
        finite = [c for c in candidates if np.isfinite(c[0])]
        if finite:
            best_lam = min(finite)[2]
            try:
                # Same main-lobe-safe search the analyser uses: a bounded
                # Brent over a +-2 Hz bracket starts its probes outside the
                # projection main lobe of a sustained partial and can slide
                # into the first sidelobe, losing most of the amplitude.
                st_p = p.get("start_time", None)
                et_p = p.get("end_time", None)
                if st_p is not None and et_p is not None:
                    t_support = max(0.0, float(et_p) - float(st_p))
                else:
                    t_support = float(y_sec.size) / float(sr)
                f_refined = _main_lobe_refine(
                    lambda fv: project(fv, best_lam)[2],
                    f0, sr, t_support, tol_hz=2.0,
                )
                a, ph, err = project(f_refined, best_lam)
                if (
                    f_refined > 10.0 and f_refined < sr / 2.0 - 1.0
                    and np.isfinite(a) and np.isfinite(ph)
                    and np.isfinite(err) and a <= a_bound
                ):
                    candidates.append((err, f_refined, best_lam, a, ph))
            except Exception:
                pass

        drop_err = float(np.sum(target ** 2))
        drop_cand = (drop_err, 0.0, 0.0, 0.0, 0.0)  # a=0 renders nothing
        scored = []
        for c in candidates + [drop_cand]:
            c_c = cand_corr(c)
            c_ss = cand_seg_snr(c)
            if c_ss >= cur_seg_snr - 0.25:
                scored.append((c_c, c_ss, c))
        scored.sort(key=lambda t: ((-t[0], -t[1], t[2][0])))
        if not scored:
            continue
        best_corr_c, best_seg_snr_c, (best_err, best_f, best_lam, best_a, best_ph) = scored[0]
        if best_corr_c > cur_corr + 1e-9:
            if best_f > 0.0:
                p["frequency"] = round(best_f, 6)
                p["envelope_decay"] = round(float(best_lam), 6)
                p["amplitude"] = round(float(best_a), 6)
                p["phase"] = round(float(best_ph), 6)
                new_own = _render_partial(p, n, sr, start_sample=start_sample, stft_half_window_s=stft_half_window_s)
                rendered_partials[id(p)] = new_own
                cur_mix = others + new_own
            else:
                # Removing the partial explains the section better than any refit.
                partials.remove(p)
                del rendered_partials[id(p)]
                cur_mix = others
            accepted += 1
        else:
            continue
    return accepted



def _parse_json_edits(text: str) -> List[Dict[str, Any]]:
    """Extract an edit list from an LLM response, tolerating code fences."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
    start = s.find("[")
    end = s.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def llm_edits_via_command(
    prompt: str,
    command: Sequence[str],
    model: Optional[str] = None,
    timeout_s: float = 300.0,
    backend: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ask a headless LLM for edits, parse JSON out.

    ``command`` entries containing {prompt_file} receive a temp file with
    the prompt. When unset, the named backend's own convention is used -
    ``agy --print``, ``opencode run``, ``gemini -p``, ``qwen <prompt>`` and
    so on, one per entry in ``src.llm_backends.py``.
    """
    edits, _receipt = llm_edits_with_receipt(
        prompt, command, model=model, timeout_s=timeout_s, backend=backend
    )
    return edits


def llm_edits_with_receipt(
    prompt: str,
    command: Sequence[str],
    model: Optional[str] = None,
    timeout_s: float = 300.0,
    backend: Optional[str] = None,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Ask a headless LLM for edits and return the edits with their receipt.

    The receipt is the point. A model that proposed nothing, a model that
    proposed twelve unusable edits, a backend that is not installed, a
    backend that timed out and a backend that answered with prose are all
    different outcomes, and the ledger has to be able to tell them apart.
    """
    receipt: Dict[str, Any] = {
        "backend": backend or ("custom" if command else "opencode"),
        "argv": [],
        "exit_code": None,
        "timed_out": False,
        "duration_s": 0.0,
        "error": "",
        "proposed": 0,
        "applied": 0,
        "refused": 0,
        "refused_reasons": [],
    }
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "res_prompt.txt"
        pf.write_text(prompt, encoding="utf-8")

        if backend == "api":
            res = run_api(
                prompt,
                model or "gpt-4o-mini",
                base_url=api_base_url or "https://api.openai.com/v1",
                api_key=api_key,
                timeout_s=timeout_s,
            )
        elif command:
            argv = [str(c).replace("{prompt_file}", str(pf)) for c in command]
            res = run_backend(argv, timeout_s=timeout_s)
        else:
            spec = find_backend(backend or "") or find_backend("opencode")
            if spec is None:
                receipt["error"] = "no such backend: %r" % (backend,)
                return [], receipt
            try:
                argv = build_command(
                    spec, prompt, model=model, prompt_file=str(pf),
                    extra_args=extra_args,
                )
            except (FileNotFoundError, ValueError) as exc:
                receipt["error"] = str(exc)
                return [], receipt
            res = run_backend(argv, timeout_s=timeout_s)

    receipt["argv"] = list(res.argv)
    receipt["exit_code"] = res.exit_code
    receipt["timed_out"] = bool(res.timed_out)
    receipt["duration_s"] = round(float(res.duration_s), 2)
    receipt["error"] = res.error
    receipt["stdout_len"] = len(res.stdout or "")
    receipt["stdout_head"] = (res.stdout or "").strip()[:400]
    if not res.ok:
        head = (res.stderr or res.stdout or "").strip().splitlines()
        if head:
            receipt["stderr_head"] = head[0][:200]
        return [], receipt
    edits = _parse_json_edits(res.stdout or "")
    receipt["proposed"] = len(edits)
    if not edits:
        receipt["no_edits_reason"] = (
            "the backend answered but the answer held no parseable edit array"
        )
    return edits, receipt


def _equations_text(ledger: Dict[str, Any]) -> str:
    """Human equation sheet: one closed-form equation per subtone.

    The model already IS a sum of independent atoms; this renders it the way
    the "sound of math functions" sheets do - a named sum y(t) over section
    equations y_s(x), each expanded to its damped-cosine terms with the fitted
    constants inline, grouped/annotated by note (inharmonic comb, Valette
    & Cuesta damping) and by transient hit (fast-decay mode bundle).
    """
    best = ledger.get("best", {})
    lines = [
        "# RES equation model - one closed-form equation per subtone",
        "# y(t)   = sum_s y_s(t - t_s)            (sum over section windows; t_s below)",
        "# y_s(x) = sum_k p_k(x)                  (x = t - t_s, section-local time)",
        "# p(x)   = A * [u(x - x0) *] exp(-lam*(x - x0)) * cos(2*pi*f*x - phi)",
        "#          u() is applied only to hypotheses that carry a one-sided",
        "#          attack; fitted partials are windowed by the support mask.",
        f"# best global: corr={float(best.get('correlation', 0.0)):.6f} "
        f"snr={float(best.get('snr_db', 0.0)):.3f}dB "
        f"iterations={len(ledger.get('iterations', []))}",
        "",
    ]
    sections = ledger.get("sections", [])
    # Top-level sum, mirroring the multi-equation sheets: y(t) = y_0 + y_1 + ...
    if len(sections) > 1:
        lines.append("y(t) = " + " + ".join(f"y_{i}(x_{i})" for i in range(len(sections))))
    note_annos: Dict[int, Dict[str, Any]] = {}
    hit_annos: list = []
    for i, sec in enumerate(sections):
        bank = sec.get("hypothesis_bank") or []
        for rec in bank:
            if not rec.get("accepted"):
                continue
            if rec.get("f0") is None:
                hit_annos.append(rec)
            else:
                note_annos[round(float(rec["f0"]), 3)] = rec

    def _partial_line(sec_start: float, j: int, p: Dict[str, Any]) -> str:
        tag = f"p_{j:03d}"
        st = p.get("start_time")
        et = p.get("end_time")
        sup = (
            f"   # support t in [{float(st):.3f}, {float(et):.3f}] s"
            if st is not None and et is not None
            else ""
        )
        if p.get("model") == "glide":
            return (
                f"{tag}(x) = glide: theta(x) = int 2*pi*f(x) dx  "
                f"[mean f = {float(p.get('frequency', 0.0)):.3f} Hz, "
                f"gain = {float(p.get('projection_gain', 1.0)):.3f}, "
                f"amplitude from smoothed trajectory]"
                + sup
            )
        if p.get("model") == "filtered_noise":
            seed = int(p.get("seed", 42))
            a = float(p.get("amplitude", 0.0))
            nb = len(p.get("band_centers", []))
            return (
                f"{tag}(x) = filtered_noise(seed={seed}, gain={a:.4f}, bands={nb})  "
                f"# Class C++ stochastic residual texture"
                + sup
            )
        a = float(p.get("amplitude", 0.0))
        lam = float(p.get("envelope_decay", 0.0))
        t0 = float(p.get("envelope_t0", 0.0))
        f = float(p.get("frequency", 0.0))
        ph = float(p.get("phase", 0.0))
        tau = t0 - sec_start
        ph_txt = f"- {ph:.4f}" if ph >= 0.0 else f"+ {abs(ph):.4f}"
        # The unit step is only real for hypotheses that carry `unit_step`;
        # a fitted partial is windowed by the support mask instead. Printing
        # u(x - x0) for every partial made the sheet describe an attack the
        # renderer never applies.
        step = "u(x - %.3f) * " % tau if p.get("unit_step") else ""
        return (
            f"{tag}(x) = {a:.4f} * {step}"
            f"exp(-{lam:.4f}*(x - {tau:.3f})) "
            f"* cos(2*pi*{f:.4f}*x {ph_txt})"
            + sup
        )

    for i, sec in enumerate(sections):
        m = sec.get("metrics", {})
        parts = sec.get("partials", [])
        t_s = float(sec["start"])
        lines.append(
            f"section {i}  t_s={t_s:.3f}  window=[{float(sec['start']):.2f}, "
            f"{float(sec['end']):.2f}] s  corr={float(m.get('correlation', 0.0)):.3f} "
            f"snr={float(m.get('snr_db', 0.0)):.1f}dB  "
            f"({len(parts)} subtone equations)"
        )
        terms = [f"p_{j:03d}" for j in range(len(parts))]
        if not terms:
            lines.append("y = 0   # laser silence: nothing left to explain")
            lines.append("")
            continue
        head = f"y_{i}(x) = "
        chunks = []
        for a in range(0, len(terms), 4):
            row = " + ".join(terms[a:a + 4])
            chunks.append(row if a == 0 else "    " + row)
        lines.append(head + chunks[0])
        lines.extend(c for c in chunks[1:])
        # Grouped closed-form annotations + per-subtone equations.
        prev_key: Any = "start"
        n_parts = len(parts)
        for j, p in enumerate(parts):
            g = p.get("note_group")
            is_hit = p.get("model") == "damped" and p.get("m") is None
            key = None if g is None else ("hit:%s" % g if is_hit else "note:%s" % g)
            if key != prev_key:
                prev_key = key
                if key is not None:
                    run = 1
                    while (
                        j + run < n_parts
                        and parts[j + run].get("note_group") == g
                        and (parts[j + run].get("m") is None) == is_hit
                    ):
                        run += 1
                    if is_hit:
                        gg = float(key.split(":", 1)[1])
                        anno = None
                        for h in hit_annos:
                            t0h = float(h.get("t0", -9.0))
                            if abs(t0h - gg) < 0.05 or abs(t0h - (gg + t_s)) < 0.05:
                                anno = h
                                break
                        mc = int(anno["m_count"]) if anno and anno.get("m_count") else run
                        lines.append(
                            f"# hit at t ~ {(gg + t_s):.3f} s: fast-decay mode bundle"
                            f" (m_count={mc}, decay grid 30/80/200)"
                        )
                    else:
                        f0g = float(key.split(":", 1)[1])
                        rec = note_annos.get(round(f0g, 3))
                        if rec is not None:
                            B = float(rec.get("B", 0.0))
                            s = f"# note: f_m = m * {f0g:.3f} * sqrt(1 + {B:.1e} * m^2)"
                            if rec.get("vc_lock"):
                                s += (
                                    f"   sigma_m = {float(rec['sigma_a']):.3f} "
                                    f"+ {float(rec['sigma_b']):.6f}*f_m  (Valette-Cuesta)"
                                )
                            else:
                                s += "   (per-harmonic fitted decays)"
                            lines.append(s)
                        else:
                            lines.append(
                                f"# note: f_m = m * {f0g:.3f} * sqrt(1 + B*m^2)"
                                "   (comb fitted to the audible harmonics)"
                            )
            lines.append("  " + _partial_line(t_s, j, p))
        lines.append("")
    return "\n".join(lines)


_DECAY_RATIO_FLOOR = 0.95


def _residual_noise_shape(
    residual: np.ndarray,
    n_fft: int = 1024,
    smooth: int = 9,
) -> np.ndarray:
    """Smoothed magnitude envelope of a residual, on a log-frequency axis."""
    n = len(residual)
    window = np.hanning(n_fft)
    hop = max(1, n_fft // 4)
    nbins = n_fft // 2 + 1
    acc = np.zeros(nbins)
    count = 0
    for start in range(0, max(1, n - n_fft), hop):
        seg = residual[start:start + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        acc += np.abs(np.fft.rfft(seg * window))
        count += 1
    acc /= max(1, count)
    kernel = np.ones(smooth) / smooth
    env = np.convolve(acc, kernel, mode="same")
    peak = float(env.max())
    return env / peak if peak > 0.0 else env


def spectral_fill(
    recon: np.ndarray,
    original: np.ndarray,
    level: float = 0.0,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Add deterministic, residual-shaped noise to a rendered reconstruction.

    This is a post-process. It is never fed back into the optimiser and never
    touches ledger correlation, because the noise is uncorrelated with the
    target by construction and only trades waveform correlation for spectral
    coverage. Measurements on this codebase showed the trade is strongly
    content-dependent: it buys large log-spectral-distance improvements on a
    broadband-poor reconstruction, but makes an already spectrally accurate one
    substantially worse. Hence the default of ``level=0.0``, which is an exact
    pass-through.

    The noise spectrum is seeded from the residual and reshaped to that
    residual's smoothed log-frequency envelope, so the same inputs and seed
    always produce byte-identical output.
    """
    level = float(level)
    if not (0.0 <= level <= 1.0):
        raise ValueError(f"spectral fill level must be in [0, 1], got {level}")
    if level == 0.0:
        return recon, {"enabled": False, "level": 0.0, "seed": int(seed)}

    n = min(len(recon), len(original))
    recon = recon[:n]
    original = original[:n]
    residual = original - recon
    env = _residual_noise_shape(residual)
    rng = np.random.default_rng(int(seed))
    spec = np.fft.rfft(rng.standard_normal(n))
    src = np.log1p(np.arange(len(env)))
    dst = np.log1p(np.arange(len(spec)))
    shaped = np.interp(dst, src, env)
    noise = np.fft.irfft(spec * shaped, n=n)
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms <= 0.0:
        return recon, {"enabled": False, "level": level, "seed": int(seed)}
    target_rms = float(np.sqrt(np.mean(original ** 2)))
    filled = recon + noise * (level * target_rms / noise_rms)
    return filled, {
        "enabled": True,
        "level": level,
        "seed": int(seed),
        "added_rms": float(level * target_rms),
        "residual_rms": float(np.sqrt(np.mean(residual ** 2))),
    }


def iteration_gains(ledger: Dict[str, Any]) -> List[float]:
    """Per-iteration correlation deltas against the previous best, in order.

    An accepted pass contributes a positive delta. A rejected pass rewinds the
    ledger to the previous best, so it contributes the negative distance the
    pass fell short - never 0.0 - and callers can read the plateau off the shape
    of the series instead of inferring it from a short list.
    """
    out: List[float] = []
    prev = ledger.get("initial_correlation")
    if prev is None:
        its = ledger.get("iterations") or []
        prev = its[0]["global"]["correlation"] if its else 0.0
    for it in ledger.get("iterations") or []:
        corr = float(it["global"]["correlation"])
        out.append(corr - float(prev))
        prev = corr
    return out


def structural_budget(
    ledger: Dict[str, Any],
    hard_min: int = 2,
    hard_max: int = 40,
) -> int:
    """Iteration guess from problem size alone, for when no trend exists yet.

    The loop re-renders the whole song every pass, so cost scales with the
    partial count while the number of independent sections sets how much has
    to be coordinated. Both push the useful budget up together.
    """
    sections = len(ledger.get("sections") or [])
    total = sum(len(s.get("partials") or []) for s in ledger.get("sections") or [])
    if sections <= 0 or total <= 0:
        return hard_min
    raw = 1.5 * sections + total / 90.0
    return int(max(hard_min, min(hard_max, round(raw))))


def recommend_iterations(
    ledger: Dict[str, Any],
    min_gain: float = 1e-3,
    max_iterations: int = 40,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """Recommend an iteration budget from the observed gain trend.

    Gains in these loops shrink roughly geometrically, so the decay ratio is
    estimated as the geometric mean of consecutive gain ratios and used to
    extrapolate both the achievable ceiling and the pass on which the
    marginal gain falls under ``min_gain``. With too few accepted samples the
    answer falls back to :func:`structural_budget`.
    """
    gains = iteration_gains(ledger)
    iters = ledger.get("iterations") or []
    best = float((ledger.get("best") or {}).get("correlation", 0.0))
    positive = [g for g in gains if g > 0.0]
    done = len(iters)
    base: Dict[str, Any] = {
        "iterations_done": done,
        "correlation": best,
        "gains": [round(g, 8) for g in gains],
        "min_gain": min_gain,
        "basis": "insufficient-data",
        "last_gain": gains[-1] if gains else 0.0,
        "decay_ratio": None,
        "predicted_ceiling": None,
        "remaining_gain": None,
        "recommended": structural_budget(ledger),
        "converged": False,
        "reason": "",
    }

    if len(positive) < min_samples:
        base["reason"] = (
            f"only {len(positive)} accepted iteration(s); falling back to a "
            "size-based guess. Run --iterations 6 or more for a real estimate."
        )
        base["recommended"] = max(base["recommended"], done + 3)
        if not gains:
            base["converged"] = True
            base["reason"] = "no accepted iterations recorded yet"
        return base

    ratios: List[float] = []
    for a, b in zip(positive, positive[1:]):
        ratios.append(b / a)
    ratio = float(np.exp(np.mean(np.log(ratios)))) if ratios else 0.0
    last = float(positive[-1])

    if not ratios:
        trend = "unknown"
    elif ratio >= 1.0:
        trend = "rising"
    elif ratio >= _DECAY_RATIO_FLOOR:
        trend = "flat"
    else:
        trend = "decaying"

    if trend == "rising":
        return {
            **base,
            "basis": "gain-trend",
            "trend": trend,
            "last_gain": last,
            "decay_ratio": None,
            "predicted_ceiling": None,
            "remaining_gain": None,
            "recommended": int(max(done + 1, max_iterations)),
            "converged": False,
            "reason": (
                f"gains are still increasing after {len(positive)} accepted "
                f"pass(es) (last {last:.6f}), so this run has not started "
                "converging - raise --iterations rather than stopping"
            ),
        }

    if trend in ("flat", "unknown"):
        if trend == "flat" and ratio:
            why = (
                f"gain ratio is {ratio:.3f}, too close to 1.0 to measure a decay "
                f"rate; extrapolating from it would fabricate a ceiling"
            )
        else:
            why = "no consecutive accepted gains to ratio"
        return {
            **base,
            "basis": "gain-trend",
            "trend": trend,
            "last_gain": last,
            "decay_ratio": ratio or None,
            "predicted_ceiling": None,
            "remaining_gain": None,
            "recommended": int(max(done + 1, max_iterations)),
            "converged": False,
            "reason": (
                f"{why} across {len(positive)} accepted pass(es) "
                f"(last {last:.6f}); budget by problem size and treat the "
                "ceiling as unknown"
            ),
        }

    ratio = float(max(0.0, min(0.97, ratio)))
    ceiling = best + (last * ratio / (1.0 - ratio) if ratio < 1.0 else float("inf"))

    if ratio <= 0.0 or last <= min_gain:
        extra = 0
    else:
        extra = int(math.ceil(math.log(min_gain / last) / math.log(ratio)))

    extra = int(max(0, min(extra, max_iterations - done)))
    recommended = int(max(done, done + extra))
    converged = extra == 0

    if converged:
        reason = (
            f"marginal gain already {last:.6f}, at or under min_gain {min_gain:g}"
            if last <= min_gain
            else "gain decay ratio collapsed; further passes will not help"
        )
    else:
        reason = (
            f"gain decays ~{ratio:.3f}x per pass from {last:.6f}; "
            f"{extra} more pass(es) to reach min_gain {min_gain:g}"
        )

    return {
        **base,
        "basis": "gain-trend",
        "trend": trend,
        "last_gain": last,
        "decay_ratio": ratio,
        "predicted_ceiling": None if ceiling is None else float(min(ceiling, 1.0)),
        "remaining_gain": None if ceiling is None else float(max(0.0, ceiling - best)),
        "recommended": recommended,
        "converged": converged,
        "reason": reason,
    }


def refine_section_parameters(
    y_sec: np.ndarray,
    partials: List[Dict[str, Any]],
    sr: int,
    start_sample: int = 0,
    stft_half_window_s: float = 0.0,
    max_nfev: int = 35,
) -> int:
    """Clustered non-linear least squares parameter refinement (Phase 4).

    Identifies collision clusters: groups of 2-4 partials within 10 Hz of each
    other that suffer from beating/phase cancellation. Optimizes only those colliding
    clusters jointly via scipy.optimize.least_squares (TRF), leaving isolated
    partials untouched to prevent phase destruction and Jacobian collapse.
    """
    n = len(y_sec)
    if n < 8 or not partials:
        return 0

    # 1. Filter active damped cosine partials
    active_indices = [
        i for i, p in enumerate(partials)
        if p.get("model") not in ("filtered_noise", "lpc_noise", "glide")
        and float(p.get("amplitude", 0.0)) > 1e-4
    ]
    if len(active_indices) < 2:
        return 0

    # 2. Build collision clusters (partials within 10 Hz of each other)
    sorted_indices = sorted(active_indices, key=lambda idx: float(partials[idx].get("frequency", 0.0)))
    clusters: List[List[int]] = []
    current_cluster: List[int] = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        f_prev = float(partials[current_cluster[-1]].get("frequency", 0.0))
        f_curr = float(partials[idx].get("frequency", 0.0))
        if abs(f_curr - f_prev) <= 10.0:
            current_cluster.append(idx)
        else:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [idx]
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)

    if not clusters:
        return 0

    t = (np.arange(n, dtype=np.float64) + start_sample) / sr
    t_fit = np.arange(n, dtype=np.float64) / sr
    two_pi = 2.0 * np.pi
    total_refined = 0

    for cluster in clusters:
        # Limit cluster size to top 4 by amplitude to keep Jacobian well-conditioned
        if len(cluster) > 4:
            cluster.sort(key=lambda idx: float(partials[idx].get("amplitude", 0.0)), reverse=True)
            cluster = cluster[:4]

        # Background: all partials in section EXCEPT this cluster
        cluster_set = set(cluster)
        bg_sig = np.zeros(n, dtype=np.float64)
        for idx in range(len(partials)):
            if idx not in cluster_set:
                bg_sig += _render_partial(
                    partials[idx], n, sr,
                    start_sample=start_sample,
                    stft_half_window_s=stft_half_window_s,
                )

        cluster_partials = [partials[i] for i in cluster]
        k_opt = len(cluster_partials)

        x0 = np.zeros(4 * k_opt, dtype=np.float64)
        lb = np.zeros(4 * k_opt, dtype=np.float64)
        ub = np.zeros(4 * k_opt, dtype=np.float64)
        masks = []
        t0_list = []

        for k, p in enumerate(cluster_partials):
            a = float(p.get("amplitude", 0.0))
            f = float(p.get("frequency", 0.0))
            ph = float(p.get("phase", 0.0))
            lam = float(p.get("envelope_decay", 0.0))
            t0 = float(p.get("envelope_t0", 0.0))
            t0_list.append(t0)

            st = p.get("start_time", None)
            et = p.get("end_time", None)
            if st is not None and et is not None and a > 0.0:
                mask = _support_mask(t, float(st), float(et), lam, a, stft_half_window_s)
            else:
                mask = None
            masks.append(mask)

            idx_p = 4 * k
            x0[idx_p + 0] = a
            x0[idx_p + 1] = f
            x0[idx_p + 2] = ph
            x0[idx_p + 3] = lam

            lb[idx_p + 0] = 0.0
            ub[idx_p + 0] = max(1.5, 3.0 * a + 0.1)

            lb[idx_p + 1] = max(10.0, f - 2.0)
            ub[idx_p + 1] = min(float(sr) / 2.0, f + 2.0)

            lb[idx_p + 2] = ph - 2.0 * np.pi
            ub[idx_p + 2] = ph + 2.0 * np.pi

            lb[idx_p + 3] = 0.0
            ub[idx_p + 3] = max(150.0, 3.0 * lam + 10.0)

        recon_init = bg_sig.copy()
        for p in cluster_partials:
            recon_init += _render_partial(
                p, n, sr,
                start_sample=start_sample,
                stft_half_window_s=stft_half_window_s,
            )
        base_corr = _corr(y_sec, recon_init)

        def _sim_cluster(params: np.ndarray) -> np.ndarray:
            sig = bg_sig.copy()
            for k in range(k_opt):
                idx_p = 4 * k
                a_val = params[idx_p + 0]
                f_val = params[idx_p + 1]
                ph_val = params[idx_p + 2]
                lam_val = params[idx_p + 3]
                t0_val = t0_list[k]
                m_val = masks[k]

                env = a_val * np.exp(-lam_val * np.maximum(t - t0_val, 0.0))
                p_sig = env * np.cos(two_pi * f_val * t_fit - ph_val)
                if m_val is not None:
                    p_sig = p_sig * m_val
                sig += p_sig
            return sig

        def residual_fn(params: np.ndarray) -> np.ndarray:
            res = _sim_cluster(params) - y_sec
            # Phase 5: Dampen clustered refinement to stop 808 flanging/phasing.
            # Add quadratic regularization penalty to frequency shift from greedy initialization
            f_reg = np.zeros(k_opt, dtype=np.float64)
            for k in range(k_opt):
                idx_p = 4 * k
                f_reg[k] = 2.0 * (params[idx_p + 1] - x0[idx_p + 1])
            return np.concatenate([res, f_reg])

        try:
            opt_res = least_squares(
                residual_fn,
                x0,
                bounds=(lb, ub),
                max_nfev=max_nfev,
                method="trf",
                ftol=1e-4,
                xtol=1e-4,
            )
            opt_params = opt_res.x
        except Exception:
            continue

        recon_opt = _sim_cluster(opt_params)
        opt_corr = _corr(y_sec, recon_opt)

        if opt_corr > base_corr + 1e-4:
            for k, p in enumerate(cluster_partials):
                idx_p = 4 * k
                p["amplitude"] = round(float(opt_params[idx_p + 0]), 6)
                p["frequency"] = round(float(opt_params[idx_p + 1]), 6)
                wrapped_ph = float((opt_params[idx_p + 2] + np.pi) % (2.0 * np.pi) - np.pi)
                p["phase"] = round(wrapped_ph, 6)
                p["envelope_decay"] = round(float(opt_params[idx_p + 3]), 6)
            total_refined += k_opt

    return total_refined


# Worker pools are cached per width and reused across loop iterations and
# across run_loop calls. Recreating a ProcessPoolExecutor per iteration makes
# Windows re-spawn every worker and re-import numpy/librosa each time.
_POOL_CACHE: Dict[int, "concurrent.futures.ProcessPoolExecutor"] = {}


@atexit.register
def _shutdown_pools_at_exit() -> None:
    """Cached pools hold live worker processes; release them before exit.

    Without this the interpreter can tear down while workers are still running,
    which shows up as a hard crash after the last test rather than a clean pass.
    """
    shutdown_worker_pools(wait=False)


def _get_pool(n_workers: int) -> "concurrent.futures.ProcessPoolExecutor":
    pool = _POOL_CACHE.get(n_workers)
    if pool is None:
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
        _POOL_CACHE[n_workers] = pool
    return pool


def shutdown_worker_pools(wait: bool = True) -> None:
    """Release every cached worker pool (call on app shutdown)."""
    for width in list(_POOL_CACHE):
        try:
            _POOL_CACHE.pop(width).shutdown(wait=wait)
        except Exception:
            pass


def _referee_ok(
    cand: Dict[str, Any],
    best: Dict[str, Any],
    corr_gain: float,
) -> bool:
    """Multi-channel referee gate for one loop iteration.

    The previous gate was asymmetric: it vetoed on segmental SNR but ignored
    every other channel. So an iteration could be rejected for a 0.31 dB
    segmental-SNR drop while another was accepted with 0.7 dB of SNR loss,
    0.54 dB of extra log-spectral distance and a 0.12 drop in onset
    correlation. The research doc asks for exactly this ("an edit is
    accepted iff section residual energy strictly decreases AND no referee
    metric degrades"), so the referee now has every channel it reports.

    Correlation is the objective, but it is not the only referee: a render
    that buys correlation by bloating its energy (the segmental-SNR symptom
    the gate was written for) or by wrecking the spectrum is not an
    improvement. Each channel gets a budget.

    The segmental-SNR budget scales with the size of the correlation gain,
    because a fixed 0.25 dB budget vetoed a large, real improvement (on the
    drums stem: correlation +0.08, onset correlation +0.20, LSD -0.17 dB,
    SNR +0.11 dB, segmental SNR -0.31 dB) over a rounding error in one
    channel. The remaining channels get a flat 1 dB budget: the class-A
    lifter legitimately trades a little spectral fidelity for waveform
    accuracy, and it was measured moving these channels by up to 0.9 dB
    while gaining 0.06-0.08 correlation, so a tighter budget would reject
    good work across the board.
    """
    if not best:
        return True
    seg_budget = 0.25 + 2.0 * max(0.0, float(corr_gain))
    db_budget = 1.0
    onset_budget = 0.25

    snr_now = float(cand.get("snr_db", 0.0))
    snr_before = float(best.get("snr_db", 0.0))
    if snr_now < snr_before - db_budget:
        return False
    if float(cand.get("log_spectral_distance_db", 0.0)) > float(
        best.get("log_spectral_distance_db", 0.0)
    ) + db_budget:
        return False
    if float(cand.get("onset_corr", 0.0)) < float(best.get("onset_corr", 0.0)) - onset_budget:
        return False
    # A frame-wise segmental-SNR drop is a *proxy* for an energy-bloated
    # render. The global SNR channel measures energy directly, so when SNR
    # improves the render is demonstrably not bloated and the proxy must not
    # veto it. Measured on mix_lost: iteration 2 improved correlation +0.034,
    # SNR +0.45 dB, LSD -0.17 dB and onset correlation +0.011, and was
    # rejected over a 0.70 dB segmental-SNR drop against a 0.32 dB budget.
    # The frame-wise channel still vetoes on its own whenever the
    # energy-based channel does not improve.
    if snr_now <= snr_before and float(cand.get("seg_snr_db", 0.0)) < float(
        best.get("seg_snr_db", 0.0)
    ) - seg_budget:
        return False
    return True


# The MR-STFT acceptance lane. Literature: the multi-resolution STFT
# distance is the selection criterion the neural-synthesis literature
# converged on (Yamamoto et al. 2019, arXiv:1910.11480; Engel et al. DDSP
# 2020 use it for model selection; the auraloss family collects it), because
# it sees tonal-vs-textural errors a single window - and a waveform metric -
# understate. The engine already computes it every iteration
# (residual._mrstft_db via _score) but only ever reported it.
#
# The measured wall it addresses (docs/PHYSICS_ROADMAP.md): corr-chaining
# rejects beneficial new structure - a candidate that fixes the spectrum
# (noise layers, inserted clean partials) can dip global waveform
# correlation transiently. Flag-gated, default OFF: the published numbers
# do not move until the lane is re-measured (same rule as M2M_EXCITER).
_MRSTFT_LANE_CORR_TOL = 0.002  # correlation may regress at most this far
_MRSTFT_LANE_GAIN_DB = 0.05    # MR-STFT LSD must improve by at least this
_MRSTFT_LANE_SNR_DB = 1.0      # SNR budget, same as the corr lane's


def _mrstft_lane_ok(
    cand: Dict[str, Any],
    best: Dict[str, Any],
    best_corr: float,
) -> bool:
    """Whether an iteration the corr lane rejected passes on the MR-STFT lane.

    Every corr-lane channel stays within its budget (SNR, LSD, onset), the
    segmental-SNR proxy keeps a flat 0.5 dB (there is no corr gain to scale
    it with here), correlation regresses at most ``_MRSTFT_LANE_CORR_TOL``,
    and the multi-resolution LSD improves by at least
    ``_MRSTFT_LANE_GAIN_DB`` - a real spectral gain, not jitter.
    """
    if not best:
        return False
    if float(cand.get("correlation", 0.0)) < best_corr - _MRSTFT_LANE_CORR_TOL:
        return False
    if float(cand.get("snr_db", 0.0)) < float(
        best.get("snr_db", 0.0)
    ) - _MRSTFT_LANE_SNR_DB:
        return False
    if float(cand.get("log_spectral_distance_db", 0.0)) > float(
        best.get("log_spectral_distance_db", 0.0)
    ) + 1.0:
        return False
    if float(cand.get("onset_corr", 0.0)) < float(
        best.get("onset_corr", 0.0)
    ) - 0.25:
        return False
    if float(cand.get("seg_snr_db", 0.0)) < float(
        best.get("seg_snr_db", 0.0)
    ) - 0.5:
        return False
    return float(cand.get("mrstft_db", 0.0)) <= float(
        best.get("mrstft_db", float("inf"))
    ) - _MRSTFT_LANE_GAIN_DB


def _process_section_task(args: Tuple[Any, ...]) -> Tuple[Dict[str, Any], List[str], Optional[np.ndarray]]:
    """Self-contained, picklable section processing worker."""
    (sec, y_sec, sr, w_start, st_hw, bank, spectral_lane, *rest) = args
    # ``enable_noise``, ``k_notes``, ``k_hits`` and ``joint_refine`` were
    # added after the worker tuple was already public; accept the older
    # tuples rather than breaking every existing caller.
    enable_noise = bool(rest[0]) if len(rest) > 0 else False
    k_notes = int(rest[1]) if len(rest) > 1 else 2
    k_hits = int(rest[2]) if len(rest) > 2 else 12
    joint_refine = bool(rest[3]) if len(rest) > 3 else False
    if len(y_sec) < 8:
        return sec, [], None

    if sec.get("converged", False):
        return sec, [], None

    prev_c = float(sec.get("metrics", {}).get("correlation", 0.0))
    _section_update(
        y_sec, sec["partials"], sr,
        start_sample=w_start, stft_half_window_s=st_hw,
    )
    if joint_refine:
        # The lifter refines each partial alone; partials within ~10 Hz of
        # each other (detuned 808 layers, unisons) beat, and no per-partial
        # refit can decouple them. The clustered TRF refiner fits each
        # collision cluster jointly - wired in from Phase 4 where it sat
        # orphaned behind a flag (opt-in: published defaults untouched).
        refine_section_parameters(
            y_sec, sec["partials"], sr,
            start_sample=w_start, stft_half_window_s=st_hw,
        )
    bank_records: List[Dict[str, Any]] = []
    new_classes: List[str] = []
    if bank:
        bank_records = tournament_section(
            y_sec, sec["partials"], sr,
            start_sample=w_start, stft_half_window_s=st_hw,
            k=k_notes, k_hits=k_hits,
            enable_noise=enable_noise,
            enable_spectral_lane=spectral_lane,
            # A degradation budget, not a fixed bar: the shipped absolute
            # 0.60 floor kept the lane inert on exactly the stems that need
            # stochastic models (drums sits at 0.42 - measured, 0 accepted
            # stochastic candidates in 277 ledgers). Relative mode lets the
            # lane run anywhere while still capping the correlation it may
            # give back. Only affects runs that opt into the lane.
            corr_floor_mode="relative" if spectral_lane else "absolute",
        )
        for r in bank_records:
            if r.get("accepted") and r.get("class"):
                new_classes.append(r["class"])
    sec["hypothesis_bank"] = bank_records
    # The bank adds partials and the lifter drops them, so the count recorded
    # at evidence time goes stale. The ledger must describe the partials it
    # actually ships.
    sec["partial_count"] = len(sec["partials"])

    # Direct render without TRF or lag shifts to eliminate phase distortion and preserve replica invariance
    recon_sec = np.zeros(len(y_sec))
    for p in sec["partials"]:
        recon_sec += _render_partial(
            p, len(y_sec), sr, start_sample=w_start,
            stft_half_window_s=st_hw,
        )
    sec["lag_samples"] = 0
    sec["lag_s"] = 0.0

    # Noise floor. A sinusoid bank leaves the section on a dead floor between
    # syllables, which is what makes a reconstructed voice sound robotic. Fit one
    # seeded band-limited noise partial to the residual, and keep it only if it
    # improves the spectral distance without costing correlation.
    noise = fit_noise_floor_partial(
        y_sec - recon_sec, sr,
        start_time=float(sec.get("start", 0.0)),
        end_time=float(sec.get("end", 0.0)),
        seed=1234 + int(w_start),
    )
    if noise is not None:
        best = _apply_noise_floor(
            y_sec, recon_sec, noise, sr, w_start, st_hw, spectral_lane
        )
        if best is not None:
            sec["partials"].append(best[0])
            recon_sec = best[2]
    # The chase: opt-in noise-layer matching pursuit on the remaining
    # residual (spectral lane only - published defaults untouched).
    chase_added, recon_sec = _chase_noise_layers(
        y_sec, recon_sec, sr, w_start, st_hw, spectral_lane,
        rounds=3,
        start_time=float(sec.get("start", 0.0)),
        end_time=float(sec.get("end", 0.0)),
    )
    if chase_added:
        sec["partials"].extend(chase_added)

    sec["metrics"] = _score(y_sec, recon_sec, sr)
    new_c = float(sec["metrics"]["correlation"])
    gain = new_c - prev_c
    gains = sec.setdefault("gain_history", [])
    gains.append(round(gain, 6))
    if len(gains) >= 2 and gains[-1] < 0.002 and gains[-2] < 0.002:
        sec["converged"] = True
        sec["frozen_reason"] = "gain < 0.002 for 2 consecutive passes"

    return sec, new_classes, recon_sec


def _apply_noise_floor(
    y_sec: np.ndarray,
    recon_sec: np.ndarray,
    noise: Dict[str, Any],
    sr: int,
    w_start: int,
    st_hw: float,
    lane: bool,
    ladder=(0.15, 0.3, 0.5, 0.8),
):
    """Try the noise floor at several levels; keep the best spectrally
    improving one. Returns ``(candidate, score, render)`` or ``None``.

    Corr-drop allowance: 0.005 shipped (a floor must be nearly free in
    waveform terms). With the spectral lane on, 0.02 - stochastic content
    is judged spectrally, the same philosophy as the bank's spectral lane.
    Opt-in: without the lane the shipped gate is byte-identical.

    The onset gate is unconditional: onset corr is a GLOBAL referee metric,
    and a floor whose band envelopes inject spectral-flux events can wreck
    it while looking free locally - measured, such a floor vetoed an
    iteration that had gained +0.52 correlation. A candidate may not give
    back more than 0.02 of the section's onset structure.
    """
    base = _score(y_sec, recon_sec, sr)
    allow = 0.02 if lane else 0.005
    best = None
    for mult in ladder:
        cand = dict(noise)
        cand["amplitude"] = round(float(noise["amplitude"]) * mult, 6)
        trial = recon_sec + _render_partial(
            cand, len(y_sec), sr, start_sample=w_start,
            stft_half_window_s=st_hw,
        )
        m = _score(y_sec, trial, sr)
        if (m["log_spectral_distance_db"] < base["log_spectral_distance_db"]
                and m["correlation"] >= base["correlation"] - allow
                and m["onset_corr"] >= base["onset_corr"] - 0.02):
            if best is None or (m["log_spectral_distance_db"]
                                < best[1]["log_spectral_distance_db"]):
                best = (cand, m, trial)
    return best


def _chase_noise_layers(
    y_sec: np.ndarray,
    recon_sec: np.ndarray,
    sr: int,
    w_start: int,
    st_hw: float,
    lane: bool,
    rounds: int = 2,
    start_time: float = 0.0,
    end_time: float = 1.0,
):
    """Noise-layer matching pursuit: stack band-envelope noise layers, each
    fitted from the REMAINING residual.

    Why this can go where the tonal atoms cannot: LSD/MR-STFT are
    phase-blind, so the magnitude spectrum is fully explainable by
    partials + noise layers even where waveform correlation is capped by
    noise orthogonality. Each round fits a fresh floor to what is left,
    and the referee's own gates (corr budget, onset structure, LSD must
    improve) decide whether the layer ships. Opt-in via the spectral lane;
    without it this returns unchanged.

    Returns ``(layers_added, recon)``.
    """
    if not lane:
        return [], recon_sec
    added: List[Dict[str, Any]] = []
    for r in range(max(0, rounds)):
        resid = y_sec - recon_sec
        if float(np.sqrt(np.mean(resid ** 2))) < 1e-4:
            break
        layer = fit_noise_floor_partial(
            resid, sr, start_time=start_time, end_time=end_time,
            seed=1234 + int(w_start) + 1000 * (r + 1),
        )
        if layer is None:
            break
        best = _apply_noise_floor(
            y_sec, recon_sec, layer, sr, w_start, st_hw, lane,
            # Chase layers keep their full measured level - the ladder's
            # 0.8 cap is for the always-on floor; a chase layer already
            # carries the residual's own scale.
            ladder=(0.3, 0.6, 1.0),
        )
        if best is None:
            break
        added.append(best[0])
        recon_sec = best[2]
    return added, recon_sec


def _repair_seams(
    signal: np.ndarray,
    boundaries: Sequence[int],
    half_ms: float = 3.0,
    sr: int = 22050,
    ratio_trigger: float = 3.0,
) -> int:
    """De-click section boundaries by interpolating across the seam.

    The OLA fade blends toward a scalar, which shrank but did not remove
    the step (measured 8.3x the typical sample delta after OLA). A
    smoothstep (C1-continuous) interpolation across the seam - only where
    the jump actually exceeds the trigger - removes the discontinuity
    without creating new edge-kinks the way a linear ramp does (measured:
    the linear ramp traded the seam click for two smaller edge clicks).
    """
    w = max(8, int(half_ms / 1000.0 * sr))
    repaired = 0
    for b in boundaries:
        if b - w < 1 or b + w >= len(signal) - 1:
            continue
        d = np.abs(np.diff(signal))
        local = float(d[max(0, b - w):b + w].max())
        typ = float(np.median(d))
        if typ <= 0 or local < ratio_trigger * typ:
            continue
        left = signal[b - w]
        right = signal[b + w]
        u = np.linspace(0.0, 1.0, 2 * w + 1)
        ramp = left + (right - left) * (u * u * (3.0 - 2.0 * u))
        signal[b - w:b + w + 1] = ramp
        repaired += 1
    return repaired


def _parity_spans(
    sections: List[Dict[str, Any]], n: int, sr: int
) -> List[Tuple[int, int]]:
    """Contiguous (start, end) sample spans for the per-section parity fit."""
    spans: List[Tuple[int, int]] = []
    prev_end = 0
    for sec in sorted(sections, key=lambda s: float(s.get("start", 0.0))):
        start = int(sec.get("start_sample", round(float(sec.get("start", 0.0)) * sr)))
        end = int(sec.get("end_sample", round(float(sec.get("end", 0.0)) * sr)))
        start = max(start, prev_end)
        end = min(end, n)
        if start < end:
            spans.append((start, end))
            prev_end = end
    return spans


def run_loop(
    ledger: Dict[str, Any],
    source_path: Union[str, Path, np.ndarray],
    out_dir: Union[str, Path],
    iterations: int = 6,
    target_corr: float = 0.98,
    provider: str = "numeric",
    llm_model: Optional[str] = None,
    llm_cmd: Optional[Sequence[str]] = None,
    llm_backend: Optional[str] = None,
    llm_timeout_s: float = 300.0,
    llm_api_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_extra_args: Optional[Sequence[str]] = None,
    max_llm_edits: int = 12,
    stop_on_no_improvement: bool = False,
    min_gain: float = 0.0,
    bank: bool = True,
    progress_fn: Optional[Callable[..., Any]] = None,
    workers: Optional[int] = None,
    spectral_lane: bool = False,
    enable_noise: bool = False,
    k_notes: int = 2,
    k_hits: int = 12,
    energy_parity: bool = False,
    parity_mode: str = "ls",
    joint_refine: bool = False,
    mrstft_lane: bool = False,
) -> Dict[str, Any]:
    """Iterate the RES loop and persist checkpoints + outputs to ``out_dir``.

    ``progress_fn(k, total, global_score, accepted)`` fires after every
    iteration (before any early-stop) so UIs can stream progress without
    touching the loop's internals.

    ``min_gain`` halts the loop once an accepted pass improves correlation by
    less than that amount, which is how a plateau is detected without waiting
    for the caller's cap. ``stop_on_no_improvement`` halts on the first
    rejected pass. Either way the best state is preserved and recorded in
    ``ledger["stop_reason"]``.

    ``enable_noise`` lets the hypothesis bank model the residual with a
    stochastic class (LPC formant noise). A sum of sinusoids cannot express
    breath, air or sibilance, so without it that content is simply absent
    from the reconstruction.
    """
    if isinstance(source_path, np.ndarray):
        y = np.asarray(source_path, dtype=np.float32)
    else:
        from src.analysis import load_audio
        y, _ = load_audio(str(source_path), target_sr=int(ledger["sample_rate"]))
    # Score only the analysed span: sections describe it, so comparing against
    # the untouched remainder of the file would fabricate a false collapse.
    n = int(ledger.get("n_samples", 0)) or len(y)
    y = y[:n]
    sr = int(ledger["sample_rate"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger_path = out / "res_ledger.json"

    init_recon = render_ledger(ledger, n, ola=False)
    init_score = _score(y, init_recon, sr)
    init_corr = float(init_score["correlation"])
    # The acceptance gate must live in the LOOP's own metric space.
    # ledger["best"]["correlation"] is the EVIDENCE score of the SMS engine's
    # own sinusoid_audio - a different signal from what this loop renders
    # (render_ledger of the same partials). Gating the loop's iterations on
    # that number rejected good work and shipped the broken initial state:
    # on the beat-quantised fixtures the evidence signal scored 0.67 while
    # the loop's own render scored 0.01, so every iteration was "worse than
    # best" and got rolled back. The loop's baseline is its own initial
    # render; the evidence score is kept for the record.
    evidence_corr = float(ledger.get("best", {}).get("correlation", 0.0))
    best_corr = init_corr
    best_metrics = dict(init_score)
    best_sections = copy.deepcopy(ledger["sections"])
    ledger.setdefault("initial_correlation", best_corr)
    ledger.setdefault("evidence_correlation", evidence_corr)
    ledger.setdefault("iterations", [])
    stop_reason = "iteration-cap"
    prev_best = best_corr
    # Continue numbering when a ledger already carries passes, so an adaptive
    # probe-and-extend run does not restart iteration ids at 1.
    start_k = max(
        (int(it.get("iteration", 0)) for it in ledger.get("iterations") or []),
        default=0,
    )
    total_k = start_k + max(1, iterations)

    for k in range(start_k + 1, total_k + 1):
        edits_applied = 0
        used = "numeric"
        llm_receipt: Dict[str, Any] = {}
        backend = llm_backend
        if provider == "auto" and not llm_cmd:
            picked = auto_backend()
            backend = picked.name if picked else None
            if picked is None:
                llm_receipt = {"backend": "auto", "error": "no LLM backend is installed"}
        if (provider == "opencode" and not backend) or backend or llm_cmd:
            prompt = (
                "You are editing DSP equations for a song reconstruction. "
                "Edits JSON array only: [{\"section\": int, \"id\": "
                "\"part_###\", \"op\": \"refit_amplitude\"|\"refit_phase\"|"
                "\"set_decay\"|\"set_frequency_offset\"|\"drop\", "
                "\"value\": float}]. Improve weak sections below. "
                "Use the section number and the part_### id exactly as they "
                "are printed below; a part id is only the id of that line, "
                "not its position in the list. Answer with the JSON array "
                "and nothing else.\n\n"
                + _prompt_text(ledger)
            )
            edits, llm_receipt = llm_edits_with_receipt(
                prompt,
                llm_cmd or [],
                model=llm_model,
                timeout_s=llm_timeout_s,
                backend=backend,
                api_base_url=llm_api_base_url,
                api_key=llm_api_key,
                extra_args=llm_extra_args,
            )
            edits = edits[:max_llm_edits]
            applied, refused = apply_edits_detailed(ledger, edits)
            edits_applied = applied
            llm_receipt["applied"] = applied
            llm_receipt["refused"] = len(refused)
            llm_receipt["refused_reasons"] = [
                {"id": str(r.get("edit", {}).get("id", "")), "reason": r.get("reason", "")}
                for r in refused[:8]
            ]
            if llm_receipt.get("proposed", 0) > max_llm_edits:
                llm_receipt["truncated_at"] = max_llm_edits
            used = "%s+numeric" % (llm_receipt.get("backend") or "llm")
        sections = ledger["sections"]
        st_hw = float(ledger.get("stft_half_window_s", 0.0))

        tasks = []
        for sec in sections:
            w_start = int(sec.get("start_sample", round(sec["start"] * sr)))
            w_end = min(n, int(sec.get("end_sample", round(sec["end"] * sr))))
            y_sec = y[w_start:w_end]
            tasks.append((sec, y_sec, sr, w_start, st_hw, bank, spectral_lane,
                          enable_noise, k_notes, k_hits, joint_refine))

        n_workers = (
            workers if workers is not None
            else min(8, os.cpu_count() or 1, max(1, len(sections)))
        )
        if n_workers > 1 and len(sections) > 1:
            try:
                executor = _get_pool(n_workers)
                results = list(executor.map(_process_section_task, tasks))
            except Exception:
                results = [_process_section_task(t) for t in tasks]
        else:
            results = [_process_section_task(t) for t in tasks]

        updated_sections = []
        classes = set(ledger.get("equation_classes") or [])
        for updated_sec, new_classes, _mix in results:
            updated_sections.append(updated_sec)
            for c in new_classes:
                classes.add(c)
        ledger["sections"] = updated_sections
        if classes:
            ledger["equation_classes"] = sorted(classes)

        # The workers already synthesised each section, so paste those buffers
        # instead of re-rendering every partial. Falls back to a full render when
        # any section was skipped (converged / too short) and has no mix.
        recon = np.zeros(n)
        assembled = True
        for task, (_usec, _nc, mix) in zip(tasks, results):
            if mix is None:
                assembled = False
                break
            w_start = task[3]
            end = min(n, w_start + len(mix))
            if end > w_start:
                recon[w_start:end] = mix[: end - w_start]
        if not assembled:
            recon = render_ledger(ledger, n)
        g = _score(y, recon, sr)
        gain = g["correlation"] - prev_best
        accepted = bool(
            g["correlation"] > best_corr + 1e-9
            and _referee_ok(g, best_metrics, gain)
        )
        referee_lane = "corr"
        if not accepted and mrstft_lane and _mrstft_lane_ok(
            g, best_metrics, best_corr
        ):
            # Accepted on spectrum, not waveform: record which lane shipped
            # it so the ledger stays an honest transcript.
            accepted = True
            referee_lane = "mrstft"
        if accepted:
            best_corr = g["correlation"]
            best_metrics = dict(g)
            best_sections = copy.deepcopy(ledger["sections"])
            ledger["best"] = g
        else:
            for i, sec in enumerate(ledger["sections"]):
                if i < len(best_sections):
                    if "gain_history" in sec:
                        best_sections[i]["gain_history"] = list(sec["gain_history"])
                    # Keep the bank audit trail of the iteration being
                    # shipped. Rolling back to `best_sections` otherwise
                    # restored the evidence-time (empty) bank records, so the
                    # ledger lost the record of every candidate the bank ever
                    # proposed.
                    if sec.get("hypothesis_bank"):
                        best_sections[i]["hypothesis_bank"] = copy.deepcopy(
                            sec["hypothesis_bank"]
                        )
                    if "partial_count" in sec:
                        best_sections[i]["partial_count"] = sec["partial_count"]
                    if sec.get("converged"):
                        best_sections[i]["converged"] = True
                        best_sections[i]["frozen_reason"] = sec.get("frozen_reason", "")
            ledger["sections"] = copy.deepcopy(best_sections)
        ledger["iterations"].append({
            "iteration": k,
            "provider": used,
            "llm_edits_applied": edits_applied,
            "llm_receipt": llm_receipt,
            "accepted": accepted,
            "gain": round(gain, 9),
            "global": g,
            "sections": [
                round(float(s["metrics"]["correlation"]), 6) for s in ledger["sections"]
            ],
            **({"referee_lane": referee_lane} if referee_lane != "corr" else {}),
        })
        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=2)
        if progress_fn is not None:
            try:
                progress_fn(k, total_k, g, accepted)
            except Exception:
                pass
        if accepted and target_corr > 0.0 and g["correlation"] >= target_corr:
            stop_reason = f"target-corr {target_corr:g} reached"
            break
        if (not accepted) and stop_on_no_improvement:
            stop_reason = f"iteration {k} produced no improvement"
            break
        if min_gain > 0.0 and gain < min_gain:
            stop_reason = (
                f"iteration {k} gained {gain:.6f}, under min_gain {min_gain:g}"
            )
            break
        prev_best = best_corr

    ledger["stop_reason"] = stop_reason

    # The note layer: the frequency-native view of what was fitted -
    # beat pairs (detuned unisons with their beat rates) and sub-bass
    # note events (the 808 as discrete notes with their attack drop).
    # Analysis only: the render and the referee are untouched.
    try:
        from src.note_layer import detect_beat_pairs, detect_note_events
        all_partials = [
            p for s in (ledger.get("sections") or [])
            for p in s.get("partials", [])
        ]
        ledger["note_layer"] = {
            "beat_pairs": detect_beat_pairs(all_partials)[:24],
            "sub_note_events": detect_note_events(y, sr)[:16],
        }
    except Exception:
        pass

    # The shipped render gets OLA crossfades: sections rephase
    # independently and the raw paste clicks (measured: a 13.8x sample
    # jump at a real section boundary - audible as a pop mid-song). The
    # score below is recomputed on this exact render, so the ledger's
    # numbers always describe the audio that ships.
    best_recon = render_ledger(
        {**ledger, "sections": best_sections}, n, ola=True
    )
    _repair_seams(
        best_recon,
        [int(round(float(s["start"]) * sr)) for s in best_sections[1:]],
        sr=sr,
    )
    ledger["energy_parity"] = {"applied": False, "gains": []}
    if energy_parity:
        # The referee optimises correlation, which is scale-invariant, so
        # the shipped render can sit at the wrong loudness (measured: the
        # drums stem's best render carried 2.1% of the source energy). One
        # gain per section restores parity. "ls" is the SNR-optimal
        # least-squares gain (it lands the render at corr x source RMS);
        # "rms" is loudness parity - the "sounds like the original" choice.
        # Off by default so published numbers do not move until re-measured.
        spans = _parity_spans(best_sections, n, sr)
        best_recon, parity_gains = apply_energy_parity(
            y, best_recon, spans, mode=parity_mode
        )
        ledger["energy_parity"] = {
            "applied": True,
            "mode": ("per-section rms" if parity_mode == "rms"
                     else "per-section least squares"),
            "gains": [round(float(g), 6) for g in parity_gains],
        }
    wave = out / "res_best.wav"
    sf.write(str(wave), np.clip(best_recon, -1.0, 1.0), sr)
    recon_final = out / "res_final.wav"
    final_full = render_ledger(ledger, n, ola=True)
    sf.write(str(recon_final), np.clip(final_full, -1.0, 1.0), sr)
    ledger["best"] = _score(y, best_recon, sr)
    with open(out / "res_equations.txt", "w", encoding="utf-8") as f:
        f.write(_equations_text(ledger) + "\n")
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
    ledger["outputs"] = {
        "ledger": str(ledger_path),
        "best_wav": str(wave),
        "final_wav": str(recon_final),
        "equations_txt": str(out / "res_equations.txt"),
    }
    return ledger


def _prompt_text(ledger: Dict[str, Any]) -> str:
    from src.res_evidence import evidence_prompt

    return evidence_prompt(ledger)


def run_stems_res_loop(
    source_path: Union[str, Path, np.ndarray],
    out_dir: Union[str, Path],
    stems_mode: str = "auto",
    stems_dir: Optional[str] = None,
    sr: int = 22050,
    max_duration: float = 20.0,
    window_s: float = 2.5,
    iterations: int = 3,
    target_corr: float = 0.98,
    workers: Optional[int] = None,
    progress_fn: Optional[Callable[..., Any]] = None,
    bank: bool = True,
    spectral_lane: bool = False,
    enable_noise: bool = False,
    k_notes: int = 2,
    k_hits: int = 12,
    energy_parity: bool = False,
    parity_mode: str = "ls",
) -> Dict[str, Any]:
    """Execute Tri-Mode Stem Separation and Multi-Track RES Tournament Loop.

    Emits:
      - res_stem_<name>.wav & res_stem_<name>_equations.txt for each stem
      - res_full_mix.wav, res_best.wav, res_final.wav for the master mix
      - res_multitrack_ledger.json (schema 'res/2.0-multitrack') & res_ledger.json
      - res_equations.txt & res_bytebeat.js
    """
    from src.res_evidence import separate_stems, build_evidence

    if isinstance(source_path, np.ndarray):
        y = np.asarray(source_path, dtype=np.float32)
        if max_duration > 0.0:
            y = y[: int(max_duration * sr)]
    else:
        from src.analysis import load_audio
        y, _ = load_audio(str(source_path), target_sr=sr)
        if max_duration > 0.0:
            y = y[: int(max_duration * sr)]
    n = len(y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Stem Separation (Directory -> Demucs -> Native Zero-Phase DSP Fallback)
    stems = separate_stems(y, sr, mode=stems_mode, stems_dir=stems_dir)
    for k in stems:
        if len(stems[k]) < n:
            stems[k] = np.pad(stems[k], (0, n - len(stems[k])))
        else:
            stems[k] = stems[k][:n]

    # 2. Compute shared section boundaries on full mix y so all stems align
    # Uniform window boundaries guarantee SMS chunk origin matches section start sample,
    # eliminating phase cancellation across section boundaries.
    win = max(int(round(window_s * sr)), 4 * 256)
    boundaries: List[int] = list(range(0, n, win))
    if boundaries[-1] != n:
        boundaries.append(n)

    # 3. Process each stem independently
    #
    # Track budgets: a stem is CLEANER than the mix it was separated from, so
    # it must not get a smaller partial budget than the mix did. The old
    # budgets (150/250/200/150 against the single-file path's 400) starved
    # the stems: measured on j-hope - MONA LISA, the drums stem scored
    # LSD 17.28 dB on 200 tracks while the same audio through the single-file
    # path scored 6.71 dB on 400, and the bass stem scored 0.5988 against
    # 0.7773. The separation already did the work; the fitter needs room.
    stem_configs = {
        "bass": {
            "n_fft": 4096,
            "hop_length": 128,
            "threshold": 0.005,
            "glide_tolerance_hz": 2.5,
            "max_tracks": 400,
        },
        "vocals": {
            "n_fft": 2048,
            "hop_length": 256,
            "threshold": 0.01,
            "glide_tolerance_hz": 6.0,
            "max_tracks": 400,
        },
        "drums": {
            "n_fft": 2048,
            "hop_length": 256,
            "threshold": 0.01,
            "glide_tolerance_hz": 6.0,
            "max_tracks": 400,
        },
        "other": {
            "n_fft": 2048,
            "hop_length": 256,
            "threshold": 0.01,
            "glide_tolerance_hz": 6.0,
            "max_tracks": 400,
        },
    }

    stem_ledgers: Dict[str, Dict[str, Any]] = {}
    recon_stems: Dict[str, np.ndarray] = {}
    stem_metrics: Dict[str, Dict[str, Any]] = {}
    all_classes: set = set()

    for s_name in ("bass", "drums", "vocals", "other"):
        y_s = stems[s_name]
        cfg = stem_configs[s_name]
        s_dir = out / f"stem_{s_name}"
        s_dir.mkdir(parents=True, exist_ok=True)

        energy = float(np.sum(y_s ** 2))
        if energy < 1e-9:
            empty_sections = []
            for bi in range(len(boundaries) - 1):
                empty_sections.append({
                    "start": round(boundaries[bi] / sr, 3),
                    "end": round(boundaries[bi + 1] / sr, 3),
                    "start_sample": boundaries[bi],
                    "partial_count": 0,
                    "partials": [],
                    "metrics": {"correlation": 1.0, "snr_db": 99.0, "log_spectral_distance_db": 0.0},
                })
            s_ledger = {
                "schema": "res/1.1",
                "sample_rate": sr,
                "n_samples": n,
                "duration": round(n / sr, 3),
                "sections": empty_sections,
                "best": {"correlation": 1.0, "snr_db": 99.0, "log_spectral_distance_db": 0.0},
            }
            recon_s = np.zeros(n, dtype=np.float32)
        else:
            ev = build_evidence(
                y_s,
                output_dir=str(s_dir),
                sr=sr,
                max_duration=max_duration,
                window_s=window_s,
                n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"],
                threshold=cfg["threshold"],
                max_tracks=cfg["max_tracks"],
                use_hpss=False,
                glide_tolerance_hz=cfg["glide_tolerance_hz"],
                boundaries=boundaries,
                beat_quantized=False,
            )
            s_ledger = run_loop(
                ev,
                y_s,
                str(s_dir),
                iterations=iterations,
                target_corr=target_corr,
                workers=workers,
                bank=bank,
                progress_fn=progress_fn,
                spectral_lane=spectral_lane,
                enable_noise=enable_noise,
                k_notes=k_notes, k_hits=k_hits,
            )
            recon_s = render_ledger(s_ledger, n, ola=True)

            # Prune low-energy junk partials
            stem_energy = float(np.sum(y_s ** 2))
            pruned_sections = []
            for sec in s_ledger.get("sections", []):
                kept_partials = []
                for p in sec.get("partials", []):
                    a = float(p.get("amplitude", 0.0))
                    lam = float(p.get("envelope_decay", 0.0))
                    span = (float(p.get("end_time", 0)) - float(p.get("start_time", 0))) or window_s
                    eff_len = min(span, 1.0 / max(2.0 * lam, 1e-4))
                    e_p = 0.5 * (a ** 2) * eff_len * sr
                    ratio = e_p / (stem_energy + 1e-12)
                    if a >= 0.001 and ratio >= 0.0005:
                        kept_partials.append(p)
                if not kept_partials and sec.get("partials"):
                    kept_partials = sorted(sec["partials"], key=lambda q: float(q.get("amplitude", 0.0)), reverse=True)[:3]
                sec["partials"] = kept_partials
                sec["partial_count"] = len(kept_partials)
                pruned_sections.append(sec)

            cand_ledger = {**s_ledger, "sections": pruned_sections}
            cand_recon = render_ledger(cand_ledger, n, ola=True)
            if _corr(y_s, cand_recon) >= _corr(y_s, recon_s) - 0.005:
                s_ledger["sections"] = pruned_sections
                recon_s = cand_recon

        for c in s_ledger.get("equation_classes", []):
            all_classes.add(c)

        for sec in s_ledger.get("sections", []):
            for p in sec.get("partials", []):
                p["stem"] = s_name

        stem_ledgers[s_name] = s_ledger
        recon_stems[s_name] = recon_s

        # Per-stem output files
        stem_wav_path = out / f"res_stem_{s_name}.wav"
        sf.write(str(stem_wav_path), np.clip(recon_s, -1.0, 1.0), sr)
        stem_eq_path = out / f"res_stem_{s_name}_equations.txt"
        with open(stem_eq_path, "w", encoding="utf-8") as f:
            f.write(_equations_text(s_ledger) + "\n")

        stem_metrics[s_name] = _score(y_s, recon_s, sr)

    # 4. Master mix synthesis and scoring
    recon_mix = np.zeros(n, dtype=np.float64)
    for r in recon_stems.values():
        recon_mix += r
    mix_metrics = _score(y, recon_mix, sr)

    # Loudness parity for the master mix. The stems pipeline had none, so
    # its combined mix shipped at the wrong level (measured: loudness
    # pillar 68.6 while tone led the whole suite at 78.0). One whole-span
    # gain - per-stem sections overlap in the mix, so per-section gains
    # here would double-apply.
    parity_info = {"applied": False, "mode": parity_mode, "gains": []}
    if energy_parity:
        if parity_mode == "rms":
            rms_y = float(np.sqrt(np.mean(y ** 2)))
            rms_r = float(np.sqrt(np.mean(recon_mix ** 2)))
            mix_gain = rms_y / rms_r if rms_r > 1e-12 else 1.0
        else:
            mix_gain = float(
                np.dot(y, recon_mix)
                / max(float(np.dot(recon_mix, recon_mix)), 1e-15)
            )
        if np.isfinite(mix_gain):
            mix_gain = min(max(mix_gain, 0.0), 50.0)
            recon_mix = recon_mix * mix_gain
            parity_info = {"applied": True, "mode": f"master-mix {parity_mode}",
                           "gains": [round(mix_gain, 6)]}
            mix_metrics = _score(y, recon_mix, sr)

    sf.write(str(out / "res_full_mix.wav"), np.clip(recon_mix, -1.0, 1.0), sr)
    sf.write(str(out / "res_best.wav"), np.clip(recon_mix, -1.0, 1.0), sr)
    sf.write(str(out / "res_final.wav"), np.clip(recon_mix, -1.0, 1.0), sr)

    # 5. Populate merged sections for backward compatibility
    merged_sections: List[Dict[str, Any]] = []
    n_secs = len(boundaries) - 1
    for bi in range(n_secs):
        w_start = boundaries[bi]
        w_end = boundaries[bi + 1]
        sec_parts: List[Dict[str, Any]] = []
        for s_name in ("bass", "drums", "vocals", "other"):
            sec_list = stem_ledgers[s_name].get("sections", [])
            if bi < len(sec_list):
                sec_parts.extend(sec_list[bi].get("partials", []))

        sec_y = y[w_start:w_end]
        sec_r = recon_mix[w_start:w_end]
        m_sec = _score(sec_y, sec_r, sr)
        merged_sections.append({
            "start": round(w_start / sr, 3),
            "end": round(w_end / sr, 3),
            "start_sample": int(w_start),
            "partial_count": len(sec_parts),
            "partials": sec_parts,
            "metrics": m_sec,
        })

    # 6. Build top-level stems dictionary
    stems_dict = {
        s_name: {
            "metrics": stem_metrics[s_name],
            "sections": stem_ledgers[s_name].get("sections", []),
            "outputs": {
                "wav": str(out / f"res_stem_{s_name}.wav"),
                "equations": str(out / f"res_stem_{s_name}_equations.txt"),
            },
        }
        for s_name in ("bass", "drums", "vocals", "other")
    }

    multitrack_ledger: Dict[str, Any] = {
        "schema": "res/2.0-multitrack",
        "source": str(source_path) if not isinstance(source_path, np.ndarray) else "in-memory",
        "sample_rate": sr,
        "duration": round(n / sr, 3),
        "n_samples": int(n),
        "window_s": window_s,
        "stems_mode": stems_mode,
        "stems": stems_dict,
        "sections": merged_sections,
        "best": mix_metrics,
        "energy_parity": parity_info,
        "initial_correlation": mix_metrics["correlation"],
        "equation_classes": sorted(list(all_classes)) or ["damped_cosine", "glide", "filtered_noise"],
        "outputs": {
            "multitrack_ledger": str(out / "res_multitrack_ledger.json"),
            "ledger": str(out / "res_ledger.json"),
            "best_wav": str(out / "res_best.wav"),
            "full_mix_wav": str(out / "res_full_mix.wav"),
            "equations_txt": str(out / "res_equations.txt"),
            "bytebeat_js": str(out / "res_bytebeat.js"),
        },
    }

    with open(out / "res_multitrack_ledger.json", "w", encoding="utf-8") as f:
        json.dump(multitrack_ledger, f, indent=2)

    with open(out / "res_ledger.json", "w", encoding="utf-8") as f:
        json.dump(multitrack_ledger, f, indent=2)

    with open(out / "res_equations.txt", "w", encoding="utf-8") as f:
        f.write(_equations_text(multitrack_ledger) + "\n")

    try:
        from tools.res_bytebeat import emit
        bb_code = emit(multitrack_ledger, 0.001)
        with open(out / "res_bytebeat.js", "w", encoding="utf-8") as f:
            f.write(bb_code + "\n")
    except Exception:
        pass

    return multitrack_ledger

