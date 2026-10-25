"""The partial schema: the single definition site for the equation atom.

A *partial* is one atom of the reconstructed model - a flat dict of string
keys representing a stationary sinusoid, a glide, a harmonic-string note, a
percussive mode, or a stochastic noise channel. The equation sheet is a list
of partials; the ledger, the renderer, and every hypothesis class read and
write them. Historically the keys were minted in four modules
(``sinusoidal_model.refine_track``, the ``res_bank`` constructors,
``res_synthesis.fit_noise_floor_partial``, ``res_percussive``) and consumed
with ``p.get(key, default)`` across six - so a typo anywhere is silent.

This module is the registry: every key any writer mints is declared here,
grouped by who needs it. ``validate_partial`` turns the schema into a check
- it does NOT change any engine behaviour, it only names violations so tests
and tooling can catch them. New model classes must register their keys here
or ``validate_partial`` will flag their output as unregistered.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Tuple

# The `model` key values. "damped_cosine" is the legacy/docs name for the
# damped atom and is tolerated on input (see note_layer._TONAL_MODELS).
MODELS: Tuple[str, ...] = (
    "constant",      # stationary sinusoid (single refined frequency)
    "damped",        # exponentially damped sinusoid (also the legacy "damped_cosine")
    "glide",         # frequency-trajectory sinusoid (vibrato, glissando)
    "filtered_noise",  # class C++ / class D / exciter: seeded band-shaped noise
    "lpc_noise",     # class E: LPC-filtered seeded noise
)
_LEGACY_MODEL_ALIASES = {"damped_cosine": "damped"}

TONAL_MODELS: FrozenSet[str] = frozenset({"constant", "damped", "glide"})
STOCHASTIC_MODELS: FrozenSet[str] = frozenset({"filtered_noise", "lpc_noise"})

# Keys every well-formed partial carries. The numeric core is what the
# renderer's fallback path reads (res_synthesis._render_partial), and
# start/end_time drive the support mask.
COMMON_KEYS: FrozenSet[str] = frozenset({
    "model",
    "frequency",
    "amplitude",
    "phase",
    "envelope_decay",
    "envelope_t0",
    "start_time",
    "end_time",
})

# Tonal-only keys, minted by sinusoidal_model.refine_track and the res_bank
# class-A/B constructors.
TONAL_KEYS: FrozenSet[str] = frozenset({
    "envelope_a0",      # analytic envelope anchor level (A(t) = a0*exp(-decay*(t-t0)))
    "projection_gain",  # gain the renderer multiplies the glide envelope by
    "freq_spread_hz",   # observed frequency dispersion that chose glide vs constant
    "phase_origin",     # the time origin the stored phase is measured against
    "phase_ref_hz",     # the anchor frequency of the glide phase reference term
    "window_end",       # far edge of the analysis window the partial was fitted in
    "trajectory_refined",  # the glide frequency path was de-chirped (heterodyne)
    "m",                # percussive mode index within a bundle (None outside bundles)
})

# Keys shared by tonal and stochastic partials.
SHARED_KEYS: FrozenSet[str] = frozenset({
    # {t, f_hz, a} on a glide (frequency path); {t, a} on a filtered_noise
    # channel (measured level envelope). Integrated by the shared kernel.
    "trajectory",
    "unit_step",     # one-sided: silent before envelope_t0
    "note_group",    # the hit/beat instant a bundle's modes belong to
})

# Stochastic-only keys, minted by the res_bank class C++/D/E constructors,
# res_percussive (the exciter), and fit_noise_floor_partial.
STOCHASTIC_KEYS: FrozenSet[str] = frozenset({
    "seed",          # deterministic RNG stream - reproducibility contract
    "band_centers",  # log-spaced shaping band centres (filtered_noise)
    "band_mags",     # per-band magnitude envelope (filtered_noise)
    "band_edges",    # measured band boundaries for per-band envelopes
    "band_traj",     # {t, a[band][seg]} per-band level trajectories
    "lpc_coeffs",    # LPC denominator coefficients (lpc_noise)
    "lpc_order",     # analysis order the coefficients were fitted at
    "sibilance",     # marker flag set by the class-D constructor
})

KNOWN_KEYS: FrozenSet[str] = (
    COMMON_KEYS | TONAL_KEYS | SHARED_KEYS | STOCHASTIC_KEYS
)


def validate_partial(p: Dict[str, Any]) -> List[str]:
    """Return a list of schema violations for one partial (empty = valid).

    Violations are names, not exceptions: callers decide whether to fail,
    log, or repair. Checks mirror what consumers actually rely on - the
    renderer's numeric reads, the support-mask time order, and the paired
    list lengths the band/glide paths index into.
    """
    problems: List[str] = []
    model = p.get("model")
    if model is None:
        return ["missing key 'model'"]
    if model not in MODELS and model not in _LEGACY_MODEL_ALIASES:
        return [f"unknown model {model!r} (known: {', '.join(MODELS)})"]

    for key in COMMON_KEYS - {"model"}:
        if key not in p:
            problems.append(f"missing key {key!r}")

    def _num(key: str) -> None:
        v = p.get(key)
        if v is None:
            return
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            problems.append(f"{key!r} is not numeric: {v!r}")
        elif v != v or v in (float("inf"), float("-inf")):
            problems.append(f"{key!r} is not finite: {v!r}")

    _num("frequency")
    _num("amplitude")
    _num("phase")
    _num("envelope_decay")
    _num("envelope_t0")
    _num("start_time")
    _num("end_time")

    if isinstance(p.get("amplitude"), (int, float)) and p["amplitude"] < 0:
        problems.append("'amplitude' is negative")
    if isinstance(p.get("envelope_decay"), (int, float)) and p["envelope_decay"] < 0:
        problems.append("'envelope_decay' is negative")
    st, et = p.get("start_time"), p.get("end_time")
    if isinstance(st, (int, float)) and isinstance(et, (int, float)) and st > et:
        problems.append("'start_time' is after 'end_time'")

    if model == "glide":
        traj = p.get("trajectory")
        if not isinstance(traj, dict):
            problems.append("glide without a 'trajectory' dict")
        else:
            t = traj.get("t", [])
            f = traj.get("f_hz", [])
            a = traj.get("a", [])
            if not (len(t) == len(f) == len(a)):
                problems.append(
                    "glide trajectory t/f_hz/a lengths differ "
                    f"({len(t)}/{len(f)}/{len(a)})"
                )

    if model == "filtered_noise":
        if p.get("seed") is None:
            problems.append("filtered_noise without 'seed'")
        bc, bm = p.get("band_centers"), p.get("band_mags")
        if isinstance(bc, list) and isinstance(bm, list) and len(bc) != len(bm):
                problems.append(
                    f"band_centers/band_mags lengths differ ({len(bc)}/{len(bm)})"
                )
        be, bt = p.get("band_edges"), p.get("band_traj")
        if isinstance(be, list) and isinstance(bt, dict):
            rows = bt.get("a", [])
            if isinstance(rows, list) and rows and len(rows) != len(be) - 1:
                problems.append(
                    f"band_traj has {len(rows)} rows for {len(be)} band edges"
                )

    if model == "lpc_noise" and not isinstance(p.get("lpc_coeffs", []), list):
        problems.append("'lpc_coeffs' is not a list")

    for key in p:
        if key not in KNOWN_KEYS:
            problems.append(
                f"unregistered key {key!r} - add it to partial_schema if intentional"
            )
    return problems
