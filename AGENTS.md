# AGENTS.md

Guidance for AI coding agents working in this repository. Human-readable
context: `README.md`. Deep context: `PROJECT_BUNDLE.md` (the full bundle),
`GLOSSARY.md` (domain vocabulary), `docs/`.

## What this project is

An audio DSP engine ("M2M / RES") that reverse-engineers songs into
editable analytic equations (damped cosines, glide trajectories, harmonic
strings, percussive modal bundles, stochastic noise classes), fitted by a
coordinate lifter plus a hypothesis tournament under a referee loop.
Python 3.10+, NumPy/librosa/scipy; Streamlit UI; CLI entry `m2m` (`main.py`).

## Commands

- Run tests: `.venv/Scripts/python.exe -m pytest tests/ -q` (a venv-less
  `python -m pytest tests/ -q` also works; the suite is green on
  ubuntu/windows/macos in CI).
- Lint (bug-class rules only, config in `pyproject.toml`): `ruff check .`
- Web app: `streamlit run app.py`
- One CLI run: `python main.py res -i data/output/_fixtures/stem_vocals.wav -o out/demo -d 8 --iterations 2 --energy-parity`
- Scoreboard provenance check: `python tools/verify_scoreboard.py`

## Conventions that will bite you if ignored

1. **Measured numbers need provenance.** Every number cited in README/docs
   comes from a scoreboard label backed by committed ledgers
   (`data/output/scoreboard.json` + `data/output/_scoreboard/<label>/`).
   Do not change engine defaults that move published numbers without
   re-running labels and committing their ledgers.
2. **Analyser/renderer parity is sacred.** `sinusoidal_model` (analyser)
   and `res_synthesis` (renderer) are deliberately kept in lockstep; tests
   pin their agreement (e.g. 1e-11 rad on trajectories). If you touch the
   render math, both sides and `tests/test_res_replica.py` must agree.
3. **The partial is a schema, not a bag.** ~25 string keys agreed across
   modules — see `GLOSSARY.md`. New model classes add keys; check every
   consumer (`p.get(key, default)` typos are silent).
4. **Physics work is flag-gated.** Experimental classes (exciter,
   heterodyne, harmonic expansion, energy parity) ship behind flags/env
   vars, default off, until re-measured (`M2M_EXCITER`,
   `RES_NO_HETERODYNE`, ...).
5. **Defect-driven development.** Measured defects are pinned as tests
   (`tests/test_res_trajectory_defects.py` D9–D12, `docs/res_research.md`
   D1–D8). Reproduce → pin → fix; keep the numbers in docstrings.

## Architecture map (hot spots)

- `src/res_loop.py` — the loop + orchestration (god module; see review)
- `src/res_bank.py` — hypothesis tournament (classes B/C/C++/D/E)
- `src/sinusoidal_model.py` — tracks → partials, envelopes, trajectories
- `src/res_synthesis.py` — the renderer kernels (shared with the analyser)
- `src/res_evidence.py` — sections, stems, initial ledger
- Roadmaps: `docs/PHYSICS_ROADMAP.md`, `docs/BEYOND_THE_LIMITS.md`,
  and the architecture review notes in `docs/AGENT_PROMPT_PHYSICS_ARCHITECTURE.md`.
