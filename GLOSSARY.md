# Domain glossary

The words this codebase already uses, written down so reviews, tests, and
refactors can speak the same language. Architecture reviews should use these
terms for the domain and the codebase-design vocabulary (module, interface,
depth, seam, adapter, leverage, locality) for the structure.

## The model

- **Partial** — one atom of the reconstructed model: a dict with ~25
  string keys (`frequency`, `amplitude`, `envelope_a0`, `envelope_decay`,
  `phase`, `model`, `trajectory`, `phase_origin`, …) representing a stationary
  sinusoid, a glide, a harmonic-string note, a percussive mode, or a
  stochastic noise band. The equation sheet is a list of partials. *The
  partial dict has no single definition site — keys are minted in
  `sinusoidal_model.refine_track`, `res_bank` hypothesis constructors,
  `res_synthesis`, and `res_percussive`, and consumed with `p.get(key,
  default)` across six modules. Treat it as a schema, not a bag.*
- **Model class** (equation class) — the `model` key of a partial:
  `damped_cosine` (stationary), `glide`, class B (one-sided harmonic string),
  percussive modes/bundles, class C++ (`filtered_noise`), class E
  (`lpc_noise`), class D (`sibilance_burst`), and the experimental resonant
  burst (the **exciter**).
- **Track** — the tracker's output (`SinusoidalTrack`): per-frame peaks
  associated across time; refined into a partial by `refine_track`.
- **Trajectory** — a glide's per-frame frequency path (`trajectory{t,f_hz,a}`),
  integrated into phase. Shared kernel: `sinusoidal_model._trajectories`
  (the renderer reaches it through `res_synthesis._glide_traj`).
- **Beat pair** — two tonal partials within ~60 cents whose supports
  overlap: one musical object (a detuned unison) with a carrier
  (average), a beat rate `f_beat = |f1 − f2|` (the intensity beat; the
  envelope swells at half of it), and independent decays (the beat dies
  as one layer dies). Detected by `src/note_layer.detect_beat_pairs`;
  rendered as the two partials (bit-identical), recorded in the ledger
  as the frequency-native view. See `docs/NOTE_LAYER.md`.
- **12-TET snap** — replacing a detected carrier with the nearest
  equal-tempered pitch (`440 · 2^((n−69)/12)`), deviation kept as data.
  Snappable = within 25 cents (every frequency is within 50 cents of
  *some* note, so the tolerance must be tighter than a quarter tone).
  Snap the carrier, never the layers of a beat pair.

## The pipeline

- **Section** — a span of the analysed audio (beat-aware, ~half-window
  rhythm). The unit the tournament and the coordinate lifter work on.
  Partials carry global times but section-local phases; `_rephase_to_section`
  re-references a partial's phase to a section origin.
- **Evidence** — the initial measurement pass (`build_evidence`): load,
  separate stems, cut sections, fit first-pass partials, score them. Writes
  the ledger skeleton.
- **Ledger** — the JSON transcript of a run: schema, source, sections,
  iterations, accepted/rejected records, best metrics, output paths. The
  provenance artifact a scoreboard row points at; rewritten every iteration.
- **Referee** — the acceptance gate. Dual time-domain gate (energy margin
  over the overfit guard + strictly better correlation) or the **spectral
  lane** (log-spectral distance) for candidates a waveform error cannot
  judge (noise, bursts).
- **Tournament** — `tournament_section`: per-section competition where every
  model class proposes hypotheses and the referee accepts or rejects each,
  appending records to the ledger.
- **Refit strategies** — the three refitters are deliberately different
  engines, not three copies of one: `refine_track` (analyser) fits a track's
  frequency and complex gain by least-squares projection inside the analysis
  window; `_section_update` (coordinate lifter) greedily re-projects one
  partial at a time against the section residual, accepting an edit only
  when section correlation strictly improves; `tournament_section` (bank)
  lets whole equation classes propose hypotheses under an overfit-guarded
  residual criterion with the spectral lane for stochastic classes. A shared
  "one refit engine" was reviewed (2026-10-25) and rejected: the acceptance
  criteria are measured behaviors, and merging them would move published
  numbers for no structural gain.
- **Referee metric family** — the four log-spectral distances are four
  distinct measurements, not duplication: `residual.compute_error_metrics`
  (canonical: coherent-gain-normalised STFT, -80 dB floor — the one scores
  and docs cite), `res_bank.section_lsd` (fast framed variant),
  `res_bank.hf_log_spectral_distance` (HF bins only, L1 — the spectral
  lane's magnitude-domain test), and `res_percussive.spectral_log_distance`
  (relative floor, for broadband exciter judgement). New code uses the
  canonical one unless it needs one of these specific behaviors.
- **Hypothesis** — one proposed candidate: a partial list, a parameter
  count, and a measured error (`_Variant`), from a class constructor
  (`filtered_noise_hypothesis`, `percussive_bundle`, …).
- **Coordinate lifter** — `_section_update`: the per-section numeric
  refinement (frequency, decay, glide trajectory, amplitude) the loop runs
  between tournaments.
- **The loop** — `run_loop`: hypothesize → render → referee → accept or
  roll back, once per iteration; an LLM edits provider can propose parameter
  edits between iterations.
- **Stems** — separated sources (bass / drums / vocals / other), each fitted
  by its own loop run, then summed to a combined mix.

## Measurement

- **Fidelity benchmark** — the "original = 100%, ours = ?" scale
  (`src/fidelity.py`, report via `tools/fidelity_report.py`). Headline =
  mean of four pillars: **tone** (waveform corr), **spectrum**
  (10^(−mrstft_db/20)), **bands** (original-energy-weighted band corr),
  **loudness** (RMS-parity, exp(−|ln ratio|)); **distance** = 100 −
  headline. Catches failures correlation is blind to (a too-quiet render
  once shipped at 2.1% of source energy while corr looked normal).
- **Scoreboard** — `data/output/scoreboard.json`: frozen, re-runnable
  measurements. A **label** is one engine run over the fixture set; each
  **row** cites the committed ledger that produced it.
- **Fixture** — a frozen input WAV under `data/output/_fixtures/` (stems,
  mixes) used by the sample tests.
- **Equation sheet** — `res_equations.txt`: the human-readable serialization
  of the partials (`_equations_text`).
- **Bytebeat export** — the JS side-artifact that replays the model in a
  browser at 8 kHz.
