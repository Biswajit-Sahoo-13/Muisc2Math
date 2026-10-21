# Agent prompt — improving the physics or the architecture of M2M (RES)

> **How to use this.** Attach `PROJECT_BUNDLE.md` to the agent session, then
> paste everything below the line as the request. The bundle is one file
> containing ten files in reading order; the agent needs no further context
> to start. `PROJECT_BUNDLE.zip` is only needed if the agent must edit the
> raw sources rather than reason about them.

---

## The request

You are working on **M2M (RES)**, a program that takes an audio file and
reverse-engineers it into **editable mathematical equations**, then renders
those equations back to audio. The output is not a waveform — it is a short
set of equations a person can open, read, edit and re-render. That is the
product.

The attached `PROJECT_BUNDLE.md` contains ten files from the repository in
reading order, with a navigation header. Read the header first; it states
the project shape, the measured numbers and the file index. Then read the
files in the order given, in full. Do not guess at what a file does from
its name — the whole point of the bundle is that you can read it.

**Your job: find the single most valuable change to the physics or the
architecture, and prove it.**

Not "improve the project". Not "add features". One change, chosen because
the evidence points at it, and then a measurement that shows whether it
worked. If the honest answer is that the best available change is small,
say so and show the measurement. A well-argued null result is a success
here; a plausible-sounding improvement that moves nothing is a failure.

### Two tracks — pick one, and say which

**Track A — physics.** Make the reconstruction better: more accurate, more
complete, closer to the representation's actual limits. This changes
published numbers.

**Track B — architecture.** Make the codebase better: clearer boundaries,
explicit contracts, testable invariants, fewer ways to lie to the user.
This usually does *not* change published numbers, and if it does, that is a
finding worth reporting.

Track B is not the lesser track. The engine's physics is already close to
what a sum-of-sinusoids representation can do (see the ceiling below), and
the most valuable remaining work may well be structural. What is *not*
acceptable is Track B as a way to avoid Track A.

If you have a strong reason to do both, do Track A first, in isolation,
measure it, and only then consider Track B on top.

---

## Ground rules

These are not stylistic preferences. Violating them makes your work
unusable.

1. **Reproduce the baseline before you touch anything.** Run the commands
   below. If your baseline differs from the numbers in the bundle header,
   stop and say so — that is itself the most important finding you could
   report.

2. **Never hand-edit a published number.** Every figure in the README, the
   research doc, the bundle header and the release notes is a row in
   `data/output/scoreboard.json` pointing at a ledger JSON produced by a
   real run. If your change moves a number, re-run the label and let the
   harness write the row. A number that was typed by hand is a number
   nobody can defend.

3. **Never weaken a test to make it pass.** If a test is wrong, fix it and
   state in the report exactly why it was wrong. A tolerance you loosened
   to get green is a hidden regression. (One example already in the
   history: `assertEqual` on a `np.corrcoef` result demanded a bit-exact
   float, which differs by platform. The right fix was to compare with a
   tolerance *and* to make the product function respect its own contract,
   which now clips the correlation to `[-1, 1]`.)

4. **Do not report a number you have not measured.** Every figure you
   state must come from a run whose output you can point at. Label
   provenance the way the project does: synthetic fixture, real material,
   or measured on a rebuilt model.

5. **One change at a time, each with its own proof.** A commit that mixes
   a physics change, a refactor and a test rewrite is unmeasurable. If a
   refactor is needed to make a change possible, land and verify the
   refactor first, separately, showing it changed no number.

6. **Do not chase a number that is physically unreachable.** See the
   ceiling below. A full commercial master caps near 0.65–0.75 waveform
   correlation. Brickwall limiting and microsecond phase jitter are not
   invertible by a sum of sinusoids. If your change takes a real mix from
   0.85 to 0.87, that is a genuine result. If it takes a real mix to 0.99,
   you have found a bug in your own measurement — the input was not what
   you thought it was.

7. **Report what did not work.** A short section listing the plausible
   ideas you tried that made no difference is worth more to the next person
   than the thing that did.

8. **Stay in scope.** Do not rename the package, restructure the CLI or
   reformat the codebase as a side effect. If you find a problem outside
   your chosen track, write it in the report and move on.

---

## The state of the art, as measured

Two different things are being measured. They are not comparable, and
conflating them is the most common way this project has been misreported.

**Synthetic input, exact fixture** — a generated triad with known
parameters:

| | |
|---|---|
| waveform correlation | **0.999921** |
| SNR | **38.03 dB** |
| log-spectral distance | **0.33** |

This is the *representation's* ceiling when the input is exactly what the
model can express. It is near-perfect, and there is almost nothing left
here.

**Real material, stem level** — correlation after reconstruction, and the
tonal energy fraction of the input:

| material | correlation | LSD | onset corr | harmonic | percussive |
|---|---|---|---|---|---|
| full mix, 10 s | **0.8518** | 3.61 dB | 0.539 | 79.9 % | 8.6 % |
| vocals | 0.7535 | 9.11 dB | 0.477 | 82.3 % | 3.6 % |
| other | 0.7376 | 9.50 dB | 0.308 | 91.5 % | 2.2 % |
| bass | 0.6160 | 3.03 dB | 0.429 | 51.8 % | 20.3 % |
| drums | **0.4107** | 18.04 dB | **0.684** | 3.1 % | 83.5 % |

Reconstruction correlation tracks the input's **tonal energy fraction at
r = +0.935** across the fixture set. The engine reconstructs what is
harmonic and degrades on what is not. That is the honest scope of the
method, not a defect in the pipeline.

Synthetic probes, to show the shape of the failure:

| probe | correlation | LSD | partials |
|---|---|---|---|
| plucked string | 1.0000 | 0.35 | 23 |
| four-on-the-floor beat (modelled kick + hat) | 0.9963 | 2.01 | 102 |
| linear sweep | 0.8738 | 5.25 | 27 |
| sustained organ | 0.8590 | 3.93 | 40 |
| sustained chord pad | 0.8495 | 3.01 | 33 |
| gated bass only | 0.5852 | 3.83 | 78 |
| vibrato tone | 0.5196 | 12.54 | 942 |
| snare roll | 0.4690 | 21.97 | 515 |
| white noise | 0.2342 | 29.52 | 821 |

Read that table carefully, because it is a map of where the physics is
weak and why:

- **The vibrato tone at 0.5196 with 942 partials is the most diagnostic
  number here.** A vibrato is a handful of sinusoids whose *frequency
  moves*. The engine spends 942 partials trying to express a moving
  frequency as a set of stationary ones, and still gets 0.52. This is a
  representation mismatch, not a fitting failure.
- **The four-on-the-floor beat reaches 0.9963 while the snare roll reaches
  0.4690.** A modelled kick and hat are well handled; a snare is not. The
  difference is that a snare's body is a *resonant* object, not a set of
  decaying tones.
- **White noise at 0.2342 with 821 partials** is the floor for stochastic
  content. The project models a noise floor separately; the question is
  whether the residual model is good enough to stop the partial budget
  being wasted on noise.
- **The sweep at 0.8738** is another frequency-trajectory problem: a
  glide inside a section.

### The honest ceiling

A full commercial master caps near **0.65–0.75** waveform correlation.
Brickwall limiting and microsecond phase jitter are not invertible by a
sum of sinusoids. The engine reports this rather than hiding it. Do not
read the 0.8518 on a 10 s mix as "the engine is 85 % of a commercial
master" — different questions.

### The history, so you do not repeat it

- **The phase-origin bug was the whole story, and it was worth +0.8095 of
  a total +0.8437 recovery.** The analyser drew every partial on the
  *analysis window's* time axis while the renderer drew it on the
  *section's*. Each partial was rotated by `2πfΔ` — hundreds of cycles for
  high partials — so the reconstruction was uncorrelated (0.008) even
  though every individual partial was correct. The fix is
  `_rephase_to_section` in `src/res_evidence.py`.
  **The lesson: when you change the time axis of anything, check both
  ends.** An analysis-time origin and a render-time origin must agree.
- **The acceptance gate everyone suspected was worth +0.0341.** The class-C
  spectral gate in `src/res_bank.py` was blamed for the collapse and
  contributed almost nothing. **Do not guess where a bug is — bisect it.**
- **Corrections to earlier claims, so you do not inherit them.** There is
  no `decay=800/6400/25600` parameter (the maximum decay in the codebase is
  58.49). `exp(+800 × 0.01) = 2980` is impossible, because the envelope
  already uses `np.maximum(t - t0, 0)`. There is no 8× mid-band boost.
  "Three fixtures improved and two collapsed" is false — all five
  collapsed at the true baseline. The `stem_vocals` baseline is
  0.670615 → 0.433915. A label called `referee2` does not exist. The
  `baseline` scoreboard label is **not reproducible** (its worktree was
  deleted); use `baseline_head`, which was re-run in-tree from the release
  commit.

### Where the time and the memory go

A 60 s input, 8 workers: 169 s end to end — 73 s evidence, 94 s loop, 2 s
render — 0.355× realtime, 2.7 GB peak RSS. A 90 s input: 224 s. Worker
scaling: 1 worker 441 s, 4 workers 207 s, **8 workers 185 s**, 16 workers
219 s. Throughput does not improve past 8 workers on this machine, and
16 is worse than 8.

Inside the evidence stage, `refine_track` / `project` / `minimize_scalar`
account for **98.4 %** of the time. Streaming the scorer was tried and
rejected: it was only 0.9 % of the total.

---

## Where the physics headroom probably is

Ranked by my estimate of value per unit of risk. Your job is to disagree
with this ranking if the code says otherwise, and to say why.

### A1. Non-stationary partials — frequency trajectory inside a section

**Why it is first.** A vibrato tone at 0.5196 with 942 partials and a sweep
at 0.8738 are the clearest evidence that the representation is the
limiting factor. If the evidence stage fits a constant frequency per
partial per section, then any frequency trajectory is being approximated
by many stationary partials, which is both inaccurate and expensive (the
942-partial vibrato case). The renderer already has glide support (see
`src/res_loop.py` and the glide-lifter tests), and there are
instantaneous-frequency and joint-frequency-refinement tests in
`tests/`, so parts of this may exist. **Establish exactly what exists
before assuming anything is missing** — `tests/test_res_instantaneous_phase.py`,
`tests/test_res_joint_freq_refine.py` and `tests/test_res_joint_refinement.py`
are the places to start.

A per-partial trajectory model — frequency as a low-order function of
time within the section — is the natural extension: it would let one
partial express a whole vibrato instead of dozens expressing an
approximation of one.

**How to measure.** The vibrato probe (0.5196, 942 partials) is the
sharpest test: a successful change should raise its correlation *and*
collapse its partial count. A change that raises correlation while
partial count rises is not an improvement.

### A2. Percussive modelling — exciter through resonant filters

**Why.** A snare at 0.4690 and drums at 0.4107 are the weakest real
material. The hypothesis bank currently proposes three classes: harmonic
strings, percussive bundles and stochastic noise. A kick is well served by
"percussive bundle" because it is close to a decaying pitch sweep. A
snare is not: it is a *noise burst exciting a resonant body*, plus a set of
modes. The physically honest model is an exciter (impulse or short noise
burst) through a bank of resonant filters (biquads), which is modal
synthesis, not a sum of stationary sinusoids.

The bank is extensible — `src/res_bank.py` already has a class taxonomy
and a per-class acceptance path — so a fourth class is a natural
extension rather than a rewrite.

**How to measure.** The snare roll and `stem_drums` fixtures. But read the
metric carefully, because there is a measurement already in the ledger
that settles it: **drums have the *best* onset correlation of all four
stems (0.684, against 0.539 for the mix, 0.477 for vocals, 0.429 for
bass) and simultaneously the worst waveform correlation (0.411) and the
worst LSD (18.0 dB).** The engine is finding the drum hits — the timing
is right — and failing to reproduce their *shape*. Waveform correlation
penalises attack-shape error far more than it rewards correct timing, so
it is the wrong target for percussion. **Log-spectral distance and onset
correlation are the metrics to watch; waveform correlation may barely
move even when the model genuinely improves.** Report all three. If you
conclude that the referee metric is the wrong target for percussion, that
is a legitimate and important finding — see A6.

### A3. Spectral splatter from truncated partials — MEASURED AND REJECTED

**This item was measured and it does not work. Do not repeat it.**

The hypothesis: the support mask in `src/res_synthesis.py::_support_mask`
builds its attack and release tapers with `np.clip(...)`, a linear ramp,
which leaves a slope discontinuity and so leaks at 1/f^2 instead of 1/f^3.
The renderer already uses a raised-cosine crossfade for section boundaries
(`apply_ola_crossfade`), so the partial-level mask looked like the odd one
out. The claim was that every partial onset is a broadband click inflating
LSD and wasting partial budget.

**The code claims were true and the numbers were not.** Both variants
(renderer-side only, and both sides) were measured on the frozen fixtures,
10 s, 6 iterations, against a baseline that reproduced the published rows
exactly:

| metric | baseline | renderer taper only | both sides |
|---|---|---|---|
| correlation (drums / other) | 0.410701 / 0.737649 | -1.59% / -1.36% | -1.70% / -2.22% |
| LSD (drums / other) | 18.035 / 9.504 | +0.10% / +0.61% | +0.17% / +0.82% |
| partials (drums / other) | 442 / 347 | 431 / 319 | 425 / 303 |

The partial count falls, as predicted, and the fidelity falls with it. LSD
gets worse, not better.

**Why the premise fails.** The splatter is not broadband at this engine's
ramp lengths. Out-of-band energy of a 440 Hz partial, as a fraction of its
own energy: at the engine's 20 ms attack and 30 ms release the linear ramp
puts **0.008-0.011%** beyond 1/L and **0.0000%** beyond 500 Hz. A 20-30 ms
ramp is a wide window, so there is no click to remove. The 1/f^2 versus
1/f^3 distinction is correct DSP and the magnitude is negligible. The
taper shape also costs accuracy: a raised cosine is gentler at the start
than a straight line, so it delays and attenuates a genuinely sharp attack.

The full record, with the out-of-band table and the reproduction commands,
is in `docs/res_research.md` § *Phase 8*.

**The related divergence that is real.** The same "the two ends must agree"
instinct points at something that *is* measurable: the model-side mask in
`src/sinusoidal_model.py::refine_track` fades over 0.25 s while the
renderer fades over `_RELEASE_FADE_S = 0.03 s`, so the model fits 0.22 s
of tail the renderer never synthesises. `tools/audit_fade_and_shelf.py`
already measures this, and aligning the renderer's fade to 0.25 s takes
the **replica** - the correlation between the analyser's own
`sinusoid_audio` and the RES render of the same partials - from 0.9962 to
**1.0000** on the bass stem and 0.9906 to 0.9992 on the full mix, with
fidelity flat (0.5189 -> 0.5196, 0.6714 -> 0.6701). So the renderer
discards fitted content and the fix makes the render an exact replica of
the model - but it does not improve fidelity, because the binding
constraint is the analyser's model of the audio, not the renderer. The
renderer's fade is also deliberately short
(`tests/test_res_whistle_suppression.py` requires `_RELEASE_FADE_S <= 0.05`
and `< 0.25`, to suppress whistles), so the alignment is not free.

**That is the better-motivated version of this idea**, and it is open: a
renderer that discards 0.22 s of fitted tail means the equations a user
edits do not fully describe the audio they hear. Whether the answer is a
longer renderer fade, a change to the analyser's extrapolation, or an
explicit "the equation describes a partial that the renderer truncates"
annotation in the equation sheet is a real design question. Measure the
replica, not just the fidelity.

### A4. Envelope shape

**Why.** The amplitude envelope is fitted log-linearly. Real envelopes have
a fast attack, a curved decay, and often amplitude modulation. A
second-order decay model, or a piecewise envelope with a separate attack
segment, is a local change to the fitting code.

**How to measure.** The plucked-string and organ probes, where the
envelope is the whole story.

### A5. Analysis resolution and frequency estimation

**Why.** The STFT window length sets a trade-off between pitch resolution
and transient smearing, and `multi_resolution_peaks` in
`src/res_evidence.py` exists to manage it — including a high-band
allocation quota, `HIGH_BAND_HZ = 2000.0` and the reserved fraction in
`select_tracks_banded` in `src/sinusoidal_model.py`. A quota is a
heuristic; allocating the track budget by band *energy* or by *entropy* is
more principled. Separately, parabolic interpolation on a windowed peak is
second-order accurate; a higher-order estimator would be more accurate at
high frequency, where the relative error of a parabolic fit is worst.

**How to measure.** The triad fixture is the sensitive test for frequency
accuracy — it should not move at all if your change is correct, because
its ground truth is exact. If the triad moves, you have changed something
you did not intend to change.

### A6. The referee defines the product

**Why this belongs on a physics list.** The RES loop accepts or rejects a
candidate by asking the referee whether it scores better. The referee
therefore *defines what better means*, and the engine will chase it. If
the referee measures waveform correlation, the engine optimises waveform
correlation, which is the wrong target for percussion and for a
sum-of-sinusoids representation in general. The metrics live in
`src/residual.py`: SNR, log-spectral distance, segmental SNR, onset
correlation.

The physically interesting question is: **what does a sum of sinusoids
approximate well that waveform correlation cannot see?** A metric in
*parameter* space — distance between the true partial set and the
estimated partial set — can only be computed on synthetic fixtures with
known ground truth, but the fixtures are there and the triad already
provides one. A metric that rewards getting the *model* right rather than
the *waveform* right would let the engine improve in places the current
referee cannot see.

**The cost, and why it is worth stating up front.** Changing the referee
changes what the loop accepts, which changes every published number. If
you do this, the entire scoreboard must be re-run, the release notes
re-issued, and the version bumped — the published numbers are the product's
credibility, and they cannot be quietly redefined.

### A7. The residual and noise model

`fit_noise_floor_partial` in `src/res_synthesis.py` and the LPC noise
tests (`tests/test_res_lpc_noise.py`) exist. A better residual model
would let the partial budget stop being spent on noise. The white-noise
probe at 0.2342 with 821 partials is the test case: a good noise model
should let that probe use *fewer* partials.

### A8. Stereo

**Determine this first, because it may dominate everything else.** If the
engine sums to mono before analysing, then spatial information is lost
before modelling begins and every real-mix number is capped by that loss
rather than by the model. A stereo-aware engine — a shared frequency set
with per-channel amplitudes, or per-channel phase — is a large change
with a correspondingly large ceiling raise. If the engine is already
stereo-aware, drop this item and say so in the report.

### A9. The honest floor

If you conclude that a sum of stationary sinusoids is simply the wrong
representation for this material and that no amount of fitting work will
move the drums number much, **that is a legitimate and valuable result**,
and it is the kind of conclusion the project would rather have than a
suspicious improvement. The README already claims a percussive ceiling;
if you can bound it better, say by how much and by what argument, that
belongs in the report.

---

## Where the architecture headroom probably is

### B1. The `src` package name

The published wheel installs a top-level package called `src`, which
collides with any other `src` on the path and is not a name anyone should
publish. Renaming it to `m2m` is mechanical but wide: **230 import
statements across 82 files** — the sources, the tools, the tests, `main.py`
and `app.py`.

**This is a good first task for an agent**: it is large, obvious, entirely
verifiable (the suite plus a clean-wheel install plus a real job), and
provably number-preserving. It is also a trap for a careless agent, because
a partial rename leaves a broken import somewhere and the failure will
look like a DSP bug. If you do it, prove it changed no number, and use the
existing "install the wheel in a clean venv and run a real job" pattern
from the CI workflow.

### B2. No static type checking

There is no type checker in CI, and the pipeline passes dictionaries
between stages with ad-hoc keys. The data contract between stages is
implicit, which is exactly the class of bug that is expensive in a
13,285-line `src/` package. Adding annotations at the stage boundaries and
a type-check job in CI is probably the single highest-value structural
change available.

### B3. Explicit stage contracts

`src/res_evidence.py` produces evidence, `src/sinusoidal_model.py`
consumes it, `src/res_bank.py` proposes candidates, `src/res_loop.py`
iterates and the referee scores, `src/res_synthesis.py` renders. The
shapes crossing those boundaries are dictionaries. Typed records (a
dataclass, a `TypedDict`, or a small schema) at the boundaries would make
the pipeline's contract explicit and catch shape drift at compile time
rather than in a ledger.

Note the file sizes, because they are the architecture statement: the
bundle's order *is* the execution order.

| file | lines |
|---|---|
| `src/res_loop.py` | 1822 |
| `src/res_bank.py` | 1625 |
| `main.py` | 1156 |
| `src/sinusoidal_model.py` | 862 |
| `src/res_evidence.py` | 752 |
| `src/res_synthesis.py` | 508 |
| `src/residual.py` | 308 |

`res_loop.py` and `res_bank.py` are the giants. Before you split them,
establish what their actual internal seams are — the ledger's transaction
records are a good source of truth for what the loop really does at
runtime, and they are better evidence than the file's layout.

### B4. Global state and resource lifetime

`src/res_loop.py` caches worker pools in a module-level `_POOL_CACHE`
with `_get_pool` and an `atexit` handler. `src/res_evidence.py` keeps
`LAST_SEPARATION_BACKEND` as a module-level mutable global. Both are
process globals with implicit lifetimes. An explicit engine or context
object would make the lifetime visible and the teardown testable.

### B5. Determinism as a tested contract

The parallel path exists (`ProcessPoolExecutor`, the cached pools, the
per-window reconstruction in `src/sinusoidal_model.py`). There is replica
coverage in `tests/test_res_replica.py`, with a phase tolerance of 1e-5.
The property worth stating and pinning explicitly is stronger: **the same
input must produce the same output bytes, at any worker count.** A test
that runs the pipeline at 1, 2 and 8 workers and compares digests is a
contract, not a check. If it currently fails, that is a significant
finding — a nondeterministic engine cannot be debugged and cannot be
trusted with a published number.

### B6. `safe_eval` is a security boundary

`src/safe_eval.py` evaluates equation strings with an AST allowlist, and
`main.py res` has a `--provider {numeric,opencode}` option, meaning
equations can be proposed by a language model. The evaluator supports
array-valued whitelisted calls and a ternary that lowers to `np.where`.
An allowlist of numpy functions, fed strings from a language model, is a
real attack surface.

The principled fix is a **capability system** rather than a function
allowlist: a fixed vocabulary of pure operations the model may reference,
with the AST pass rejecting attribute access and arbitrary calls outright,
plus a hostile-input test suite. A capability system is easier to reason
about and easier to audit than a list of "safe" numpy functions.

### B7. Generate every published number from the ledger

This is the strongest part of the codebase and the most extensible idea in
it. `src/scoreboard.py` already holds `load`, `save`, `check_row`,
`verify` and `run_label`; `tools/verify_scoreboard.py` is a thin CLI over
it; CI runs the verifier and fails if a row was edited to a number its
ledger does not support, or if a label that must improve stops improving.

The extension: make the README numbers, the bundle header, the release
notes and any pitch material **generated from the scoreboard** rather than
typed by hand. Then no published number can drift from its evidence,
because there is nowhere to type one. This kills an entire class of
defect permanently.

### B8. A performance gate

The speed figures (169 s for 60 s of audio, 0.355× realtime, 2.7 GB peak)
are not in CI. A change could double the runtime and every quality number
would stay identical. A perf gate on a fixed short input, with a generous
threshold, is cheap and would catch that.

### B9. The CLI is a 1156-line dispatcher

`main.py` exposes 19 commands: `benchmark`, `synthesize`, `analyze`,
`verify`, `refine`, `visualize`, `extract-notes`, `tone-id`, `tone-remake`,
`sms-remake`, `res`, `res-recommend`, `math-daw`, `math-daw-demo`,
`fit-formula`, `discover-equations`, `recreate-song`, `viral-phonk`,
`run-all`. A command registry — each command a function with its own help
and arguments — would make `main.py` a dispatcher and would let the
release notes and the bundle header be generated from the registry, so
they cannot fall out of date either.

### B10. One engine, two front ends

`app.py` is a Streamlit interface and `main.py` is a CLI. `src/res_ui.py`
already holds some of the job and payload plumbing, and there are job
APIs in both. A single job service with progress, cancellation and a typed
result would let the app and the CLI drive the same path, so a bug fixed
in one is fixed in both.

### B11. Error and uncertainty states are strings

The tool prints things like "only 1 accepted iteration(s); falling back to
a size-based guess" and reports a basis of "insufficient-data". That
graceful degradation is good. But a basis expressed as a string in a
printout cannot be asserted on. Make the uncertainty a typed state the
caller can branch on and the tests can pin.

### B12. The data layout

`data/output/` mixes frozen fixtures, scoreboard labels, per-run ledgers
and one-off outputs. A clearer separation — fixtures / runs / published —
would make it obvious which files are frozen evidence and which are
scratch, and would stop a scratch output being mistaken for a fixture.

---

## How to work

### Establish the baseline

```bash
python -m pytest tests/ -q                    # 408 passed, 2 skipped
python tools/verify_scoreboard.py             # CONSISTENT=True improved=True
python -m build                               # wheel + sdist
```

Run a real job and read its ledger:

```bash
python main.py res -i data/output/_fixtures/stem_drums.wav -o out/drums -d 3 --iterations 1
```

The ledger `out/drums/res_ledger.json` records every attempt, accepted or
rejected, with its score. **Read the ledger before you read the code.** It
tells you what the engine actually did, which is frequently not what the
source suggests.

Re-run a scoreboard label after a change:

```bash
python tools/rebuild_scoreboard_label.py --label after
```

Note that the scoreboard harness drives the same CLI a user would type, so
any label is reproducible by hand as well as from the app. If you change
the engine's behaviour, re-run the affected labels; do not edit the rows.

### Make the change

One change, on a branch, with a test that pins it. A change without a test
that fails before it and passes after it has not been demonstrated — it
has only been asserted.

### Prove it

Run the full suite, the scoreboard verifier, and the affected labels.
Then report:

- the metric before and after, both from a ledger, not from memory
- the full metric set, not just the one that moved
- the partial count and the runtime, because a change that improves
  correlation by doubling the cost or the partial count is a trade, not a
  win
- whether the triad fixture moved (it should not, unless your change
  touches frequency accuracy)
- whether the *other* fixtures moved at all

### Report

Write it in this shape:

1. **Claim.** One sentence: what you changed and what it should have done.
2. **Why this was the change.** The evidence that pointed here, with the
   number that pointed at it.
3. **What you did.** The mechanism, at the level of a function or two.
4. **Measurement.** Before and after, from the ledgers, with the full
   metric set and the cost. Include the fixtures that did *not* move.
5. **What it cost.** Runtime, memory, partial count, complexity.
6. **What you tried that did not work.** The honest list.
7. **What is still open.** The next three things, ranked.

If you changed the referee (A6) or the scoreboard, say so prominently in
the report and state exactly which published numbers are now stale, because
they will need re-running and re-issuing before the next release.

---

## The infrastructure you can lean on

- **CI runs the suite on Linux, Windows and macOS**, on Python 3.10, 3.11
  and 3.12, and additionally builds the wheel, installs it into a clean
  virtual environment, and runs a real job from it. If your change passes
  locally but breaks the install, CI will tell you.
- **A failing test is diagnosable without log access.** The test step
  posts the failing test names to a commit status, readable without a
  token, and the job summary carries the pytest output. Use it.
- **`tools/build_project_bundle.py` regenerates `PROJECT_BUNDLE.md` and
  `PROJECT_BUNDLE.zip`** from the same file list, and the bundle is now
  byte-identical on every platform. If you change a bundled file, rebuild
  the bundle; if the bundle and the source disagree, the bundle is the
  artefact that misleads.
- **A release is one tag.** `.github/workflows/release.yml` checks the tag
  against the package version, builds, installs the wheel in a clean
  environment, runs a real job with it, and publishes the wheel, the sdist
  and both bundle files. You do not need to touch the release process.

## What this project values

Accuracy over appearance. A number with a ledger behind it is worth more
than a number that looks good. The engine's own reports are graded by
whether they help a person understand what happened, and a person who
catches a discrepancy loses trust in every other number the engine prints.
When you are unsure whether something improved, measure it again on a
different fixture and say what you found. When you are unsure whether a
claim is true, say you are unsure.

The scope is the method's real limits, stated plainly. The best outcome
from this work is a sharper statement of what a sum of sinusoids can and
cannot do — and, if the evidence supports it, a way past the limit that
currently binds.
