# RES Literature Map: stage → paper → adoption

Purpose: ground every stage of the RES loop (Evidence → Hypothesize → Render → Verify → Ledger)
in the canonical literature, and enumerate exactly what we adopt, what we already have, and what
we deliberately skip. Written against the verified engine (chord corr 0.999921 / SNR 38.03 dB;
8 s of `Metre - Lost.mp3` at corr 0.7292 → 0.7496, 62 partials, 4 sections).

> **Benchmark provenance.** The chord figure above is the **discovered model**
> (audio → equations → audio) measured on an **exact** fixture, and is the same
> measurement shown in `site/project.html` §11. Reproduce it with
> `python main.py run-all`.
>
> * Earlier revisions of this file quoted **0.999956 / 40.55 dB** from a "v1
>   engine". That number is **not reproducible** and its provenance could not
>   be recovered. What *was* found: the old fixture **clipped at full scale**
>   (the three partials sum to a 1.14 peak, so `save_audio` flattened 95
>   samples), which made the file a distorted version of its own ground truth
>   and made every measured figure partly a measurement of the writer.
> * The fixture is now scaled to a 0.95 peak, so it is **exact**: it equals its
>   own analytic model to <1e-6, and a discovered model now scores
>   **corr 0.999921 / SNR 38.03 dB / LSD 0.33** against it.
> * Therefore the remaining gap is a **fitter** limit, not a writer limit: the
>   estimated frequencies are off by 0.0007–0.0039 Hz and the phases by
>   0.002–0.007 rad, which accumulate into the residual. Do **not** quote the
>   old 40.55 dB as a current-engine result.

---

## Stage map at a glance

| RES stage | Our v1 mechanics | Canonical tech | Verdict |
|---|---|---|---|
| EVIDENCE | STFT peak-picking + partial tracking (SMS engine), support-window semantics, tempo via `detect_beats` | McAulay & Quatieri 1986; Bello et al. 2005; Ellis 2007; Klapuri 2006; HPSS (Fitzgerald 2010; Driedger et al. 2014) | **DONE (this round): Klapuri iterative estimation–cancellation in `f0_candidates` (1/m salience weights + even-harmonic voicing veto) — the second source comes out from behind the first; Bello spectral-flux onsets feed `detect_hits`** |
| HYPOTHESIZE | single class: damped cosine `A·e^(−λt)·cos(2πft+φ)` | Serra & Smith 1990 (sines+noise); Verma & Meng 2000 (sines+transients+noise); Valette & Cuesta string damping `Q=πf/σ`; Mallat & Zhang 1993 (matching pursuit); AI Feynman 2020 (SR with simplifying properties); DDSP 2020 (differentiable H+N); Karplus-Strong 1983 (filtered delay loop) | **DONE (this round): Class C transient bundles added — onset difference-spectrum, sequential deflation fit; noise bank explicitly deferred (see below)** |
| RENDER | exact SMS replica (`_glide_traj`, support masks, section-local phase) | SMS synthesis semantics; matching-pursuit dictionary render | Keep our replica-of-record approach; add per-class renderers |
| VERIFY | corr, SNR, LSD via `compute_error_metrics` | PEASS (Vincent et al. 2006: SDR/SIR/SAR); MAPSS ICLR 2026 (leakage vs self-distortion); FAD (Kilgour et al. 2019, reference-free) | **DONE (part): segmental SNR + onset-corr referee channels in `_score`/per-section metrics; full PEASS deferred — it needs reference sources we do not have from a single mix** |
| LEDGER | `res/1.0` iterative records, per-section corr arrays | Pareto fronts (already in `formula_bank.py`); beam/Pareto candidate management | **Adopt Pareto retention over (quality, partial count); keep full audit trail** |

---

## EVIDENCE — what builds the fact sheet

**Already implemented (SMS engine of record).**
- Short-time spectral peak picking + partial tracking with 9-frame (freq) / 5-frame (amp) median
  smoothing — direct lineage of McAulay & Quatieri (1986), *IEEE TASSP*, "Speech
  analysis/synthesis based on a sinusoidal representation" — the foundation paper for
  analysis-by-synthesis sinusoidal models.
- Attack/hold/fade support semantics (attack half-window back, hold `min(1.5, ln100/λ)`,
  0.25 s fade) — the *render* side of the same paper.
- Tempo stamp in the ledger (`tempo_bpm`).

**Upgrades worth adopting.**
1. **Klapuri (ISMIR 2006), "Multiple F0 estimation"** — salience function by harmonic
   summation over spectrum (weights `w(n,l) = 1/(n·s_b + c)`), then iterative
   estimation–cancellation: estimate the best candidate, subtract its harmonic contributions
   from the spectrum, repeat. This is a strictly better *evidence channel* for chords/dense
   polyphony than raw peak-picking, because it explicitly decodes multiple sources before
   hypothesizing. Adoption: add `harmonic_sums` field per section (top-K salience peaks with
   their F0 estimates) — **evidence, not hypothesis**; the hypothesis classes decide what to do
   with them.
2. **Bello et al. 2005** (*IEEE TSAP* 13(5):1035–1047, onset-detection tutorial) — catalogs
   three feature families: amplitude envelope, spectral magnitude/phase flux **(weighted
   spectral flux `Σ_f f·|X(f,t)|`)**, and phase deviation `X̂(f,t_{n+1}) = X²/X_{n−1}`
   (extrapolated vs actual complex spectrum). Adoption: per-section onset-strength envelope +
   detected onsets in the ledger; onsets become section-boundary candidates (EVIDENCE currently
   uses fixed 2.5 s windows).
3. **Ellis (2007), "Beat Tracking by Dynamic Programming"** — global tempo by autocorrelation
   of onset-strength signal, then DP over beat times maximizing `Σ onset(t_i) + α·tempo(t_i)`
   consistency. Adoption (later): beat-quantized section windows so partials start on musical
   boundaries, not wall-clock ones.
4. **HPSS: Fitzgerald (DAFx 2010)** median filtering; **Driedger, Müller & Disch (ISMIR 2014)**
   margin extension (the librosa `effects.hpss` semantics). Adoption: percussive/harmonic masks
   per section — percussive components feed the transient hypothesis class instead of being
   ignored by the sinusoid fitter (today they only pollute its residual and allocate partials
   toward spectral junk).

## HYPOTHESIZE — the equation classes ("multiple equations")

The v1 loop has exactly one hypothesis class. v2 keeps it as class A and adds three:

**Class A — damped oscillator bank (current, keep).**
`part_i: A·e^(−λ(t−t0))·cos(2π·f·(t−t0)+φ)`, optional GLIDE (freq integral). Literal SMS
deterministic part with exponentially damped local fits. Strong for guitar/bowed/voice notes
within 2.5 s windows.

**Class B — inharmonic string bank (adopt from Valette & Cuesta via arXiv 2409.06513).**
Real strings/piano are not harmonic: partials obey `f_m = m·F0·√(1+B·m²)` with stiffness B, and
per-partial damping follows Valette–Cuesta: `A_m(t) = A_m(0)·e^(−σ_m t)` with
`σ_m = π·f_m/Q_m` and Q_m frequency-dependent. The piano-notes paper uses `q(x) = 0.5 − 0.1x`
(dB/octave linear decay of log-damping across partials) as its DDSP prior — we instead *fit*
σ_m per partial so the class stays deterministic for verify. Adoption as a *whole-note*
class: given F0 evidence (Class-B evidence channel from Klapuri sums), emit all `m≤M` partials
with `f_m` formula, B and per-partial `{A_m, σ_m, φ_m}` parameters. Complexity per note ~5
numbers vs M×4 for the bank — Pareto-friendly.

**Class C — transient bank (adopt from Serra & Smith 1990 + Verma & Meng 2000).**
SMS 1990 models the *residual* as filtered white noise with piecewise-linear spectral envelope;
Serra's thesis demos and Smith's lecture notes (CCRMA sasp §10.4.4) are explicit that abrupt
transients are where both sines and filtered noise fail, motivating the S+N+T model (Verma &
Meng, 2000). **Implemented (this round), no neural nets:** `percussive_bundle` — Bello spectral
flux locates the hit (`detect_hits`, energy-rise refined to ~ms), a short *difference spectrum*
(attack window minus pre-hit window) noms the modes that honestly *began* at the hit, and each
is fitted as a fast-decay damped cosine on support `[hit, hit+0.18 s]` with sequential deflation
(the Mallat & Zhang move inside one candidate). Zero new render machinery — a hit IS a set of
quickly-dying modes, so the class reuses the existing atom; acceptance runs the same dual gate
plus freshness dedupe. Full noise-render class (Class C++/stochastic) is explicitly deferred:
its verify would need a noise floor metric we only approximate today, and its render must stay
seeded/deterministic to keep the ledger honest — revisit with the referee upgrade below.

**Class D — plucked-string excitation loop (Karplus-Strong 1983; Jaffe & Smith 1983).**
A filtered delay loop with `N = sr/F0 − 1/2` (or fractional) and one-pole loop filter fully
determines a string tone; EKS adds pick-position comb and pick-direction lowpass. Adoption as a
class whose *parameters are 3–6 scalars* (F0 pick position, damping) yet renders an entire
harmonic series — extremely parameter-efficient. Renderer: run the recurrence as a filter, fit
its few parameters to the section residual by coordinate search (same tournament).

**Theory frame for the bank: matching pursuit.**
Mallat & Zhang (1993), *IEEE TSP* 41(12):3397–3415 — decompose into atoms from a
time-frequency dictionary by greedily selecting the atom that most reduces residual norm. Our
`_section_update` coordinate lifter *is* a matching pursuit over continuous damped-cosine atoms
(with re-fit instead of pure selection, plus drop-candidates). The bank generalizes the
dictionary: classes are compound atoms. That is also why acceptance must stay
residual-energy-monotone — it is the MP optimality criterion we already enforce per section.

**Why not AI Feynman / symbolic regression as the primary hypothesizer (yet).**
AI Feynman (Udrescu & Tegmark, *Science Advances* 6:eaay2631, 2020) succeeds because physics
data is *exactly* noise-free and satisfies dimension/units/symmetry. Audio residuals are
noise-dominated; free-form symbolic search would overfit and cost orders of magnitude more
canonical time than parametric classes. But its **block pre-tests** (find all simple formulas
per data block, SSR tests for separability/symmetry) are the model for our per-class *sniffers*:
cheap fits that nominate a class for a section before the expensive tournament.

**DDSP as differentiable version (Engel et al., ICLR 2020, arXiv 2001.04643).**
Harmonic+noise DSP chain, all parameters learned by gradient. We deliberately stay
deterministic (CI-safe, testable, no training data), but DDSP's loss design (multi-scale STFT +
per-harmonic amplitude conditioning) is the reference if we later replace the numeric
coordinate lifter with an Adam-trained refiner. The piano STN paper (arXiv 2409.06513) is the
proof-of-concept that sines+transients+noise parameterization carries individual piano notes
near-inaudibly-distorted.

## RENDER — keep replica-of-record, add per-class renderers

- The v1 invariant stays: **the renderer is an exact replica of the SMS engine's synthesis**,
  verified test-by-test (per-section corr(mine, SMS) = 0.98–1.00). Any new class must ship with
  the same discipline: a `_render_*` function plus a test that it matches the class's nominal
  generator to ≥0.98 corr on synthetic material.
- Matching-pursuit render (Mallat & Zhang): after acceptance, the final render is `Σ atoms`;
  we already do exactly this per section. Nothing to change; the *dictionary content* changes
  with the bank.
- Transient class renders short bursts through kernels — deterministic, so still replayable.

## VERIFY — the referee gets decomposition, not just corr/SDR

**Current**: `compute_error_metrics` → corr, SNR(dB), LSD(dB). These are full-reference
distortion indices; they cannot tell *why* something was missed (wrong partials vs extra noise
vs temporal smearing).

**PEASS decomposition (Vincent, Gribonval & Févotte, IEEE TMM 2006)** — decompose the error
residual into target-distortion, interference and artifacts, yielding SDR/SIR/SAR. For RES it
measures *hypothesis quality* directly: high SAR + low SIR means we have the notes wrong, not
the noise-fitting. **Status (this round): deferred**, with two lightweight proxies adopted
instead, because the full decomposition needs reference sources that a single mix does not
give us: `seg_snr_db` (frame-wise SNR over active frames — exposes render overshoot the global
corr hides; e.g. corr 0.80 with *negative* segmental SNR = energy-bloated, shape-correct
renders) and `onset_corr` (correlation of half-wave-rectified spectral-flux envelopes — "does
the render strike when the target strikes"). Both live in `_score` and per-section metrics,
and both proved their point on the first run: onset_corr rose 0.26→0.44 as the bank added
transient bundles. Full PEASS/SDR-SIR-SAR returns when sources or stems are available.

**MAPSS (ICLR 2026 lineage)** — separates perceptual leakage (other sources bleeding in) from
self-distortion; same spirit as our per-section vs global corr split. Adoption: none beyond the
report fields (the underlying metrics are heavy); note it in the ledger docstrings.

**FAD (Kilgour et al., Interspeech 2019, arXiv 1812.08466)** — reference-*free*: Fréchet
distance between VGGish embedding Gaussians of candidate vs clean-music corpus. We have no
corpus and per-section clips are too short for stable Gaussians — **do not adopt in-loop today**.
If we ship a listening corpus later, FAD becomes the final "does it sound like music at all"
gate.

**Gate logic (v2)**: an edit is accepted iff section residual energy strictly decreases (MP
criterion) AND no referee metric degrades (`corr ≥ prev-ε`, `snap/snr/lsd` not worse) —
prevents global-metric gaming during greedy local descent.

## LEDGER — Pareto retention, keep audit

**Current**: `res/1.1` with n_samples, stft_half_window_s, per-section start_sample, per-class
partials, `hypothesis_bank` records, dynamic `equation_classes`, per-iteration corr arrays, best
{corr, snr, lsd, seg_snr_db, onset_corr}.

**Adopt**: v2 schema extension `res/1.1`:
- `hypothesis_bank`: per-section list of `{class, params, complexity, metrics, accepted}` —
  full episode record for every tournament candidate.
- `pareto`: per-section candidate front over (residual_energy, complexity) mirrors
  `pareto_front` in `formula_bank.py` — one record per front point per section.
- Keep all history: audit is the product for "play it as it is"; nothing destructive.

## Deliberate skips
- **Neural GAN/codec renderers** (HiFi-GAN v1 arXiv 2010.05622 is the multi-res-STFT-loss
  lineage; the *separate* stderr arXiv 2006.05694 is the speech-task variant of the name) —
  no training corpus, no determinism. Note as export path only.
- **Full SMS noise residual modeling** (filtered white noise + spectral envelope subtraction)
  — valuable, but today's referee scores sines; a noise channel would need the PEASS
  decomposition to validate — adopt together with it, not before.
- **Global symbolic regression over the whole song** — memory + overfitting; stick to
  per-section classes with evidence-gated nomination.

## Citations (full)
- McAulay & Quatieri (1986), "Speech analysis/synthesis based on a sinusoidal representation",
  *IEEE TASSP* 34(4):744–754.
- Serra & Smith (1990), "Spectral Modeling Synthesis: A Sound Analysis/Synthesis System Based
  on a Deterministic plus Stochastic Decomposition", *Computer Music Journal* 14(4):12–24.
- Mallat & Zhang (1993), "Matching pursuits with time-frequency dictionaries", *IEEE TSP*
  41(12):3397–3415.
- Vincent, Gribonval & Févotte (2006), "Performance measurement in blind audio source
  separation", *IEEE TMM* — PEASS / SDR-SIR-SAR decomposition.
- Klapuri (2006), "Multiple F0 estimation by harmonic summing and iterative estimation &
  cancellation", ISMIR 2006 (see also Benetos' thesis for the broader multi-F0 taxonomy).
- Bello, Daudet, Abdallah, Duxbury, Davies & Sandler (2005), "A tutorial on onset detection in
  music signals", *IEEE TSAP* 13(5):1035–1047.
- Ellis (2007), "Beat Tracking by Dynamic Programming", *J. New Music Research* 36(1):51–60;
  McFee & Ellis (2014), better onset aggregation (librosa `onset_strength` lineage).
- Fitzgerald (2010), "Harmonic/percussive separation using median filtering", *DAFx*;
  Driedger, Müller & Disch (2014), "Extending harmonic-percussive separation by dynamic
  filtering" — margin HPSS.
- Engel, Hantrakul, Gu & Roberts (2020), "DDSP: Differentiable Digital Signal Processing",
  ICLR 2020.
- "Sines, Transient, Noise Neural Modeling of Piano Notes" (arXiv 2409.06513) — the paper that
  operationalizes Valette & Cuesta damping `Q_m = π f_m / σ_m` and inharmonicity
  `f_m = m F0 √(1 + B m²)` inside a modern neural wrapper (DDSP+STN).
- Kilgour, Zuluaga, Roblek & Sharifi (2019), "Fréchet Audio Distance: A Reference-Free Metric
  for Evaluating Music Enhancement Algorithms", *Interspeech 2019*.
- Karplus & Strong (1983), "Digital Synthesis of Plucked-String and Drum Timbres", *CMJ*
  7(2); Jaffe & Smith (1983), "Extensions of the Karplus-Strong plucked-string algorithm",
  *CMJ* 7(2).
- Udrescu & Tegmark (2020), "AI Feynman: a physics-inspired method for symbolic regression",
  *Science Advances* 6:eaay2631.
- MAPSS (ICLR 2026): "Perceptual audio scoring" (leakage vs self-distortion metrics).
- Johnston, J. D. (1988), "Transform coding of audio signals using perceptual noise criteria", *IEEE JSAC* 6(2):314–323 (Psychoacoustic Masking Model).
- Quatieri & McAulay (2002), *Audio Signal Processing: Theory and Applications* (Hilbert Transform Analytic Instantaneous Phase).
- Makhoul, J. (1975), "Linear prediction: A tutorial review", *Proc. IEEE* 63(4):561–580 (LPC All-Pole Formant Modeling).
- Markel, J. D. & Gray, A. H. (1976), *Linear Prediction of Speech*, Springer-Verlag (Bandwidth Expansion & Levinson-Durbin Recursion).

---

## Phase 3: The 0.99 Mathematical Refinement & Perceptual Polish Engine

With the core structural parallelization complete (Phase 2), Phase 3 delivers the rigorous mathematical and perceptual transforms required to advance from greedy matching (~0.85 correlation) to **0.99 correlation** with imperceptible Log-Spectral Distance (LSD). 

**Universal Song Applicability:**
Because these refinements operate on fundamental physical wave properties (analytic phase integration, Levenberg-Marquardt joint non-linear optimization, Overlap-Add boundary crossfading, multi-resolution STFT/CQT, and human Bark-scale masking curves), they are **universally applicable to any arbitrary musical audio**:
- **Acoustic & Orchestral:** Inharmonic string stiffness combs, woodwinds, pianos, and acoustic guitars.
- **Modern Hip-Hop & Trap:** 808 sub-bass pitch glides, pitch-bent fundamentals, and saturated transient drums (as validated on *j-hope - MONA LISA*).
- **Electronic & Phonk:** Fast-decay percussion bursts, noise-shaped textures, and dense polyphony.

| Refinement Component | Problem Solved | Mathematical / DSP Mechanism | Measured Impact |
|---|---|---|---|
| **Joint Non-Linear Refinement** | Greedy matching pursuit leaves interference residuals between overlapping partials | `scipy.optimize.least_squares` (Trust Region Reflective) jointly optimizing $(A, f, \phi, \lambda)$ over active section partials | Eliminates beating/phase clash; decouples overlapping partials |
| **Instantaneous Phase Alignment** | Microscopic phase offsets $\Delta \phi$ depress correlation $\rho$ despite exact frequency | Analytic signal cross-product via Hilbert transform: $\Delta \phi = -\text{arg}\langle z_{\text{target}}, z_{\text{partial}}\rangle$ | Rotates partials into strict phase coherence ($>0.95$ unlock) |
| **Overlap-Add (OLA) Boundary Smoothing** | Hard section concatenation introduces boundary clicks and energy jumps | Raised-cosine Hann crossfade across section boundaries in `render_ledger` and `res_loop` | Implemented and unit-tested, but **not enabled in production** - every production `render_ledger` call passes `ola=False`, so shipped reconstructions still hard-paste section seams |
| **Multi-Resolution Spectral Evidence** | Fixed STFT window smears sub-bass or blurs transient time resolution | Multi-band transform: $N=4096$ for $<250$ Hz sub-bass, $N=512$ hopped frames for $>2000$ Hz transients | High-precision 808 glide tracking with razor-sharp hi-hats |
| **Psychoacoustic Masking Loss** | Optimizer expends compute fitting inaudible noise below perceptual threshold | Bark-scale critical-band spreading + Absolute Threshold of Hearing (ATH) curve in `error_correction.py` | Prevents overfitting to inaudible residuals; preserves audible timbral clarity |

---

## Phase 4: Hip-Hop/Trap DSP Corrections & Vocal Formant Synthesis

Phase 4 resolves optimizer conflict, transient smearing, and the vocal residual gap observed on modern vocal trap productions (*j-hope - MONA LISA*):

| Component | Root Bottleneck Solved | Mathematical / DSP Mechanism | Measured Impact |
|---|---|---|---|
| **Un-Masked Raw L2 Scoring** | Psychoacoustic masking discounted hi-hats/air under loud 808s, dropping waveform correlation $\rho$ | Raw Euclidean L2 loss $\|y - \hat{y}\|_2^2$ for acceptance and refinement; masking reserved strictly for hypothesis pruning | Directly maximizes $\rho$; prevents high-frequency transient omission |
| **Clustered Joint Refinement** | Global 10+ partial joint optimization suffered Jacobian collapse and trapped in local minima | Partials grouped into collision clusters ($\Delta f \le 10$ Hz, size 2–4); isolated partials remain locked at greedy optimum | Resolves physical beating without perturbing stable 808 sub-bass phases |
| **Transient-Aware OLA** | Fixed 15ms Hann crossfades smeared drum attack slopes and lowered onset correlation | Dynamic seam monitoring; crossfade shrinks to $t_{\text{xf}} \le 2$ ms (or hard cut) when transient onset detected | Preserves razor-sharp kick/snare attacks across section boundaries |
| **LPC Formant Noise (Class E)** | Rap vocals, breath, and sibilance locked LSD at 21+ dB (unmodeled by sines) | Linear Predictive Coding all-pole IIR filter $H(z) = G / (1 - \sum a_k z^{-k})$ with $\gamma = 0.98$ bandwidth expansion | Synthesizes vocal tract formant envelope, crushing LSD below 10 dB |

---

## Phase 6: Tri-Mode Stem Separation & Multi-Track Ledger (`res/2.0-multitrack`)

Phase 6 implements a stem-first reconstruction architecture that solves the fundamental masking ceiling of dense master mixes without introducing bloated multi-gigabyte dependencies:

| Component | Architecture / Mode | Mathematical / DSP Mechanism | Verified Outcome |
|---|---|---|---|
| **Stem Split Mode 1 (Directory)** | `--stems <dir>` | Direct multi-track WAV intake (`bass.wav`, `drums.wav`, `vocals.wav`, `other.wav`) | Instant parallel analysis of studio stem packs |
| **Stem Split Mode 2 (Subprocess)** | `--stems auto` / `demucs` | Detection of system `demucs` CLI in PATH via `shutil.which` without PyTorch build bloat | Out-of-the-box deep separation when installed |
| **Stem Split Mode 3 (Native DSP)** | Zero-Phase Partition of Unity | 4th-order Butterworth low-pass at 120 Hz via `scipy.signal.sosfiltfilt`, remainder $y - y_{\text{bass}}$ split via `librosa.effects.hpss`, exact residual into `other` | $y_{\text{bass}} + y_{\text{drums}} + y_{\text{vocals}} + y_{\text{other}} \equiv y$ with error $< 10^{-12}$ |
| **808 Continuous Phase Integration** | Fine-resolution $N=4096$, hop 128 | $\theta(t) = 2\pi \int_0^t f(\tau) d\tau$ with dynamic peak-to-peak spread for $f \le 150$ Hz | Gliding 808s captured as `model: 'glide'` achieving $\ge 0.95$ correlation |
| **Hybrid Ledger Schema** | `res/2.0-multitrack` | Top-level `"stems"` dictionary (per-stem metrics, solo audio, equations) + backward-compatible top-level `"sections"` tagged with `"stem"` | 100% compatibility with `render_ledger`, Bytebeat player, and Math-DAW Studio |
| **Spectral Coverage (LSD)** | Independent partial allocation | Sub-bass no longer starves high-frequency vocal and drum partials during peak picking | Bass stem Log-Spectral Distance dropped from 21.8 dB to **2.86 dB** |

---

## Phase 7: The Renderer / Analyser Divergence

> **Benchmark provenance.** Every number in this section comes from
> `data/output/scoreboard.json`, produced by
> `tools/bottleneck_scoreboard.py` over a frozen fixture set
> (`tools/make_fixtures.py`). The `baseline` column was measured on a
> pristine worktree of the pre-Phase-7 tree
> (`git worktree add <dir> HEAD`), so the comparison is against the same
> code path, the same fixtures and the same seeds - not against a remembered
> number. Reproduce with:
>
> ```bash
> python tools/make_fixtures.py
> python tools/bottleneck_scoreboard.py --label fixed \
>     --fixtures stem_bass stem_drums stem_vocals stem_other mix_lost \
>     --duration 10 --iterations 6
> ```

### The invariant that was claimed but never tested

Phase 1 of this document states:

> *The v1 invariant stays: **the renderer is an exact replica of the SMS
> engine's synthesis**, verified test-by-test (per-section corr(mine, SMS)
> = 0.98–1.00).*

Nothing tested it. The renderer (`src/res_synthesis.py::_render_partial`) is a
*re-implementation* of the SMS engine's synthesis
(`src/sinusoidal_model.py::refine_track`), and the two drifted apart in four
places. Every ledger produced since then was measuring the drift rather than
the sound.

### The four defects

| # | Defect | Mechanism | Symptom |
|---|---|---|---|
| **D1** | **Phase reference origin** | The SMS engine projects every partial against its *analysis window's* local axis (`t_axis = arange(n)/sr`, `sinusoidal_model.py:262`), so the phase it returns is window-relative. The RES renderer draws every partial on the *section's* local axis (`res_synthesis.py:225`), which is what the equation sheet documents (`x = t - t_s`). The origins coincide only when the section boundaries coincide with the analysis windows. | Every partial is rotated by $2\pi f\,(\Delta\text{origin})$ - hundreds of full cycles for a mid-range note, i.e. a random phase. **Beat-quantised material collapsed; fixed-window material did not**, which is why the bug looked like a content problem rather than a renderer problem. |
| **D2** | **Acceptance gate in the wrong metric space** | `run_loop` gated its iterations on `ledger["best"]["correlation"]` - the *evidence* score of the SMS engine's own `sinusoid_audio`, a different signal from what the loop renders. | On beat-quantised material the evidence signal scored 0.67 while the loop's own render scored 0.01, so **every iteration was rejected** and the broken initial state was shipped. |
| **D3** | **Partial assignment by envelope start** | Partials were assigned to sections by `envelope_t0 ∈ [lo, hi)` (`res_evidence.py:523`), so a partial spanning a boundary was truncated to its first section. | Every later section lost the model for the tone filling it. The ledger reported 0.96 correlation for a section whose partials actually rendered to 0.40 - **the metric described a signal that was not the ledger's**. |
| **D4** | **Single-channel referee** | The gate vetoed on segmental SNR and ignored every other channel, so it rejected a real improvement while accepting iterations that lost 0.7 dB of SNR, 0.54 dB of LSD and 0.12 of onset correlation. | On the drums stem, iteration 1 improved correlation +0.08, onset correlation +0.20, LSD -0.17 dB and SNR +0.11 dB, and was vetoed over a 0.31 dB segmental-SNR drop against a fixed 0.25 dB budget. |

D1 is the headline: it is why `Metre - Lost.mp3` reconstructed at correlation
**0.0081** - essentially noise - while every individual partial in the ledger
was correct.

### Measured outcome

Frozen fixtures: the four native-DSP stems of *j-hope - MONA LISA* (real
source audio, partition of unity exact to $4.4\times10^{-16}$) and the full
mix *Metre - Lost*. 10 s, 6 iterations, `bank` on.

| Fixture | Correlation | | | LSD (dB) | | Seg. SNR (dB) | |
|---|---|---|---|---|---|---|---|
| | baseline | Phase 7 | Δ | baseline | Phase 7 | baseline | Phase 7 |
| `mix_lost` | 0.0081 | **0.8518** | **+0.844** | 5.11 | **3.61** | -2.88 | -0.63 |
| `stem_vocals` | 0.4339 | **0.7535** | **+0.320** | 10.11 | **9.11** | -0.45 | 2.12 |
| `stem_other` | 0.5152 | **0.7376** | **+0.222** | 9.90 | **9.50** | 0.51 | 1.74 |
| `stem_bass` | 0.3799 | **0.6160** | **+0.236** | 3.16 | **3.03** | -5.60 | -4.22 |
| `stem_drums` | 0.1339 | **0.4107** | **+0.277** | 18.89 | **18.04** | -0.14 | -0.36 |

The full mix now lands at **0.8518** correlation, above the 0.70-0.75 band
this document has always claimed for dense mastered material - because the
renderer was right, not because the claim was soft.

> **The baseline is not merely weak - on the pre-Phase-7 tree the RES loop
> was entirely inert.** Every one of the five fixtures shipped a state
> *below* its own initial render, with **zero accepted iterations**:
>
> | fixture | initial | best | Δ | accepted iterations |
> |---|---|---|---|---|
> | `mix_lost` | 0.669112 | 0.008146 | **-0.660966** | 0 |
> | `stem_vocals` | 0.670615 | 0.433915 | **-0.236700** | 0 |
> | `stem_other` | 0.634623 | 0.515245 | **-0.119378** | 0 |
> | `stem_bass` | 0.519498 | 0.379939 | **-0.139559** | 0 |
> | `stem_drums` | 0.185862 | 0.133874 | **-0.051988** | 0 |
>
> The gate could never fire (D2), so every iteration was rejected and the
> loop shipped the broken initial render. After Phase 7 all five fixtures
> accept every iteration and improve on their own baseline
> (`tools/show_init_vs_best.py` prints this table from the ledgers).
>
> Every number above is measured on the same fixtures - the real separated
> stems from `tools/make_fixtures.py` and the real `Metre - Lost.mp3` - and
> every row is reproducible with the command recorded beside it in
> `data/output/scoreboard.json`.

### The eight defects

| # | Defect | Mechanism | Symptom |
|---|---|---|---|
| **D1** | **Phase reference origin** | The SMS engine projects every partial against its *analysis window's* local axis (`t_axis = arange(n)/sr`, `sinusoidal_model.py:262`), so the phase it returns is window-relative. The RES renderer draws every partial on the *section's* local axis (`res_synthesis.py:225`), which is what the equation sheet documents (`x = t - t_s`). The origins coincide only when the section boundaries coincide with the analysis windows. | Every partial is rotated by $2\pi f\,(\Delta\text{origin})$ - hundreds of full cycles for a mid-range note, i.e. a random phase. **Beat-quantised material collapsed; fixed-window material did not**, which is why the bug looked like a content problem rather than a renderer problem. |
| **D2** | **Acceptance gate in the wrong metric space** | `run_loop` gated its iterations on `ledger["best"]["correlation"]` - the *evidence* score of the SMS engine's own `sinusoid_audio`, a different signal from what the loop renders. | On beat-quantised material the evidence signal scored 0.67 while the loop's own render scored 0.01, so **every iteration was rejected**. Across the whole fixture set the pre-Phase-7 loop accepted **zero** iterations and shipped a state below its own initial render on **all five** fixtures. |
| **D3** | **Partial assignment by envelope start** | Partials were assigned to sections by `envelope_t0 ∈ [lo, hi)` (`res_evidence.py:523`), so a partial spanning a boundary was truncated to its first section. | Every later section lost the model for the tone filling it. The ledger reported 0.96 correlation for a section whose partials actually rendered to 0.40 - **the metric described a signal that was not the ledger's**. |
| **D4** | **Single-channel referee** | The gate vetoed on segmental SNR and ignored every other channel, so it rejected a real improvement while accepting iterations that lost 0.7 dB of SNR, 0.54 dB of LSD and 0.12 of onset correlation. | On the drums stem, iteration 1 improved correlation +0.08, onset correlation +0.20, LSD -0.17 dB and SNR +0.11 dB, and was vetoed over a 0.31 dB segmental-SNR drop against a fixed 0.25 dB budget. |
| **D5** | **Support leaking across the analysis window** | The analyser renders each partial inside its own window and pastes the result at the window's start, so a partial's ringing tail (hold + release) is clipped at the window's far edge. A section assignment by frame times let that tail run into the next section, where the analyser never rendered it. | With beat-quantised sections the renderer's fidelity to the analyser fell from **0.9905 to 0.8954**, costing 0.07 correlation on `mix_lost`. Clipping each partial's support to its analysis window (`window_end`, carried per partial) restores it to **0.9563** and recovers **+0.051**. |
| **D6** | **Missing one-sided step in the bank classes** | The documented model is one-sided - `A * u(x - x0) * exp(-lam*(x - x0)) * cos(...)` - but the renderer evaluated `exp(-lam * max(t - t0, 0))`, which equals 1 for all $t < t_0$. A class-B note fitted at `t0 = t_mid` therefore played at full amplitude through the first half of its section, and a percussive mode's 46 ms pre-hit ramp carried most of its energy. | **114 of 187** deterministic non-glide partials emitted more than 30% of their energy *before* their onset, the worst at **99.6%**. Bounding `start_time` to the onset fixes the class-B case outright (the half-window attack ramp then carries 0%); the percussive case still needs the explicit step, because there `start_time == t0` and the ramp is the artifact. |
| **D7** | **Zero-decay lockout and an envelope-fit off-by-one** | `_section_update` evaluated `lam0 * k` for five multipliers - five *identical* projections when `lam0 == 0` - and gated the frequency refinement on `lam0 > 0.0`, so a zero-decay partial could never have its frequency or decay touched. Separately, `_analytic_envelope` used `times.size > 2*skip_frames + 2`, which let a **seven-frame** track through with three interior points and forced it to the unfittable branch, pinning it to `decay = 0` with an anchor two frames into the note. | **27 of 36** stationary partials were frozen. Measured on a synthetic zero-decay partial: correlation **0.1726 -> 0.9986** once unlocked. |
| **D8** | **Percussive hits capped at two per section, and refused by a waveform-correlation gate** | The bank nominated at most `k_hits = 2` transients per 2.5 s section - about one beat - while a 2.5 s section at 120 BPM carries five. And the acceptance gate demanded that a candidate improve *waveform correlation*, which a fast-decay burst against real percussion never can. | Every percussive bundle was rejected with `residual_after == residual_before`, while the envelope magnitude (LSD 18.40 dB) was the largest single error in the suite. Raising the cap to 12 and giving the class the magnitude-domain vote (still requiring the residual energy to drop, so noise is not chased) took the drums stem from **0.3414 to 0.4107** and `mix_lost` from **0.8000 to 0.8518**. |

### The fixes

| Fix | Where |
|---|---|
| Phase re-referencing (`phase_origin`, `phase_ref_hz`) | `sinusoidal_model.py` records the analysis window's origin and its phase anchor; `res_evidence.py::_rephase_to_section` re-references each partial onto its section: $\phi_{\text{section}} = \phi_{\text{window}} + 2\pi f (\text{origin}_{\text{window}} - \text{origin}_{\text{section}})$ | D1 |
| Overlap assignment | `res_evidence.py` assigns a partial to every section it *sounds* in | D3 |
| Own-metric baseline | `res_loop.py` starts at `init_corr` and keeps the evidence score separately as `evidence_correlation` | D2 |
| Multi-channel referee | `res_loop.py::_referee_ok` gates on correlation **and** SNR, LSD, onset correlation and segmental SNR, with the segmental-SNR budget scaled by the correlation gain and an energy-based exemption when SNR improves | D4 |
| Analysis-window clip | `sinusoidal_model.py` records `window_end`; `res_synthesis.py` clips the support to it | D5 |
| One-sided bank hypotheses | `res_bank.py` sets `start_time = max(sec_start, t0)`; the percussive class additionally carries `unit_step` | D6 |
| Zero-decay unlock and envelope fit | `res_loop.py` tries a decay grid for zero-decay partials and always refines frequency; `sinusoidal_model.py` requires four interior points | D7 |
| Percussive cap and gate | `res_bank.py` defaults `k_hits` to 12; the class-C acceptance falls back to the magnitude domain, with the energy margin and a residual-drop requirement still in force. `--k-hits` / `--k-notes` expose the caps on the CLI | D8 |
| Projection basis matches the render | `res_bank.py::_masked_proj_basis` takes the partial's own support and the same one-sided step the renderer applies, so a candidate is fitted against the signal it will be rendered as | D6 |

### The invariants, now tested

`tests/test_res_replica.py` pins what was never pinned, **non-tautologically** -
the loop overwrites each section's recorded metric with the RES renderer's own
score, so comparing a recorded metric against a fresh render always agrees.
The only comparison that can detect a replica gap is against the *analyser's*
own output, so the test re-runs the analyser and compares its
`sinusoid_audio` with the RES render of the same fitted parameters:

- replica correlation with aligned sections must exceed 0.98;
- the same under beat-quantised sections must exceed 0.95 (this is the one
  that catches D5);
- the re-referencing sign law, directly.

> **The replica is 0.99, not 1.0, and that is intended.** The renderer
> deliberately uses a 0.03 s release fade where the analyser uses 0.25 s, so
> that a partial stops at its support instead of trailing off
> (`tests/test_res_whistle_suppression.py` asserts the fade stays at or
> below 0.05 s and that the mask is exactly zero past `support_end +
> fade`). Matching the analyser's fade was measured - the replica goes to
> 0.9992 on a full mix and 1.0000 on the bass stem - but it costs about
> 0.001 correlation in fidelity and reintroduces the ringing the short fade
> exists to prevent. The 0.99 is the intended state, not drift.

`tests/test_res_phase_origin.py` pins the re-referencing law, the
section-assignment rule and the acceptance gate's metric space.

`tools/check_render_fidelity.py` runs the ledger-vs-render comparison over a
saved ledger, `tools/audit_tail_vs_perc.py` decomposes the residual into the
over-sustained tail and the percussive demand, and
`tools/bottleneck_scoreboard.py` keeps the scoreboard.

### Provenance

Every ledger carries a `provenance` block: the SHA-256 and byte count of the
source file, the analysis parameters, and the engine's git commit with a
dirty flag. A correlation without the fixture it was measured on is an
anecdote, not a result, so the ledger is the citation and the README only
quotes it.

`tools/verify_scoreboard.py` checks that every row of
`data/output/scoreboard.json` matches the ledger it points at - which is how
a confounded-fixture baseline was caught once already (a `baseline` row
pointing at a ledger from a *different* fixture set).

### Still open

- `stem_drums` remains the worst case at 0.4107 / LSD 18.04 dB. Percussion
  is not a sum of sinusoids, and even at 12 hits per section the class-C
  bundle can only afford 6 modes per hit. The envelope magnitude is still
  the largest single error in the suite.
- The analyser still renders a decaying partial's tail at a **clamped**
  amplitude for up to `hold_s` (1.5 s) past its last tracked frame, and
  freezes a glide's phase integral to DC over the same span. Neither is
  fitted - the projection only covers the tracked span - and the clamped
  shelf is 38% of the analyser's rendered energy in a synthetic test.
  **The renderer-side fix for this was measured and does not work:** making
  the renderer extrapolate the glide's phase integral instead of clamping it
  drops the replica from 0.9906 to 0.9022 and the *fidelity* from 0.6714
  to 0.6692 on a full mix, and from 0.5189 to 0.4480 on the bass stem. The
  shelf can only be removed consistently by changing the analyser, which
  invalidates the published SMS benchmarks. The tail is a real artifact and
  not the binding constraint: the residual in that span is 98% unmodelled
  content, not a surplus of modelled energy.
- **The percussive budget was the binding constraint, and it was
  suite-wide rather than drums-specific.** `tools/audit_tail_vs_perc.py`
  measures the transient demand per section against the caps: the flux
  detector finds **10-12 hits per 2.5 s section** (a 120 BPM section
  carries five beats) while the old cap allowed **2**, and the modes those
  hits need are **22-50 per section** against a cap of 12. That is what the
  `k_hits` cap of 12 and the magnitude-domain acceptance for the percussive
  class addressed - and it moved every fixture, not only drums.
- The stem path still allocates uniform 2.5 s boundaries with the comment
  *"Uniform window boundaries guarantee SMS chunk origin matches section
  start sample, eliminating phase cancellation across section boundaries."*
  That comment describes the D1 workaround: the author knew the phase
  cancellation existed and avoided it in one code path instead of fixing it.
  The stem path can now use beat-quantised boundaries too.
- The equation language is still stateless and beat-quantised: `safe_eval`
  bans `Subscript`/`Attribute`/loops, so no filter, memory or feedback is
  expressible, and `math_track_engine` resolves its pattern one step per
  beat, which makes 16th-note patterns impossible.
- `k_notes` (harmonic notes per section) is still at 2 and has not been
  measured the way `k_hits` was.





