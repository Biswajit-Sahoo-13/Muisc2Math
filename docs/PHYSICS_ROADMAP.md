# The 100% Roadmap: physics and architecture for full song recreation

Status: analysis, 2026-10-23. Everything below is either measured from this
repo's own 277 ledgers (file paths cited), pinned in code (file:line), or
sourced from the literature (links). No projections without a number behind
them. Companion to `docs/res_research.md` (defect log D1–D12) and the
architecture review of 2026-10-23.

---

## Part I — What "100% recreation" can physically mean

Three different goals hide inside "100%". They have different endpoints and
different costs, and the engine currently measures none of them explicitly.

**1. Bit-exact / waveform correlation 1.0.**
Always *constructible* — any n-sample signal is exactly n/2 sinusoids (the
DFT is itself an equation representation), and greedy pursuit converges to it
(Mallat & Zhang 1993; Tropp 2007). But the operational floor is lossless
coding: FLAC carries music at ~4–8 bits/sample ≈ 90–160 kbps at 22.05 kHz
mono ≈ **3,000–5,000 float parameters per second**. The engine today carries
**~180–300 params/s** (measured, §II) — 15–25× below the exactness floor.
Corr 1.0 is therefore bought with ~20× more equations, not a better fitter.
At that point the "equation sheet" is a lossless codec competing with FLAC,
which ships bit-exact at ~50–70% of PCM size. This is the honest cost of
literal 100%.

**2. Perceptual transparency — "sounds 100% identical".**
Reachable at far lower budgets. Neural codecs hit near-transparent music
(MUSHRA ~92 vs reference 93–97) at 6–12 kbps (EnCodec, SoundStream, DAC) by
letting a learned decoder supply physics; classical parametric sines+noise
(HILN) **failed** MPEG verification on general music — winning only on
sparse/tonal material — and the documented fixes were exactly: a transient
layer, time-varying noise envelopes, and phase-locked harmonics
(Levine & Smith 1998; Serra & Smith SMS; Simionato & Fasciani 2024, whose
differentiable sines+transient+noise piano is distinguishable from real
"but not substantially" — weakest at attacks, same as this engine's drums).

**3. Compact equations that are waveform-exact on stochastic content.**
Information-theoretically impossible below the lossless floor: two
independent realizations of the same noise process are orthogonal
(E[⟨x,x̂⟩]=0), so a re-synthesized cymbal wash **cannot** sample-correlate
with the original no matter how correct its spectrum is. This is why drums
corr is structurally stuck low while its onset-corr is 0.71 (§II): the
timing is right, the texture is stochastic. Demanding corr→1 on noise is
demanding bit-exact noise realizations — i.e. goal 1.

**The separator ceiling.** Per-stem reconstruction rides on separation.
HT-Demucs ft SDR: drums 10.8 dB, bass 10.5, vocals 9.2–9.4, other 6.4 dB →
max waveform corr sqrt(1/(1+10^(−SDR/10))): **drums 0.96, bass 0.958,
vocals ~0.945, other 0.90**. Measured vocals 0.79 / bass 0.78 are already
within separation-artifact territory; drums 0.42 is *below* it — dominated
by the engine's own stochastic-modeling gap, not by separation.

**Consequences for the goal.** Split the metric, then split the target:
- tonal stems (bass, vocals, mid mix): waveform corr, ceiling ≈ separator
  ceiling ≈ 0.95–0.96;
- stochastic content (drums, >2 kHz texture, tails): multi-resolution STFT
  distance + band-envelope match + LSD (the DDSP/Parallel-WaveGAN loss
  family), where "100% perceptual" is achievable;
- literal corr 1.0: a "chase mode" (Part III, P2) that runs pursuit to
  convergence and *publishes the params-vs-corr curve* — the engine's
  rate-distortion identity — so "100%" becomes a dial with a printed price.

> **Update (2026-10-23, later):** a follow-up literature sweep
> (`docs/BEYOND_THE_LIMITS.md`) found documented breakers for each of the
> three walls in this Part — PRNG seed recovery makes bit-exact
> machine-generated noise *arithmetically* recoverable; repetition-aware and
> structure-aware coding (a 2026 lossless neural codec beats FLAC on music
> by 31–370%) dissolves the "arbitrary signal" entropy floor for real
> songs; and the separation ceiling is now ~0.975 (drums 12.91 dB,
> BS-RoFormer class) and routable via mixture-consistency fitting and
> note-layer priors. Read Part I as the conservative bound, and
> BEYOND_THE_LIMITS.md as where the walls actually stand.

---

## Part II — Where every missing decimal actually lives (measured)

Best label per fixture, from 55 scoreboard ledgers + residual analysis:

| fixture | best corr | missing | where it lives (measured) |
|---|---|---|---|
| stem_drums | 0.4194 | 0.581 | **Uniform in time** (section spread only 0.368–0.481; residual 80–100% of source RMS in every 0.5 s bin). Residual = 90.2% of source energy; **the render carries only 2.1%** (≈7× too quiet; optimal rescale still leaves 82.9%). ≥98.3% of 2–8 kHz and 99.5% of >8 kHz unexplained. Onset corr 0.71 ≫ waveform 0.41: timing captured, texture not. LSD 18.0 dB (suite worst). |
| stem_other | 0.7829 | 0.217 | 68% of 2–8 kHz + 95.8% of >8 kHz unexplained; worst onset-corr 0.354 (tempo 2× error). |
| stem_bass | 0.7839 | 0.216 | 100% of source energy is <150 Hz; 42.6% of it unexplained (band corr 0.788). 263 of 364 equations are glides. Render leaks 15× the source's tiny 150 Hz–2 kHz energy. |
| stem_vocals | 0.7904 | 0.210 | 34.6% of 150 Hz–2 kHz unexplained (98.6% of energy there); 82.2% of 2–8 kHz (sibilance) unexplained; 6.3% of residual is spurious sub-150 Hz render output. |
| mix_lost | 0.8554 | 0.145 | Time-concentrated: corr falls 0.901→0.762 across 10 s as arrangement density doubles (partials/s 29→60). |

Cross-cutting, from the same corpus:

- **Every stochastic hypothesis class has a 0% acceptance rate, ever**:
  filtered_noise 12 offered/0 accepted, lpc_noise 10/0, sibilance 7/0 across
  277 ledgers. They are off by default *and* structurally refused when on
  (Part III P0).
- **The bank is not where gains come from**: median accepted residual
  reduction 0.000% (percussive) / 0.4–0.9% (harmonic_string). Essentially all
  loop gain is the numeric coordinate lifter `_section_update`.
- **The LLM proposer is dead weight in production**: 0 edits applied in all
  332 scoreboard iterations (1 edit repo-wide, on a run that ended worse;
  other backends produced no parseable edits at 26–34 s/call).
- **Iteration budget is not binding**: 100% of runs stop on `iteration-cap`,
  yet the 8-iteration experiment gained +0.0002/iteration at iterations 7–8.
  Marginal return is flat — more iterations is not the missing dial.
- **Model size is not binding**: ~180–300 params/s ≈ 85 samples/param. The
  residual is a *model-class* deficiency, not parameter starvation.

---

## Part III — The roadmap

Ordered by (measured impact) ÷ (effort). Architecture enablers from the
2026-10-23 review are referenced where they gate a step.

### P0 — Turn on the engine you already built (days, ~zero new math)

The single most damning finding of this analysis: the engine contains its
own fixes, and they are switched off or locked out by constants.

1. **Open the spectral lane for low-correlation stems.** The lane's
   acceptance floor is absolute 0.60 (`src/res_bank.py:1239-1260`) — drums at
   0.41 can *never* use the one referee path that can accept stochastic
   models. Make the floor relative (`corr_floor_mode="relative"`, the mode
   already exists) and enable `enable_noise`/`spectral_lane` by default for
   percussive-dominant sections.
2. **Ship the exciter.** `resonant_burst_hypothesis` (the broadband strike —
   most of a snare's energy) is complete behind `M2M_EXCITER=0`
   (`src/res_bank.py:45`). Gate it per-section on detected hit density
   instead of an env var. The exciter-acceptance spectral lane (this week's
   fixed code path) is its only viable referee.
3. **Energy parity.** Correlation is scale-invariant, so the loop never
   enforces loudness: the drums render ships at 2.1% of source energy (SNR
   0.45 dB, LSD inflated). Add a per-section gain refit (one scalar, fitted
   by least squares on the accepted mix) before scoring/writing. Expected:
   large SNR/LSD gains, zero corr change, honest numbers.
4. **OLA the shipped renders.** `apply_ola_crossfade` exists, is tested, and
   every production write uses `ola=False` (`src/res_loop.py:1456,1644-1651`).
   Flip it; then replace the scalar-tail fade with a true two-block
   crossfade (the worker already returns each section's tail buffer).
5. **Decide the LLM's job.** Measured: 0 production edits. Retire the
   per-partial editor (the lifter already refits those parameters
   optimally); keep the receipt plumbing and reposition the provider as a
   *strategy selector* (choose k_notes/k_hits/lane flags/chase budgets per
   section from the evidence summary) — or drop it and simplify `run_loop`.

**P0 exit criteria (scoreboard label):** drums corr ≥ 0.55 and LSD ≤ 12 dB;
every stem SNR ≥ 6 dB; zero sections frozen while their LSD > 10 dB.

### P1 — The physics that is missing (weeks)

1. **Time-varying noise envelopes — the SMS/HILN fix, and the single
   highest-transfer change in this roadmap.** All stochastic classes carry
   *static, whole-section* envelopes (`envelope_decay=0`,
   `src/res_bank.py:984-997,1127-1141,1161-1166`). Real noise is
   transient-modulated: a fill is modeled as a stationary hiss at its
   time-average — correct spectrum, near-zero sample and envelope match.
   The plumbing already generalizes: glide partials carry
   `trajectory{t,f_hz,a}` and `_trajectories` already interpolates an
   amplitude trajectory (`src/sinusoidal_model.py:261-303`); the
   `filtered_noise` render branch needs `np.interp(t, traj.t, traj.a)`
   per band, plus a fitting side that measures per-band short-time
   envelopes from the residual (`_residual_noise_shape`,
   `src/res_loop.py:777-799`, is the germ). Converts the drums problem from
   impossible (noise orthogonality) to regression (envelope match).
2. **Wet tails — reverb, first increment.** Nothing models room energy
   today (longest decay hold: 1.5 s, `src/res_synthesis.py:174-178`). A
   "wet tail" atom is *already expressible*: filtered_noise + decay +
   unit_step renders a one-sided ringing-down band tail; nominate it from
   post-hit residual energy decay, accept on the spectral lane. True
   reverb comes in P3 as one shared FDN (~50–150 params for the whole
   song — Bai et al. 2015; late reverb *is* exponentially-decaying noise,
   so sinusoid budget spent on tails is provably wasted).
3. **Beat and detune: wire the orphaned refiner.** `refine_section_parameters`
   (TRF joint least-squares for ≤10 Hz clusters, `src/res_loop.py:1034-1211`)
   has no production caller. Detuned unisons currently cannot even be
   *nominated* (peak min-distance ≈ 32 Hz, tracker merge 25 Hz,
   `_dedupe` 1.5 Hz) — this is the bass sub-150 Hz gap and every synth
   stack. Wire it into `_process_section_task` between the lifter and the
   tournament.
4. **Shallow vibrato.** The glide gate requires >6 Hz total spread
   (`src/sinusoidal_model.py:383-398`); real vocal/strings vibrato is often
   ±1–2 Hz and is flattened to a stationary median-frequency partial with
   constant phase — wrong by construction, and vocals are the loudest
   mid-band source. Run the (new) heterodyne refinement below the gate too,
   or lower the gate where the de-chirp validates.
5. **Attack shapes.** Every fitted partial gets a fixed 46 ms linear attack
   (`src/res_synthesis.py:159-166`); sub-35 ms tonal events do not exist
   (`src/tracking.py:55`). Add an attack-time parameter (the exciter's 12 ms
   burst is the template) and phase-lock harmonic groups at onsets — the
   documented fix for pre-echo in every sines+noise system since 1998.

**P1 exit criteria:** vocals ≥ 0.85; bass ≥ 0.85; drums corr ≥ 0.65 *and*
MR-STFT distance (new metric, below) at or under the mix's; a fill played
in A/B is no longer audible as "hiss instead of hits".

### P2 — Fitting machinery (weeks–months)

1. **Chase mode — guaranteed convergence, honest by construction.** The
   tournament already *is* matching pursuit; five caps prevent it from
   converging: (a) `sec["converged"]` freeze never reset
   (`src/res_loop.py:1392-1394`); (b) global rollback discards per-section
   wins (:1591-1609); (c) quotas k_notes=2/k_hits=12; (d) bank gates
   (overfit margin, +1e-5 corr); (e) the 0.60 absolute lane floor. A
   `chase=True` run_loop variant: clear freezes on accept, accept
   per-section, escalate quotas, relative floors, stop on residual-energy
   target instead of iteration cap. This is the literal corr→1.0 dial of
   Part I §1 — every run publishes its (params, corr) pair, and repeated
   runs trace the engine's rate-distortion curve.
2. **Gradient fitting — the replica makes it possible.** Measured this
   session: `_render_partial` is pure NumPy; finite-difference gradients of
   all fitted parameters are finite and smooth (~20–90 ms/partial/gradient
   at 1 s sections). First step: the orphaned TRF refiner (P1.3) already
   *is* the analytic version — wire it. Second: an autodiff port of the
   renderer held to the existing parity bar (`tests/test_res_replica.py`:
   corr > 0.98) gives DDSP-style refinement (DDSP reached near-accurate
   resynthesis of real violin with 100× fewer parameters than waveform
   models). The 2024–25 differentiable-piano result confirms the stack
   (quasi-harmonic module + DCT transients + time-varying filtered noise,
   trained with MR-STFT losses) — nobody has yet done it for full mixes
   with closed-form equations; this engine is structurally positioned for
   it (analyser/renderer parity is the hard prerequisite, already built).
3. **Referee upgrade — measure what "identical" means.** Add the
   multi-resolution STFT loss family (L1 magnitude + spectral convergence,
   FFT sizes 64→4096; Yamamoto et al. 2019 / DDSP) and band-envelope
   correlation as first-class metrics (`src/residual.py`), reported per
   section in the ledger and gateable per lane: corr for tonal stems,
   MR-STFT + envelope + LSD for stochastic ones. Without this the
   tournament keeps rejecting perceptually-correct noise (measured: it has
   rejected 100% of stochastic candidates ever offered).
4. **Absorb separator debris instead of fighting it.** With per-stem fits,
   the mix-minus-stems residual (bleed, musical noise — the separator
   ceiling of 0.90–0.96) is exactly the stochastic debris the referee
   fights. Fit an explicit "separator residual" model on it, or report
   per-stem corr relative to the separator's own ceiling.

**P2 exit criteria:** chase mode runs to corr ≥ 0.98 on any tonal stem at
≤2,000 params/s; the params-vs-corr curve is a published scoreboard label;
ABX on 10 s clips: mix not distinguishable at P1+P2 physics with MR-STFT
referee.

### P3 — The full-fidelity ladder (months)

1. **44.1 kHz full band.** `--sr` exists (`main.py:1081`); the pipeline is
   sr-parameterized. Retune n_fft (2048→4096), max_tracks (400→800), and
   the absolute band masks (`src/res_evidence.py:228-353`). ~2× compute.
   Everything above 11 kHz is currently discarded at load — the missing
   octave of cymbal sheen and vocal air.
2. **Stereo as mid/side.** Mono is baked in end-to-end
   (`src/analysis.py:18-22` → ledger → UI). Cheapest route: run the
   untouched mono stack twice (M, S) on the stems rails, interleave at WAV
   write. Out-of-phase side content is currently *unrecoverable* — an
   information-theoretic ceiling on any stereo master.
3. **One shared FDN reverb** (8–16 delay lines, structured matrix, ~50–150
   params/song) applied to dry stems; requires the cross-section noise-RNG
   stream fix (today a stochastic partial re-rendered in the next section
   *restarts* its RNG from sample 0, `src/res_synthesis.py:376-377`).
4. **Saturation atoms.** Brickwalled masters carry clipped energy no sum of
   source partials reproduces; a memoryless Chebyshev/Wiener block per stem
   closes it. (Only worth it after reverb — wet energy dominates.)
5. **Amplitude modulation as physics, not noise.** AM is explicitly
   median-filtered away today (`src/sinusoidal_model.py:95-108,279`) —
   tremolo, ensemble swell, compressor pumping all become residual.
   Tremolo is one trajectory (reuse the glide `a` channel).

### Architecture prerequisite (from the 2026-10-23 review)

P1 adds new keys to the partial (band-envelope trajectories, wet tails,
attack times); P2 adds new lanes to the tournament and a second renderer
(autodiff); P3 adds a channel axis and an FDN block. Do **candidate #1 —
give the partial a real home** (one definition site for its schema) before
P1, and **candidate #2 — one refit engine** before P2's gradient work. The
cost of skipping them is measured: this week's referee mis-gate was a
schema/bookkeeping failure inside a five-lane 246-line function.

---

## Part IV — The honest endpoint

- **Perceptual 100% ("indistinguishable")**: achievable. The physics list is
  closed and known — noise envelopes, transients/attacks, reverb tail, full
  band, stereo, AM — each with a literature-proven fix and a measured share
  of this repo's own residual. At ~2–10× today's parameter budget
  (~400–2,000 params/s), with the split referee.
- **Corr 1.0 on tonal stems**: bounded by separation (~0.95–0.96 with
  HT-Demucs; higher only with better separation or fitting the mix
  directly). The engine is at 0.78–0.86; the gap is P1 physics, not the
  ceiling.
- **Corr 1.0 everywhere, bit-exact**: purchasable at ~3,000–5,000 params/s
  via chase mode (P2.1) — at which point the equation sheet is a lossless
  codec and its competitor is FLAC. Publish it as the top of the dial.
- **Corr 1.0 on stochastic content from compact equations**: impossible,
  provably (noise-realization orthogonality). The 100% that matters there
  is perceptual, and it is reachable.

The engine's own measurements already agree with every line of this
literature: deterministic tonal physics is excellent (main-lobe-safe search,
heterodyne de-chirp, inharmonicity, VC damping); everything stochastic is
off, static, or referee-locked. The path to 100% is not a better fitter —
it is turning on the physics that exists, adding the physics that's missing,
and then measuring "identical" the way ears do.

## Sources

- Matching pursuit convergence: Mallat & Zhang, IEEE TSP 41(12), 1993; Tropp, "Signal recovery from random measurements via OMP", 2007.
- Lossless floor: FLAC ~4–8 bits/sample — IETF cellar draft; en.wikipedia.org/wiki/FLAC.
- Parametric coding: Purnhagen & Meine, HILN (AES/MPEG-4); MPEG-4 verification tests (sound.media.mit.edu/resources/mpeg4/audio/public/w2425.html); den Brinker et al., SSC overview (EURASIP 2009).
- Sines+transients+noise: Levine & Smith, AES 1998 (mp3-tech.org/programmer/docs/aes_98_levine_smith.pdf); Serra & Smith SMS; Verma & Meng, DAFx 98; Simionato & Fasciani, "Sines, Transient, Noise Neural Modeling of Piano Notes", arXiv:2409.06513 (2024).
- Differentiable DSP: Engel et al., DDSP, ICLR 2020 (arxiv.org/abs/2001.04643).
- MR-STFT loss: Yamamoto et al., Parallel WaveGAN, arXiv:1910.11480.
- Reverb as equations: Bai, Carabias-Orti et al., IEEE/ACM TASLP 2015 (FDN); Smith, "Late Reverberation Approximations" (dsprelated.com/freebooks/pasp).
- Neural codec transparency: Défossez et al., EnCodec (arXiv:2210.13438); Zeghidour et al., SoundStream; Descript Audio Codec.
- Separation ceiling: Défossez et al., HT-Demucs (arXiv:2211.08553), MUSDB18-HQ SDR table.
- Perceptual metrics: PEAQ (ITU-R BS.1387); ViSQOL; FAD (Kilgour et al., arXiv:1812.08466).
