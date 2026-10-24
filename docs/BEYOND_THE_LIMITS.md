# Beyond the limits: the "impossible," made possible

Companion to `docs/PHYSICS_ROADMAP.md` (2026-10-23). That roadmap stated
three walls. A sweep of the research literature (cryptanalysis, codec
theory, psychoacoustics, separation SOTA — sources below) found documented
breakers for all three. This file states each wall, what breaks it, the
evidence, and what it means for this engine.

---

## Wall 1 — "Noise cannot be waveform-exactly reconstructed from compact
parameters" (orthogonality of independent realizations)

### Breaker 1A: much "noise" is not random — it is PRNG output, and PRNG
state is algebraically recoverable

The orthogonality argument assumes noise with ~infinite entropy rate.
Drum-machine and synth noise has **zero conditional entropy given its
state**:

- The **Roland TR-909 noise generator is a 32-stage LFSR** (shift registers
  IC32/IC33); the **NES APU noise channel is a 15-bit LFSR**. LFSR
  feedback polynomial + state fall to **Berlekamp–Massey from ~2n output
  bits** — decades-old arithmetic. An 808/909 hi-hat/cymbal track collapses
  to {taps, clock rate, trigger offsets, filter bank} — Pearson **r = 1.000,
  bitwise-exact**, in pure NumPy.
- LCG seeds: linear algebra, or LLL lattice reduction for truncated outputs
  (Fouque et al. 2013). NumPy's legacy MT19937: untemper 624 consecutive
  outputs → full state.
- Synthetic-vs-acoustic noise is classifiable (AI-audio detection
  benchmarks), so the engine can decide *when* seed recovery applies.

**For the engine:** a "generator hypothesis" class — attempt LFSR/LCG fits
on residual noise segments; when a fit locks, the residual is explained
*exactly* by (generator id, seed, clock, envelope). The engine already
stores seeds for its own noise renders; the cross-section RNG restart bug
(`res_synthesis.py:376-377`) must be fixed regardless — it is the same
determinism this breaker exploits. Their own fixtures are pop/hip-hop —
exactly the genre where hats/snares are machine-generated.

### Breaker 1B: perceptual interchangeability of noise realizations — the
psychoacoustic proof

McDermott & Simoncelli (Neuron 2011): synthetic textures from **fresh noise
matched in ~4 classes of cochlear statistics** are identified by listeners
as well as the real recordings (~90% both). McWalter & McDermott (Nature
Comms 2019) and TexStat (2025, differentiable) extend it. **Corr → 0 with
matched statistics is perceptually indistinguishable** for washes, tails,
ambience.

**For the engine:** the P1 noise-envelope work should target these
statistics (envelope moments, cross-band envelope correlations, modulation
power) as the noise atoms' fitting objective — a *differentiable* referee
lane, not just per-band RMS.

### Breaker 1C: waveform-exact by construction — parametric + lossless
residual

Keep the analytic model; entropy-code `x − model(x)`. The parametric layer
removes the deterministic bits; the stochastic remainder costs its entropy:
**~7.7–8.6 bits/sample for 16-bit music even for a 2026 LM-based lossless
codec** (18% better than FLAC); at 24-bit LSB noise, FLAC is already
Rice-optimal. Standardized lineage: HILN 4 kbps (this engine's exact
decomposition, 1999); SBR halves HE-AAC bitrates by regenerating the top
octane parametrically; CELT/Opus fills collapsed bands with envelope-coded
noise.

**For the engine:** a "chase + code the remainder" mode is the guaranteed
corr-1.0 product: equations for the deterministic structure, a FLAC-class
residual stream for what remains, priced honestly per second.

### Honesty corner

Neural implicit representations do **not** break this wall: Siamese-SIREN
audio ≈ 243 kbps with ViSQOL 2.1 — overfitting repackages entropy, it does
not beat it. Phase-retrieval uniqueness theorems (STFT magnitude determines
generic signals up to global phase, Jaganathan et al.) help tonal content
only — noise phases are i.i.d. uniform. And generative codec priors
(WavTokenizer MUSHRA 96.1 at 0.9 kbps vs DAC 92.8 at 9 kbps; MuCodec 0.35
kbps stereo) do not recover the original realization — they *hallucinate a
plausible one*, which is Breaker 1B at industrial scale.

---

## Wall 2 — "Bit-exact music needs ~4–8 bits/sample (the FLAC floor,
~3–5k params/s)"

### Breaker 2A: songs are programs, not signals

The floor is the entropy of a waveform *treated as an arbitrary signal*.
Songs are outputs of small generative programs (score + patches + sample
bank + mix chain), and conditional on that structure their description is
10–1000× smaller:

- **MPEG-4 Structured Audio** (ISO 1999) standardized music as
  orchestra+score code: **0.01–10 kbps**.
- **Demoscene**: a full 3-minute song in a 4 KB executable (Ergon, 4klang —
  ~0.2 bits/s); bytebeat minutes from <100 characters.
- Knuth, *The Complexity of Songs* (1977) — the joke formalized.

### Breaker 2B: repetition is redundancy, and it is codable

- **ACER** (Cunningham & Grout 2009–2019): LZ-style dictionary coding of
  audio — repeated sections become pointers; listeners found no perceptible
  difference vs PCM/AAC on repetitive material.
- **Trilobyte (2026): a lossless neural codec that beats FLAC on music** —
  +31% on MusDB18 at 16-bit, **+370%** at 8-bit — precisely because the
  predictor learns musical repetition. (At 24-bit LSB noise FLAC wins
  again: the wall's honest residue.)

**For the engine:** the equation sheet is already a parameter stream —
detect repeated parameter blocks (verse/chorus, loops) and store once +
references. Bit-exact where repeats are exact, and it compounds with
everything else.

### Breaker 2C: sample-based music is event-described, not sample-described

32–38% of chart hip-hop literally contains reused audio (Tracklib/Statista);
sample detection is at mAP 44.2% (ISMIR 2025) and robust to effects. A
track built from sample playback is **bit-exactly regenerable from
(one-shot bank + trigger grid + per-hit params)** — description length
scales with events (~100s of numbers), not samples (~10M).

**For the engine:** fingerprint-cluster its own fitted/rendered segments
(the repo already has `toneprint.py`, a Shazam-style fingerprinter) into an
implicit one-shot bank; re-express drum tracks as (sample-id, onset, gain,
pitch) grids. For their hip-hop/pop fixtures this may be the single largest
compression+corr win available.

### Breaker 2D: transcription + resynthesis — for electronic music, audio
is a function of few parameters

Piano transcription onset F1 96.72% (Kong et al. 2021); MT3/YourMT3+
multi-instrument; MIDI-DDSP round-trip **statistically indistinguishable
from ground truth** in listening tests; synth-parameter estimation from
audio exists (InverSynth; differentiable-DSP sound matching, ISMIR 2021;
2026 CMA-ES subtractive matching). Deterministic synths are exactly
invertible in principle — only estimation error keeps it "near-".

**The open gap (the prize):** no published system reconstructs a **DAW
project from a final mix** — all components exist separately (AMT, sound
matching, blind effect estimation — Barchiesi & Reiss "Reverse Engineering
of a Mix" 2010; Martinez-Ramirez mixing-style transfer 2023; blind AFX
estimation 2024). An equation engine with a discrete event layer is
structurally the right machine to claim it.

---

## Wall 3 — "Separation caps per-stem correlation at ~0.95"

### Breaker 3A: the ceiling was a 2022 checkpoint, not physics

BS-RoFormer L=12 + extra data (SDX'23 config), MUSDB18-HQ median SDR:
**drums 12.91 dB → ρ 0.975, bass 13.32 → 0.9775, vocals 12.72 → 0.974,
other 9.01 → 0.943**. Band-split architectures (BSRNN → BS-RoFormer →
Mel-RoFormer) are what lifted drums from 8.2 → 12.9 dB — overlapped
subbands give resolution where cymbals live. Drums is no longer the hardest
stem — **"other" is**.

**For the engine:** the fitter should be fed BS/Mel-RoFormer-class stems
(ZFTurbo training ecosystem), not HT-Demucs; and per-band residual
weighting in its own loss is the same insight natively.

### Breaker 3B: mix-minus consistency beats independent stems

The SDX'23 winner's "other" came from **subtracting the other three stems
from the mix** — 9.01 dB vs 7.93 for a dedicated separator. Stems-sum-to-mix
is a constraint an equation engine can impose natively (fit against the
mixture residual; per-stem models share the budget). Literature proof of
concept: unsupervised separation by fitting differentiable synthesis models
directly to the mixture (arXiv:2201.09592).

### Breaker 3C: transcribe-then-render — notes have no waveform ceiling

YourMT3+/SheetSage2-class systems turn mixes into note events/chords/
structure — exact analytic parameters with zero separation error. GuideSep
(ISMIR 2025) already conditions separation on MIDI. For this engine: a
discrete note layer (from AMT) as prior equations, continuous atoms fitted
on the perceptually-weighted residual below.

### Breaker 3D: change the objective the ceiling is computed on

The field's own 2026 verdict (Music Source Restoration Challenge, ICASSP
2026): **SDR dropped entirely** ("phase oversensitivity") for Multi-Mel-SNR
+ Zimtohrli + FAD-CLAP; L1/multi-res-STFT losses beat GAN training. Waveform
ρ is capped (~0.975 best stem) — perceptual-grade reconstruction is
open-ended. Same conclusion as the roadmap's Part I, now with the field's
2026 consensus behind it.

---

## The updated verdict on "100% of any song"

1. **Electronic / sample-based / repetitive music** (their fixture genres):
   corr 1.0 is *conditionally achievable* — LFSR/PRNG recovery for machine
   noise, sample-bank grids for sampled drums, repetition references for
   structure, equations for tonal content. The wall only stands for
   genuinely acoustic noise.
2. **Acoustic recordings**: perceptual 100% via texture statistics +
   noise envelopes + reverb tails + full band (roadmap P1/P3); waveform
   100% only via the priced residual (Breaker 1C).
3. **The separator ceiling** is now ~0.975 and falling, and can be routed
   around entirely by mixture-consistent fitting and note-layer priors.
4. **The unclaimed prize**: "mix → DAW project" reconstruction. Every
   component is published; nobody has assembled it. This engine — discrete
   event layer + continuous analytic atoms + mix chain + referee — is the
   right architecture to be first.

### New roadmap items (extending PHYSICS_ROADMAP.md)

- **P4 "program layer"**: repetition references in the equation sheet;
  sample-bank re-encoding via `toneprint.py` fingerprints; PRNG/LFSR
  generator hypotheses on residual noise; optional note-event prior layer
  (YourMT3-class) feeding the tournament; mixture-consistency (sum-of-stems
  = mix) fitting. Each is independently testable against the scoreboard.
- Update to Part I of the roadmap: the lossless floor holds for *arbitrary*
  signals; real songs are generated programs, and the program is the
  equation sheet.

## Sources (key)

- LFSR/PRNG recovery: Berlekamp–Massey (en.wikipedia.org/wiki/Berlekamp–Massey_algorithm);
  TR-909 noise = LFSR (electricdruid.net/tr-909-noise-generator);
  truncated-LCG lattice recovery (Fouque et al. 2013, inria.hal.science/hal-01094296v1);
  MT19937 untempering (programmingpraxis.com/2011/09/09/mersenne-twister).
- Texture statistics: McDermott & Simoncelli, Neuron 2011
  (mcdermottlab.mit.edu); McWalter & McDermott, Nat Comms 2019; TexStat
  arXiv:2506.04073.
- Lossless floor and its breaking: Trilobyte, arXiv:2603.08683 (2026) —
  beats FLAC on music losslessly; FLAC/SBR/CELT lineage.
- Codecs: WavTokenizer arXiv:2408.16532 (ICLR 2025); SemantiCodec
  arXiv:2405.00233; MuCodec arXiv:2409.13216; EnCodec arXiv:2210.13438;
  SoundStream arXiv:2107.03312; DAC arXiv:2306.06546.
- Programs-as-descriptions: MPEG-4 Structured Audio (ISO 14496-3:1999);
  Knuth, SIGACT News 1977; demoscene (ctrl-alt-test.fr; code4k.blogspot.com).
- Repetition coding: ACER (Cunningham & Grout — doaj/ResearchGate 2009–2019,
  subjective eval doi:10.1155/2019/8265301); Lotfaliei 2022.
- Sample detection: Gururani & Lerch ISMIR 2017; GNN sample ID
  arXiv:2506.14684 (ISMIR 2025); Tracklib State of Sampling 2019.
- Transcription/resynthesis: Kong et al. arXiv:2106.05076 (onset F1 96.72%);
  MT3 arXiv:2111.03017; YourMT3+ arXiv:2407.04822; MIDI-DDSP arXiv:2112.09312;
  InverSynth arXiv:1812.06349; Masuda & Saito ISMIR 2021.
- Mix inversion: Barchiesi & Reiss 2010; Martinez-Ramirez et al.
  arXiv:2211.02247; blind AFX estimation arXiv:2310.11781.
- Separation SOTA: BS-RoFormer arXiv:2309.02612; Mel-RoFormer
  arXiv:2310.01809; SDX'23 (TISMIR doi:10.5334/tismir.171); SCNet
  arXiv:2401.13276; HT-Demucs ft arXiv:2211.08553; MSR Challenge ICASSP 2026
  arXiv:2601.04343 (drops SDR); SDR critique — Le Roux et al., ICASSP 2019;
  Bake-Off arXiv:2507.06917.
- Mixture-fitting: differentiable-DSP unsupervised separation
  arXiv:2201.09592; GuideSep (MIDI-conditioned) arXiv:2507.01339.
