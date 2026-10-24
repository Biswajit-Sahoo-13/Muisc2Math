# The Note Layer: frequency-native equations

Status: design + first increment (detector), 2026-10-24. This is the
user's idea — "detect the frequency and convert it into the mathematical
equation" — made precise, measured against the engine's own failures, and
checked against the research literature. Companion to
`PHYSICS_ROADMAP.md` and `BEYOND_THE_LIMITS.md`.

---

## The idea, stated precisely

Today the engine fits *free* frequencies and hopes they land on music.
The note layer inverts that: **detect the frequencies first, then write
the equations from them**:

1. **12-TET snapping** — a detected carrier becomes the nearest
   equal-tempered pitch, `f = 440 · 2^((n−69)/12)`; the residual detune
   stays a parameter.
2. **Harmonic-native form** — partials written as stacks of a detected
   f0 (the engine already has this for class B; extend it to notes).
3. **Beat pairs (the new model class)** — two partials within ~60 cents
   of each other are not two accidents; they are ONE musical object (a
   detuned unison) whose *beat frequency is itself a parameter*.
   cos A + cos B folds to `2·cos((A+B)/2)·cos((A−B)/2)`: a carrier, a
   beat rate, and — when the decays differ — a beat that deepens or dies
   over the note.

## The measured evidence (MONA LISA, first 4 s)

From the frequency-reachability audit (2026-10-24, `out/nbn_ear3`):

- **Unreachable frequencies**: 46.01 Hz (F#1), 36.84 Hz (D1), 47.36 Hz —
  among the strongest residual peaks with **0% partial coverage**;
  effectively unreachable: 42.73, 38.19, 64.35 Hz. The residual is
  83.1% concentrated below 150 Hz.
- **The beat comb**: every top sub-bass carrier carries envelope
  components at **1.0–1.35 Hz** — the 808's detune/beat structure. Three
  of the strongest beat lines have no two-partial explanation in the
  ledger.
- **Temperament drift**: only 2/20 of the strongest fitted partials sit
  within ±5 cents of 12-TET; mean |deviation| 27 cents, max 47.5. The
  engine tracks layered spectral peaks, not musical pitch.
- **The 808 glide, measured**: four notes (t ≈ 0.62, 1.30, 2.19, 2.80 s),
  each starting 43.2–43.6 Hz and bending down ~1.3–1.6 Hz (≈60 cents)
  over ~250 ms, sustaining at 42.0 ± 0.5 Hz. The engine's glides smear
  trajectories across 32.8–79.5 Hz — **5–15× the true excursion** — and
  one is 130 cents flat.
- **Beat grid**: true tempo 136 BPM; the ledger's 68.7 half-time grid
  drifts 0.25–0.37 s by t = 3.5 s; the section boundary sits 23 ms off a
  true beat.
- **First detector run** (`src/note_layer.detect_beat_pairs` on the
  ledger): **70 beat pairs**, incl. 83.6 Hz (E2 +24c) with beat 2.25 Hz
  and decays 44.1/23.3 — the layered-808 signature: the beat dies as one
  layer decays, which no single-exponential envelope can express.

## What the research says (and doesn't)

- Constraining frequencies to a grid is precedented by **score-informed
  NMF** (Ewert & Müller, ICASSP 2012) — but the score provides the grid;
  harmonic-NMF (INRIA) estimates the tuning reference from the recording
  instead of assuming 440.
- **No published parametric coder snaps detected partials to the grid
  and re-synthesizes the snapped values.** Quantization exists only for
  labeling. The snap-then-resynthesize move with residual detune kept as
  a parameter appears novel.
- **Beating as a parameter is effectively unshipped.** The closest
  precedents: the differentiable piano model (arXiv:2409.06513) that
  synthesizes TWO partial-sets per note with a small detuning δf —
  beating and double decay *emerge* from the pair; Goodwin & Vetterli's
  damped-sinusoid matching pursuit, where close atoms pair up
  implicitly. Nobody promotes f_beat to a first-class schema key.
- **Honest math warning**: the single-AM equation is exact only when both
  layers share amplitude AND decay. General case:
  `A1 e^(−λ1t) cos ω1t + A2 e^(−λ2t) cos ω2t` needs cosh/sinh envelope
  corrections. **Therefore the render primitive is the PAIR, not the
  collapsed AM equation** — the renderer already sums damped cosines, so
  parity risk is zero; the pair view is analysis/schema, the two
  partials are render.
- **f_beat semantics**: the envelope swells at (f1−f2)/2 but the
  *perceived* beat rate is f1−f2 (CCRMA, Feynman I-48). The schema
  defines f_beat as the intensity beat rate f1−f2.
- **Perceptual stakes**: two subs a few Hz apart sit in the Plomp-Levelt
  maximum-roughness zone — the beat is one of the most salient features
  of the low end, and the magnitude spectrum is nearly blind to it.
  Waveform corr is not — which is why the referee can arbitrate.
- **Production reality**: supersaw detune is symmetric around a
  true-pitch master (25–30 cents typical), so **snap the carrier, never
  the layers** — snapping layers would destroy the beat, the very
  percept that defines the sound. Pure 808 subs are usually NOT detuned;
  when they are, it's intentional (the growl), and the slow beat IS the
  sound.

## The ladder (each step independently measurable)

1. ✅ **Detector** (shipped, `src/note_layer.py`): scan the fitted
   partials, emit beat-pair records `{f_carrier, f_beat, A1, A2, λ1,
   λ2, carrier_note, cents_off, snappable}` into the ledger view.
   Renderer untouched (bit-identical output).
2. **12-TET snap as a hypothesis class**: re-emit a pair with f_c
   snapped (carrier <25 cents off) and let the referee arbitrate on
   waveform corr + LSD. If snapping hurts, the referee says so — same
   tournament structure as every other class.
3. **808 note events**: pitch-continuous sub-bass tracking (one
   oscillator per note with the measured attack drop of ~60 cents over
   250 ms), replacing the smeared 32.8–79.5 Hz glide trajectories —
   targets the measured 0% coverage at 36.8/46.0/47.4 Hz and the 6.2 dB
   sub-bass error.
4. **Beat-grid sections**: quantize section boundaries to the true beat
   grid (136 BPM here, not the ledger's half-time 68.7) so seams land on
   beats, not between them.
5. **Tuning reference estimation**: fit the recording's A4 (the INRIA
   move) instead of assuming 440, then snap against that grid.

## Sources (key)

- Score-informed NMF: Ewert & Müller, ICASSP 2012; harmonic/inharmonic
  NMF with tuning estimation: INRIA HAL inria-00544183.
- Beating: CCRMA "Sinusoidal AM" (jos); Feynman Lectures I-48; Boashash
  on instantaneous frequency (the AM/FM swing of unequal pairs).
- Two-partial detune as parameters: arXiv:2409.06513 (piano STN, §3.2);
  Goodwin & Vetterli, ICASSP 1997 (damped-sinusoid MP).
- Roughness salience: Plomp & Levelt 1965; AM phase-locking: Joris et
  al., Physiological Reviews 2004.
- Production detune practice: supersaw history (Perfect Circuit);
  808 layering threads (Gearspace/Serum 2 guides).
