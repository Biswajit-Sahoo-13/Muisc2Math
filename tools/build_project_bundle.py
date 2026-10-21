"""Build a single uploadable bundle that explains the project to another AI.

Ten files, in reading order, with a navigation header. An unannotated dump of
8,500 lines would waste most of an LLM's context, so the header carries the
project shape, the measured numbers and the file index.
"""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PROJECT_BUNDLE.md"
ZIP = ROOT / "PROJECT_BUNDLE.zip"

FILES = [
    ("README.md",
     "Start here. What the project is, the measured results, the honest limits, "
     "the quickstart and the CLI reference."),
    ("docs/math_model.md",
     "The equation model: what a partial is, how envelopes, glides, notes and "
     "transients are expressed, and how the renderer evaluates them."),
    ("src/res_evidence.py",
     "Stage 1 - audio to evidence. STFT with window normalisation, parabolic peak "
     "interpolation, harmonic/percussive split, stem separation, band-reserved "
     "peak selection, and the phase re-referencing that fixes the time origin."),
    ("src/sinusoidal_model.py",
     "Stage 2 - sinusoidal modelling synthesis. Track continuation, log-linear "
     "envelope fitting, the refinement inner loop (the project's hot spot), and "
     "the per-window parallel reconstruction."),
    ("src/res_bank.py",
     "Stage 3 - the hypothesis bank. What classes may be proposed (harmonic "
     "strings, percussive bundles, stochastic noise), and how a candidate is "
     "accepted or refused."),
    ("src/res_loop.py",
     "Stage 4 - the RES loop. Section iteration, the referee budget, the "
     "worker protocol, the stems pipeline, and the equation-sheet renderer."),
    ("src/res_synthesis.py",
     "Stage 5 - equations back to audio. How one partial is rendered, the "
     "support mask, the glide trajectory, and the noise-floor partial."),
    ("src/residual.py",
     "The referee. Every published number comes from these metrics: SNR, "
     "log-spectral distance, segmental SNR, onset correlation."),
    ("docs/res_research.md",
     "The evidence base. Literature map per stage, ablations, and the provenance "
     "rules for every published number."),
    ("main.py",
     "The CLI surface: every capability the engine exposes, and therefore the "
     "complete list of what the project can do."),
]

HEADER = """# M2M (RES) - project bundle for an AI assistant

This file contains ten files from the repository, concatenated in the order they
should be read. The headings below are the real repository paths, so anything
quoted back can be located exactly.

---

## 1. What this project does

It takes an audio file and reverse-engineers it into **editable mathematical
equations** (JSON), then renders those equations back to audio.

```
audio file
  -> Evidence            (src/res_evidence.py)   spectral analysis: partials,
                         amplitudes, frequencies, phases, per section
  -> Sinusoidal model    (src/sinusoidal_model.py)  track continuation and
                         envelope fitting, the DSP core
  -> Hypothesis bank     (src/res_bank.py)   proposes candidate classes:
                         harmonic strings, percussive bundles, stochastic noise
  -> RES loop            (src/res_loop.py)    iterates sections, scores every
                         candidate against a referee, accepts or rolls back
  -> Render              (src/res_synthesis.py)  equations -> waveform
  -> Ledger              (data/output/.../res_ledger.json)  every attempt,
                         accepted or rejected, with its score
```

The output is not an audio file - it is a **short set of equations a person can
open, read, edit, and re-render**. That is the product.

---

## 2. The numbers that matter

Measured on frozen fixtures; every one is a row in `data/output/scoreboard.json`
pointing at a ledger.

| material | correlation | LSD |
|---|---|---|
| synthetic triad (ground truth) | **0.99992** | 0.33 |
| plucked string | 1.0000 | 0.35 |
| real full mix, 10 s | **0.8518** | 3.61 |
| vocals stem | 0.7535 | 9.11 |
| bass stem | 0.6160 | 3.03 |
| drums stem | **0.4107** | 18.04 |

Reconstruction correlation tracks the input's **tonal energy fraction** at
r = +0.935 across the fixtures. Percussion is the weak point because a kick and
a snare are not sums of sinusoids.

**Honest ceiling.** A full commercial master caps near 0.65-0.75 correlation:
brickwall limiting and microsecond phase jitter are not invertible by a sum of
sinusoids. The engine reports this rather than hiding it.

**The one bug that mattered.** The analyser drew each partial on the
*analysis window's* time axis while the renderer drew it on the *section's*.
Every partial was rotated by hundreds of cycles, so the reconstruction was
uncorrelated (0.008) even though each partial was individually correct.
`_rephase_to_section` in `src/res_evidence.py` fixes it; it is worth +0.844 of
the total recovery. The acceptance gate everyone suspected was worth +0.034.

---

## 3. How to read this bundle

Read in order. The first three files are self-contained: the README states the
project and its results, `math_model.md` states the mathematics, and
`res_evidence.py` shows the stage that everything downstream depends on. The
`src/` files then follow the pipeline in execution order. `main.py` is last
because it is a reference of capabilities, not a place to start.

If you only have room for part of it, take: **README.md**, then
**docs/math_model.md**, then **src/res_evidence.py** and **src/res_bank.py**.
That is enough to explain the concept and the two hardest parts of the engine.

---

## 4. Vocabulary

| term | meaning |
|---|---|
| **partial** | one modelled sinusoid: amplitude, frequency, phase, decay, support |
| **section** | a time window the engine analyses and refits independently |
| **evidence** | what the analysis stage measured: partials and their metrics |
| **hypothesis bank** | the set of candidate classes the engine may propose |
| **transaction** | one candidate proposal and its accept/reject receipt |
| **ledger** | the JSON record of every iteration, candidate and score |
| **referee** | the metric set that decides whether a candidate is better |
| **render** | evaluate the equations back into a waveform |

---

## 5. File index

"""

def _lang(rel):
    if rel.endswith(".py"):
        return "python"
    if rel.endswith(".md"):
        return "markdown"
    return "text"


def _read(rel):
    """Read a file with normalised line endings.

    The bundle quotes source code, so it has to be byte-identical whichever
    platform built it. A Windows checkout has CRLF in the working tree and a
    Linux one has LF; without this the same commit produced two different
    bundles.
    """
    return (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


parts = [HEADER]
for i, (rel, role) in enumerate(FILES, 1):
    p = ROOT / rel
    if not p.exists():
        continue
    n = len(_read(rel).splitlines())
    parts.append(f"| {i} | `{rel}` | {n} lines | {role} |\n")
parts.append("\n---\n\n# File contents\n")
for i, (rel, _role) in enumerate(FILES, 1):
    p = ROOT / rel
    if not p.exists():
        continue
    body = _read(rel)
    parts.append(f"\n---\n\n## File {i}: `{rel}`\n\n```{_lang(rel)}\n{body}\n```\n")

# newline="\n" so the bundle is byte-identical on every platform. Text mode
# otherwise rewrites LF to CRLF on Windows, which made the same commit
# produce two different bundles.
OUT.write_text("".join(parts), encoding="utf-8", newline="\n")
print(f"WROTE {OUT}")
print(f"size  {OUT.stat().st_size:,} bytes")
print(f"lines {len(''.join(parts).splitlines()):,}")

# The zip carries the same files as the markdown, so the bundle can be taken
# as one archive instead of the repository. The sdist already has the whole
# source tree; this is the readable subset plus the bundle itself.
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(OUT, OUT.name)
    for rel, _role in FILES:
        p = ROOT / rel
        if p.exists():
            zf.write(p, rel)
print(f"WROTE {ZIP}")
print(f"size  {ZIP.stat().st_size:,} bytes")
