# M2M RES 0.1.0

Mathematical audio reverse engineering: songs in, editable equations out.

This is the first tagged release. The DSP pipeline is frozen at this version;
what changes after it is packaging, documentation and interfaces, not the
reconstruction behaviour.

## What it does

Given an audio file, M2M RES separates the stems, fits a sinusoidal model to
each one, discovers the mathematical form of every partial, and renders the
result back to audio. The output is a set of equations and a ledger you can
edit and re-render, not a waveform.

## Measured results

Two different things are being measured, and they are not comparable. Both
are labelled so they cannot be confused.

**Synthetic input, exact fixture** (`data/output/_fixtures/`):

| | |
|---|---|
| waveform correlation | 0.999921 |
| SNR | 38.03 dB |
| log-spectral distance | 0.33 |

**Real material, stem level** (correlation after reconstruction):

| material | waveform correlation | tonal energy |
|---|---|---|
| mixed stem | 0.8518 | 79.9 % |
| vocals | 0.7535 | 82.3 % |
| other | 0.7376 | 91.5 % |
| bass | 0.6160 | 51.8 % |
| drums | 0.4107 | 3.1 % |

The correlation tracks the fraction of tonal energy in the material at
r = +0.935. M2M RES reconstructs what is harmonic and degrades on what is
not. That is the honest scope of the method, not a defect in the pipeline.

**Speed**, 8 workers, one 60 s input: 169 s end to end (73 s evidence, 94 s
loop, 2 s render), 0.355x realtime, 2.7 GB peak memory. A 90 s input takes
224 s. Throughput does not improve past 8 workers on this machine.

## Install

```bash
pip install m2m-res
m2m --help
```

Or from a checkout: `pip install -e ".[dev]"`. See `INSTALL.md` for the
optional extras (`separators` for Demucs-based stem separation, `notes` for
note extraction).

## Try it

```bash
m2m res -i path/to/song.wav -o out/ -d 3
```

The Streamlit app (`app.py`) does the same thing with a live stage timeline,
per-section equations, and a player for the result.

## In this release

- Sinusoidal model fitting with band-reserved high-band selection.
- Support-windowed rendering, so only the support of a partial is
  synthesised.
- Evidence generation with stem separation, spectral analysis and
  hypothesis acceptance.
- A sample-test harness with a scoreboard: every published number is a row
  pointing at a ledger, and CI fails if a row is edited to a number its
  ledger does not support.
- A project bundle (`PROJECT_BUNDLE.md`) that puts the whole project in one
  readable file.
- CI on Linux, Windows and macOS, including an install of the built wheel
  and a real job run from it.
