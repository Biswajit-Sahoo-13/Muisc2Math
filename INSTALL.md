# Installing and running M2M (RES)

Everything below was verified from a **clean virtualenv on a machine with
nothing installed** — clone, install, run.

---

## What is a CLI, and how does it work?

A **CLI (Command Line Interface)** is a program you drive by typing commands
into a terminal instead of clicking buttons.

Two ways to use this project:

| | the web app | the CLI |
|---|---|---|
| start it | `streamlit run app.py` | `python main.py <command> ...` |
| you use it in | a browser | your terminal |
| good for | uploading a song, hearing the result, running the sample tests | scripting, batch work, CI, anything you want to repeat |

### How a CLI actually works

When you type `python main.py res -i song.mp3 -d 15`, three things happen:

1. **Python starts** and runs the file `main.py`.
2. **`main.py` parses your text** into an *argument parser* — a structured
   description of every command and flag. `res` is a sub-command; `-i` and `-d`
   are its options. This is why `python main.py --help` works: the help text is
   generated from that same description, so it can never drift out of date.
3. **Your command is dispatched** to a function, which does the work and prints
   a result.

So a CLI is just a program with a structured front door. The output is text
and files, which is why it composes: you can pipe it, schedule it, or call it
from a script. That is exactly what the benchmark harness does — it calls the
*same* `main.py res` command you would type by hand, which is why every
published number is reproducible by hand.

---

## Install

You need **Python 3.10 or newer**. Check with `python --version`.

### 1. Get the code

```bash
git clone https://github.com/Biswajit-Sahoo-13/Muisc2Math.git
cd Muisc2Math
```

### 2. Make a virtual environment

A virtual environment keeps this project's packages separate from everything
else on your machine. **Do this** — it is the step people skip and then regret.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`.

### 3. Install

```bash
pip install -e .
```

That reads `pyproject.toml`, works out the dependencies, and installs them.
The `-e` ("editable") means the code stays linked to your clone, so edits take
effect immediately.

Or, without the packaging metadata:

```bash
pip install -r requirements.txt
```

### 4. Optional extras

| extra | adds | when you need it |
|---|---|---|
| `separators` | torch, torchaudio, demucs | real source separation (per-stem correlations in the 0.9+ range instead of the HPSS band split) |
| `notes` | basic-pitch | better note extraction (falls back to pYIN without it) |
| `dev` | pytest | running the test suite |

```bash
pip install -e ".[separators,dev]"
```

`separators` pulls in PyTorch, which is a **~200 MB download** and is the
only heavy part. The engine works fully without it.

---

## Run it

### The web app

```bash
streamlit run app.py
```

Open <http://localhost:8501>. Two tabs: **RES** (upload a song, watch it
reconstruct, listen to Original vs Recreated vs Residual) and **Benchmarks**
(the frozen-fixture sample-test harness).

### The CLI

```bash
# reconstruct a song into equations and audio
python main.py res -i "song.mp3" -o data/output/run -d 15 --iterations 3

# with real source separation
python main.py res -i "song.mp3" -o data/output/run -d 15 --iterations 3 --stems demucs

# the end-to-end self-check
python main.py run-all

# every published number, verified against its ledger
python tools/verify_scoreboard.py
```

`python main.py --help` lists all 19 commands.

### Run the tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

---

## If something goes wrong

**`ModuleNotFoundError: No module named 'streamlit'`** — you are not in the
virtualenv. Run `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate`
(macOS/Linux) and try again.

**`pip install -e .` fails on Windows with a permissions error** — do not use
`sudo` or admin mode. Use a venv (step 2) and it goes away.

**`ModuleNotFoundError: No module named 'src'`** — you are running from the
wrong directory. `cd` into the `Muisc2Math` folder, or install the package
(`pip install -e .`) and import it from anywhere.

**The app starts but a run never finishes** — the RES loop is CPU-heavy.
Lower "Analyse seconds" and "Iterations" in the UI, or watch the live stage
timeline to see where the time is going.

**`--stems demucs` does nothing** — Demucs is not installed. Run
`pip install -e ".[separators]"`. The app warns about this in the tab rather
than silently falling back.
