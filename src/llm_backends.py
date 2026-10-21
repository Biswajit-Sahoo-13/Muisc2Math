"""LLM backends for the RES hypothesis provider.

The RES loop can ask a language model to propose edits to the fitted
equations. The model is a *proposer*: the referee still scores every
iteration and the loop still rolls back what does not help. So the backend
only has to do one thing reliably - take a prompt and answer with the
edit array described in the loop's prompt.

Why a registry rather than one hardcoded command:

* Different CLIs take the prompt differently. ``agy --print`` and
  ``claude -p`` take it as an argument, ``qwen`` takes it positionally
  (``-p`` is deprecated there), ``opencode run`` takes it positionally
  after a subcommand.
* Not every CLI is installed, and not every installed CLI is signed in.
  ``src/res_loop.py`` used to claim that Antigravity "cannot be called
  headlessly from a script"; ``agy --print`` does exactly that, so the
  claim was wrong and the backend is now listed.
* A CLI that spawns children can outlive its own timeout while holding
  the output pipe open, which makes ``subprocess.run(timeout=...)`` hang
  forever instead of timing out. ``run_backend`` kills the whole process
  tree and bounds the final read.

The prompt is a few kilobytes for a short fixture, so it rides in argv.
The Windows command line is capped at 32767 characters, so
``build_command`` refuses to build a command that cannot fit rather than
letting the OS truncate it silently.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Windows caps a command line at 32767 characters. Stay under it with room
# to spare for the executable and the rest of the arguments.
_MAX_CMD_CHARS = 30000 if os.name == "nt" else 100000


@dataclass(frozen=True)
class Backend:
    """One way of asking a language model for edit proposals."""

    name: str
    binary: str
    #: Builds the argument list after the executable. ``(prompt, model)``.
    build: Any
    #: A distinctive string the version output should contain, so a name
    #: collision with an unrelated tool is caught rather than trusted.
    identity: str
    #: What to tell the user when this backend is installed but refuses to
    #: run, which is almost always an auth problem and not a code problem.
    auth_hint: str = ""
    #: Whether the invocation has been observed to work on this machine.
    confirmed: bool = False
    #: A note on the convention, kept honest about what was verified.
    notes: str = ""
    #: Extra arguments this backend needs to work at all, beyond the
    #: prompt convention. Caller-supplied arguments come after these.
    extra_args: Tuple[str, ...] = ()


def _arg(prompt: str, model: Optional[str]) -> List[str]:
    return [prompt]


def _opencode(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["run"]
    if model:
        argv += ["--model", model]
    argv.append(prompt)
    return argv


def _agy(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["--print"]
    if model:
        argv += ["--model", model]
    argv.append(prompt)
    return argv


def _gemini(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["-p", prompt]
    if model:
        argv += ["-m", model]
    return argv


def _qwen(prompt: str, model: Optional[str]) -> List[str]:
    # qwen-code deprecates -p in favour of the positional prompt.
    argv = [prompt]
    if model:
        argv += ["-m", model]
    return argv


def _claude(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["-p", prompt]
    if model:
        argv += ["--model", model]
    return argv


def _codex(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["exec", prompt]
    if model:
        argv += ["-m", model]
    return argv


def _crush(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["run", prompt]
    if model:
        argv += ["--model", model]
    return argv


def _goose(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["run", "-t", prompt]
    if model:
        argv += ["--model", model]
    return argv


def _amp(prompt: str, model: Optional[str]) -> List[str]:
    argv = ["-x", prompt]
    if model:
        argv += ["--model", model]
    return argv


def _ollama(prompt: str, model: Optional[str]) -> List[str]:
    # ollama run needs a model name; without one it cannot be built.
    argv = ["run", model or "llama3.1", prompt]
    return argv


BACKENDS: Dict[str, Backend] = {b.name: b for b in (
    Backend(
        name="agy",
        binary="agy",
        build=_agy,
        identity="agy",
        auth_hint="run 'agy' once interactively to sign in",
        confirmed=True,
        notes="Antigravity CLI. --print runs a single prompt headlessly, but "
              "it is an agent: on the RES equation prompt it reaches for a "
              "command tool, and headless mode auto-denies that, so it "
              "answers with nothing. '--dangerously-skip-permissions' lets "
              "it act (and then it stops answering quickly), and a "
              "permissions.allow rule for one command target is the safer "
              "route. Pass it with --llm-extra-args.",
    ),
    Backend(
        name="opencode",
        binary="opencode",
        build=_opencode,
        identity="opencode",
        auth_hint="run 'opencode' once to sign in",
        confirmed=False,
        notes="opencode run <prompt>. Spawns node children, so the "
              "process tree has to be killed on timeout.",
    ),
    Backend(
        name="gemini",
        binary="gemini",
        build=_gemini,
        identity="gemini",
        auth_hint="run 'gemini' once to sign in; a gemini-cli version "
                  "older than the account allows will refuse to run",
        confirmed=False,
        notes="gemini -p <prompt> -m <model>.",
    ),
    Backend(
        name="qwen",
        binary="qwen",
        build=_qwen,
        identity="qwen",
        auth_hint="run 'qwen' once to pick an auth type",
        confirmed=False,
        notes="qwen <prompt> -m <model>. -p is deprecated there.",
    ),
    Backend(
        name="claude",
        binary="claude",
        build=_claude,
        identity="claude",
        auth_hint="install Claude Code and run 'claude' once to sign in",
        confirmed=False,
        notes="claude -p <prompt> --model <model>.",
    ),
    Backend(
        name="codex",
        binary="codex",
        build=_codex,
        identity="codex",
        auth_hint="install the Codex CLI and run 'codex' once to sign in",
        confirmed=False,
        notes="codex exec <prompt>.",
    ),
    Backend(
        name="crush",
        binary="crush",
        build=_crush,
        identity="crush",
        auth_hint="install Crush and run 'crush' once to sign in",
        confirmed=False,
        notes="crush run <prompt>.",
    ),
    Backend(
        name="goose",
        binary="goose",
        build=_goose,
        identity="goose",
        auth_hint="install Goose and run 'goose configure'",
        confirmed=False,
        notes="goose run -t <prompt>.",
    ),
    Backend(
        name="amp",
        binary="amp",
        build=_amp,
        identity="amp",
        auth_hint="install Amp and run 'amp' once to sign in",
        confirmed=False,
        notes="amp -x <prompt>.",
    ),
    Backend(
        name="ollama",
        binary="ollama",
        build=_ollama,
        identity="ollama",
        auth_hint="ollama needs no account; pull a model first "
                  "(ollama pull llama3.1)",
        confirmed=False,
        notes="ollama run <model> <prompt>. A local model, so no key and "
              "no network.",
    ),
)}

#: Preference order used by ``--provider auto``.
AUTO_ORDER = ["agy", "opencode", "claude", "gemini", "qwen", "codex",
              "crush", "goose", "amp", "ollama"]


def backend_names() -> List[str]:
    """Every backend name, in a stable order."""
    return list(BACKENDS.keys())


def find_backend(name: str) -> Optional[Backend]:
    return BACKENDS.get(name.strip().lower())


def resolve_binary(backend: Backend) -> Optional[str]:
    """The executable for a backend, or None if it is not installed.

    ``shutil.which`` uses PATHEXT, which on Windows covers ``.CMD`` but not
    ``.PS1``. The npm CLIs ship a ``.CMD`` shim, so they are found; a
    PowerShell-only shim is not, and the doctor says so.
    """
    return shutil.which(backend.binary)


def build_command(
    backend: Backend,
    prompt: str,
    model: Optional[str] = None,
    prompt_file: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """The full argv for a backend, or a clear refusal.

    Refuses rather than truncating: a prompt that will not fit on the
    command line is a bug in the caller, and a silently shortened prompt
    would make the model answer a different question.
    """
    exe = resolve_binary(backend)
    if not exe:
        raise FileNotFoundError(
            "%s is not installed (looked for %r on PATH)" % (backend.name, backend.binary)
        )
    argv = [exe] + list(backend.extra_args) + list(extra_args or ())
    argv += list(backend.build(prompt, model))
    if prompt_file:
        argv = [a.replace("{prompt_file}", prompt_file) for a in argv]
    length = sum(len(a) + 3 for a in argv)
    if length > _MAX_CMD_CHARS:
        raise ValueError(
            "the prompt is %d characters and will not fit on the command line "
            "(limit %d); pass a shorter section or use a backend that reads "
            "stdin" % (length, _MAX_CMD_CHARS)
        )
    return argv


@dataclass
class RunResult:
    """The receipt of one backend call."""

    argv: List[str]
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.error


def _kill_tree(proc: "subprocess.Popen") -> None:
    """Kill a process and everything it started.

    A CLI that spawns children can keep the output pipe open after it is
    killed, so killing the parent alone is not enough to make the read
    return.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=20,
            )
            return
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def run_backend(
    argv: Sequence[str],
    stdin_text: Optional[str] = None,
    timeout_s: float = 300.0,
) -> RunResult:
    """Run a backend with a timeout that actually bounds the read.

    ``subprocess.run(timeout=...)`` kills the process and then reads the
    pipes with no timeout of its own, so a CLI with surviving children
    hangs the caller forever. Here the whole tree is killed and the final
    read is bounded.
    """
    argv = [str(a) for a in argv]
    result = RunResult(argv=list(argv))
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    started = time.time()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            **kwargs,
        )
    except FileNotFoundError as exc:
        result.error = "not found: %s" % exc
        result.duration_s = time.time() - started
        return result
    except Exception as exc:
        result.error = "%s: %s" % (type(exc).__name__, exc)
        result.duration_s = time.time() - started
        return result
    try:
        out, err = proc.communicate(input=stdin_text, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        result.timed_out = True
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = "", ""
        result.stdout = out or ""
        result.stderr = err or ""
        result.duration_s = time.time() - started
        return result
    except Exception as exc:
        result.error = "%s: %s" % (type(exc).__name__, exc)
        result.duration_s = time.time() - started
        return result
    result.exit_code = proc.returncode
    result.stdout = out or ""
    result.stderr = err or ""
    result.duration_s = time.time() - started
    return result


def probe_backend(backend: Backend, timeout_s: float = 20.0) -> Dict[str, Any]:
    """Check one backend: installed, present, and the right tool.

    A name can collide with an unrelated program - ``fabric`` on this
    machine is Lightning Fabric, the Python ML framework, not the prompt
    runner of the same name. The identity check catches that.
    """
    info: Dict[str, Any] = {
        "name": backend.name,
        "binary": backend.binary,
        "installed": False,
        "path": None,
        "version": None,
        "identity_ok": None,
        "usable": False,
        "notes": backend.notes,
        "auth_hint": backend.auth_hint,
        "confirmed_here": backend.confirmed,
    }
    exe = resolve_binary(backend)
    if not exe:
        info["reason"] = "not installed"
        return info
    info["installed"] = True
    info["path"] = exe
    res = run_backend([exe, "--version"], timeout_s=timeout_s)
    if res.error:
        info["reason"] = res.error
        return info
    text = ((res.stdout or "") + (res.stderr or "")).strip()
    info["version"] = text.splitlines()[0].strip() if text else ""
    # Most of these CLIs print a bare version number, so the name is not in
    # the version output. The help text is where the name appears.
    found = backend.identity.lower() in text.lower()
    if not found:
        helped = run_backend([exe, "--help"], timeout_s=timeout_s)
        help_text = ((helped.stdout or "") + (helped.stderr or "")).lower()
        found = backend.identity.lower() in help_text
    info["identity_ok"] = found
    if not found:
        info["reason"] = (
            "installed, but it does not look like %s (version output was %r)"
            % (backend.name, info["version"])
        )
        return info
    if res.timed_out:
        info["reason"] = "installed and identified, but --version timed out"
        return info
    if res.exit_code != 0:
        info["reason"] = "installed and identified, but --version exited %s" % res.exit_code
        return info
    info["usable"] = True
    info["reason"] = "installed and identified"
    return info


def probe_all() -> List[Dict[str, Any]]:
    """Probe every backend, in the auto-preference order."""
    rows = []
    for name in AUTO_ORDER:
        backend = BACKENDS.get(name)
        if backend is None:
            continue
        rows.append(probe_backend(backend))
    for name in backend_names():
        if name not in AUTO_ORDER:
            rows.append(probe_backend(BACKENDS[name]))
    return rows


def auto_backend() -> Optional[Backend]:
    """The first usable backend, in preference order."""
    for name in AUTO_ORDER:
        backend = BACKENDS.get(name)
        if backend is None:
            continue
        if resolve_binary(backend):
            return backend
    return None


# ---------------------------------------------------------------- API path

def run_api(
    prompt: str,
    model: str,
    base_url: str = "https://api.openai.com/v1",
    api_key: Optional[str] = None,
    timeout_s: float = 120.0,
) -> RunResult:
    """Call an OpenAI-compatible chat completions endpoint directly.

    This is the backend that needs an API key. The CLI backends do not:
    they carry their own authentication, so a key in this project's
    environment is only needed when there is no CLI to use, or when a
    direct HTTP call to a specific endpoint is wanted (a local Ollama
    server, a gateway, a self-hosted model).
    """
    import urllib.error
    import urllib.request

    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
    result = RunResult(argv=["POST", base_url + "/chat/completions", model])
    if not key:
        result.error = "no API key: set OPENAI_API_KEY or pass --llm-api-key"
        return result
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
        },
    )
    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        result.error = "HTTP %s %s" % (exc.code, detail)
        result.duration_s = time.time() - started
        return result
    except Exception as exc:
        result.error = "%s: %s" % (type(exc).__name__, exc)
        result.duration_s = time.time() - started
        return result
    result.exit_code = 0
    result.duration_s = time.time() - started
    try:
        result.stdout = payload["choices"][0]["message"]["content"] or ""
    except Exception as exc:
        result.error = "unexpected response shape: %s" % exc
    return result
