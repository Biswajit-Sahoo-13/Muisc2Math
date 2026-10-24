# How to present this project — the decision

Question asked: "how will I present my project — through harness, CLI, or
app?" Answer, after a sweep of how high-adoption audio/DSP projects present
(DDSP, demucs, Music2Latent, WavTokenizer, Descript DAC, Coqui, librosa)
and a 2026 check of hosting realities:

**None of the three is the face. The README is the face, and it funnels to
an audible demo. "Harness vs CLI vs app" is the wrong split — each is a
face for a different audience, and they share one front door.**

## The funnel (and who each layer is for)

| Layer | Audience | What it must do |
|---|---|---|
| **README (front door)** | everyone, first 60 s | badges, a 10–15 s demo GIF/MP4 of the app loop, "hear it" link, one-command quickstart. Currently: opens with a mermaid diagram — visually inert, no demo media. |
| **Listening-samples page** (static, GitHub Pages) | recruiters, judges, skeptics | 4–6 excerpts: original vs reconstruction A/B players + per-excerpt scoreboard metrics. This is the audio-community norm (demucs, DAC, Music2Latent all ship one); a GIF cannot sell audio. |
| **Hosted app** (Streamlit Community Cloud) | casual visitors | one-click demo — now exists as the built-in "Try the demo" button (15 s tracked vocals fixture). Best free host in 2026: no request timeout, ~2 cores; **HF Spaces free tier is now PRO-paywalled** for compute apps. Deploy via `scripts/wizard_deploy_app.sh`. |
| **CLI (`m2m`)** | engineers, reviewers, agents | already the strongest asset: 20 subcommands, 3-OS CI, scoreboard provenance, wheel-install tested. Needs visibility, not work: record an asciinema cast. |
| **Agent harness (AGENTS.md + bundle)** | AI agents | `AGENTS.md` is the recognized standard (read by Codex/Cursor/Gemini CLI); PROJECT_BUNDLE.md is a custom name no agent looks for — keep the bundle, add the standard entry point (done — see repo root). A thin documented CLI + AGENTS.md beats a bolted-on MCP server in 2026 consensus. |
| **Docs (res_research, roadmap, limits)** | researchers | already paper-grade; frame the provenance in judge-recognized terms: committed ledgers + CI = ACM *Artifacts Available/Functional/Reproduced* vocabulary. |

## The decision

If you invest in one thing next: **the README front door + the listening
page** — the claim this project makes ("songs as equations, measured") is
*audible*, and neither a GIF alone nor a CLI transcript can carry it. The
app is deployed behind them as the interactive proof, the CLI as the
reproducibility proof, AGENTS.md as the machine door.

## The play, in order (status)

1. ✅ Built-in one-click demo in the app (button + tracked fixture,
   guarded by a structure test) — a hosted demo works with zero upload.
2. ✅ Deploy wizard: `bash scripts/wizard_deploy_app.sh` walks the
   Streamlit Community Cloud deploy and captures the URL into `.env`.
3. ✅ `AGENTS.md` at repo root — the standard agent entry point, pointing
   at PROJECT_BUNDLE.md and the docs.
4. ⬜ Record the 10–15 s README demo GIF/MP4 (upload or demo button →
   equations → playback) and put it under the badge row.
5. ⬜ Listening-samples page (GitHub Pages): 4–6 A/B excerpts with their
   scoreboard numbers — lift `data/output/_scoreboard/after/*` renders.
6. ⬜ "Artifact & reproducibility" README section in ACM-badge vocabulary +
   an asciinema cast of one `m2m res` run.

## The 60-second live demo script (for when you present in person)

1. Open the hosted app (or `streamlit run app.py`).
2. Click **"Try the built-in demo (15 s vocals stem)"** → preset *Fast* →
   **Recreate**. narrate: evidence → sections → tournament → referee while
   the progress panel ticks.
3. Play the **Original**, then the **Recreated** player; point at the corr
   metric and say "every number on screen is reproduced by CI from
   committed ledgers."
4. Open **All equations** — "that text file *is* the song."
5. Close on the honest line: "drums are at 0.42 today; the roadmap to
   0.95+ is measured, sourced, and in `docs/`."

Sources for the claims above: research sweep of 2026-10-23 (case-study
READMEs fetched; HF Spaces policy change documented on the HF forums and
changelog; AGENTS.md standard at agents.md; ACM badging at acm.org).
