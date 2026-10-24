"""Visualization helpers for the Math-DAW multi-track equation engine.

Generates:
1. Stacked per-track + master waveform plots
2. Per-track spectrogram grid
3. Equation annotation overlays
4. DAW-style arrangement timeline (tracks x time grid)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TIMELINE_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#fd79a8",
]


def plot_tracks(
    rendered_tracks: List[Tuple[str, np.ndarray]],
    master: np.ndarray,
    sr: int = 44100,
    bpm: int = 140,
    title: str = "Math-DAW: Multi-Track Waveforms",
    out_path: Optional[Union[str, Path]] = None,
    equations: Optional[List[str]] = None,
) -> None:
    """Plot stacked per-track waveforms with the master bus at the bottom.

    Parameters
    ----------
    rendered_tracks : list of (name, ndarray)
        Per-track rendered audio.
    master : ndarray
        The mixed master audio.
    sr : int
        Sample rate.
    bpm : int
        Tempo for beat grid lines.
    title : str
        Plot title.
    out_path : str or Path, optional
        If given, saves the plot to this path.
    equations : list of str, optional
        Equation strings to annotate each track panel.
    """
    n_tracks = len(rendered_tracks)
    n_panels = n_tracks + 1  # +1 for master

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(14, 2.2 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    colors = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
        "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    ]
    beat_sec = 60.0 / bpm

    for idx, (ax, (name, y_track)) in enumerate(zip(axes[:-1], rendered_tracks)):
        t_axis = np.arange(len(y_track)) / sr
        color = colors[idx % len(colors)]
        ax.plot(t_axis, y_track, lw=0.4, color=color, alpha=0.9)

        # Beat grid
        max_t = t_axis[-1]
        beat_times = np.arange(0, max_t, beat_sec)
        for bt in beat_times:
            ax.axvline(bt, color="#cccccc", lw=0.3, alpha=0.5)

        ax.set_ylabel(name, fontsize=9, fontweight="bold", color=color)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, axis="y", alpha=0.2)
        ax.tick_params(axis="y", labelsize=7)

        # Equation annotation
        if equations and idx < len(equations):
            eq_text = equations[idx]
            if len(eq_text) > 60:
                eq_text = eq_text[:57] + "..."
            ax.text(
                0.01, 0.92, f"y = {eq_text}",
                transform=ax.transAxes,
                fontsize=7, fontfamily="monospace",
                color=color, alpha=0.8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

    # Master bus
    t_master = np.arange(len(master)) / sr
    axes[-1].plot(t_master, master, lw=0.5, color="#2c3e50")
    axes[-1].set_ylabel("MASTER", fontsize=10, fontweight="bold", color="#2c3e50")
    axes[-1].set_xlabel("Time (seconds)", fontsize=10)
    axes[-1].set_ylim(-1.05, 1.05)
    axes[-1].grid(True, alpha=0.3)

    # Beat grid on master
    max_t = t_master[-1]
    beat_times = np.arange(0, max_t, beat_sec)
    for bt in beat_times:
        axes[-1].axvline(bt, color="#cccccc", lw=0.3, alpha=0.5)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_track_spectrograms(
    rendered_tracks: List[Tuple[str, np.ndarray]],
    master: np.ndarray,
    sr: int = 44100,
    title: str = "Math-DAW: Track Spectrograms",
    out_path: Optional[Union[str, Path]] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float = 8000.0,
) -> None:
    """Plot spectrograms for each track + master in a grid.

    Parameters
    ----------
    rendered_tracks : list of (name, ndarray)
        Per-track rendered audio.
    master : ndarray
        Mixed master audio.
    sr : int
        Sample rate.
    title : str
        Plot title.
    out_path : str or Path, optional
        If given, saves the plot.
    """
    import librosa
    import librosa.display

    n_panels = len(rendered_tracks) + 1
    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    all_signals = list(rendered_tracks) + [("MASTER", master)]

    for i, (name, audio) in enumerate(all_signals):
        r, c = divmod(i, cols)
        ax = axes[r, c]

        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)

        librosa.display.specshow(
            spec_db, sr=sr, hop_length=hop_length,
            x_axis="time", y_axis="linear",
            fmax=fmax, cmap="magma", ax=ax,
        )
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylim(0, fmax)

    # Hide unused axes
    for i in range(len(all_signals), rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_timeline_figure(
    project: Any,
    title: Optional[str] = None,
    dark: bool = True,
) -> "plt.Figure":
    """Build a DAW-style arrangement timeline figure for a Math-DAW project.

    Rows are tracks, the x-axis is time, and each colored block is one active
    beat-step (pattern note). Rests stay empty — exactly like an FL Studio
    playlist timeline.
    """
    tracks = list(project.tracks)
    bpm = float(project.bpm)
    duration = float(project.duration)
    beat = 60.0 / bpm
    n_steps = max(1, int(np.ceil(duration / beat)))

    bg = "#07090e" if dark else "white"
    fg = "#c5c6c7" if dark else "black"
    grid_light = "#2a2f3a" if dark else "#cccccc"
    grid_bar = "#4a5160" if dark else "#999999"

    fig_height = max(2.5, 0.9 * len(tracks) + 1.2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    for row, track in enumerate(tracks):
        pattern = track.pattern or []
        color = TIMELINE_COLORS[row % len(TIMELINE_COLORS)]
        # Lane background strip
        ax.add_patch(Rectangle(
            (0, row + 0.08), duration, 0.84,
            facecolor=color, alpha=0.07, edgecolor="none", zorder=1,
        ))
        if pattern:
            for step in range(n_steps):
                note = pattern[step % len(pattern)]
                if note is None:
                    continue
                x0 = step * beat
                w = min(beat * 0.96, duration - x0)
                if w <= 0:
                    break
                ax.add_patch(Rectangle(
                    (x0, row + 0.12), w, 0.76,
                    facecolor=color, edgecolor="white" if dark else "black",
                    lw=0.4, alpha=0.9, zorder=3,
                ))
                if w > 0.28:
                    ax.text(
                        x0 + w / 2, row + 0.5, str(note),
                        ha="center", va="center", fontsize=6,
                        color="white" if dark else "black", zorder=4,
                    )

    # Beat grid (light) + bar grid every 4 beats (strong)
    for b in range(n_steps + 1):
        t = b * beat
        ax.axvline(t, color=grid_bar if b % 4 == 0 else grid_light,
                   lw=0.8 if b % 4 == 0 else 0.4, alpha=0.9, zorder=2)
    for bar in range(n_steps // 4 + 1):
        ax.text(bar * 4 * beat, len(tracks) + 0.05, f"Bar {bar + 1}",
                fontsize=7, color=fg, alpha=0.7, ha="left", va="bottom")

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.15, len(tracks) + 0.35)
    ax.set_yticks([r + 0.5 for r in range(len(tracks))])
    ax.set_yticklabels([t.name for t in tracks], fontsize=9, color=fg)
    ax.set_xlabel("Time (seconds)", fontsize=10, color=fg)
    ax.tick_params(colors=fg, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(grid_light)
    ax.set_title(
        title or f"Arrangement Timeline: {project.name} ({bpm:g} BPM)",
        fontsize=12, fontweight="bold", color=fg, pad=12,
    )
    fig.tight_layout()
    return fig


def plot_project_timeline(
    project: Any,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    dark: bool = True,
) -> None:
    """Render a DAW-style arrangement timeline and save it to ``out_path``."""
    fig = build_timeline_figure(project, title=title, dark=dark)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_timeline_player_html(
    project: Any,
    master: np.ndarray,
    sr: int = 44100,
    preview_sr: int = 22050,
    title: Optional[str] = None,
    uid: str = "tl",
) -> str:
    """Build a self-contained realtime timeline player (HTML + Web Audio).

    Returns an HTML page/fragment with a canvas arrangement timeline, an
    embedded audio preview of the master mix, and a playhead that sweeps the
    timeline while the audio plays. Active blocks light up; clicking the
    canvas seeks. ``uid`` keeps element IDs unique when several players share
    one page.
    """
    import base64
    import io
    import json
    import wave

    import librosa

    tracks = list(project.tracks)
    bpm = float(project.bpm)
    duration = float(project.duration)
    beat = 60.0 / bpm
    n_steps = max(1, int(np.ceil(duration / beat)))

    lanes = []
    for row, track in enumerate(tracks):
        pattern = track.pattern or []
        blocks = []
        if pattern:
            for step in range(n_steps):
                note = pattern[step % len(pattern)]
                if note is None:
                    continue
                x0 = step * beat
                w = min(beat, duration - x0)
                if w <= 0:
                    break
                blocks.append({"x0": round(x0, 4), "w": round(w, 4), "midi": int(note)})
        lanes.append({
            "name": track.name,
            "color": TIMELINE_COLORS[row % len(TIMELINE_COLORS)],
            "blocks": blocks,
        })

    # Downmixed preview audio embedded as a data URI.
    mono = np.asarray(master, dtype=np.float64).ravel()
    if sr != preview_sr and len(mono) > 0:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=preview_sr)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak > 1.0:
        mono = mono / peak * 0.9
    pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(preview_sr)
        wf.writeframes(pcm.tobytes())
    audio_uri = "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    layout = {
        "lanes": lanes,
        "bpm": bpm,
        "beat": round(beat, 4),
        "duration": round(duration, 4),
    }
    lane_h, gutter, top_pad = 44, 150, 34
    canvas_h = top_pad + lane_h * len(lanes) + 8
    heading = (title or f"{project.name} ({bpm:g} BPM) — realtime timeline")

    return (
        f"<div style='font-family:sans-serif;color:#c5c6c7;'>"
        f"<div style='font-weight:700;margin-bottom:6px;'>{heading}</div>"
        f"<button id='{uid}_play' style='font-size:15px;padding:6px 18px;margin-right:8px;'>Play timeline</button>"
        f"<button id='{uid}_stop' style='font-size:15px;padding:6px 18px;'>Stop</button>"
        f"<span id='{uid}_t' style='margin-left:10px;font-variant-numeric:tabular-nums;'></span>"
        f"<div style='overflow-x:auto;margin-top:8px;'>"
        f"<canvas id='{uid}_cv' width='1200' height='{canvas_h}' style='background:#07090e;border-radius:8px;cursor:pointer;'></canvas>"
        f"</div>"
        f"<audio id='{uid}_au' src='{audio_uri}' preload='auto'></audio>"
        "</div>"
        "<script>"
        f"const L={json.dumps(layout)};"
        f"const cv=document.getElementById('{uid}_cv'),ctx=cv.getContext('2d');"
        f"const au=document.getElementById('{uid}_au'),lbl=document.getElementById('{uid}_t');"
        f"const W=cv.width,H=cv.height,G={gutter},TP={top_pad},LH={lane_h};"
        "const X=t=>G+(W-G-10)*(t/L.duration);"
        "function draw(now){"
        "ctx.fillStyle='#07090e';ctx.fillRect(0,0,W,H);"
        "ctx.font='11px sans-serif';"
        "for(let b=0;b*L.beat<=L.duration+1e-6;b++){"
        "const x=X(b*L.beat);ctx.strokeStyle=b%4===0?'#4a5160':'#232833';"
        "ctx.lineWidth=b%4===0?1.2:0.6;ctx.beginPath();ctx.moveTo(x,TP-14);ctx.lineTo(x,H-4);ctx.stroke();"
        "if(b%4===0){ctx.fillStyle='#8b93a3';ctx.fillText('Bar '+(b/4+1),x+3,TP-18);}}"
        "L.lanes.forEach((lane,r)=>{"
        "const y=TP+r*LH;"
        "ctx.fillStyle=lane.color+'22';ctx.fillRect(G,y+3,W-G-10,LH-6);"
        "ctx.fillStyle='#e6e8ec';ctx.fillText(lane.name,8,y+LH/2+4);"
        "lane.blocks.forEach(bl=>{"
        "const active=now>=bl.x0&&now<bl.x0+bl.w;"
        "ctx.globalAlpha=active?1.0:0.55;ctx.fillStyle=lane.color;"
        "const bx=X(bl.x0),bw=Math.max(2,(X(bl.x0+bl.w)-bx)-2);"
        "ctx.fillRect(bx,y+6,bw,LH-12);ctx.globalAlpha=1.0;"
        "if(bw>26){ctx.fillStyle='#fff';ctx.fillText(String(bl.midi),bx+bw/2-8,y+LH/2+4);}});});"
        "const px=X(Math.min(now,L.duration));"
        "ctx.strokeStyle='#66fcf1';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(px,TP-14);ctx.lineTo(px,H-4);ctx.stroke();"
        "lbl.textContent=now.toFixed(2)+'s / '+L.duration.toFixed(2)+'s';}"
        "let raf=null;"
        "function loop(){draw(au.currentTime||0);raf=requestAnimationFrame(loop);}"
        f"document.getElementById('{uid}_play').onclick=()=>{{au.play();if(!raf)loop();}};"
        f"document.getElementById('{uid}_stop').onclick=()=>{{au.pause();if(raf){{cancelAnimationFrame(raf);raf=null;}}}};"
        "au.onended=()=>{if(raf){cancelAnimationFrame(raf);raf=null;}draw(L.duration);};"
        f"cv.onclick=e=>{{const r=cv.getBoundingClientRect();const fx=(e.clientX-r.left)*(cv.width/r.width);au.currentTime=Math.min(L.duration,Math.max(0,(fx-G)/(W-G-10)*L.duration));draw(au.currentTime);}};"
        "draw(0);"
        "</script>"
    )


def save_timeline_player(
    project: Any,
    master: np.ndarray,
    sr: int,
    out_path: Union[str, Path],
    title: Optional[str] = None,
) -> None:
    """Save a self-contained realtime timeline player as an HTML file."""
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Math-DAW realtime timeline</title></head>"
        "<body style='background:#0c0f17;margin:0;padding:16px;'>"
        + build_timeline_player_html(project, master, sr, title=title, uid="tl")
        + "</body></html>"
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
