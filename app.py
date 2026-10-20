"""Math-DAW: Interactive Multi-Track Equation Studio & Viral Math Phonk Suite (Streamlit UI).

Features:
- Viral Math Phonk: The exact Reel equations (#7/10 SINC DROP, Modulo Cowbell, Logistic 808, etc.)
- Math-DAW Studio: Live multi-track equation mixer with real-time waveform decomposition and WAV export
- Formula Bank Explorer: Acoustic dictionary with live waveform, FFT spectrum & audio preview
- Spectrogram Math: CQT vs Reassigned vs STFT spectrogram transforms & polyphonic peak extraction
- Bytebeat Player: paste floatbeat functions and play them in-browser via Web Audio
"""

import sys
import io
import json
import time
import wave
from pathlib import Path
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from src.math_track_engine import (
    Project, Track, render,
    demo_sinc_drop, demo_math_remix, demo_ambient,
)
from src.formula_bank import FORMULA_BANK, fit_best_formula_for_segment
from src.analysis import load_audio
from src.spectrogram_math import (
    compute_cqt_spectrogram,
    compute_reassigned_spectrogram,
    extract_notes_cqt_polyphonic,
)
from src.viral_math_reels import (
    REEL_EQUATIONS,
    synthesize_exact_sinc_drop,
    build_viral_math_phonk_project,
    render_all_viral_reels,
)

# ---------------------------------------------------------------------------
# Premium inline SVG icon system (replaces all emojis)
# ---------------------------------------------------------------------------
SVG_ICONS = {
    "bolt": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
    "sliders": '<line x1="6" y1="4" x2="6" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/><line x1="18" y1="4" x2="18" y2="20"/><circle cx="6" cy="9" r="2.2"/><circle cx="12" cy="15" r="2.2"/><circle cx="18" cy="7" r="2.2"/>',
    "flask": '<path d="M10 2v7.5a2 2 0 0 1-.2.9L4.7 20.5a1 1 0 0 0 .9 1.5h12.8a1 1 0 0 0 .9-1.5L14.2 10.4a2 2 0 0 1-.2-.9V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M8 17v-6"/><path d="M13 17V7"/><path d="M18 17v-3"/>',
    "gamepad": '<line x1="6" x2="10" y1="11" y2="11"/><line x1="8" x2="8" y1="9" y2="13"/><line x1="15" x2="15.01" y1="12" y2="12"/><line x1="18" x2="18.01" y1="10" y2="10"/><path d="M17.32 5H6.68a4 4 0 0 0-3.978 3.59c-.006.052-.01.101-.017.152C2.604 9.416 2 14.456 2 16a3 3 0 0 0 3 3c1 0 1.5-.5 2-1l1.414-1.414A2 2 0 0 1 9.828 16h4.344a2 2 0 0 1 1.414.586L17 18c.5.5 1 1 2 1a3 3 0 0 0 3-3c0-1.545-.604-6.584-.685-7.258-.007-.05-.011-.1-.017-.151A4 4 0 0 0 17.32 5z"/>',
    "testtube": '<path d="M14.5 2v17.5a2.5 2.5 0 0 1-5 0V2"/><path d="M8.5 2h7"/><path d="M14.5 16h-5"/>',
    "gear": '<circle cx="12" cy="12" r="3"/><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>',
    "package": '<path d="M16.5 9.4 7.55 4.24"/><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.29 7 12 12 20.71 7"/><line x1="12" x2="12" y1="22" y2="12"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>',
    "trash": '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    "plus": '<circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="16"/><line x1="8" x2="16" y1="12" y2="12"/>',
    "play": '<polygon points="6 3 20 12 6 21 6 3"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "speaker": '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
    "book": '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "film": '<rect x="2" y="2" width="20" height="20" rx="2.18"/><path d="M7 2v20"/><path d="M17 2v20"/><path d="M2 12h20"/><path d="M2 7h5"/><path d="M2 17h5"/><path d="M17 17h5"/><path d="M17 7h5"/>',
    "music": '<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>',
    "waves": '<path d="M2 6c.6.5 1.2 1 2.5 1C7 7 7 5 9.5 5c2.6 0 2.4 2 5 2 2.5 0 2.437-2 5.5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 12c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.437-2 5.5-2 1.3 0 1.9.5 2.5 1"/><path d="M2 18c.6.5 1.2 1 2.5 1 2.5 0 2.5-2 5-2 2.6 0 2.4 2 5 2 2.5 0 2.437-2 5.5-2 1.3 0 1.9.5 2.5 1"/>',
    "warning": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "check": '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/>',
}


def svg_icon(name: str, size: int = 18, color: str = "#66fcf1") -> str:
    """Render a premium stroke-style SVG icon as inline HTML."""
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
        f"stroke='{color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' "
        f"style='vertical-align:-3px;margin-right:7px;'>{SVG_ICONS[name]}</svg>"
    )


def svg_dot(color: str, size: int = 12) -> str:
    """Render a small solid color dot (track bullets)."""
    return (
        f"<svg width='{size}' height='{size}' viewBox='0 0 12 12' "
        f"style='vertical-align:-1px;margin-right:7px;'><circle cx='6' cy='6' r='5' fill='{color}'/></svg>"
    )

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
_favicon = Path(__file__).resolve().parent / "assets" / "favicon.png"
st.set_page_config(
    page_title="Math-DAW: Viral Math Phonk & Equation Studio",
    page_icon=str(_favicon) if _favicon.exists() else None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a sleek dark aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #07090e 0%, #151922 50%, #0c0f17 100%);
    }
    .main-title {
        background: linear-gradient(90deg, #ff007f, #ff7597, #66fcf1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 0.1rem;
        letter-spacing: -0.5px;
    }
    .subtitle {
        color: #c5c6c7;
        text-align: center;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .reel-banner {
        background: linear-gradient(90deg, rgba(255, 0, 127, 0.15), rgba(102, 252, 241, 0.1));
        border: 1px solid rgba(255, 0, 127, 0.4);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .reel-title {
        color: #ff007f;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }
    .equation-code {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        background: rgba(11, 12, 16, 0.85);
        border-left: 4px solid #ff007f;
        padding: 0.8rem;
        border-radius: 0 8px 8px 0;
        color: #66fcf1;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] {
        background: rgba(21, 25, 34, 0.6);
        border: 1px solid rgba(102, 252, 241, 0.2);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f'<div class="main-title">{svg_icon("bolt", size=34, color="#ff007f")}MATH FUNCTIONS: Every Sound is a Graph</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Viral Math Phonk Suite • The Sinc Drop • Pure Equation Audio Synthesis</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_pattern(pattern_str: str):
    """Parse comma-separated MIDI pattern string."""
    if not pattern_str.strip():
        return None
    items = []
    for item in pattern_str.split(","):
        item = item.strip()
        if item.lower() in ("none", "rest", "_", "-", ""):
            items.append(None)
        else:
            try:
                items.append(int(item))
            except ValueError:
                items.append(None)
    return items if items else None


def _pattern_to_str(pattern) -> str:
    """Format pattern list to string."""
    if not pattern:
        return ""
    return ", ".join("None" if p is None else str(p) for p in pattern)


def _build_project(bpm_val: float = 140.0, dur_val: float = 6.0, seed_val: int = 7) -> Project:
    """Reconstruct Project from current session state."""
    tracks_data = st.session_state.get("tracks", [])
    p = Project(
        bpm=st.session_state.get("bpm", bpm_val),
        duration=st.session_state.get("duration", dur_val),
        seed=seed_val,
    )
    for tr in tracks_data:
        p.add(
            name=tr["name"],
            equation=tr["equation"],
            pattern=tr.get("pattern"),
            gain=tr.get("gain", 1.0),
            pan=tr.get("pan", 0.0),
        )
    return p


def _render_bytebeat_player(suffix: str = "") -> None:
    """Render the self-contained Web Audio bytebeat player.

    Suffix parametrizes widget/session keys so multiple independent copies
    can live in different tabs.
    """
    import streamlit.components.v1 as components

    code_key = f"bb_code{suffix}"
    preset_key = f"bb_preset{suffix}"

    BB_VIDEO_EQ = "x=t/8000*2,(3.2*Math.exp(-.5*(x%4))*(1-(2*((x%1)-1))**2)-1.6)/8"
    BB_PHONK = (
        "S=8000,B=60/140*S,\n"
        "K=Math.exp(-(t%B)/140)*Math.sin(6.283*48*t/S),\n"
        "N=[55,55,65.41,49][((t/(B/2))|0)%4],\n"
        "W=Math.tanh(2.5*Math.sin(6.283*N*t/S))*Math.exp(-(t%(B/2))/700),\n"
        "A=[220,277.18,329.63,440,329.63,277.18][((t/(B/4))|0)%6],\n"
        "L=.5*Math.sin(6.283*A*t/S+2*Math.sin(12.566*A*t/S))*Math.exp(-(t%(B/4))/300),\n"
        "H=((t*15923%997)/997-.5)*Math.exp(-(t%(B/4))/70),\n"
        "(K+W+L*.8+H*.5)/3"
    )
    bb_presets = {
        "Video equation (math remix)": BB_VIDEO_EQ,
        "Math phonk loop": BB_PHONK,
    }
    ai_remake = Path("data/output/video_remake/audio_bytebeat.txt")
    if ai_remake.exists():
        try:
            lines = [
                ln for ln in ai_remake.read_text(encoding="utf-8").splitlines()
                if not ln.startswith("//")
            ]
            if lines:
                bb_presets["AI remake of your video song"] = "\n".join(lines)
        except Exception:
            pass

    preset_name = st.selectbox("Preset function", list(bb_presets.keys()), key=f"bb_preset_sel{suffix}")
    if code_key not in st.session_state or st.session_state.get(preset_key) != preset_name:
        st.session_state[code_key] = bb_presets[preset_name]
        st.session_state[preset_key] = preset_name

    bb_code = st.text_area(
        "Floatbeat function f(t) → sample in [-1, 1]",
        value=st.session_state[code_key],
        height=220,
        key=f"bb_code_area{suffix}",
    )
    st.session_state[code_key] = bb_code

    col_bb1, col_bb2 = st.columns([1, 3])
    with col_bb1:
        bb_dur = st.slider("Play length (seconds)", 2.0, 30.0, 8.0, step=1.0, key=f"bb_dur{suffix}")
    with col_bb2:
        st.download_button(
            "Download function (.txt)",
            data=bb_code,
            file_name="bytebeat_function.txt",
            mime="text/plain",
            key=f"bb_dl{suffix}",
        )

    play_svg = svg_icon("play", size=15, color="#ffffff").replace("margin-right:7px;", "")
    stop_svg = svg_icon("stop", size=15, color="#ffffff").replace("margin-right:7px;", "")
    bb_html = (
        "<button id='bb_play" + suffix + "' style='font-size:16px;padding:6px 18px;'>" + play_svg + " Play</button> "
        "<button id='bb_stop" + suffix + "' style='font-size:16px;padding:6px 18px;'>" + stop_svg + " Stop</button> "
        "<label>Vol <input id='bb_vol" + suffix + "' type='range' min='0' max='100' value='70'></label> "
        "<span id='bb_status" + suffix + "' style='margin-left:10px;'></span>"
        "<script>"
        "let bbCtx" + suffix + "=null,bbSrc" + suffix + "=null;"
        "const BB_CODE=" + json.dumps(bb_code) + ";"
        "const BB_DUR=" + json.dumps(float(bb_dur)) + ";"
        "const BB_SR=8000;"
        "function bbStop" + suffix + "(){if(bbSrc" + suffix + "){try{bbSrc" + suffix + ".stop();}catch(e){}bbSrc" + suffix + "=null;}}"
        "document.getElementById('bb_play" + suffix + "').onclick=()=>{try{"
        "if(!bbCtx" + suffix + ")bbCtx" + suffix + "=new (window.AudioContext||window.webkitAudioContext)();"
        "if(bbCtx" + suffix + ".state==='suspended')bbCtx" + suffix + ".resume();"
        "bbStop" + suffix + "();"
        "const fn=new Function('t','return ('+BB_CODE+')');"
        "const n=Math.floor(BB_SR*BB_DUR);"
        "const buf=bbCtx" + suffix + ".createBuffer(1,n,BB_SR);"
        "const d=buf.getChannelData(0);"
        "for(let t=0;t<n;t++){let v=fn(t);"
        "d[t]=Math.max(-1,Math.min(1,(typeof v==='number'&&isFinite(v))?v:0));}"
        "bbSrc" + suffix + "=bbCtx" + suffix + ".createBufferSource();bbSrc" + suffix + ".buffer=buf;"
        "const g=bbCtx" + suffix + ".createGain();"
        "g.gain.value=document.getElementById('bb_vol" + suffix + "').value/100;"
        "bbSrc" + suffix + ".connect(g);g.connect(bbCtx" + suffix + ".destination);bbSrc" + suffix + ".start();"
        "document.getElementById('bb_status" + suffix + "').textContent='playing '+BB_DUR+'s @ 8000Hz';"
        "}catch(e){document.getElementById('bb_status" + suffix + "').textContent='Error: '+e.message;}};"
        "document.getElementById('bb_stop" + suffix + "').onclick=()=>{bbStop" + suffix + "();"
        "document.getElementById('bb_status" + suffix + "').textContent='stopped';};"
        "</script>"
    )
    components.html(bb_html, height=120)
    st.caption("Tip: switch presets or edit the code, then press Play. Nothing is uploaded anywhere.")


def _estimate_seconds(dur: float, iters: int, stems: str, bank: bool) -> float:
    """Estimated wall-clock for a RES run, from measurements on this machine.

    Measured on an 8 s dense clip, no stem separation:
        bank on : 2 iterations -> 18.3 s,  4 iterations -> 63.5 s
        bank off: 2 iterations -> 12.2 s,  4 iterations -> 19.9 s
    The cost grows super-linearly with iteration count (the bank proposes more
    hypotheses each pass), so it is fitted as a power law rather than a guess.
    Duration scales the fixed analysis cost linearly.
    """
    base, power = (18.3, 1.795) if bank else (12.2, 0.70)
    total = base * (max(1, iters) / 2.0) ** power * (max(dur, 1.0) / 8.0)
    if stems in ("auto", "demucs"):
        total *= 3.5   # separation plus fitting each stem separately
    return total


def _audio_player(data: bytes, mime: str = "audio/wav", height: int = 54) -> None:
    """Render an <audio> player that actually plays.

    Streamlit's /media/ endpoint answers the browser's follow-up Range request
    with 204 No Content, which Chromium reports as MEDIA_ERR_NETWORK -- the
    track plays for ~0.4 s and then stops. A base64 data URI is not an HTTP
    request at all, so there is no Range header and no 204 to trip over.
    """
    import base64
    b64 = base64.b64encode(data).decode("ascii")
    st.components.v1.html(
        f'<audio controls style="width:100%;height:{height}px" '
        f'src="data:{mime};base64,{b64}"></audio>',
        height=height + 8,
    )


def _audio_to_bytes(y: np.ndarray, sr: int = 44100) -> bytes:
    """Convert float audio array into 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        peak = np.max(np.abs(y))
        y_norm = y / (peak + 1e-12) * 0.9 if peak > 1.0 else y
        pcm = (np.clip(y_norm, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.getvalue()


def _res_render_results(p: dict) -> None:
    """Render the original -> equations -> recreation story from a RES payload."""
    b = p["best"]
    lsd = b.get("log_spectral_distance_db")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("correlation", f"{b['correlation']:.4f}")
    m2.metric("SNR", f"{b['snr_db']:.2f} dB")
    m3.metric("LSD", "n/a" if lsd is None else f"{lsd:.2f} dB")
    m4.metric("segmental SNR", f"{(b.get('seg_snr_db') or 0.0):.2f} dB")
    m5.metric("onset corr", f"{(b.get('onset_corr') or 0.0):.3f}")
    st.caption(
        f"equation classes in the final model: "
        f"{', '.join(p['equation_classes']) or 'damped_cosine'}"
    )
    if p.get("separation_note"):
        note = p["separation_note"]
        if "HPSS" in note or "unavailable" in note:
            st.warning("⚠️ " + note)
        else:
            st.caption(note)

    st.markdown(f"#### {svg_icon('music')}1 - Listen", unsafe_allow_html=True)
    tabs = st.tabs(["Original", "Recreated", "Residual"])
    with tabs[0]:
        _audio_player(p.get("orig_bytes") or b"", "audio/wav")
        st.caption(
            "The analysed span only (the slider length) - not the whole song."
        )

    st.markdown(f"#### {svg_icon('flask')}2 · The math equations", unsafe_allow_html=True)
    st.markdown(r"$s_k(t)=A_k\,e^{-\lambda_k (t-t_{0,k})}\,\cos(2\pi f_k t-\varphi_k)$"
                "&nbsp;&nbsp;+ whole notes "
                r"$(f_m=mF_0\sqrt{1+B\,m^{2}},\ \sigma_m=a+b f_m)$"
                "&nbsp;+ transient hit bundles.")
    n_eq_lines = p["equations"].count("\n") + 1
    sec_eqs = p.get("section_equations") or []
    if sec_eqs:
        _labels = [
            f"Section {i+1}  "
            f"({s.get('start', 0.0):.1f}s-{s.get('end', 0.0):.1f}s, "
            f"corr {s.get('corr', 0.0):.3f})"
            for i, s in enumerate(p.get("sections") or [])
        ]
        if len(_labels) != len(sec_eqs):
            _labels = [f"Section {i+1}" for i in range(len(sec_eqs))]
        _pick = st.selectbox(
            "Equations - pick a section", options=list(range(len(sec_eqs))),
            format_func=lambda i: _labels[i], key="res_eq_pick")
        st.code(sec_eqs[_pick], language="text")
    with st.expander(f"All equations ({n_eq_lines} lines)"):
        head = "\n".join(p["equations"].splitlines()[:40])
        st.code(head + ("\n... (truncated; full text below)" if n_eq_lines > 40 else ""),
                language="text")
        st.download_button("Download all equations", p["equations"],
                           file_name="res_equations.txt", key="res_dl_eq")
    if p["accepted"]:
        with st.expander(f"Hypothesis bank — {len(p['accepted'])} accepted record(s)"):
            st.dataframe(pd.DataFrame(p["accepted"]), hide_index=True,
                         use_container_width=True)

    with tabs[1]:
        if p["best_bytes"]:
            _audio_player(p.get("best_bytes") or b"", "audio/wav")
            st.caption(
                f"evidence corr {p['start_corr']:.4f} -> best "
                f"{b['correlation']:.4f} after the loop; every accepted and "
                f"rejected try is in the ledger."
            )
        else:
            st.warning("No reconstruction WAV was produced.")

    with tabs[2]:
        if p.get("residual_bytes"):
            _audio_player(p.get("residual_bytes") or b"", "audio/wav")
            st.caption(
                "Original minus reconstruction. This is where the remaining "
                "error lives: transients, noise floor and high-frequency air "
                "that a sum of sinusoids cannot represent."
            )
        else:
            st.info("No residual was computed.")

    stems = p.get("stems")
    if not stems and Path(p.get("ledger_path", "")).exists():
        try:
            raw_led = json.loads(Path(p["ledger_path"]).read_text(encoding="utf-8"))
            stems = raw_led.get("stems")
        except Exception:
            pass

    if stems:
        st.markdown(f"#### {svg_icon('sliders')}Multi-Track Stems Breakdown (Solo & Metrics)", unsafe_allow_html=True)
        stem_cols = st.columns(len(stems))
        for col, (stem_name, s_data) in zip(stem_cols, stems.items()):
            with col:
                st.markdown(f"**{stem_name.upper()}**")
                sm = s_data.get("metrics", {})
                st.caption(
                    f"Corr: {sm.get('correlation', 0.0):.4f}\n\n"
                    f"SNR: {sm.get('snr_db', 0.0):.1f} dB\n\n"
                    f"LSD: {sm.get('log_spectral_distance_db', 0.0):.1f} dB"
                )
                stem_wav = s_data.get("outputs", {}).get("wav")
                if stem_wav and Path(stem_wav).exists():
                        _audio_player(Path(stem_wav).read_bytes(), "audio/wav")

    with st.expander("Iteration trail (accept / rollback receipts)"):
        st.dataframe(pd.DataFrame(p["trail"]), hide_index=True,
                     use_container_width=True)
    with st.expander("Per-section referee scores"):
        sec_df = pd.DataFrame(p["sections"])
        if not sec_df.empty:
            st.bar_chart(sec_df.set_index("start")[["corr"]], height=170)
        st.dataframe(sec_df, hide_index=True, use_container_width=True)
        st.caption(f"evidence start per section: {p['start_sections']}")

    d1, d2, d3 = st.columns([1, 1, 2])
    if p["best_bytes"]:
        d1.download_button("Recreated WAV", p["best_bytes"],
                           file_name="res_recreation.wav", key="res_dl_wav")
    if Path(p["ledger_path"]).exists():
        d2.download_button("Ledger JSON", Path(p["ledger_path"]).read_bytes(),
                           file_name="res_ledger.json", key="res_dl_ledger")
    d3.caption(f"ledger: `{p['ledger_path']}` · recreated WAV: `{p['best_wav']}`")


# ---------------------------------------------------------------------------
# Sidebar: Project controls & presets
# ---------------------------------------------------------------------------
# ===========================================================================
# TAB 0: RES - Song -> Equations -> Replay (the agentic loop)
# ===========================================================================
# ---------------------------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------------------------
tab_res, tab_bench, = st.tabs([
    "RES: Song -> Equations -> Replay",
    "Benchmarks: sample tests",
])

# ===========================================================================
# TAB 1: Benchmarks - the sample-test harness, run and read in the browser
# ===========================================================================
with tab_bench:
    st.markdown(
        f"### {svg_icon('flask')}Benchmarks - the sample-test harness",
        unsafe_allow_html=True,
    )
    st.caption(
        "A *label* is one run of the engine over the frozen fixture set; a "
        "*row* is one fixture inside it. Every label is re-runnable from the "
        "command line - this tab is a view over `data/output/scoreboard.json` "
        "plus a button that runs the same CLI in the background."
    )

    from src.scoreboard import FIXTURE_ORDER, FIXTURES, load, verify

    b1, b2, b3 = st.columns([1, 1, 1.2])
    with b1:
        bench_label = st.text_input("Label name", value="after",
                                    key="bench_label",
                                    help="The name this run is stored under.")
    with b2:
        bench_dur = st.slider("Seconds analysed", 4.0, 30.0, 10.0, 1.0,
                              key="bench_dur")
    with b3:
        bench_iters = st.slider("Iterations", 1, 8, 6, key="bench_iters")
    bench_fixtures = st.multiselect("Fixtures", list(FIXTURES.keys()),
                                    default=list(FIXTURE_ORDER),
                                    key="bench_fixtures")

    st.caption(
        "This is a **harness**, not an executable: it drives the same "
        "`main.py res` CLI you can type by hand, so any label here can be "
        "reproduced outside the app. Each fixture takes 30-200 s, so the run "
        "happens in the background."
    )

    run_pressed_b = st.button("Run sample test", type="primary",
                               disabled=not bench_fixtures,
                               key="bench_run_btn")

    if run_pressed_b:
        from src.res_ui import spawn_scoreboard_job
        spawn_scoreboard_job(bench_label, fixtures=bench_fixtures,
                             duration=bench_dur, iterations=bench_iters,
                             root=Path(__file__).resolve().parent)
        st.rerun()

    from src.res_ui import done_scoreboard_job, running_scoreboard_jobs
    _sb_running = running_scoreboard_jobs()
    if _sb_running:
        _job = _sb_running[0]
        _bar = st.progress(0.0)
        _ph = st.empty()
        while _job.get("state") in ("queued", "running"):
            _tot = max(1, int(_job.get("total") or 1))
            _done_n = min(_tot, int(_job.get("done") or 0))
            _bar.progress(_done_n / _tot)
            _el = max(0.0, time.monotonic() - _job.get("t0", time.monotonic()))
            _c = _job.get("corr")
            _ph.write(
                f"label **{_job.get('label')}** - {_job.get('stage')} "
                f"· {_done_n}/{_tot} fixtures"
                f"{f' · corr {_c:.4f}' if isinstance(_c, float) else ''}"
                f" · {int(_el // 60)}m {int(_el % 60)}s elapsed"
            )
            time.sleep(1.0)
        _bar.progress(1.0)
        if _job.get("state") == "error":
            st.error(f"sample test failed: {_job.get('error')}")
        else:
            st.success(f"label '{_job.get('label')}' written to the scoreboard")

    _sb_done = done_scoreboard_job()
    if _sb_done and _sb_done.get("state") == "error":
        st.error(f"sample test failed: {_sb_done.get('error')}")

    st.divider()
    st.markdown("#### Recorded labels", unsafe_allow_html=True)
    _sb = load()
    _labels = sorted((_sb.get("labels") or {}).keys())
    if not _labels:
        st.info("No labels recorded yet - run one above.")
    else:
        _hdr = "".join(
            f"<th style='text-align:right'>{n}</th>" for n in FIXTURE_ORDER)
        _rows_html = ""
        for _lbl in _labels:
            _rows = (_sb["labels"][_lbl] or {})
            _cells = ""
            for _n in FIXTURE_ORDER:
                _c = (_rows.get(_n) or {}).get("correlation")
                _cells += (f"<td style='text-align:right'>{_c:.6f}</td>"
                           if isinstance(_c, float) else "<td>-</td>")
            _note = (_sb.get("expect_improvement") or {}).get(_lbl)
            _tag = ("  <span style='color:#b91c1c'>captured regression</span>"
                    if _note is False else "")
            _rows_html += (f"<tr><td class='mono'>{_lbl}</td>{_cells}"
                           f"<td style='color:#64748b'>{_note}</td></tr>")
        st.markdown(
            f"<table style='width:100%;border-collapse:collapse'>"
            f"<tr><th style='text-align:left'>label</th>{_hdr}"
            f"<th style='text-align:right'>expect improvement</th></tr>"
            f"{_rows_html}</table>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Verification", unsafe_allow_html=True)
    _res = verify()
    for _lbl, _v in sorted(_res["labels"].items()):
        _cls = "ok" if _v["consistent"] else "INCONSISTENT"
        _imp = "improved" if _v["improved"] else "did not improve"
        if _v["expect_improvement"] is False:
            _imp += " (expected: captured regression)"
        st.caption(f"**{_lbl}** - consistent={_v['consistent']} · {_imp}")
    if _res["unexpected_regressions"]:
        st.error("UNEXPECTED REGRESSION (must improve): "
                 + ", ".join(_res["unexpected_regressions"]))
    else:
        st.success("VERDICT: OK" if _res["ok"] else "VERDICT: FAIL")

    with st.expander("Reproduce a label from the command line"):
        st.code(
            ".venv\\Scripts\\python.exe main.py res "
            "-i data/output/_fixtures/stem_drums.wav "
            "-o data/output/_scoreboard/after/stem_drums "
            "-d 10 --iterations 6\n\n"
            ".venv\\Scripts\\python.exe tools\\verify_scoreboard.py",
            language="text",
        )

with tab_res:
    st.markdown(
        f"### {svg_icon('waves')}RES - reverse engineer a song into equations, then replay it",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.15], gap="large")

    # ---------------------------------------------------------------- controls
    with left:
        up = st.file_uploader(
            "Song (mp3 / wav / ogg / flac)",
            type=["mp3", "wav", "ogg", "flac", "m4a"], key="res_upload",
        )
        # Play the upload from disk so the <audio> URL stays stable across
        # re-renders (passing bytes makes Streamlit mint a new media file each
        # time, and the browser then restarts playback from zero).
        p_full_upload = None
        if up is not None:
            import tempfile as _tf
            _d = _tf.mkdtemp(prefix="res_upload_")
            _f = Path(_d) / (Path(up.name).name or "upload.wav")
            _f.write_bytes(up.getvalue())
            p_full_upload = str(_f)
            _audio_player(up.getvalue(), up.type or "audio/wav")
            st.caption(
                "Your upload, in full. The reconstruction below covers only "
                "the analysed span."
            )
        # Preset first: it seeds the sliders below, so the cost of a run is
        # never a guess. Timings are measured on this machine.
        PRESETS = {
            "Fast": (6.0, 2, True),
            "Balanced": (8.0, 4, True),
            "Quality": (15.0, 6, True),
        }
        res_preset = st.selectbox(
            "Preset", list(PRESETS), index=1, key="res_preset",
            help="Measured on this machine: Fast ~11 s, Balanced ~60 s, "
                 "Quality ~3 min for a dense clip (no stem separation).",
        )
        if st.session_state.get("_res_preset_seen") != res_preset:
            _d, _it, _bk = PRESETS[res_preset]
            st.session_state["res_dur"] = _d
            st.session_state["res_iters"] = _it
            st.session_state["res_bank"] = _bk
            st.session_state["_res_preset_seen"] = res_preset

        c1, c2 = st.columns(2)
        with c1:
            res_dur = st.slider("Analyse seconds", 4.0, 30.0, 8.0, 1.0,
                                key="res_dur")
        with c2:
            res_iters = st.slider("Iterations", 1, 6, 4, key="res_iters")
        c3, c4 = st.columns(2)
        with c3:
            res_stems = st.selectbox(
                "Stem separation", ["none", "auto", "demucs"], key="res_stems",
                help="none = one mix. auto/demucs = source separation first, "
                     "which reaches much higher per-stem correlation.",
            )
        with c4:
            res_bank = st.toggle("Hypothesis bank", value=True, key="res_bank")
        c5, c6 = st.columns(2)
        with c5:
            res_noise = st.toggle(
                "Noise model", value=False, key="res_noise",
                help="Model the leftover residual with a stochastic class "
                     "(LPC formant noise). A sum of sinusoids cannot express "
                     "breath, air or sibilance, so that content is absent "
                     "without it.",
            )
        with c6:
            res_lane = st.toggle(
                "Spectral lane", value=False, key="res_lane",
                help="Let stochastic bursts be accepted on log-spectral "
                     "distance instead of waveform correlation.",
            )

        from src.res_ui import demucs_status
        demucs_ok, demucs_msg = demucs_status()
        if res_stems in ("auto", "demucs"):
            if demucs_ok:
                st.caption("\u2705 " + demucs_msg)
            else:
                st.warning("**Stem separation will silently degrade.** " + demucs_msg)

        est = _estimate_seconds(res_dur, res_iters, res_stems, res_bank)
        run_pressed = st.button(
            "\u25b6 Recreate", type="primary", disabled=up is None,
            use_container_width=True, key="res_run_btn",
        )
        st.caption(
            f"~{est / 60:.0f} min estimated for {res_dur:.0f} s at "
            f"{res_iters} iteration(s)"
            f"{' with source separation' if res_stems != 'none' else ''}."
        )
        with st.expander("What the numbers mean"):
            st.markdown(
                "**Honest ceiling.** A full commercial master caps near "
                "0.65-0.75 correlation: brickwall limiting and phase jitter are "
                "not invertible by a sum of sinusoids. Real source separation is "
                "what lifts the *per-stem* numbers into the 0.9+ range. That is a "
                "physics limit, not a tuning failure.\n\n"
                "**Live progress is per-stage and per-iteration**, not instant "
                "synthesis: the engine analyses, fits, then iterates."
            )

    # ------------------------------------------------------------ live status
    with right:
        from src.res_ui import running_jobs as _running_jobs

        if _running_jobs():
            # The live panel is a self-refreshing fragment, but it is only
            # mounted while a job is actually running. An always-on fragment
            # re-renders the page every second forever, which remounts the
            # <audio> elements below and makes playback restart from zero
            # after about a second.
            @st.fragment(run_every=1.0)
            def _res_live_panel() -> None:
                from src.res_ui import running_jobs
                running = running_jobs()
                if not running:
                    st.rerun(scope="app")
                    return
                job = running[0]
                st.markdown("**Running**")
                stages = job.get("stages") or []
                if stages:
                    rows = [{
                        "stage": s.get("name", ""),
                        "detail": s.get("detail", ""),
                        "pct": f"{100 * float(s.get('pct', 0.0)):.0f}%",
                        "sec": f"{float(s.get('seconds', 0.0)):.1f}",
                    } for s in stages]
                    st.dataframe(pd.DataFrame(rows), hide_index=True,
                                 use_container_width=True, height=190)
                cur = job.get("stage") or "working"
                pct = float(job.get("stage_pct") or 0.0)
                detail = job.get("stage_detail") or ""
                st.progress(min(0.99, max(0.02, pct)))
                st.caption(f"{cur} - {detail}" if detail else cur)
                m1, m2, m3 = st.columns(3)
                corr = job.get("corr")
                m1.metric("correlation",
                          "-" if corr is None else f"{corr:.4f}")
                hist = job.get("corr_history") or []
                snr = hist[-1]["snr_db"] if hist else None
                m2.metric("SNR", "-" if snr is None else f"{snr:.2f} dB")
                lsd = hist[-1]["lsd_db"] if hist else None
                m3.metric("LSD", "-" if lsd is None else f"{lsd:.2f} dB")
                if hist:
                    st.line_chart(
                        pd.DataFrame(hist).set_index("iteration")[
                            ["corr", "snr_db", "lsd_db"]], height=170)

            _res_live_panel()
        else:
            st.info("Upload a song and press **Recreate**. The engine "
                    "reports each stage as it runs.")

    if run_pressed and up is not None:
        from src.res_ui import spawn_job
        key = (f"{up.name}:{up.size}:{round(res_dur)}:{res_iters}:{res_bank}"
               f":{res_stems}:{res_noise}:{res_lane}")
        spawn_job(key, up.getvalue(), up.name, res_dur, res_iters, res_bank,
                  root=Path(__file__).resolve().parent, stem_mode=res_stems,
                  enable_noise=res_noise, spectral_lane=res_lane)
        st.rerun()

    from src.res_ui import done_job as _done_job
    _done = _done_job()
    if _done:
        _res_render_results(_done["payload"])

# ===========================================================================
# TAB 1: Viral Math Phonk (The Exact Reel Equations)
# ===========================================================================
# ===========================================================================
# TAB 6: Song A/B (original vs recreation vs error-corrected)
# ===========================================================================
# ===========================================================================
# TAB 3: Formula Bank Explorer
# ===========================================================================
# ===========================================================================
# TAB 4: Spectrogram Math (CQT vs Reassigned vs STFT)
# ===========================================================================
# ===========================================================================
# TAB 5: Bytebeat Player (own implementation, Web Audio — no external code)
# ===========================================================================
# ===========================================================================
# TAB 2: Math-DAW Studio
# ===========================================================================
