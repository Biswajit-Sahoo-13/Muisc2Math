"""The note layer: frequency-native equations (the user's idea, made precise).

Instead of fitting free frequencies and hoping they land on music, DETECT
the frequencies and write the equations FROM them:
  - 12-TET snapping: a detected f0 becomes the nearest equal-tempered
    pitch (f = 440 * 2^((n-69)/12)); the residual detune stays a parameter.
  - Beat pairs: two partials within ~60 cents of each other are not two
    accidents - they are ONE musical object (a detuned unison) whose
    beat frequency is itself a parameter of the sound. cos A + cos B
    folds to 2 cos((A+B)/2) cos((A-B)/2), so a pair has a carrier, a
    beat rate, and - when the decays differ - a beat that deepens or
    dies over the note. That beat is one of the most salient features
    of a mix's low end (Plomp-Levelt: two subs a few Hz apart sit in
    maximum roughness), and the magnitude spectrum is nearly blind to it.

First increment (zero render risk): DETECT and RECORD. The renderer keeps
emitting the two partials bit-identically; the ledger gains the beat-pair
view so the referee and the human can see what the free-frequency fit was
hiding. Measured on the user's clip: the sub-bass residual carries a comb
of lines spaced ~1.0-1.35 Hz (the 808's beat), three of its strongest
peaks (36.8, 46.0, 47.4 Hz) have NO partial at all, and 18/20 of the
strongest fitted partials sit 10-48 cents off any musical pitch.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

A4_HZ = 440.0
# A pair must be close in PITCH (cents) as well as in Hz: 15 Hz is 470
# cents at 55 Hz (would merge genuinely different notes) but only 64
# cents at 400 Hz. The window is min of the two bounds.
_PAIR_MAX_HZ = 15.0
_PAIR_MAX_CENTS = 60.0
_MIN_AMPLITUDE = 1e-3
_TONAL_MODELS = ("damped", "damped_cosine", "constant")


def hz_to_note(f: float) -> Dict[str, Any]:
    """Nearest 12-TET note (A4 = 440) for a frequency."""
    if f <= 0:
        return {"midi": 0, "note": "?", "note_hz": 0.0, "cents": 0.0}
    midi = 69.0 + 12.0 * np.log2(f / A4_HZ)
    nearest = int(round(midi))
    note_hz = A4_HZ * 2.0 ** ((nearest - 69) / 12.0)
    names = ["C", "C#", "D", "D#", "E", "F", "F#",
             "G", "G#", "A", "A#", "B"]
    name = f"{names[nearest % 12]}{nearest // 12 - 1}"
    cents = 1200.0 * np.log2(f / note_hz) if note_hz > 0 else 0.0
    return {"midi": nearest, "note": name, "note_hz": round(note_hz, 3),
            "cents": round(cents, 1)}


def _support(p: Dict[str, Any]) -> tuple:
    return (float(p.get("start_time", 0.0)),
            float(p.get("end_time", 0.0)))


def _overlap_frac(a: tuple, b: tuple) -> float:
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi <= lo:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return (hi - lo) / shorter


def detect_beat_pairs(
    partials: List[Dict[str, Any]],
    max_hz: float = _PAIR_MAX_HZ,
    max_cents: float = _PAIR_MAX_CENTS,
    min_overlap: float = 0.5,
) -> List[Dict[str, Any]]:
    """Find detuned-unison pairs among the fitted partials.

    Two tonal partials are a beat pair when their frequencies sit within
    ``min(max_hz, max_cents of the lower f)`` of each other and their
    supports overlap. The record carries the carrier (average), the beat
    rate |f1 - f2| (the intensity beat - the envelope swells at half of
    it; CCRMA/Feynman), both amplitudes and decays (independent: the
    beat deepening as one layer dies is the physics of layered 808s),
    and the 12-TET snap of the carrier with its cents deviation.
    """
    tonal = [
        p for p in partials
        if p.get("model", "damped") in _TONAL_MODELS
        and float(p.get("amplitude", 0.0)) > _MIN_AMPLITUDE
        and float(p.get("frequency", 0.0)) > 0.0
    ]
    tonal.sort(key=lambda p: float(p["frequency"]))
    records: List[Dict[str, Any]] = []
    used: set = set()
    for i in range(len(tonal)):
        if id(tonal[i]) in used:
            continue
        f1 = float(tonal[i]["frequency"])
        for j in range(i + 1, len(tonal)):
            if id(tonal[j]) in used:
                continue
            f2 = float(tonal[j]["frequency"])
            cents = 1200.0 * np.log2(f2 / f1) if f1 > 0 else 0.0
            if abs(cents) > max_cents or (f2 - f1) > max_hz:
                break  # sorted by f: everything after is farther
            if _overlap_frac(_support(tonal[i]), _support(tonal[j])) < min_overlap:
                continue
            f_avg = 0.5 * (f1 + f2)
            snap = hz_to_note(f_avg)
            records.append({
                "f_carrier_hz": round(f_avg, 4),
                "f_beat_hz": round(f2 - f1, 4),
                "f1_hz": round(f1, 4),
                "f2_hz": round(f2, 4),
                "a1": round(float(tonal[i].get("amplitude", 0.0)), 6),
                "a2": round(float(tonal[j].get("amplitude", 0.0)), 6),
                "lam1": round(float(tonal[i].get("envelope_decay", 0.0)), 4),
                "lam2": round(float(tonal[j].get("envelope_decay", 0.0)), 4),
                "t1": round(_support(tonal[i])[0], 4),
                "t2": round(_support(tonal[j])[0], 4),
                "carrier_note": snap["note"],
                "carrier_note_hz": snap["note_hz"],
                "carrier_cents_off": snap["cents"],
                # Every frequency is within 50 cents of SOME note, so the
                # snap tolerance must be tighter than a quarter tone: past
                # 25 cents the carrier is a glide or a non-440 tuning, not
                # a pitch to snap.
                "snappable": bool(abs(snap["cents"]) < 25.0),
            })
            used.add(id(tonal[i]))
            used.add(id(tonal[j]))
            break
    records.sort(key=lambda r: -max(r["a1"], r["a2"]))
    return records


def snap_carrier(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Move a beat pair's carrier to its 12-TET note, preserving the beat.

    The snap is symmetric: f1' = f_c' - f_beat/2, f2' = f_c' + f_beat/2 —
    the detune (the percept that defines the sound) is untouched, the
    master lands on the grid. Refuses carriers deeper than a quarter
    tone off (a glide or non-440 tuning, not a pitch): returned unchanged
    with ``applied=False`` so the referee can arbitrate.
    """
    f_c = float(pair.get("f_carrier_hz", 0.0))
    f_beat = float(pair.get("f_beat_hz", 0.0))
    note_hz = float(pair.get("carrier_note_hz", 0.0))
    cents = abs(float(pair.get("carrier_cents_off", 100.0)))
    out = dict(pair)
    if f_c <= 0 or note_hz <= 0 or cents >= 25.0:
        half = f_beat / 2.0
        out.update({
            "applied": False,
            "f1_snapped_hz": round(f_c - half, 4),
            "f2_snapped_hz": round(f_c + half, 4),
        })
        return out
    half = f_beat / 2.0
    out.update({
        "applied": True,
        "f1_snapped_hz": round(note_hz - half, 4),
        "f2_snapped_hz": round(note_hz + half, 4),
    })
    return out


def detect_note_events(
    y: np.ndarray,
    sr: int,
    f_lo: float = 30.0,
    f_hi: float = 150.0,
    min_note_s: float = 0.12,
    gap_s: float = 0.06,
    attack_s: float = 0.08,
) -> List[Dict[str, Any]]:
    """Segment a sub-bass band into note events with their glide law.

    The 808 is not a smeared trajectory: it is discrete notes, each with a
    short attack drop (~60 cents over ~250 ms, measured) then a quasi-
    stable sustain. Bandpass to [f_lo, f_hi], take the analytic signal's
    instantaneous frequency where the envelope is active, and segment into
    events. Returns ``{t0_s, t1_s, f_attack_hz, f_sustain_hz, drop_hz,
    drop_cents}`` per note, time-ordered.
    """
    from scipy.signal import hilbert

    x = np.asarray(y, dtype=np.float64)
    if x.size < int(sr * min_note_s):
        return []
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, 1.0 / sr)
    xb = np.fft.irfft(X * ((f >= f_lo) & (f < f_hi)), n=x.size)
    analytic = hilbert(xb)
    env = np.abs(analytic)
    inst_f = np.diff(np.unwrap(np.angle(analytic))) / (2.0 * np.pi) * sr
    inst_f = np.concatenate([inst_f[:1], inst_f])
    # Smooth the IF over ~10 ms: sample-rate IF is noisy inside the band.
    smooth_n = max(3, int(0.01 * sr) | 1)
    kernel = np.ones(smooth_n) / smooth_n
    inst_f = np.convolve(inst_f, kernel, mode="same")

    peak = float(env.max())
    if peak <= 1e-9:
        return []
    active = env > 0.25 * peak
    # Close small gaps (< gap_s) so one note is not split by envelope dips.
    min_gap = int(gap_s * sr)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    events: List[Dict[str, Any]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > min_gap:
            events.append((start, prev))
            start = i
        prev = i
    events.append((start, prev))

    out: List[Dict[str, Any]] = []
    min_len = int(min_note_s * sr)
    atk = max(8, int(attack_s * sr))
    for i0, i1 in events:
        if i1 - i0 < min_len:
            continue
        # Median IF is robust to the wobble at event edges.
        f_attack = float(np.median(inst_f[i0:min(i0 + atk, i1)]))
        f_sustain = float(np.median(inst_f[min(i0 + atk, i1):i1])) if i1 > i0 + atk else f_attack
        f_attack = max(f_attack, 1.0)
        f_sustain = max(f_sustain, 1.0)
        out.append({
            "t0_s": round(i0 / sr, 4),
            "t1_s": round(i1 / sr, 4),
            "f_attack_hz": round(f_attack, 3),
            "f_sustain_hz": round(f_sustain, 3),
            "drop_hz": round(f_attack - f_sustain, 3),
            "drop_cents": round(1200.0 * np.log2(f_attack / f_sustain), 1),
        })
    return out


def sub_bass_note_partials(
    y: np.ndarray,
    sr: int,
    f_lo: float = 30.0,
    f_hi: float = 150.0,
) -> List[Dict[str, Any]]:
    """Render-ready glide partials built FROM the detected note events.

    This is the frequency-native move applied to the sub-bass: one
    oscillator per detected note, its trajectory the measured law
    (attack f -> sustain f over ~250 ms, then sustain), its amplitude
    envelope the source's own band envelope during the note - instead of
    the engine's free glides that smear 5-15x the true excursion.
    """
    from scipy.signal import hilbert

    events = detect_note_events(y, sr, f_lo=f_lo, f_hi=f_hi)
    if not events:
        return []
    X = np.fft.rfft(y)
    f = np.fft.rfftfreq(y.size, 1.0 / sr)
    xb = np.fft.irfft(X * ((f >= f_lo) & (f < f_hi)), n=y.size)
    analytic = hilbert(xb)
    hop = max(1, int(0.02 * sr))
    n_frames = y.size // hop
    if n_frames < 4:
        return []
    env = np.array([
        float(np.sqrt(np.mean(xb[i * hop:(i + 1) * hop] ** 2)))
        for i in range(n_frames)
    ])
    out: List[Dict[str, Any]] = []
    for ev in events:
        i0 = int(ev["t0_s"] * sr)
        i1 = min(int(ev["t1_s"] * sr), y.size - 1)
        if i1 <= i0:
            continue
        f0 = int(ev["t0_s"] / (hop / sr))
        f1 = min(int(ev["t1_s"] / (hop / sr)), n_frames - 1)
        seg = env[f0:f1 + 1]
        peak = float(seg.max()) if seg.size else 0.0
        if peak <= 1e-9:
            continue
        drop_t = ev["t0_s"] + min(0.25, (ev["t1_s"] - ev["t0_s"]) / 2.0)
        tt = sorted({round(ev["t0_s"], 4), round(drop_t, 4),
                     round(ev["t1_s"], 4)})
        def env_at(t: float) -> float:
            i = min(max(int(t / (hop / sr)), 0), n_frames - 1)
            return env[i] / peak
        # Phase anchoring: the render's phase integral starts at the note
        # onset. The bandpassed source is Re[analytic] = |A| cos(angle(t))
        # with angle(t) = angle(t0) + integral, so setting the partial's
        # phase to -angle(t0) makes cos(theta - phase) track the source
        # exactly, whatever convention built it.
        src_phase = float(np.angle(analytic[min(i0, y.size - 1)]))
        phase = float((-src_phase + np.pi) % (2.0 * np.pi) - np.pi)
        out.append({
            "model": "glide",
            # The band envelope is an RMS measure; a cosine partial's
            # amplitude is a peak - convert, or the note renders sqrt(2)
            # too quiet.
            "amplitude": round(peak * 2.0 ** 0.5, 6),
            "projection_gain": 1.0,
            "phase": round(phase, 4),
            "frequency": ev["f_sustain_hz"],
            "trajectory": {
                "t": tt,
                "f_hz": [round(ev["f_attack_hz"], 3),
                         round(ev["f_sustain_hz"], 3),
                         round(ev["f_sustain_hz"], 3)],
                "a": [round(min(env_at(t), 1.0), 4) for t in tt],
            },
            "start_time": ev["t0_s"],
            "end_time": ev["t1_s"],
            "envelope_decay": 0.0,
            "envelope_t0": ev["t0_s"],
        })
    return out
