"""Polyphony and Note Tracking Engine.

Integrates:
1. librosa.pyin (Probabilistic YIN) for monophonic pitch and melody extraction.
2. basic-pitch (Spotify) with automatic graceful fallback if tensorflow/basic-pitch is omitted.
3. Harmonic grouping: maps sinusoidal tracks to Note Objects with fundamental pitch + overtones.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import librosa


@dataclass
class NoteEvent:
    """Represents a detected musical note event."""
    pitch_midi: int
    pitch_hz: float
    start_time: float
    end_time: float
    duration: float
    confidence: float = 1.0
    partials: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch_midi": self.pitch_midi,
            "pitch_hz": round(self.pitch_hz, 2),
            "start_time": round(self.start_time, 4),
            "end_time": round(self.end_time, 4),
            "duration": round(self.duration, 4),
            "confidence": round(self.confidence, 3),
            "partial_count": len(self.partials),
            "partials": self.partials,
        }


def extract_melody_pyin(
    y: np.ndarray,
    sr: int = 44100,
    fmin: float = 65.41,  # C2
    fmax: float = 2093.0,  # C7
    frame_length: int = 2048,
    hop_length: int = 512,
    min_note_duration: float = 0.08,
) -> List[NoteEvent]:
    """Extracts monophonic melody and notes using librosa.pyin (Probabilistic YIN)."""
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
        fill_na=None,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    notes: List[NoteEvent] = []

    current_note_f0: List[float] = []
    current_times: List[float] = []
    current_probs: List[float] = []

    for t, f, v, p in zip(times, f0, voiced_flag, voiced_probs):
        if v and f is not None and not np.isnan(f):
            # Check if frequency jumped significantly (new note or pitch discontinuity > 1 semitone)
            if current_note_f0:
                prev_f = current_note_f0[-1]
                ratio = f / prev_f
                semitone_diff = abs(12.0 * np.log2(ratio))
                if semitone_diff > 1.2:
                    # Flush previous note
                    _flush_pyin_note(notes, current_note_f0, current_times, current_probs, min_note_duration)
                    current_note_f0 = []
                    current_times = []
                    current_probs = []

            current_note_f0.append(float(f))
            current_times.append(float(t))
            current_probs.append(float(p))
        else:
            if current_note_f0:
                _flush_pyin_note(notes, current_note_f0, current_times, current_probs, min_note_duration)
                current_note_f0 = []
                current_times = []
                current_probs = []

    # Final flush
    if current_note_f0:
        _flush_pyin_note(notes, current_note_f0, current_times, current_probs, min_note_duration)

    return notes


def _flush_pyin_note(
    notes_list: List[NoteEvent],
    f0_arr: List[float],
    times_arr: List[float],
    probs_arr: List[float],
    min_dur: float,
) -> None:
    dur = times_arr[-1] - times_arr[0]
    if dur < min_dur:
        return

    median_f = float(np.median(f0_arr))
    midi_note = int(np.round(librosa.hz_to_midi(median_f)))
    conf = float(np.mean(probs_arr))

    notes_list.append(
        NoteEvent(
            pitch_midi=midi_note,
            pitch_hz=median_f,
            start_time=times_arr[0],
            end_time=times_arr[-1],
            duration=dur,
            confidence=conf,
        )
    )


def extract_notes_polyphonic(
    y: np.ndarray,
    sr: int = 44100,
    audio_path: Optional[Union[str, Path]] = None,
) -> List[NoteEvent]:
    """Extracts polyphonic notes using Spotify's basic-pitch if available, else pyin with harmonic decomposition."""
    try:
        # Check if basic_pitch is installed
        import basic_pitch.inference as bp_inf  # type: ignore

        if audio_path is None:
            import tempfile
            import soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name
            sf.write(temp_wav, y, sr)
            target_path = temp_wav
        else:
            target_path = str(audio_path)

        model_output, midi_data, note_events = bp_inf.predict(target_path)
        notes: List[NoteEvent] = []
        for onset, offset, note_pitch, amp, _ in note_events:
            notes.append(
                NoteEvent(
                    pitch_midi=int(note_pitch),
                    pitch_hz=float(librosa.midi_to_hz(note_pitch)),
                    start_time=float(onset),
                    end_time=float(offset),
                    duration=float(offset - onset),
                    confidence=float(amp),
                )
            )
        return notes

    except ImportError:
        # Fallback to CQT polyphonic peak extraction across musical bins
        from src.spectrogram_math import extract_notes_cqt_polyphonic
        return extract_notes_cqt_polyphonic(y, sr=sr)


def group_partials_into_notes(
    tracks: List[Any],
    notes: List[NoteEvent],
    max_harmonic_num: int = 16,
    harmonic_tolerance: float = 0.03,  # 3% tolerance
) -> List[NoteEvent]:
    """Groups sinusoidal tracks under detected notes as physical harmonics: f_partial ~= h * f_0."""
    for tr in tracks:
        tr_freq = tr.mean_frequency
        tr_start = tr.times[0]
        tr_end = tr.times[-1]

        best_note: Optional[NoteEvent] = None
        best_harmonic = 0
        min_harmonic_err = float("inf")

        for note in notes:
            # Check temporal overlap
            overlap_start = max(tr_start, note.start_time)
            overlap_end = min(tr_end, note.end_time)
            if overlap_end <= overlap_start:
                continue

            f0 = note.pitch_hz
            h = round(tr_freq / f0)
            if 1 <= h <= max_harmonic_num:
                expected_f = h * f0
                rel_err = abs(tr_freq - expected_f) / expected_f
                if rel_err < harmonic_tolerance and rel_err < min_harmonic_err:
                    min_harmonic_err = rel_err
                    best_note = note
                    best_harmonic = int(h)

        if best_note is not None:
            best_note.partials.append({
                "harmonic_number": best_harmonic,
                "frequency": round(tr_freq, 2),
                "peak_amplitude": round(tr.peak_amplitude, 5),
                "duration": round(tr.duration, 4),
            })

    return notes
