"""Spectrogram Mathematics and Polyphonic CQT Extraction Engine.

Addresses the core DSP requirements:
1. CQT (Constant-Q Transform): Logarithmically spaced frequency bins (12, 24, 36 per octave)
   matching Western musical pitch scales. Optimal for detecting notes, chords, and bass lines.
2. Reassigned Spectrogram: Snaps STFT energy to exact instantaneous frequency and time
   using phase derivatives, yielding razor-sharp frequency curves without filter smearing.
3. STFT (Short-Time Fourier Transform): Linear frequency grid for measuring high harmonic overtones.
4. Polyphonic Peak Extraction: Uses per-frame scipy.signal.find_peaks on CQT / Reassigned frames,
   eliminating the flawed `argmax(axis=0)` trap that destroys chords and polyphony.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import librosa

from src.note_extraction import NoteEvent


@dataclass
class SpectrogramResult:
    """Holds computed spectrogram matrix and coordinate grids."""
    spectrogram_type: str  # "cqt", "reassigned", "stft"
    matrix: np.ndarray      # Magnitudes (linear or dB)
    frequencies: np.ndarray # Frequency bins in Hz
    times: np.ndarray       # Time frames in seconds
    sr: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def magnitude(self) -> np.ndarray:
        return self.matrix

    @property
    def cqt_mag(self) -> np.ndarray:
        return self.matrix

    @property
    def freqs(self) -> np.ndarray:
        return self.frequencies

    def __contains__(self, item: str) -> bool:
        return item in (
            "spectrogram_type", "matrix", "magnitude", "cqt_mag",
            "frequencies", "freqs", "times", "sr", "metadata"
        )

    def __getitem__(self, item: str) -> Any:
        if item in ("matrix", "magnitude", "cqt_mag"):
            return self.matrix
        elif item in ("frequencies", "freqs"):
            return self.frequencies
        elif item == "times":
            return self.times
        elif item == "sr":
            return self.sr
        elif item == "spectrogram_type":
            return self.spectrogram_type
        elif item == "metadata":
            return self.metadata
        raise KeyError(item)


def compute_cqt_spectrogram(
    y: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    fmin: Optional[float] = None,
    n_bins: int = 84,  # 7 octaves
    bins_per_octave: int = 12,
) -> SpectrogramResult:
    """Computes Constant-Q Transform with logarithmic musical note frequency spacing.

    Parameters:
        y: Audio signal array
        sr: Sample rate
        hop_length: Number of samples between successive frames
        fmin: Minimum frequency (default C1 ~= 32.7 Hz)
        n_bins: Total frequency bins (84 = 7 octaves * 12 semitones)
        bins_per_octave: Bins per octave (12 = semitone resolution, 24/36 = sub-semitone)
    """
    if fmin is None:
        fmin = float(librosa.note_to_hz("C1"))

    C = np.abs(librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    ))

    freqs = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave,
    )
    times = librosa.frames_to_time(
        np.arange(C.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return SpectrogramResult(
        spectrogram_type="cqt",
        matrix=C,
        frequencies=freqs,
        times=times,
        sr=sr,
        metadata={"fmin": fmin, "bins_per_octave": bins_per_octave, "n_bins": n_bins},
    )


def compute_reassigned_spectrogram(
    y: np.ndarray,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> SpectrogramResult:
    """Computes the reassigned spectrogram using instantaneous phase derivatives.

    Reassignment maps blurred STFT energy centroids directly to their true instantaneous
    frequencies and arrival times, producing crisp trajectory tracks.
    """
    freqs_stft, times_stft, mags = librosa.reassigned_spectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    # Convert linear frame indices to times
    frames_t = librosa.frames_to_time(np.arange(mags.shape[1]), sr=sr, hop_length=hop_length)
    f_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    return SpectrogramResult(
        spectrogram_type="reassigned",
        matrix=mags,
        frequencies=f_bins,
        times=frames_t,
        sr=sr,
        metadata={"reassigned_freqs": freqs_stft, "reassigned_times": times_stft, "n_fft": n_fft},
    )


def extract_notes_cqt_polyphonic(
    y: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    bins_per_octave: int = 12,
    fmin: Optional[float] = None,
    n_bins: int = 84,
    peak_prominence: float = 0.08,
    min_note_duration: float = 0.08,
    max_polyphony: int = 6,
    peak_threshold: Optional[float] = None,
) -> List[NoteEvent]:
    """Extracts polyphonic notes and chords using Constant-Q Spectrogram peak detection.

    Avoids the destructive `argmax(axis=0)` trap by running `scipy.signal.find_peaks`
    per time frame across logarithmic musical bins, grouping temporal contours into
    distinct note objects.
    """
    if peak_threshold is not None:
        peak_prominence = peak_threshold

    if fmin is None:
        fmin = float(librosa.note_to_hz("C1"))

    cqt_res = compute_cqt_spectrogram(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
    )

    C = cqt_res.matrix
    freqs = cqt_res.frequencies
    times = cqt_res.times

    # Normalize CQT matrix
    norm_max = np.max(C)
    if norm_max > 1e-12:
        C_norm = C / norm_max
    else:
        return []

    # Detect frame-by-frame polyphonic peaks
    num_frames = C.shape[1]
    active_tracks: List[Dict[str, Any]] = []  # List of ongoing note tracks
    completed_notes: List[NoteEvent] = []

    for frame_idx in range(num_frames):
        col = C_norm[:, frame_idx]
        t_curr = float(times[frame_idx])

        # Find local peaks across frequency bins in this frame
        peak_indices, properties = signal.find_peaks(
            col,
            height=0.03,
            prominence=peak_prominence,
            distance=2,
        )

        if len(peak_indices) > max_polyphony:
            # Keep the top prominent peaks
            sorted_by_mag = np.argsort(-col[peak_indices])[:max_polyphony]
            peak_indices = peak_indices[sorted_by_mag]

        frame_peaks = [
            (int(bin_idx), float(freqs[bin_idx]), float(col[bin_idx]))
            for bin_idx in peak_indices
        ]

        # Match detected peaks to active note tracks
        unmatched_peaks = list(frame_peaks)
        still_active: List[Dict[str, Any]] = []

        for track in active_tracks:
            last_bin = track["bin"]
            best_match = None
            best_dist = 3.0  # within ~2 bins (approx 2 semitones)

            for p in unmatched_peaks:
                p_bin, p_f, p_mag = p
                dist = abs(p_bin - last_bin)
                if dist < best_dist:
                    best_dist = dist
                    best_match = p

            if best_match is not None:
                unmatched_peaks.remove(best_match)
                track["times"].append(t_curr)
                track["freqs"].append(best_match[1])
                track["mags"].append(best_match[2])
                track["last_frame"] = frame_idx
                track["bin"] = best_match[0]
                still_active.append(track)
            else:
                # Track ended or silent frame
                dur = track["times"][-1] - track["times"][0]
                if dur >= min_note_duration:
                    med_f = float(np.median(track["freqs"]))
                    midi_p = int(np.round(librosa.hz_to_midi(med_f)))
                    completed_notes.append(
                        NoteEvent(
                            pitch_midi=midi_p,
                            pitch_hz=med_f,
                            start_time=track["times"][0],
                            end_time=track["times"][-1],
                            duration=dur,
                            confidence=float(np.mean(track["mags"])),
                        )
                    )

        # Start new tracks for unmatched peaks
        for p_bin, p_f, p_mag in unmatched_peaks:
            still_active.append({
                "bin": p_bin,
                "times": [t_curr],
                "freqs": [p_f],
                "mags": [p_mag],
                "last_frame": frame_idx,
            })

        active_tracks = still_active

    # Flush remaining active tracks
    for track in active_tracks:
        dur = track["times"][-1] - track["times"][0]
        if dur >= min_note_duration:
            med_f = float(np.median(track["freqs"]))
            midi_p = int(np.round(librosa.hz_to_midi(med_f)))
            completed_notes.append(
                NoteEvent(
                    pitch_midi=midi_p,
                    pitch_hz=med_f,
                    start_time=track["times"][0],
                    end_time=track["times"][-1],
                    duration=dur,
                    confidence=float(np.mean(track["mags"])),
                )
            )

    # Sort notes by onset start time
    completed_notes.sort(key=lambda n: n.start_time)
    return completed_notes
