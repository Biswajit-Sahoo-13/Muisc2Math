"""Shazam-style audio fingerprinting plus precise tone identification.

Two complementary pieces:

1. ``ToneprintDB``: a landmark-based fingerprint (constellation map of
   time-frequency peaks hashed into ``(f1, f2, delta_t)`` pairs) used to match
   a short query clip against registered references, robust to time shifts.
2. ``identify_tones``: harmonic-sieve transcription that reports the active
   musical tones (MIDI + Hz + start/end) per segment.

No method can literally guarantee perfection on noisy polyphonic audio, so the
design targets near-exact results on clean tonal material while degrading
gracefully (confidence scores) elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.signal as signal
import librosa


# ---------------------------------------------------------------------------
# Constellation map (Shazam-style spectral peaks)
# ---------------------------------------------------------------------------

def _to_mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    return y.ravel()


def extract_constellation(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 4096,
    hop_length: int = 512,
    fmin: float = 60.0,
    fmax: float = 9000.0,
    top_per_frame: int = 6,
    threshold_ratio: float = 0.03,
    max_peaks: int = 3000,
    min_time_gap: float = 0.03,
    min_freq_gap: float = 25.0,
) -> List[Dict[str, float]]:
    """Extract a sparse constellation map of prominent time-frequency peaks."""
    y = _to_mono(y)
    if y.size < n_fft:
        return []

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann", center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)
    global_max = float(np.max(stft))
    if global_max <= 1e-12:
        return []

    min_height = threshold_ratio * global_max
    min_bin_distance = max(1, int(round(min_freq_gap / (sr / n_fft))))
    candidates: List[Dict[str, float]] = []
    for frame_idx in range(stft.shape[1]):
        col = stft[:, frame_idx]
        bins, _ = signal.find_peaks(col, height=min_height, distance=min_bin_distance)
        if bins.size == 0:
            continue
        order = np.argsort(-col[bins])[:top_per_frame]
        for b in bins[order]:
            f = float(freqs[b])
            if f < fmin or f > fmax:
                continue
            candidates.append({
                "time": float(times[frame_idx]),
                "frequency": f,
                "amplitude": float(col[b]),
            })

    # Greedy density control: keep strongest peaks separated in time/frequency.
    candidates.sort(key=lambda p: p["amplitude"], reverse=True)
    accepted: List[Dict[str, float]] = []
    for peak in candidates:
        keep = True
        for other in accepted:
            if abs(peak["time"] - other["time"]) < min_time_gap and \
               abs(peak["frequency"] - other["frequency"]) < min_freq_gap:
                keep = False
                break
        if keep:
            accepted.append(peak)
        if len(accepted) >= max_peaks:
            break
    accepted.sort(key=lambda p: p["time"])
    return accepted


def fingerprint_constellation(
    peaks: Sequence[Dict[str, float]],
    fanout: int = 6,
    min_dt: float = 0.02,
    max_dt: float = 1.2,
) -> List[Tuple[Tuple[int, int, int], float]]:
    """Hash ordered peak pairs into Shazam-like ``(f1, f2, dt)`` landmarks."""
    ordered = sorted(peaks, key=lambda p: p["time"])
    hashes: List[Tuple[Tuple[int, int, int], float]] = []
    for i, anchor in enumerate(ordered):
        paired = 0
        for other in ordered[i + 1:]:
            dt = other["time"] - anchor["time"]
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            key = (
                int(round(anchor["frequency"])),
                int(round(other["frequency"])),
                int(round(dt * 100.0)),
            )
            hashes.append((key, float(anchor["time"])))
            paired += 1
            if paired >= fanout:
                break
    return hashes


class ToneprintDB:
    """In-memory reference database for Shazam-style clip identification."""

    def __init__(self, offset_bin: float = 0.05):
        self.offset_bin = float(offset_bin)
        self._index: Dict[Tuple[int, int, int], List[Tuple[int, float]]] = {}
        self._names: List[str] = []

    def add(self, name: str, y: np.ndarray, sr: int = 22050, **kwargs: Any) -> int:
        """Register a reference clip. Returns the number of hashes stored."""
        peaks = extract_constellation(y, sr=sr, **kwargs)
        hashes = fingerprint_constellation(peaks)
        ref_id = len(self._names)
        self._names.append(name)
        for key, anchor_time in hashes:
            self._index.setdefault(key, []).append((ref_id, anchor_time))
        return len(hashes)

    def match(
        self,
        y: np.ndarray,
        sr: int = 22050,
        top_k: int = 3,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Match a query clip against references using offset histograms."""
        peaks = extract_constellation(y, sr=sr, **kwargs)
        hashes = fingerprint_constellation(peaks)
        votes: Dict[int, Dict[int, int]] = {}
        for key, query_time in hashes:
            for ref_id, ref_time in self._index.get(key, []):
                offset = ref_time - query_time
                offset_bin = int(round(offset / self.offset_bin))
                votes.setdefault(ref_id, {}).setdefault(offset_bin, 0)
                votes[ref_id][offset_bin] += 1

        results: List[Dict[str, Any]] = []
        denom = max(1, len(hashes))
        for ref_id, hist in votes.items():
            best_bin = max(hist, key=lambda b: hist[b])
            count = hist[best_bin]
            results.append({
                "name": self._names[ref_id],
                "votes": int(count),
                "confidence": round(float(count) / denom, 4),
                "offset": round(best_bin * self.offset_bin, 4),
                "query_hashes": len(hashes),
            })
        results.sort(key=lambda r: r["votes"], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Precise tone identification (harmonic sieve)
# ---------------------------------------------------------------------------

@dataclass
class ToneHit:
    """A detected musical tone with time span and confidence."""
    pitch_midi: int
    pitch_hz: float
    start_time: float
    end_time: float
    duration: float
    confidence: float = 1.0
    harmonics: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch_midi": int(self.pitch_midi),
            "pitch_hz": round(float(self.pitch_hz), 2),
            "start_time": round(float(self.start_time), 4),
            "end_time": round(float(self.end_time), 4),
            "duration": round(float(self.duration), 4),
            "confidence": round(float(self.confidence), 3),
            "harmonics": int(self.harmonics),
        }


def _refined_spectrum_peaks(
    y_seg: np.ndarray,
    sr: int,
    n_fft: int = 16384,
    fmin: float = 40.0,
    fmax: float = 6000.0,
    height_ratio: float = 0.04,
    min_peak_hz: float = 18.0,
) -> List[Dict[str, float]]:
    """Zero-padded spectrum with log-parabolic sub-bin refinement."""
    y_seg = _to_mono(y_seg)
    if y_seg.size < 64:
        return []
    windowed = y_seg * np.hanning(y_seg.size)
    mags = np.abs(np.fft.rfft(windowed, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    peak_max = float(np.max(mags))
    if peak_max <= 1e-12:
        return []
    bin_hz = sr / n_fft
    distance = max(1, int(round(min_peak_hz / bin_hz)))
    bins, _ = signal.find_peaks(mags, height=height_ratio * peak_max, distance=distance)
    eps = 1e-12
    out: List[Dict[str, float]] = []
    for k in bins:
        if k <= 0 or k >= mags.size - 1:
            continue
        f = float(freqs[k])
        if f < fmin or f > fmax:
            continue
        alpha = np.log(max(float(mags[k - 1]), eps))
        beta = np.log(max(float(mags[k]), eps))
        gamma = np.log(max(float(mags[k + 1]), eps))
        denom = alpha - 2.0 * beta + gamma
        p = float(np.clip(0.5 * (alpha - gamma) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
        out.append({
            "frequency": f + p * bin_hz,
            "amplitude": float(mags[k]),
        })
    out.sort(key=lambda d: d["amplitude"], reverse=True)
    return out


def _nearest_peak(
    peaks: Sequence[Dict[str, float]],
    target: float,
    tol: float,
    skip: Optional[set] = None,
) -> Optional[int]:
    best: Optional[int] = None
    best_err = tol
    for i, peak in enumerate(peaks):
        if skip is not None and i in skip:
            continue
        err = abs(peak["frequency"] - target)
        if err <= best_err:
            best_err = err
            best = i
    return best


def identify_tones_in_segment(
    y_seg: np.ndarray,
    sr: int = 22050,
    max_polyphony: int = 6,
    max_harmonic: int = 8,
    min_fundamental_ratio: float = 0.08,
    min_score_ratio: float = 0.12,
    explained_harmonic_ratio: float = 0.45,
) -> List[Dict[str, Any]]:
    """Identify active tones inside one short audio segment."""
    peaks = _refined_spectrum_peaks(y_seg, sr)
    if not peaks:
        return []
    peak_max = max(p["amplitude"] for p in peaks)

    # Score every MIDI pitch with a harmonic sieve anchored on its fundamental.
    scored: List[Dict[str, Any]] = []
    for midi in range(21, 109):
        f0 = float(librosa.midi_to_hz(midi))
        if f0 < 40.0 or f0 > 5000.0:
            continue
        fund_tol = max(2.0, 0.004 * f0)
        fund_idx = _nearest_peak(peaks, f0, fund_tol)
        if fund_idx is None:
            continue
        fund_amp = peaks[fund_idx]["amplitude"]
        if fund_amp < min_fundamental_ratio * peak_max:
            continue
        score = fund_amp
        matched = [fund_idx]
        for h in range(2, max_harmonic + 1):
            expected = h * f0
            if expected > 6000.0:
                break
            idx = _nearest_peak(peaks, expected, max(3.0, 0.006 * expected))
            if idx is None or idx in matched:
                continue
            amp = peaks[idx]["amplitude"]
            if amp < 0.03 * peak_max:
                continue
            score += amp / (h ** 0.8)
            matched.append(idx)
        if score >= min_score_ratio * peak_max:
            scored.append({
                "midi": midi,
                "f0": f0,
                "fund_idx": fund_idx,
                "fund_freq": peaks[fund_idx]["frequency"],
                "score": score,
                "matched": matched,
            })
    scored.sort(key=lambda c: c["score"], reverse=True)

    # Greedy acceptance with harmonic explanation. A higher tone whose
    # fundamental is merely a quiet harmonic of an accepted lower tone is
    # suppressed; loud octave doublings survive the amplitude check.
    accepted: List[Dict[str, Any]] = []
    explained: set = set()
    for cand in scored:
        if len(accepted) >= max_polyphony:
            break
        if cand["fund_idx"] in explained:
            continue
        fund_amp = peaks[cand["fund_idx"]]["amplitude"]
        accepted.append({
            "midi": cand["midi"],
            "pitch_hz": cand["fund_freq"],
            "score": cand["score"],
            "harmonics": len(cand["matched"]),
        })
        fund_peak_amp = fund_amp
        for idx in cand["matched"][1:]:
            h_ratio = peaks[idx]["frequency"] / max(cand["fund_freq"], 1e-9)
            h = max(2, int(round(h_ratio)))
            # Only explain harmonics that are plausibly quiet overtones.
            if peaks[idx]["amplitude"] <= explained_harmonic_ratio * fund_peak_amp / np.sqrt(h):
                explained.add(idx)

    best = max((a["score"] for a in accepted), default=1.0)
    tones = []
    for item in accepted:
        tones.append({
            "pitch_midi": item["midi"],
            "pitch_hz": round(float(item["pitch_hz"]), 2),
            "confidence": round(float(item["score"] / best), 3),
            "harmonics": int(item["harmonics"]),
        })
    tones.sort(key=lambda t: t["pitch_midi"])
    return tones


def identify_tones(
    y: np.ndarray,
    sr: int = 22050,
    segment: float = 0.25,
    **kwargs: Any,
) -> List[ToneHit]:
    """Split audio into segments, identify tones, and merge repeats into hits."""
    y = _to_mono(y)
    if y.size == 0:
        return []
    seg_len = max(64, int(segment * sr))
    hits: List[ToneHit] = []
    open_notes: Dict[int, Dict[str, Any]] = {}

    n_segs = int(np.ceil(len(y) / seg_len))
    for s in range(n_segs):
        seg = y[s * seg_len:(s + 1) * seg_len]
        start = s * seg_len / sr
        end = min(len(y) / sr, (s + 1) * seg_len / sr)
        detected = identify_tones_in_segment(seg, sr, **kwargs)
        active = {d["pitch_midi"]: d for d in detected}
        # Close notes that vanished.
        for midi in [m for m in open_notes if m not in active]:
            note = open_notes.pop(midi)
            hits.append(ToneHit(
                pitch_midi=midi,
                pitch_hz=note["hz"],
                start_time=note["start"],
                end_time=note["end"],
                duration=note["end"] - note["start"],
                confidence=note["conf"],
                harmonics=note["harmonics"],
            ))
        # Extend or open notes.
        for midi, det in active.items():
            if midi in open_notes:
                open_notes[midi]["end"] = end
                open_notes[midi]["conf"] = max(open_notes[midi]["conf"], det["confidence"])
                open_notes[midi]["harmonics"] = max(open_notes[midi]["harmonics"], det["harmonics"])
            else:
                open_notes[midi] = {
                    "hz": det["pitch_hz"],
                    "start": start,
                    "end": end,
                    "conf": det["confidence"],
                    "harmonics": det["harmonics"],
                }
    for midi, note in open_notes.items():
        hits.append(ToneHit(
            pitch_midi=midi,
            pitch_hz=note["hz"],
            start_time=note["start"],
            end_time=note["end"],
            duration=note["end"] - note["start"],
            confidence=note["conf"],
            harmonics=note["harmonics"],
        ))
    hits.sort(key=lambda h: (h.start_time, h.pitch_midi))
    return hits


def identify_file(
    path: Union[str, Path],
    target_sr: int = 22050,
    segment: float = 0.25,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Identify tones directly from an audio file."""
    y, _ = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float64)
    return [h.to_dict() for h in identify_tones(y, sr=target_sr, segment=segment, **kwargs)]


__all__ = [
    "ToneHit",
    "ToneprintDB",
    "extract_constellation",
    "fingerprint_constellation",
    "identify_file",
    "identify_tones",
    "identify_tones_in_segment",
]
