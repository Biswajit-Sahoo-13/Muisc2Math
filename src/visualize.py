"""Visual inspection helpers replacing and complementing Audacity's inspection tools.

Generates:
1. Waveform plots
2. dB Spectrograms with frequency scale
3. FFT Spectrum plots (matching Audacity's 'Analyze -> Plot Spectrum')
4. Comprehensive 3x2 closed-loop comparison dashboard (Original, Reconstructed, Residual)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_waveform(
    audio: np.ndarray,
    sr: int = 44100,
    title: str = "Audio Waveform",
    out_path: Optional[Union[str, Path]] = None,
) -> None:
    """Plots time-domain waveform."""
    t = np.arange(len(audio)) / sr
    plt.figure(figsize=(10, 3))
    plt.plot(t, audio, color="#2563eb", linewidth=0.8)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectrogram(
    audio: np.ndarray,
    sr: int = 44100,
    title: str = "Audio Spectrogram (dB)",
    out_path: Optional[Union[str, Path]] = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float = 8000.0,
) -> None:
    """Plots magnitude spectrogram in decibels."""
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    spec_db = librosa.amplitude_to_db(spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(
        spec_db,
        sr=sr,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        fmax=fmax,
        cmap="magma",
    )
    plt.colorbar(img, format="%+2.0f dB")
    plt.title(title, fontsize=12, fontweight="bold")
    plt.ylim(0, fmax)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectrum(
    audio: np.ndarray,
    sr: int = 44100,
    title: str = "Frequency Spectrum (Audacity Style)",
    out_path: Optional[Union[str, Path]] = None,
    max_freq: float = 5000.0,
) -> None:
    """Plots FFT spectrum magnitude in dB vs Frequency, matching Audacity's 'Plot Spectrum'."""
    n = len(audio)
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(audio * w))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    eps = 1e-12
    spec_db = 20.0 * np.log10(spec / (np.max(spec) + eps) + eps)

    mask = freqs <= max_freq
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[mask], spec_db[mask], color="#059669", linewidth=1.0)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(-80, 5)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison_dashboard(
    orig: np.ndarray,
    recon: np.ndarray,
    sr: int = 44100,
    out_path: Optional[Union[str, Path]] = None,
    title: str = "Closed-Loop Reconstruction & Residual Verification",
    metrics: Optional[dict] = None,
) -> None:
    """Generates a complete 3x2 multi-panel comparison dashboard:

    Top row: Waveforms (Original, Reconstructed, Residual)
    Bottom row: Spectrograms (Original, Reconstructed, Residual)
    """
    min_len = min(len(orig), len(recon))
    orig = orig[:min_len]
    recon = recon[:min_len]
    residual = orig - recon
    t = np.arange(min_len) / sr

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Waveforms
    axes[0, 0].plot(t, orig, color="#2563eb", lw=0.7)
    axes[0, 0].set_title("Original Waveform x(t)")
    axes[0, 0].set_ylim(-1.05, 1.05)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, recon, color="#16a34a", lw=0.7)
    axes[0, 1].set_title("Reconstructed Waveform x_hat(t)")
    axes[0, 1].set_ylim(-1.05, 1.05)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(t, residual, color="#dc2626", lw=0.7)
    subtitle = "Residual r(t) = x(t) - x_hat(t)"
    if metrics and "snr_db" in metrics:
        subtitle += f" (SNR: {metrics['snr_db']:.1f} dB)"
    axes[0, 2].set_title(subtitle)
    axes[0, 2].set_ylim(-1.05, 1.05)
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].grid(True, alpha=0.3)

    # Spectrograms
    n_fft = 2048
    hop = 512
    fmax = 4000.0

    s_orig = librosa.amplitude_to_db(np.abs(librosa.stft(orig, n_fft=n_fft, hop_length=hop)), ref=np.max)
    s_recon = librosa.amplitude_to_db(np.abs(librosa.stft(recon, n_fft=n_fft, hop_length=hop)), ref=np.max)
    s_res = librosa.amplitude_to_db(np.abs(librosa.stft(residual, n_fft=n_fft, hop_length=hop)), ref=np.max)

    img1 = librosa.display.specshow(s_orig, sr=sr, hop_length=hop, x_axis="time", y_axis="linear", ax=axes[1, 0], fmax=fmax, cmap="magma")
    axes[1, 0].set_title("Original Spectrogram")
    axes[1, 0].set_ylim(0, fmax)
    fig.colorbar(img1, ax=axes[1, 0], format="%+2.0f dB")

    img2 = librosa.display.specshow(s_recon, sr=sr, hop_length=hop, x_axis="time", y_axis="linear", ax=axes[1, 1], fmax=fmax, cmap="magma")
    axes[1, 1].set_title("Reconstructed Spectrogram")
    axes[1, 1].set_ylim(0, fmax)
    fig.colorbar(img2, ax=axes[1, 1], format="%+2.0f dB")

    img3 = librosa.display.specshow(s_res, sr=sr, hop_length=hop, x_axis="time", y_axis="linear", ax=axes[1, 2], fmax=fmax, cmap="magma")
    axes[1, 2].set_title("Residual Spectrogram")
    axes[1, 2].set_ylim(0, fmax)
    fig.colorbar(img3, ax=axes[1, 2], format="%+2.0f dB")

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
    plt.close()
