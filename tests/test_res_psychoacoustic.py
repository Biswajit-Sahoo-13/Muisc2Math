"""Tests for Slice 5: Psychoacoustic Masking-Weighted Loss in error_correction."""

import numpy as np
import pytest

from src.error_correction import (
    psychoacoustic_masking_threshold,
    masking_weighted_residual_error,
)


def test_masking_threshold_peaks_near_tonal_masker():
    """Verify masking threshold rises substantially around a loud masker tone."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr

    # Loud tone at 1000 Hz
    masker_freq = 1000.0
    signal = 0.8 * np.cos(2 * np.pi * masker_freq * t)

    freqs, threshold = psychoacoustic_masking_threshold(signal, sr, n_fft=1024)

    # Threshold near 1000 Hz must be much higher than at far frequencies (e.g. 4000 Hz)
    idx_1000 = np.argmin(np.abs(freqs - 1000.0))
    idx_4000 = np.argmin(np.abs(freqs - 4000.0))

    assert threshold[idx_1000] > 10.0 * threshold[idx_4000], (
        f"Masking threshold near masker ({threshold[idx_1000]}) should exceed "
        f"remote threshold ({threshold[idx_4000]}) by >10x"
    )


def test_masking_weighted_residual_penalizes_inaudible_noise():
    """Verify inaudible residual beneath masking curve contributes far less loss."""
    sr = 22050
    n = 2048
    t = np.arange(n) / sr

    # Reference has loud 1000 Hz tone
    reference = 0.8 * np.cos(2 * np.pi * 1000.0 * t)

    # Inaudible error: weak 1030 Hz tone (within masking cone of 1000 Hz)
    resid_inaudible = 0.02 * np.cos(2 * np.pi * 1030.0 * t)

    # Audible error: same amplitude weak tone at 3500 Hz (outside masking cone)
    resid_audible = 0.02 * np.cos(2 * np.pi * 3500.0 * t)

    loss_inaudible = masking_weighted_residual_error(resid_inaudible, reference, sr)
    loss_audible = masking_weighted_residual_error(resid_audible, reference, sr)

    assert loss_audible > 2.0 * loss_inaudible, (
        f"Audible error loss ({loss_audible:.6f}) must exceed "
        f"masked inaudible loss ({loss_inaudible:.6f})"
    )


def test_ath_quiet_threshold_shape():
    """Verify quiet threshold has lowest threshold (highest sensitivity) near 2-4 kHz."""
    sr = 22050
    silent = np.zeros(2048)

    freqs, threshold = psychoacoustic_masking_threshold(silent, sr, n_fft=1024)

    idx_50 = np.argmin(np.abs(freqs - 50.0))
    idx_3000 = np.argmin(np.abs(freqs - 3000.0))

    # Human ear is much less sensitive at 50 Hz than at 3000 Hz (threshold is higher at 50 Hz)
    assert threshold[idx_50] > threshold[idx_3000], (
        f"Threshold at 50 Hz ({threshold[idx_50]}) should be higher than at 3000 Hz ({threshold[idx_3000]})"
    )
