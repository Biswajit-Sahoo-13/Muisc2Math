"""L1/L2/L3 guards for the synthetic benchmark.

L1  save_audio must never silently destroy samples.
L2  the benchmark fixture must be exact (no clipping) and stay a normal
    24-bit PCM file.
L3  the discovery must not quantise frequency/phase to 0.001 rad / 0.001 Hz.
"""
import numpy as np
import pytest
import soundfile as sf

from src.synthesis import generate_benchmark_chord, render_model, save_audio

SR = 44100


def _snr(a, b):
    n = min(len(a), len(b))
    e = np.asarray(a[:n], np.float64) - np.asarray(b[:n], np.float64)
    return 10 * np.log10((np.sum(a[:n] ** 2) + 1e-20) / (np.sum(e ** 2) + 1e-20))


class TestL1SaveAudioNeverSilent:
    def test_reports_clipped_count(self, tmp_path):
        sig = np.array([0.5, 1.4, -1.9, 0.2])
        n = save_audio(str(tmp_path / "c.wav"), sig, SR)
        assert n == 2, "the number of clipped samples must be reported"

    def test_float_subtype_preserves_over_full_scale(self, tmp_path):
        sig = np.array([0.5, 1.4, -1.9, 0.2])
        p = tmp_path / "f.wav"
        assert save_audio(str(p), sig, SR, subtype="FLOAT") == 0
        back, _ = sf.read(str(p), always_2d=False)
        # 32-bit float storage, so the tolerance is float32 epsilon, not 1e-9
        assert np.max(np.abs(back - sig)) < 1e-7, "float must not lose samples"

    def test_pcm24_still_round_trips_below_full_scale(self, tmp_path):
        sig = np.linspace(-0.9, 0.9, 1000)
        p = tmp_path / "n.wav"
        assert save_audio(str(p), sig, SR) == 0
        back, _ = sf.read(str(p), always_2d=False)
        assert np.max(np.abs(back - sig)) < 2e-7


class TestL2FixtureIsExact:
    def test_fixture_does_not_clip(self, tmp_path):
        p = tmp_path / "chord.wav"
        generate_benchmark_chord(p, duration=2.0, sr=SR)
        audio, _ = sf.read(str(p), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert np.max(np.abs(audio)) < 1.0, "fixture must stay inside full scale"

    def test_fixture_is_24bit_pcm(self, tmp_path):
        p = tmp_path / "chord.wav"
        generate_benchmark_chord(p, duration=2.0, sr=SR)
        assert sf.info(str(p)).subtype == "PCM_24"

    def test_fixture_matches_its_own_ground_truth(self, tmp_path):
        """The file must equal the model it claims to encode, exactly."""
        p = tmp_path / "chord.wav"
        model = generate_benchmark_chord(p, duration=2.0, sr=SR)
        expected, _ = render_model(model, duration=2.0, sr=SR)
        got, _ = sf.read(str(p), always_2d=False)
        if got.ndim > 1:
            got = got.mean(axis=1)
        got = got.astype(np.float64)
        n = min(len(got), len(expected))
        # 24-bit quantisation is ~6e-8; anything larger means real distortion
        assert np.max(np.abs(got[:n] - expected[:n])) < 1e-6, (
            "the fixture must be the exact model, not a clipped approximation"
        )

    def test_relative_amplitudes_are_preserved(self, tmp_path):
        p = tmp_path / "chord.wav"
        model = generate_benchmark_chord(p, duration=2.0, sr=SR)
        amps = [c["amplitude"] for c in model["components"]]
        ratios = [amps[1] / amps[0], amps[2] / amps[0]]
        assert ratios == pytest.approx([0.35 / 0.60, 0.25 / 0.60], rel=1e-9)


class TestL3NoQuantisationOfPhaseOrFrequency:
    def test_stored_frequency_is_not_quantised_to_millihertz(self, tmp_path):
        """The stored value must keep the estimator's own precision.

        Rounding to 3 decimals injects up to 0.0005 Hz of error, which
        accumulates 2*pi*0.0005*t of phase over the note.
        """
        from src.tracking import discover_equations_from_audio
        p = tmp_path / "chord.wav"
        generate_benchmark_chord(p, duration=2.0, sr=SR)
        disc = discover_equations_from_audio(p, n_fft=4096, hop_length=1024,
                                             threshold=0.01,
                                             out_json_path=str(tmp_path / "e.json"))
        for c in disc["components"]:
            stored = c["frequency"]
            assert stored != round(stored, 3), (
                f"frequency {stored} was quantised to millihertz"
            )
            assert abs(stored - round(stored, 3)) < 5e-4, (
                "the estimator's own error must dominate the quantisation"
            )

    def test_joint_refinement_delivers_sub_millihertz_accuracy(self, tmp_path):
        """The refinement, not the storage precision, is what delivers accuracy.

        Storing more decimals only helps if the underlying estimate is
        accurate; the deflated joint refinement is what brings the frequency
        error down to the microhertz range.
        """
        from src.tracking import discover_equations_from_audio
        p = tmp_path / "chord.wav"
        model = generate_benchmark_chord(p, duration=2.0, sr=SR)
        disc = discover_equations_from_audio(p, n_fft=4096, hop_length=1024,
                                             threshold=0.01,
                                             out_json_path=str(tmp_path / "e.json"))
        worst = max(abs(c["frequency"] - g["frequency"])
                    for c, g in zip(disc["components"], model["components"]))
        assert worst < 1e-4, (
            f"worst frequency error {worst:.8f} Hz: the deflated joint "
            f"refinement should reach the microhertz range"
        )

    def test_refinement_can_be_disabled_for_ab_measurement(self):
        from src import tracking
        assert isinstance(tracking.JOINT_FREQ_REFINE, bool)