"""Verify the report's four physics claims against the CURRENT tree.

Claim 1: the 9-frame (104.5 ms) median filter crushes 6 Hz vibrato by ~45%.
Claim 2: upper vibrato harmonics shatter into stationary-phase caustics.
Claim 3: static-median outlier filtering truncates wide linear sweeps.
Claim 4: the +-2 Hz bounded Brent search jumps into the first sidelobe.
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sinusoidal_model import (  # noqa: E402
    _prepare_frames, _smooth_envelope, refine_track, SinusoidalTrack,
)

SR = 22050
HOP = 256
NFFT = 2048

print("=" * 74)
print("CLAIM 1: 9-frame median filter attenuates 6 Hz vibrato")
print("=" * 74)
T_w = 9 * HOP / SR
print(f"  window span = 9 * {HOP}/{SR} = {T_w*1000:.2f} ms")
print(f"  6 Hz half-period = {0.5/6*1000:.2f} ms  -> window is "
      f"{T_w/(0.5/6):.2f}x the half-period")
tt = np.arange(0, 2.5, HOP / SR)
vib = 20.0 * np.sin(2 * np.pi * 6.0 * tt)
sm = _smooth_envelope(vib, 9)
amp_in = float(np.max(np.abs(vib)))
amp_out = float(np.max(np.abs(sm)))
print(f"  measured peak excursion: in={amp_in:.4f} Hz  out={amp_out:.4f} Hz")
print(f"  measured attenuation = {100*(1-amp_out/amp_in):.1f}%")
boxcar = abs(np.sinc(6.0 * T_w))
print(f"  report's formula 1-cos(pi f_v T_w/2) = "
      f"{100*(1-math.cos(math.pi*6.0*T_w/2)):.1f}%")
print(f"  exact boxcar (moving-average) response |sinc(f T_w)| = {boxcar:.4f} "
      f"-> attenuation {100*(1-boxcar):.1f}%")
print(f"  NOTE: a median filter is neither; the report's formula is the "
      f"boxcar's shape, so the number is an approximation.")

print()
print("=" * 74)
print("CLAIM 3: static-median truncation of a wide linear sweep")
print("=" * 74)
n = int(2.5 * SR)
t = np.arange(n) / SR
f_inst = 200.0 + (400.0 / 2.5) * t
y = 0.8 * np.cos(2 * np.pi * np.cumsum(f_inst) / SR)
frame_times = np.arange(0, n - NFFT, HOP) / SR
freqs = 200.0 + (400.0 / 2.5) * frame_times
med_f = float(np.median(freqs))
band = max(25.0, 0.35 * med_f)
keep = np.abs(freqs - med_f) <= band
print(f"  sweep 200->600 Hz over 2.5 s, {freqs.size} frames")
print(f"  med_f={med_f:.1f} Hz, band=0.35*med_f={0.35*med_f:.1f} Hz")
print(f"  kept band = [{med_f-band:.1f}, {med_f+band:.1f}] Hz")
print(f"  frames kept: {int(keep.sum())}/{keep.size}")
if keep.any():
    tk = frame_times
    print(f"  kept span = [{tk[keep][0]:.4f}, {tk[keep][-1]:.4f}] s  "
          f"(report: start 0.383, end 2.113)")
    print(f"  DELETED head = {tk[keep][0]:.4f} s, tail = {2.496-tk[keep][-1]:.4f} s")
    dropped = int((~keep).sum())
    print(f"  dropped frames = {dropped} ({100*dropped/keep.size:.1f}%)")
    print(f"  frequency span kept: [{freqs[keep][0]:.1f}, {freqs[keep][-1]:.1f}] Hz "
          f"of the 200->600 Hz swept")
# and through the real _prepare_frames path
tr = SinusoidalTrack(0, 0, {"frequency": float(freqs[0]), "amplitude": 0.8,
                            "phase": 0.0}, float(frame_times[0]))
for i in range(1, len(freqs)):
    tr.update(i, {"frequency": float(freqs[i]), "amplitude": 0.8, "phase": 0.0},
              float(frame_times[i]))
times, fr, amps, *_ = _prepare_frames(tr)
print(f"  through _prepare_frames: {fr.size}/{freqs.size} frames kept, "
      f"span [{times[0]:.4f}, {times[-1]:.4f}] s  "
      f"freq [{fr[0]:.1f}, {fr[-1]:.1f}] Hz")

print()
print("=" * 74)
print("CLAIM 4: the +-2 Hz bounded Brent search jumps into the first sidelobe")
print("=" * 74)

T = 2.5
print(f"  first zero of the projection kernel: 1/T = {1/T:.4f} Hz")
print(f"  first sidelobe minimum: 1.4303/T = {1.4303/T:.4f} Hz")
# golden-section probe positions for scipy 'bounded' on [f-2, f+2]
a, b = 220.0 - 2.0, 220.0 + 2.0
gr = (math.sqrt(5) - 1) / 2
print(f"  scipy 'bounded' first probes: {a + (1-gr)*(b-a):.4f} and "
      f"{a + gr*(b-a):.4f} Hz  (report: f_guess +- 0.4721)")

dur = 2.5
nn = int(dur * SR)
tt = np.arange(nn) / SR
f_true = 220.0
ys = 0.65 * np.cos(2 * np.pi * f_true * tt - 0.3)
frame_times = np.arange(0, nn - NFFT, HOP) / SR
tr2 = SinusoidalTrack(0, 0, {"frequency": 220.098, "amplitude": 0.65,
                             "phase": 0.0}, float(frame_times[0]))
for i in range(1, len(frame_times)):
    tr2.update(i, {"frequency": 220.098, "amplitude": 0.65, "phase": 0.0},
                float(frame_times[i]))
sig, params = refine_track(tr2, ys, SR, refine_frequency=True,
                           stft_half_window_s=NFFT / 2 / SR)
err = float(params["frequency"]) - f_true
print(f"  refine_track(220.098) -> {params['frequency']:.4f} Hz "
      f"(error {err:+.4f} Hz)")
print(f"  projection_gain = {params['projection_gain']:.4f}   "
      f"amplitude = {params['amplitude']:.5f}  "
      f"env_a0 = {params['envelope_a0']:.5f}")
print(f"  report says: slid to 220.5722, gain 0.217, amplitude -79%")
