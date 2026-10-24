"""Compare the expansion's behaviour on the bass (good) and 'other' (bad)."""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import load_audio  # noqa: E402
from src.sinusoidal_model import sinusoidal_reconstruction  # noqa: E402

SR = 22050


def probe(name: str) -> None:
    y, _ = load_audio(str(ROOT / f"data/output/_fixtures/{name}.wav"),
                      target_sr=SR)
    y = y[: int(6.0 * SR)]
    res = sinusoidal_reconstruction(y, SR, n_fft=2048, hop_length=256,
                                   add_noise_layer=False, window_s=2.5,
                                   workers=1)
    parts = res["fitted_parameters"]
    refined = [p for p in parts if p.get("trajectory_refined")]
    print(f"\n{name}: {len(parts)} partials, {len(refined)} refined, "
          f"corr {res['correlation']:.4f}")
    # group by rounded base frequency to see the harmonic families
    fams: dict = {}
    for p in refined:
        f = float(p["frequency"])
        key = round(f / 50.0) * 50.0
        fams.setdefault(key, []).append(p)
    for key in sorted(fams)[:6]:
        row = sorted(fams[key], key=lambda p: float(p["frequency"]))
        desc = ", ".join(
            f"{float(p['frequency']):.0f}Hz/g{float(p['projection_gain']):.2f}"
            for p in row[:6]
        )
        print(f"   base~{key:6.0f}: {len(row):3d} refined  [{desc}]")


probe("stem_bass")
probe("stem_other")
