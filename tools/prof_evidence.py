"""Profile the evidence build to find where the new physics costs time."""
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import load_audio  # noqa: E402
from src.res_evidence import build_evidence  # noqa: E402

SR = 22050
y, _ = load_audio(str(ROOT / "data/output/_fixtures/stem_vocals.wav"),
                  target_sr=SR)
y = y[: int(10.0 * SR)]

t0 = time.time()
pr = cProfile.Profile()
pr.enable()
build_evidence(y, str(ROOT / "data/output/_prof"), sr=SR, max_duration=10.0,
               window_s=2.5)
pr.disable()
print(f"build_evidence: {time.time()-t0:.1f}s")
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
print(s.getvalue())
