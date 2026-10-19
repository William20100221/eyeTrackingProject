"""
Self-check for the report figures (no camera, never data/):  python -m eyetracking.testing.test_report_figures

A temp data folder with two small synthetic calibrations (an Original 5x5 and an Original 3x3 of a simulated
eye, core/line_model.features_for) and one synthetic gaze check (a multiple check of a model trained on one
calibration) and a Sweep's head step. report_figures runs on it; every PNG and CSV must exist, the PNGs
real ones; the smoothing figure's maths is checked on a made-up step (no video here).
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from eyetracking.core.calibration import point_grid
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.services import storage
from eyetracking.testing import report_figures

tmp = Path(tempfile.mkdtemp())
data, out = tmp / "data", tmp / "out"
rng = np.random.default_rng(2)
screen = ScreenGeometry(27.7, 18.9)


def frames(t, n, spread=(1, 1, 1, 1, 1, 1)):
    return tuple(CalibrationSample(features_for(screen, t, list(rng.normal([0, 0, 0, 0, -5, -65], spread)), (2, -1), 0.9, rng),
                                   *t) for _ in range(n))


def dots(n):
    return tuple(CalibrationSample(features_for(screen, t, list(rng.normal([0, 0, 0, 0, -5, -65], 1)), (2, -1), 0.9, rng), *t)
                 for t in point_grid(n) for _ in range(12))


storage.save_calibration(CalibrationRecording((), (), dots(5)), "original-5x5",
                         data / "Ann" / "original-5x5" / "original-5x5_2026-10-20_100000.json")
storage.save_calibration(CalibrationRecording((), (), dots(3)), "original-3x3",
                         data / "Ann" / "original-3x3" / "original-3x3_2026-10-20_100500.json")
rings = point_grid(6)
storage.save_gaze_check([(i, tx + rng.normal(0, .012), ty + rng.normal(0, .012), tx, ty) for i, (tx, ty) in enumerate(rings)]
                        + [(36, 1.2, 0.5, 0.5, 0.5)],                               # one prediction off screen
                        "original-5x5", data / "Ann" / "original-5x5", calibrations=1, tag="multi")

head = frames((0.5, 0.5), 120, spread=(8, 6, 3, 2, 2, 4))                      # a Sweep's head step, for k_matrix
storage.save_calibration(CalibrationRecording(dots(3)[:30], tuple(head), ()), "sweep",
                         data / "Ann" / "sweep" / "sweep_2026-10-20_101000.json")
# a blink test: open eyes with one blink (0.2 -> 0.85 for 3 frames), and the other states held
blink = [("open", 33 * i, s, s) for i, s in enumerate([0.2] * 40 + [0.85] * 3 + [0.2] * 40)]
blink += [(state, 33 * i, v, v) for state, v in (("down", 0.3), ("half", 0.55), ("closed", 0.9)) for i in range(30)]
storage.save_blink_scores(blink, data / "Ann" / "blink-test")
assert report_figures.main([str(data), "--out", str(out), "--person", "Ann"]) == 0
names = ["multi_check", "one_vs_two", "across_vs_down", "ring_map", "versions", "feature_correlation",
         "feature_subsets", "heat_25", "lodo", "k_matrix", "one_euro", "same_vs_next", "blink_timeline", "blink_states"]
for n in names:
    assert (out / f"{n}.png").read_bytes()[:4] == b"\x89PNG", n
    assert (out / f"{n}.csv").exists() and (out / f"{n}.csv").stat().st_size > 0, n
multi = (out / "multi_check.csv").read_text(encoding="utf-8").splitlines()
assert len(multi) == 2 and multi[1].startswith("Original 5×5,1,37,") and multi[1].endswith(",1"), multi   # 1 off screen
heat = (out / "heat_25.csv").read_text(encoding="utf-8").splitlines()
assert 20 <= len(heat) - 1 <= 25, heat                                               # the dots the quality check kept
versions = (out / "versions.csv").read_text(encoding="utf-8").splitlines()
assert len(versions) == 4 and all(line.split(",")[2] == "" for line in versions[1:]), versions   # 2 files + mean; no 18-coordinate number
k = (out / "k_matrix.csv").read_text(encoding="utf-8")
assert "built-in (DEFAULT_K)" in k and "Ann (1 head step)" in k, k                   # the built-in K and Ann's own
lodo = (out / "lodo.csv").read_text(encoding="utf-8").splitlines()
assert len(lodo) > 20 and all(line.split(",")[2] for line in lodo[1:]), lodo        # a held-out error for every dot
same = (out / "same_vs_next.csv").read_text(encoding="utf-8")
assert "same sitting" in same and "next sitting" in same and "the old 12-feature curve" in same, same
assert (out / "one_euro.csv").read_text(encoding="utf-8").startswith("trace,")   # no video here

# the smoothing figure's maths, on a made-up step: the eye jumps from 0.2 to 0.8 at 1 s, with jitter
ms = list(range(0, 3000, 33))
raw = [(0.2 if t < 1000 else 0.8) + rng.normal(0, 0.01) for t in ms]
traces = report_figures.smoothing_traces(ms, [(v, 0.5) for v in raw])
settle = {n: report_figures.settling_ms(ms, [p[0] for p in v], 1000) for n, v in traces.items()}
assert settle["raw"] < settle["moving average 30 %"] < 1000 and settle["1-euro"] < 1000, settle
assert report_figures.settling_ms(ms, [0.5] * len(ms), 1000) is None                # it never moved
regions = [r[4] for r in csv.reader((out / "blink_timeline.csv").open(encoding="utf-8"))][1:]
assert "closed" in regions and "recovery" in regions and regions.index("recovery") > regions.index("closed"), regions
states = (out / "blink_states.csv").read_text(encoding="utf-8")
assert "eyes open,left,83," in states and "today's cut-off (BLINK_THRESHOLD)" in states, states
print("report figures ok:", ", ".join(f"{n}.png/.csv" for n in names))
