"""
Self-check for blink handling (no webcam):  python -m eyetracking.testing.test_blinks

1. EyeTracker skips a blink coming (lids closing), the blink, and 250 ms
   of reopening; a rise that never becomes a blink (looking down) is let
   through again after 200 ms
2. CalibrationSession drops the frames just BEFORE a blink
3. A whole calibration (Sweep, the longest method) with frequent blinks: every dot still
   gets enough good frames, and no closing-lid frame is saved
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from eyetracking.core.models import FrameResult, HeadPose
from eyetracking.services.tracker import EyeTracker

# ---- 1. tracker: frames 33 ms apart (30 fps) ---------------------------------
lm = np.random.default_rng(0).uniform(0.2, 0.8, (478, 3))
def run(scores: list[float]) -> str:
    """Feed blink scores one frame at a time; '.' = frame used, '-' = skipped."""
    t = EyeTracker.__new__(EyeTracker)                   # no camera: feed frames by hand
    t.face_found, t._open_level, t._onset_ms, t._blink_ms = False, None, None, None
    frames = iter(FrameResult(33 * i, True, HeadPose(0, 0, 0), s, s, lm) for i, s in enumerate(scores))
    t.read = lambda: next(frames)
    out = "".join("." if t.read_features() is not None else "-" for _ in scores)
    assert t.face_found                                  # a blink is still a face
    return out

# open, lids closing (0.3, 0.45), blink (0.9 x2), reopening (0.1 for 250 ms), open
blink = run([0.1] * 4 + [0.3, 0.45, 0.9, 0.9] + [0.1] * 10)
assert blink == "...." + "----" + "-------" + "...", blink    # 7 x 33 ms < 250 ms reopening
# looking down: score rises to 0.3 and STAYS there (no blink comes)
down = run([0.1] * 4 + [0.3] * 12)
assert down == "...." + "-------" + ".....", down             # held < 200 ms, then let through
print("1. tracker: blink", blink, "| looking down", down)

# ---- 2. session ---------------------------------------------------------------
from eyetracking.core.calibration import BLINK_LOOKBACK, CalibrationSession
s = CalibrationSession([(0.1, 0.1), (0.9, 0.9)])
for v in range(10):
    s.add_sample([float(v)])                             # good frames 0..9
s.add_sample(None)                                       # blink: drops the held-back 7, 8, 9
for v in range(20, 25):
    s.add_sample([float(v)])                             # 20..24: the last 3 are still held back
s.next_target()                                          # ...and dropped on moving on
kept = [x.features[0] for x in s.samples()]
assert kept == [float(v) for v in range(10 - BLINK_LOOKBACK)] + [20.0, 21.0], kept
print(f"2. session: last {BLINK_LOOKBACK} frames before the blink dropped; kept {kept}")

# ---- 3. a whole calibration, blinking every ~1.2 s --------------------------
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
import eyetracking.core.calibration as cal
import eyetracking.ui.calibration_window as cw
from eyetracking.core.methods import METHOD_BY_ID
cal.SWEEP_SETTLE_MS, cal.SWEEP_CROSS_MS = 100, 250        # speed up the moving sweep
cal.HEAD_SETTLE_MS, cal.HEAD_MOVE_MIN_MS, cal.HEAD_MOVE_MAX_MS = 100, 100, 300   # the fake head never moves: max it is
cw.SETTLE_MS, cw.COLLECT_MS, cw.DOT_SAMPLES, cw.MAX_COLLECT_MS, cw.TICK_MS = 50, 300, 10, 2000, 10

qt = QApplication.instance() or QApplication([])
n = [0]
def get_features():
    """Blinks of 4 frames every 12 frames; the 2 frames before each blink are
    'closing' (marked -1) and must never end up in the saved recording."""
    n[0] += 1
    k = n[0] % 12
    if k >= 8:
        return None                                       # blink (tracker gives None)
    return (-1.0 if k >= 6 else 1.0,) + (0.5,) * 11       # 6, 7 = lids closing
win = cw.CalibrationWindow(METHOD_BY_ID["sweep"], get_features=get_features)
out = []
win.finished.connect(lambda rec: (out.append(rec), qt.quit()))
win.resize(800, 600); win.show(); win.start()
QTimer.singleShot(120000, qt.quit)
qt.exec()
rec = out[0]
counts = {}
for x in rec.dots:
    counts[(x.target_x, x.target_y)] = counts.get((x.target_x, x.target_y), 0) + 1
closing = sum(x.features[0] < 0 for part in (rec.sweep, rec.head, rec.dots) for x in part)
print(f"3. calibration: {len(counts)} dots, fewest samples {min(counts.values())} "
      f"(need {cw.DOT_SAMPLES}); closing-lid frames saved: {closing}")
assert len(counts) == 25 and min(counts.values()) >= cw.DOT_SAMPLES and closing == 0

print("ok")
