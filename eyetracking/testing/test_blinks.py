"""
Self-check for blink handling (no webcam):  python -m eyetracking.testing.test_blinks

1. EyeTracker skips the blink AND the frames just after it
2. CalibrationSession drops the frames just BEFORE a blink
3. A whole calibration with frequent blinks: every dot still gets enough
   good frames, and no closing-lid frame is saved
4. The app holds the gaze point during a blink instead of freezing/jumping
"""
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from eyetracking.core.models import FrameResult, HeadPose
from eyetracking.services.tracker import BLINK_RECOVERY_FRAMES, EyeTracker

# ---- 1. tracker -------------------------------------------------------------
lm = np.random.default_rng(0).uniform(0.2, 0.8, (478, 3))
def frame(blink: float) -> FrameResult:
    return FrameResult(0, True, HeadPose(0, 0, 0), blink, blink, lm)
seq = [0.1, 0.1, 0.9, 0.9] + [0.1] * (BLINK_RECOVERY_FRAMES + 2)
t = EyeTracker.__new__(EyeTracker)                       # no camera: feed frames by hand
t.face_found, t._recovering = False, 0
frames = iter(seq)
t.read = lambda: frame(next(frames))
got = [t.read_features() is not None for _ in seq]
want = [True, True, False, False] + [False] * BLINK_RECOVERY_FRAMES + [True, True]
assert got == want, got
assert t.face_found                                      # a blink is still a face
print("1. tracker: blink + recovery frames skipped:", got)

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
if hasattr(cal, "SWEEP_CROSS_MS"):                        # branches with the moving sweep
    cal.SWEEP_SETTLE_MS, cal.SWEEP_CROSS_MS = 100, 250    # speed it up
    cal.SWEEP_AXIS_MS = cal.SWEEP_SETTLE_MS + cal.SWEEP_CROSS_MS * cal.SWEEP_CROSSINGS
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
    return (-1.0 if k >= 6 else 1.0,) + (0.5,) * 9        # 6, 7 = lids closing
win = cw.CalibrationWindow(get_features=get_features)
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

# ---- 4. app holds the point through a blink --------------------------------
import eyetracking.app as appmod
fed = []
feats = iter([(0.2, 0.2), None, None, None])               # one prediction, then a blink
app = types.SimpleNamespace(
    mouse=False, smoothed=None, reading=None,
    clock=types.SimpleNamespace(restart=lambda: 33),
    tracker=types.SimpleNamespace(read_features=lambda: next(feats), face_found=True),
    ball=types.SimpleNamespace(set_tracking_state=lambda f: None, look_at=lambda p: None), reader=None,
    dwell=types.SimpleNamespace(feed=lambda p: fed.append((p.x(), p.y()))),
    predict=lambda f: f)
for _ in range(4):
    appmod.App.tick(app)
assert len(fed) == 4 and len(set(fed)) == 1, fed           # same point fed through the blink
print("4. app: point held through the blink:", fed[0], "x", len(fed))
print("ok")
