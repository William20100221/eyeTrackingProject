"""
Self-check for blink handling on master (no webcam):  python -m eyetracking.testing.test_blinks

1. EyeTracker skips a blink coming (lids closing), the blink, and 250 ms of
   reopening; a rise that never becomes a blink (looking down) is let through
   again after 200 ms
2. The app holds the gaze point through a blink instead of freezing or jumping (issue #9)
"""
import os
import types

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

# ---- 2. app holds the point through a blink --------------------------------
from PySide6.QtWidgets import QApplication
qt = QApplication.instance() or QApplication([])
import eyetracking.app as appmod
from eyetracking.core.smoothing import GazeSmoother
fed = []
face = types.SimpleNamespace(face_found=True)              # the FrameResult: a blink is still a face
feats = iter([((0.2, 0.2), face, 0.0)] + [(None, face, i / 30) for i in (1, 2, 3)])   # one prediction, then a blink
app = types.SimpleNamespace(
    mouse=False, smoothed=None, smoother=GazeSmoother(), reading=None, reader=None,
    clock=types.SimpleNamespace(restart=lambda: 33),
    tracker=object(), _tick_frames=lambda: next(feats),   # (features, FrameResult, time) from the camera's thread
    ball=types.SimpleNamespace(set_tracking_state=lambda f: None, look_at=lambda p: None),
    dwell=types.SimpleNamespace(feed=lambda p: fed.append((p.x(), p.y()))),
    predict=lambda f: f)
for _ in range(4):
    appmod.App.tick(app)
assert len(fed) == 4 and len(set(fed)) == 1, fed           # same point fed through the blink
print("2. app: point held through the blink:", fed[0], "x", len(fed))
print("ok")
