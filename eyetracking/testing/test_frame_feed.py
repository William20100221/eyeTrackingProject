"""
Self-check for the camera read on its own thread (no webcam):  python -m eyetracking.testing.test_frame_feed

1. FrameFeed (services/tracker.py): each reader gets every new frame once, the newest if several came
   since it last asked, and NO_FRAME in between; "stamped" adds when it was read, "result" the
   FrameResult; stop() ends the reads.
2. The windows between two camera frames (NO_FRAME): the calibration records each frame once and
   doesn't take NO_FRAME for a blink (a blink drops the frames held back before it); the gaze check
   keeps its point and counts frames, not ticks.
"""
import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from eyetracking.core.models import NO_FRAME
from eyetracking.services.tracker import FrameFeed


# ---- 1. the feed --------------------------------------------------------------------------
class FakeTracker:
    """A camera that gives its next frame when the test lets it (go.release())."""

    def __init__(self):
        self.go, self.reads, self.last_result = threading.Semaphore(0), 0, None

    def read_features(self):
        self.go.acquire()
        self.reads += 1
        self.last_result = f"result {self.reads}"
        return (float(self.reads),)


def frames_in(feed, n):
    end = time.perf_counter() + 2
    while feed._n < n:
        assert time.perf_counter() < end, "the feed thread stalled"
        time.sleep(0.001)


tracker = FakeTracker()
feed = FrameFeed(tracker)
a, b, r = feed.reader(), feed.reader("stamped"), feed.reader("result")
assert a() is NO_FRAME and b() is NO_FRAME and r() is NO_FRAME                # nothing read yet
tracker.go.release()
frames_in(feed, 1)
assert a() == (1.0,) and a() is NO_FRAME                                      # a frame once, then NO_FRAME
features, t = b()
assert features == (1.0,) and 0 <= time.perf_counter() - t < 1               # when it was read (live_clock's clock)
assert r() == "result 1" and r() is NO_FRAME
tracker.go.release()
tracker.go.release()
frames_in(feed, 3)
assert a() == (3.0,) and a() is NO_FRAME                                      # two came: the newest only
assert feed._thread.daemon and feed._thread.name == "camera"
threading.Timer(0.05, tracker.go.release).start()                            # the read in progress ends...
feed.stop()                                                                    # ...and stop() waits for it
assert not feed._thread.is_alive()
reads = tracker.reads
tracker.go.release()
time.sleep(0.05)
assert tracker.reads == reads                                                  # no reads after stop()
print("1. feed ok")

# ---- 2. the windows, ticking faster than the camera ---------------------------------------
import eyetracking.core.calibration as cal
import eyetracking.ui.calibration_window as cw
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.ui.gaze_window import GazeWindow

cal.SWEEP_SETTLE_MS, cal.SWEEP_CROSS_MS = 50, 150                             # quick, as in test_blinks
cal.HEAD_SETTLE_MS, cal.HEAD_MOVE_MIN_MS, cal.HEAD_MOVE_MAX_MS = 50, 50, 150
cw.SETTLE_MS, cw.COLLECT_MS, cw.DOT_SAMPLES, cw.MAX_COLLECT_MS, cw.TICK_MS = 30, 150, 5, 1000, 5
qt = QApplication.instance() or QApplication([])
ticks, frame = [0], [0]


def get_features():
    """A new frame every third tick, each with its own number; NO_FRAME in between."""
    ticks[0] += 1
    if ticks[0] % 3:
        return NO_FRAME
    frame[0] += 1
    return (float(frame[0]),) + (0.5,) * 11


def calibrate(method_id):
    win = cw.CalibrationWindow(METHOD_BY_ID[method_id], get_features=get_features)
    out = []
    win.finished.connect(lambda rec: (out.append(rec), qt.quit()))
    win.resize(800, 600)
    win.show()
    win.start()
    QTimer.singleShot(60000, qt.quit)
    qt.exec()
    assert out, f"the {method_id} calibration didn't finish"
    return out[0]


sweep, grid = calibrate("sweep"), calibrate("original-4x4")       # a special's steps (no dots), a grid's dots
ids = [s.features[0] for rec in (sweep, grid) for part in (rec.sweep, rec.head, rec.dots) for s in part]
per_dot = {}
for s in grid.dots:
    per_dot[(s.target_x, s.target_y)] = per_dot.get((s.target_x, s.target_y), 0) + 1
assert len(ids) == len(set(ids)), "a frame recorded twice"
assert len(sweep.sweep) >= 20 and len(sweep.head) >= 20, ("NO_FRAME taken for a blink?", len(sweep.sweep), len(sweep.head))
assert len(per_dot) == METHOD_BY_ID["original-4x4"].dots and min(per_dot.values()) >= cw.DOT_SAMPLES, per_dot
print(f"2. calibration ok: {len(ids)} frames recorded once each over {ticks[0]} ticks")

seq = iter([NO_FRAME, (0.2, 0.3), NO_FRAME, NO_FRAME, None, NO_FRAME])
gaze = GazeWindow.single(lambda: next(seq))
gaze._tick()
assert gaze.point is None and gaze._frames == 0                               # nothing yet
gaze._tick()
gaze._tick()
gaze._tick()
assert gaze.point == (0.2, 0.3) and not gaze._stale and gaze._frames == 1     # one frame, three ticks
gaze._tick()
gaze._tick()
assert gaze.point == (0.2, 0.3) and gaze._stale and gaze._frames == 2         # a blink: held, grey
print("3. windows ok")
