"""
Self-check for the camera read on its own thread (no webcam):  python -m eyetracking.testing.test_frame_feed

1. FrameFeed (services/tracker.py): each reader gets every new frame once, the newest if several came
   since it last asked, and NO_FRAME in between; "stamped" adds when it was read, "result" is the
   FrameResult, "frame" all three; stop() ends the reads.
2. The windows between two camera frames (NO_FRAME): the calibration records each frame once and
   isn't thrown by NO_FRAME; the gaze check keeps its point and counts frames, not ticks; the blink
   test records a row per frame.
3. App.tick (the overlay, Go Reading, the sample text): each frame is smoothed once, on its own time;
   NO_FRAME isn't a lost face; the reading time of the ticks in between isn't lost, and the time the
   camera was off isn't reading time.
"""
import os
import sys
import threading
import time
import types

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
a, b, r, f = feed.reader(), feed.reader("stamped"), feed.reader("result"), feed.reader("frame")
assert a() is NO_FRAME and b() is NO_FRAME and r() is NO_FRAME and f() is NO_FRAME   # nothing read yet
tracker.go.release()
frames_in(feed, 1)
assert a() == (1.0,) and a() is NO_FRAME                                      # a frame once, then NO_FRAME
features, t = b()
assert features == (1.0,) and 0 <= time.perf_counter() - t < 1               # when it was read (the smoother's clock)
assert r() == "result 1" and r() is NO_FRAME
assert f() == ((1.0,), "result 1", t) and f() is NO_FRAME                     # all three, of the same frame
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
import eyetracking.ui.calibration_window as cw
from eyetracking.core.calibration import point_grid
from eyetracking.ui.blink_window import BlinkWindow
from eyetracking.ui.gaze_window import GazeWindow

import eyetracking.core.calibration as cal
cw.SETTLE_MS, cw.COLLECT_MS, cw.TICK_MS = 30, 250, 5                           # quick
cal.HEAD_SETTLE_MS, cal.HEAD_MOVE_MIN_MS, cal.HEAD_MOVE_MAX_MS = 30, 20, 120   # the head step too
qt = QApplication.instance() or QApplication([])
errors = []
sys.excepthook = lambda *e: errors.append(e[1])                                # a tick's exception is only printed
ticks, frame = [0], [0]


def get_features():
    """A new frame every third tick, each with its own number; NO_FRAME in between."""
    ticks[0] += 1
    if ticks[0] % 3:
        return NO_FRAME
    frame[0] += 1
    return (float(frame[0]),) + (0.5,) * 11


win = cw.CalibrationWindow(get_features=get_features)
out = []
win.finished.connect(lambda samples: (out.append(samples), qt.quit()))
win.resize(800, 600)
win.show()
win.start()
QTimer.singleShot(60000, qt.quit)
qt.exec()
sys.excepthook = sys.__excepthook__
assert not errors, (len(errors), errors[0])                                    # NO_FRAME passed on raw raises
assert out, "the calibration didn't finish"
rec = out[0]                                                                   # line dots, head step, 4x4 grid
ids = [s.features[0] for s in rec.sweep + rec.head + rec.dots]
per_dot = {}
for s in rec.dots:
    per_dot[(s.target_x, s.target_y)] = per_dot.get((s.target_x, s.target_y), 0) + 1
assert len(ids) == len(set(ids)), "a frame recorded twice"
assert len(per_dot) == len(point_grid(cal.GRID)) and min(per_dot.values()) >= 3, per_dot
assert len({(s.target_x, s.target_y) for s in rec.sweep}) == 13 and rec.head      # 14 line dots (the centre twice)
order = list(dict.fromkeys((s.target_x, s.target_y) for s in rec.dots))
assert order != point_grid(cal.GRID), "the grid wasn't shuffled"                 # random order
print(f"2. calibration ok: {len(ids)} frames recorded once each over {ticks[0]} ticks")

seq = iter([NO_FRAME, (0.2, 0.3), NO_FRAME, NO_FRAME, None, NO_FRAME])
gaze = GazeWindow(get_point=lambda: next(seq))
gaze._tick()
assert gaze._point is None and gaze._frames == 0                              # nothing yet
gaze._tick()
gaze._tick()
gaze._tick()
assert gaze._point == (0.2, 0.3) and not gaze._stale and gaze._frames == 1    # one frame, three ticks
gaze._tick()
gaze._tick()
assert gaze._point == (0.2, 0.3) and gaze._stale and gaze._frames == 2        # a blink: held, grey

scores = iter([(0.1, 0.2), NO_FRAME, NO_FRAME, (0.3, 0.4), NO_FRAME, None])
blink = BlinkWindow(get_scores=lambda: next(scores))
blink._phase = "record"
blink._clock.start()
for _ in range(6):
    blink._tick()
assert [row[2:] for row in blink.rows] == [(0.1, 0.2), (0.3, 0.4)] and blink._scores is None, blink.rows
print("2. gaze check and blink test ok")

# ---- 3. App.tick, ticking faster than the camera ------------------------------------------
import eyetracking.app as appmod

face = types.SimpleNamespace(face_found=True)                                  # the FrameResult: a blink is a face too
seq = iter([NO_FRAME, ((0.2, 0.2), face, 1.0), NO_FRAME, NO_FRAME, ((0.4, 0.4), face, 1.05), NO_FRAME,
            (None, face, 1.1), NO_FRAME, (None, None, 1.15), NO_FRAME])         # two frames, a blink, no face
now, smoothed, halo, reading = [0], [], [], []


class Clock:
    """QElapsedTimer's restart(): the ms since the last restart."""
    last = 0

    def restart(self):
        ms, self.last = now[0] - self.last, now[0]
        return ms


app = types.SimpleNamespace(
    mouse=False, tracker=object(), _tick_frames=lambda: next(seq), smoothed=None, reader=None, clock=Clock(),
    predict=lambda features: features, smoother=types.SimpleNamespace(feed=lambda p, t: smoothed.append((p, t)) or p),
    ball=types.SimpleNamespace(set_tracking_state=halo.append, look_at=lambda p: None),
    dwell=types.SimpleNamespace(feed=lambda p: None), reading=types.SimpleNamespace(feed=lambda p, dt: reading.append(dt)))
for _ in range(10):
    now[0] += 16                                                               # a tick every 16 ms
    appmod.App.tick(app)
assert smoothed == [((0.2, 0.2), 1.0), ((0.4, 0.4), 1.05)], smoothed          # each frame once, on its own time
assert halo == [True, True, True, False], halo                                 # set per frame: NO_FRAME isn't a lost face
assert [round(dt, 3) for dt in reading] == [0.032, 0.048, 0.032], reading     # every ms until the face was lost
app.tracker, now[0] = None, 10000                                              # the camera off for ten seconds...
appmod.App.tick(app)
app.tracker, seq = object(), iter([((0.4, 0.4), face, 2.0)])
now[0] += 16
appmod.App.tick(app)
assert round(reading[-1], 3) == 0.016, reading                                 # ...isn't reading time
print("3. App.tick ok")
