"""
Camera + FaceLandmarker glued together: one call reads a frame and gives back
eye features, or None when the frame is no use: no face, a blink coming, a
blink, or the eyes still reopening after one.

No Qt in here. app.py reads it on a thread of its own (FrameFeed, below), so the
windows never wait for the camera.

How a blink is handled, and why (the numbers come from these sources):
  * Closing is fast (~0.15 s), reopening slower (~0.25 s):
      Espinosa et al. 2018, https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0196125
      Kwon et al. 2013, https://royalsocietypublishing.org/doi/10.1098/rsif.2013.0227
  * The eyes rotate down and inward just BEFORE the lids close, so the gaze
    point drifts down before a blink is even detected:
      Bour et al. 2000, https://doi.org/10.1152/jn.2000.83.1.166
  * Eye openness is judged against the person's own level, and a blink is a
    fast change in it, not one fixed number:
      Soukupova & Cech 2016, https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
  * Eye trackers drop data on both sides of a blink (~50-150 ms before,
    ~150-300 ms after):
      Hershman et al. 2018, https://link.springer.com/article/10.3758/s13428-017-1008-1
      https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0261463
So: a blink score rising fast above the user's normal open-eye level = a
blink is coming (skip); a full blink (skip); then 250 ms of reopening (skip).
"""
from __future__ import annotations

import threading
import time
from typing import Self

from eyetracking.core.analysis import is_blinking
from eyetracking.core.features import gaze_features
from eyetracking.core.models import NO_FRAME, FrameResult
from eyetracking.services.camera import Camera
from eyetracking.services.landmarker import FaceLandmarker

BLINK_ONSET_RISE = 0.15   # blink score this far above the user's open-eye level = lids closing
BLINK_ONSET_MAX_MS = 200  # closing takes ~150 ms: a rise that isn't a blink by then was
                          # something else (looking down, squinting) and becomes the new normal
BLINK_RECOVERY_MS = 250   # reopening takes ~250 ms: the lids still cover part of the iris
OPEN_LEVEL_RATE = 0.05    # how fast the open-eye level follows slow changes (e.g. looking down)
# ponytail: rise/level thresholds are first guesses from the research, not measured on this camera


class EyeTracker:
    def __init__(self, camera_index: int | None = None) -> None:
        self.face_found = False              # set by read_features(); a blink still counts as a face
        self.last_result: FrameResult | None = None   # read_features()'s frame, before the blink rules
        self._open_level: float | None = None   # the user's normal blink score with eyes open
        self._onset_ms: int | None = None    # when the lids started closing (blink coming)
        self._blink_ms: int | None = None    # last frame that was a full blink
        self._camera = Camera(camera_index)   # None = ask in the console
        try:
            self._landmarker = FaceLandmarker()
        except Exception:
            self._camera.release()           # don't leave the camera on if MediaPipe fails
            raise

    def read(self) -> FrameResult | None:
        """One camera frame through MediaPipe. None = no frame, or no face."""
        frame = self._camera.read_rgb()
        if frame is None:
            return None
        try:
            return self._landmarker.detect(frame)
        except Exception as e:                   # one bad frame mustn't kill the app: skip it
            print(f"Skipped a frame: {e}")
            return None

    def read_features(self) -> tuple[float, ...] | None:
        """Features for one frame, or None if this frame should be skipped:
        no face, a blink coming (lids closing), a blink, or the eyes still
        reopening after one. Also sets self.face_found, so callers can tell a
        blink (face still there: hold the last point) from a lost face."""
        result = self.last_result = self.read()
        self.face_found = result is not None and result.face_found
        if not self.face_found:
            return None
        t = result.timestamp_ms
        score = max(result.blink_score_left, result.blink_score_right)   # either lid closing

        if is_blinking(result):                       # 1. a full blink
            self._blink_ms, self._onset_ms = t, None
            return None
        if self._blink_ms is not None and t - self._blink_ms < BLINK_RECOVERY_MS:
            return None                               # 2. still reopening

        if self._open_level is None:
            self._open_level = score
        if score - self._open_level > BLINK_ONSET_RISE:   # 3. lids closing: a blink is coming
            if self._onset_ms is None:
                self._onset_ms = t
            if t - self._onset_ms < BLINK_ONSET_MAX_MS:
                return None
            self._open_level = score                  # no blink came: this is the new normal
        self._onset_ms = None
        # eyes open: let the open level follow slow changes (looking down raises it a little)
        self._open_level += OPEN_LEVEL_RATE * (score - self._open_level)
        return gaze_features(result)

    def close(self) -> None:
        self._landmarker.close()
        self._camera.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class FrameFeed:
    """The tracker read on its own thread, as fast as the camera gives frames, the newest kept.
    With the camera read on the GUI thread, a full-screen window could only tick when a frame came:
    the sweep's dot moved 17-19 times a second here, with stalls of half a second, and the camera
    was read 18 times a second of its 30. Now the window ticks at its own pace (61 a second) while
    OpenCV and MediaPipe work, as they let go of Python's lock, and every one of the 30 frames is read.
    Each window gets its own reader(): every tick it asks for the newest frame and gets NO_FRAME when
    none came since its last ask, so it records each frame once. Frames are fresh, too: none waits
    in the camera's buffer while a window does something else."""

    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self._lock = threading.Lock()
        self._n = 0                                  # frames read so far
        self._last: tuple = (None, None, 0.0)        # (features or None, FrameResult or None, perf_counter s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera")
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            features = self.tracker.read_features()  # waits for the camera's next frame
            result = getattr(self.tracker, "last_result", None)
            with self._lock:
                self._n, self._last = self._n + 1, (features, result, time.perf_counter())
            time.sleep(0.001 if result is not None else 0.03)   # no frame at all (a camera gone): don't spin

    def reader(self, part: str = "features"):
        """A get() for one window: the newest frame's `part` ("features": the features or None;
        "stamped": (features or None, when it was read: perf_counter seconds, live_clock's clock);
        "result": the FrameResult or None), or NO_FRAME when no frame came since its last call."""
        seen = [0]

        def get():
            with self._lock:
                n, (features, result, t) = self._n, self._last
            if n == seen[0]:
                return NO_FRAME
            seen[0] = n
            return features if part == "features" else (features, t) if part == "stamped" else result
        return get

    def stop(self) -> None:
        """No more reads. Waits for the read in progress (a frame), so the tracker can be closed after."""
        self._stop.set()
        self._thread.join()
