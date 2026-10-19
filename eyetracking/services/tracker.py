"""
Camera + FaceLandmarker glued together: one call reads a frame and gives back
eye features, or None when the frame is no use (no face, a blink, or the
eyes still reopening after one).

No Qt in here. app.py reads it on a thread of its own (FrameFeed, below), so the
windows and the tracking loop never wait for the camera.
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

# frames still skipped after the eyes reopen: the lids are partly closed for a
# moment, which throws the iris-between-the-lids numbers off
# ponytail: counted in frames (~150 ms at 20 fps), not time; fine while the frame rate is steady
BLINK_RECOVERY_FRAMES = 3


class EyeTracker:
    def __init__(self, camera_index: int | None = None) -> None:
        self.face_found = False              # set by read_features(); a blink still counts as a face
        self.last_result: FrameResult | None = None   # read_features()'s frame, before the blink rules
        self._recovering = 0                 # frames left to skip after a blink
        self._camera = Camera(camera_index)  # None = ask in the console
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
        no face, a blink, or one of the BLINK_RECOVERY_FRAMES just after one.
        Also sets self.face_found, so callers can tell a blink (face still
        there: hold the last point) from a lost face."""
        result = self.last_result = self.read()
        self.face_found = result is not None and result.face_found
        if not self.face_found:
            return None
        if is_blinking(result):
            self._recovering = BLINK_RECOVERY_FRAMES
            return None
        if self._recovering:                     # eyes just reopened: not trusted yet
            self._recovering -= 1
            return None
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
    With the camera read on the GUI thread, the tracking loop and every full-screen window could only
    tick when a frame came, and a slow frame held them all. Now they tick at their own pace while
    OpenCV and MediaPipe work, as they let go of Python's lock, and every camera frame is read.
    Each user (a window, App.tick) gets its own reader(): every tick it asks for the newest frame and
    gets NO_FRAME when none came since its last ask, so it uses each frame once. Frames are fresh,
    too: none waits in the camera's buffer while a window does something else."""

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
        """A get() for one user: the newest frame's `part` ("features": the features or None;
        "stamped": (features or None, when it was read: perf_counter seconds, the smoother's clock);
        "result": the FrameResult or None; "frame": (features, FrameResult, time), all three), or
        NO_FRAME when no frame came since its last call."""
        seen = [0]

        def get():
            with self._lock:
                n, (features, result, t) = self._n, self._last
            if n == seen[0]:
                return NO_FRAME
            seen[0] = n
            return {"features": features, "stamped": (features, t), "result": result, "frame": (features, result, t)}[part]
        return get

    def stop(self) -> None:
        """No more reads. Waits for the read in progress (a frame), so the tracker can be closed after."""
        self._stop.set()
        self._thread.join()
