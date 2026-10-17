"""
Camera + FaceLandmarker glued together: one call reads a frame and gives back
eye features, or None when the frame is no use (no face, a blink, or the
eyes still reopening after one).

No Qt in here. The calibration window and app.py call read_features()
directly; the same object could run on a QThread if the UI ever stutters.
"""
from __future__ import annotations

from typing import Self

from eyetracking.core.analysis import is_blinking
from eyetracking.core.features import gaze_features
from eyetracking.core.models import FrameResult
from eyetracking.services.camera import Camera
from eyetracking.services.landmarker import FaceLandmarker

# frames still skipped after the eyes reopen: the lids are partly closed for a
# moment, which throws the iris-between-the-lids numbers off
# ponytail: counted in frames (~150 ms at 20 fps), not time; fine while the frame rate is steady
BLINK_RECOVERY_FRAMES = 3
LONG_BLINK_MS = 300       # eyes shut this long = on purpose (a natural blink is ~100-150 ms)


class EyeTracker:
    def __init__(self, camera_index: int | None = None) -> None:
        self.face_found = False              # set by read_features(); a blink still counts as a face
        self._recovering = 0                 # frames left to skip after a blink
        self._closed_ms: int | None = None   # when the current blink started
        self._long_blink = False             # a deliberate blink ended: see blinked()
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
        result = self.read()
        self.face_found = result is not None and result.face_found
        if not self.face_found:
            return None
        t, blinking = result.timestamp_ms, is_blinking(result)
        if blinking and self._closed_ms is None:
            self._closed_ms = t                       # the lids closed on this frame
        elif not blinking and self._closed_ms is not None:
            if t - self._closed_ms >= LONG_BLINK_MS:  # held shut on purpose: see blinked()
                self._long_blink = True
            self._closed_ms = None
        if blinking:
            self._recovering = BLINK_RECOVERY_FRAMES
            return None
        if self._recovering:                     # eyes just reopened: not trusted yet
            self._recovering -= 1
            return None
        return gaze_features(result)

    def blinked(self) -> bool:
        """True once after each deliberate blink (eyes shut LONG_BLINK_MS or more),
        on the frame they reopen. The gaze check records a sample on it."""
        flag, self._long_blink = self._long_blink, False
        return flag

    def close(self) -> None:
        self._landmarker.close()
        self._camera.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
