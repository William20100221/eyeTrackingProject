"""
Camera + FaceLandmarker glued together: one call reads a frame and gives back
eye features, or None when the frame is no use (no face, or a blink).

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


class EyeTracker:
    def __init__(self, camera_index: int | None = None) -> None:
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
        """Features for one frame, or None if this frame should be skipped."""
        result = self.read()
        if result is None or not result.face_found or is_blinking(result):
            return None
        return gaze_features(result)

    def close(self) -> None:
        self._landmarker.close()
        self._camera.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
