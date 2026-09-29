"""
Camera + FaceLandmarker glued together: one call reads a frame and gives back
eye features, or None when the frame is no use (no face, or a blink).

No Qt in here. The calibration window calls read_features() directly, and
later ui/worker.py can run the same object on a QThread for live tracking.
"""
from __future__ import annotations

from typing import Self

from eyetracking.core.analysis import is_blinking
from eyetracking.core.features import eye_points
from eyetracking.core.models import FrameResult
from eyetracking.services.camera import Camera
from eyetracking.services.landmarker import FaceLandmarker


class EyeTracker:
    def __init__(self) -> None:
        self._camera = Camera()              # asks which camera, in the console
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
        return self._landmarker.detect(frame)

    def read_features(self) -> tuple[float, ...] | None:
        """Features for one frame, or None if this frame should be skipped."""
        result = self.read()
        if result is None or not result.face_found or is_blinking(result):
            return None
        points = eye_points(result)
        if points is None:
            return None
        # PLACEHOLDER: the 6 raw eye points (x, y, z each = 18 numbers).
        # Swap this for your normalised features from features.py when ready.
        return tuple(points.flatten().tolist())

    def close(self) -> None:
        self._landmarker.close()
        self._camera.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
