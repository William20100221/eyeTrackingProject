import numpy as np
from eyetracking.core.models import FrameResult

# right eye: outer corner, inner corner, iris centre | left eye: inner corner, outer corner, iris centre
EYE_POINTS = [33, 133, 468, 362, 263, 473]


def eye_points(frame_result: FrameResult) -> np.ndarray | None:
    if frame_result.landmarks is None:
        return None
    return frame_result.landmarks[EYE_POINTS]