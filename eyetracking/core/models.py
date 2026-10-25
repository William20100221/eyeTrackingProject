"""
Personalized dataclasses for describing face landmark data
"""

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class HeadPose:
    yaw: float
    "left/right yaw"
    pitch: float
    "up/down pitch"
    roll: float
    "tilting your ear toward your shoulder"
    # where the head IS (not where it points), from MediaPipe's facial
    # transformation matrix, roughly in cm from the camera. Defaults keep
    # HeadPose(yaw, pitch, roll) working for code that only has angles.
    x: float = 0.0
    "left/right of the camera"
    y: float = 0.0
    "up/down"
    z: float = 0.0
    "distance from the camera (negative: in front of it)"

@dataclass(frozen=True)
class FrameResult:
    timestamp_ms: int
    face_found: bool
    head_pose: HeadPose | None
    blink_score_left: float
    blink_score_right: float
    landmarks: np.ndarray | None

@dataclass(frozen=True)
class CalibrationSample:
    features: tuple[float, ...]   # what the eyes looked like (model INPUT)
    target_x: float               # where the dot was, 0..1 across (model ANSWER)
    target_y: float               # 0..1 down

@dataclass(frozen=True)
class CalibrationRecording:
    """Everything one calibration recorded (core/calibration.py's three steps), as RAW features
    (gaze_features()). Raw on purpose: the head correction is worked out when the model trains,
    so improving that maths never needs a recalibration. "sweep" is the lab's name for step 1."""
    sweep: tuple[CalibrationSample, ...]   # step 1: eyes on the 14 still dots across + down the middle, head still
    head: tuple[CalibrationSample, ...]    # step 2: eyes on the centre dot, head moving
    dots: tuple[CalibrationSample, ...]    # step 3: the 4x4 grid in random order: the model's main training data


# A frame source's answer when no new camera frame has come since the last ask (services/tracker.py
# FrameFeed). Not a blink, not a lost face: nothing happened yet. A window draws on and records nothing.
NO_FRAME = object()
