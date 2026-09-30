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
