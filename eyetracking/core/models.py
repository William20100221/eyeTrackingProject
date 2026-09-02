

from dataclasses import dataclass

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