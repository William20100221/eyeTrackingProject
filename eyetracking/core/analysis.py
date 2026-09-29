"""
Checking Blink
"""

from .models import FrameResult

BLINK_THRESHOLD = 0.5

def is_blinking(result: FrameResult) -> bool:
    if not result.face_found:
        return False
    return (result.blink_score_left > BLINK_THRESHOLD
            and result.blink_score_right > BLINK_THRESHOLD)


def count_blinks(results: list[FrameResult]) -> int:
    count = 0
    was_blinking = False
    for r in results:
        now = is_blinking(r)
        if now and not was_blinking:
            count += 1
        was_blinking = now
    return count


def is_looking_away(result: FrameResult, yaw_limit: float = 30.0) -> bool:
    if not result.face_found or result.head_pose is None:
        return True
    return abs(result.head_pose.yaw) > yaw_limit