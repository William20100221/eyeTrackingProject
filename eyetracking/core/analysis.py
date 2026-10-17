"""
Checking for a blink. (count_blinks / is_looking_away were removed in Oct 2026:
nothing called them; see GitHub issue #3.)
"""

from .models import FrameResult

BLINK_THRESHOLD = 0.5

def is_blinking(result: FrameResult) -> bool:
    if not result.face_found:
        return False
    return (result.blink_score_left > BLINK_THRESHOLD
            and result.blink_score_right > BLINK_THRESHOLD)
