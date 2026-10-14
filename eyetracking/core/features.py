"""
One camera frame -> the RAW gaze features, 10 numbers.

    0-3  eye:  where each iris sits inside its eye
               [right x, left x, right y, left y]
    4-6  head angles:   yaw, pitch, roll (degrees)
    7-9  head position: x, y, z (~cm from the camera)

Raw = straight from the face, not yet adjusted for this user. The
GazeProfile (core/calibration.py) turns them into the model's 4 inputs: it
uses the head numbers to remove the part of the eye numbers caused by head
movement, then rescales them to the eye range measured in the sweep.
"""
from eyetracking.core.models import FrameResult

# MediaPipe indices. "right" = the person's right eye (left side of the image).
R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS = 33, 133, 159, 145, 468
L_INNER, L_OUTER, L_UPPER, L_LOWER, L_IRIS = 362, 263, 386, 374, 473

N_FEATURES = 10        # how many numbers gaze_features() returns
EYE = slice(0, 4)      # the eye part
HEAD = slice(4, 10)    # the head part (angles + position)


def gaze_features(r: FrameResult) -> tuple[float, ...] | None:
    """Raw features for one frame, or None (no face / eye closed).

    The eye numbers are ratios INSIDE the eye (0 at one corner, 1 at the
    other), so where the face sits in the picture mostly cancels out.
    Each ratio uses one axis only, so the camera's aspect ratio cancels too.
    """
    if r.landmarks is None or r.head_pose is None:
        return None
    lm = r.landmarks
    x, y = lm[:, 0], lm[:, 1]

    def ratio(v, i, a, b):         # where i sits: 0 at landmark a, 1 at landmark b
        span = v[b] - v[a]
        return (v[i] - v[a]) / span if abs(span) > 1e-6 else None

    eye = [ratio(x, R_IRIS, R_OUTER, R_INNER),     # left/right inside the eye
           ratio(x, L_IRIS, L_INNER, L_OUTER),
           ratio(y, R_IRIS, R_UPPER, R_LOWER),     # up/down between the lids
           ratio(y, L_IRIS, L_UPPER, L_LOWER)]
    if any(f is None for f in eye):                # eye closed / degenerate
        return None

    hp = r.head_pose
    head = [hp.yaw, hp.pitch, hp.roll, hp.x, hp.y, hp.z]
    return tuple(float(f) for f in eye + head)
