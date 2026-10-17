"""
One camera frame -> the RAW gaze features, 12 numbers.

    0-3   eye:  where each iris sits inside its eye
                [right x, left x, right y, left y]  (y = between the LIDS)
    4-6   head angles:   yaw, pitch, roll (degrees)
    7-9   head position: x, y, z (~cm from the camera)
    10-11 iris height vs the eye CORNERS, in cm, down = positive [right, left]

Why 10-11 exist: the lids follow the eye when you look up or down, so
"iris between the lids" (2-3) hardly changes with vertical gaze. Measured on
two real calibrations (15 Oct): its average per dot row stayed at
0.44-0.49 from the top row to the bottom row. The eye corners are fixed to
the skull, so the iris height measured from them does move with gaze.

Raw = straight from the face, not yet adjusted for this user. The line
model (core/line_model.py) turns them into a 3D line from each eye.
"""
from typing import Sequence

from eyetracking.core.models import FrameResult

# MediaPipe indices. "right" = the person's right eye (left side of the image).
R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS = 33, 133, 159, 145, 468
L_INNER, L_OUTER, L_UPPER, L_LOWER, L_IRIS = 362, 263, 386, 374, 473
NOSE_TIP = 1

# From MediaPipe's own canonical face model (the cm-scale 3D face shipped in
# models/face_landmarker.task): eye-corner midpoint y = 2.625, nose tip
# y = -1.127, so the nose tip is 3.75 cm below the eye corners. Used as a
# head-fixed ruler: it turns image units into cm without knowing the
# camera's resolution or aspect ratio.
CORNER_TO_NOSE_CM = 3.752

N_FEATURES = 12            # how many numbers gaze_features() returns
EYE = slice(0, 4)          # the eye part (ratios)
HEAD = slice(4, 10)        # the head part (angles + position)
IRIS_DOWN = slice(10, 12)  # iris height vs the corners, cm


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

    # iris height below the line through its two eye corners (image y points down)
    corners_y = ((y[R_OUTER] + y[R_INNER]) / 2, (y[L_INNER] + y[L_OUTER]) / 2)
    ruler = y[NOSE_TIP] - (corners_y[0] + corners_y[1]) / 2   # 3.75 cm in image units
    if ruler <= 1e-6:
        return None
    down = [(y[R_IRIS] - corners_y[0]) * CORNER_TO_NOSE_CM / ruler,
            (y[L_IRIS] - corners_y[1]) * CORNER_TO_NOSE_CM / ruler]
    return tuple(float(f) for f in eye + head + down)


def describe_features(f: Sequence[float]) -> str:
    """One frame's features in words, for the gaze-data viewer."""
    rx, lx, ry, ly, yaw, pitch, roll, x, y, z, dr, dl = f
    return (f"eyes: iris {rx:.0%} across, {ry:.0%} down in the right eye, {lx:.0%} / {ly:.0%} in the left; "
            f"head: turned {yaw:+.0f}°, nodded {pitch:+.0f}°, tilted {roll:+.0f}°, "
            f"{z:.0f} cm from the camera ({x:+.0f} cm across, {y:+.0f} cm up)" + f"; iris {dr:+.2f} / {dl:+.2f} cm below the eye corners")


def head_degrees(f: Sequence[float]) -> tuple[float, float]:
    """(yaw, pitch) in degrees: how far the head was turned and nodded."""
    return f[4], f[5]
