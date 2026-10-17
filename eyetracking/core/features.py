from typing import Sequence

from eyetracking.core.models import FrameResult

# MediaPipe indices. "right" = the person's right eye (left side of the image).
R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS = 33, 133, 159, 145, 468
L_INNER, L_OUTER, L_UPPER, L_LOWER, L_IRIS = 362, 263, 386, 374, 473
NOSE_TIP = 1

# how many numbers gaze_features() returns. Saved calibrations with a different
# count come from an older version of this file and can't be trained on.
N_FEATURES = 10


def gaze_features(r: FrameResult) -> tuple[float, ...] | None:
    """Head-invariant gaze features: WHERE the iris sits inside each eye.

    Raw landmark coords are mostly "where is the face in the picture", so a
    1 cm head shift swamps the iris moving. Ratios inside the eye cancel that.
    Each ratio uses one axis only, so the camera's aspect ratio cancels too.
    """
    if r.landmarks is None or r.head_pose is None:
        return None
    lm = r.landmarks
    x, y = lm[:, 0], lm[:, 1]

    def ratio(v, i, a, b):         # where i sits: 0 at landmark a, 1 at landmark b
        span = v[b] - v[a]
        return (v[i] - v[a]) / span if abs(span) > 1e-6 else None

    feats = []
    for i, (a, b) in ((R_IRIS, (R_OUTER, R_INNER)), (L_IRIS, (L_INNER, L_OUTER))):
        feats.append(ratio(x, i, a, b))                 # left/right inside the eye
    for i, (a, b) in ((R_IRIS, (R_UPPER, R_LOWER)), (L_IRIS, (L_UPPER, L_LOWER))):
        feats.append(ratio(y, i, a, b))                 # up/down between the lids
    if any(f is None for f in feats):                # eye closed / degenerate
        return None

    face_w = abs(x[L_OUTER] - x[R_OUTER])            # ~ distance to the screen
    hp = r.head_pose
    feats += [hp.yaw / 30, hp.pitch / 30, hp.roll / 30,   # head turned = eyes compensate
              x[NOSE_TIP], y[NOSE_TIP], face_w * 5]       # head moved / leaned in
    return tuple(float(f) for f in feats)


def describe_features(f: Sequence[float]) -> str:
    """One frame's features in words, for the gaze-data viewer."""
    rx, lx, ry, ly, yaw, pitch, roll, nx, ny, fw = f
    return (f"eyes: iris {rx:.0%} across, {ry:.0%} down in the right eye, {lx:.0%} / {ly:.0%} in the left; "
            f"head: turned {yaw * 30:+.0f}°, nodded {pitch * 30:+.0f}°, tilted {roll * 30:+.0f}°; "
            f"nose {nx:.0%} across, {ny:.0%} down the picture; face {fw / 5:.2f} of the picture wide (bigger = closer)")


def head_degrees(f: Sequence[float]) -> tuple[float, float]:
    """(yaw, pitch) in degrees: how far the head was turned and nodded."""
    return f[4] * 30, f[5] * 30
