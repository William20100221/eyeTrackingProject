r"""
Vergence: the angle between the two eyes' lines of sight. Both eyes aim at
the point being looked at, so their two lines meet there:

          right eye  o\
                       \  theta/2
    <- IPD ->           * <- the point on the screen, d away
                       /  theta/2
          left eye   o/

    theta = 2 * atan((IPD / 2) / d)

Why it "starts at 6 degrees": not a magic number. With the average adult
IPD (~63 mm) and a typical desk distance (60 cm):
    2 * atan(3.15 / 60) = 6.0 degrees.
Both inputs are named constants below, and live distance comes from
MediaPipe's head position, so the angle follows the user: 45 cm -> 8.0
degrees, 95 cm -> 3.8 degrees.

Sources (checked 16 Oct 2026):
  * Average IPD ~63 mm (US Army ANSUR data, 3976 adults; quoted values range
    58-70 mm): Dodgson 2004, https://doi.org/10.1117/12.529999
  * MediaPipe's own canonical face model puts the two eye-corner midpoints
    6.30 cm apart: the same number, so the face model and the formula agree.
  * Typical screen distance 60 cm (webcam gaze work starts from z0 = 60 cm):
    https://arxiv.org/pdf/2508.19544
  * The eye's line of sight (visual axis) is itself ~5 degrees nasal of the
    eye's optical axis ("angle kappa"), different per person: one reason the
    line model is calibrated per user. Mean 5.0 +- 1.4 degrees in 977 adults:
    https://pmc.ncbi.nlm.nih.gov/articles/PMC4341357

Units: inside the maths everything is radians and cm; degrees only for
display, and names say which (angle_deg, distance_cm).
"""
from __future__ import annotations

import math

IPD_CM = 6.30                # distance between the pupils (adult average, = MediaPipe's face model)
DEFAULT_DISTANCE_CM = 60.0   # typical eye-to-screen distance at a desk
MIN_DISTANCE_CM = 45.0       # valid head positions: closer or further is clamped,
MAX_DISTANCE_CM = 95.0       # so one noisy frame can't produce an impossible angle


def vergence_angle_rad(distance_cm: float, ipd_cm: float = IPD_CM) -> float:
    """Angle between the two lines of sight when looking at a point
    `distance_cm` away. Distances outside 45-95 cm (including 0 or negative,
    e.g. a bad frame) are clamped into that range."""
    d = min(max(distance_cm, MIN_DISTANCE_CM), MAX_DISTANCE_CM)
    return 2 * math.atan((ipd_cm / 2) / d)


def vergence_angle_deg(distance_cm: float, ipd_cm: float = IPD_CM) -> float:
    """Same, in degrees: for display only."""
    return math.degrees(vergence_angle_rad(distance_cm, ipd_cm))


# the starting value, worked out from the constants above (~6.0 degrees)
DEFAULT_ANGLE_DEG = vergence_angle_deg(DEFAULT_DISTANCE_CM)


if __name__ == "__main__":
    # self-check: the numbers the reasoning above promises
    for d, want in ((60, 6.0), (45, 8.0), (95, 3.8)):
        got = vergence_angle_deg(d)
        print(f"{d} cm -> {got:.2f} deg (want ~{want})")
        assert abs(got - want) < 0.05
    assert vergence_angle_deg(0) == vergence_angle_deg(-10) == vergence_angle_deg(MIN_DISTANCE_CM)
    assert vergence_angle_deg(500) == vergence_angle_deg(MAX_DISTANCE_CM)
    print(f"default: {DEFAULT_ANGLE_DEG:.2f} deg | 0 or negative distance -> clamped to 45 cm")
    print("ok")
