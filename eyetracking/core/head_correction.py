"""
Head-rotation compensation for the eye features: the one thing a regression cannot learn
from a calibration, because the head hardly moves during one. Ported from calibration-lab
("Models, step 2", 18 Oct 2026); the measurements and the why: docs/model_improvements.md.

When the head turns by a degree while the gaze stays on a point, the eye turns the other
way by a degree, so the iris moves inside the eye as if the gaze had moved. Measured on
the head steps of the 3-step calibrations (gaze fixed on the centre dot, the head turning,
nodding, tilting): each eye feature shifts by a fixed amount per degree of yaw, pitch and
roll, and the amounts are the same for both people measured. So one default serves anyone.
Corrected eye = eye - (head angles - reference angles) @ K, with the reference = the
calibration's mean head angles: every frame is turned back to the posture of the
calibration, and the regression only ever sees eye features at that posture
(core/gaze_model.py does this for every frame, trained or live).

Why only rotation, and only the eye features (the lab's bench_*.py on 14 real calibrations:
train on one sitting, test on another of the same person): the head POSITION numbers barely
move within a calibration, so a model given them extrapolates wildly when the next sitting
is 10 cm further away (2800% of the screen); the eye features alone 32%; with this
correction 23%. The same idea as "data normalisation" in appearance-based gaze estimation
(Zhang, Sugano & Bulling, ETRA 2018, https://perceptualui.org/publications/zhang18_etra/):
cancel the head pose before learning, instead of asking the model to learn it.

This branch's features (core/features.py): the 4 iris ratios, then yaw/30, pitch/30,
roll/30 (DEG turns those back into degrees, so K is per degree here too), then the nose
and the face width. The plain 25-dot calibration has no head step, so K is always
DEFAULT_K: measured on the 10 head steps on disk on 18 Oct 2026 (two people, 2665 frames:
the 3-step branches' two 10-feature calibrations and the lab's eight 12-feature ones, whose
first 4 numbers are these same iris ratios and whose angles are these in degrees); the lab's
own 6-step matrix agrees within 0.0013 per degree.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from eyetracking.core.models import CalibrationSample

EYE_FEATURES = [0, 1, 2, 3]    # right x, left x, right y, left y
ANGLES = [4, 5, 6]             # yaw, pitch, roll...
DEG = 30.0                     # ...each divided by 30 in the features

# per degree of yaw, pitch, roll (rows): the change of [right x, left x, right y, left y]
DEFAULT_K = np.array([[-0.00245, -0.00254, -0.00128, -0.00041],
                      [-0.00052, -0.00005, -0.00101, -0.00075],
                      [-0.00179, -0.00178, -0.00419,  0.00112]])


def reference_angles(samples: Sequence[CalibrationSample]) -> np.ndarray:
    """The posture a calibration was made in: its mean yaw, pitch, roll (degrees)."""
    return np.array([s.features for s in samples], float)[:, ANGLES].mean(axis=0) * DEG


def corrected(features: Sequence[float] | np.ndarray, K: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """The eye features of one frame (or of a table of frames, one per row), as they would be
    at the reference posture."""
    f = np.asarray(features, float)
    return f[..., EYE_FEATURES] - (f[..., ANGLES] * DEG - ref) @ K


if __name__ == "__main__":
    # a frame at a head turned 10 degrees (yaw/30 in the features) has its eye features shifted by
    # 10 @ K from the posture's 0.5: corrected back, they are 0.5 again; a table of frames row by row
    turned = 0.5 + np.array([10.0, 0, 0]) @ DEFAULT_K
    f = (*turned, 10 / DEG, 0, 0, 0.5, 0.5, 1.0)
    assert np.abs(corrected(f, DEFAULT_K, np.zeros(3)) - 0.5).max() < 1e-9
    assert corrected([f, f], DEFAULT_K, np.zeros(3)).shape == (2, 4)
    dots = [CalibrationSample(f, 0.5, 0.5), CalibrationSample((*turned, 20 / DEG, 0, 0, 0.5, 0.5, 1.0), 0.5, 0.5)]
    assert np.allclose(reference_angles(dots), [15, 0, 0])                   # the posture, in degrees
    print("ok: a 10 degree turn corrected back to the calibration's posture")
