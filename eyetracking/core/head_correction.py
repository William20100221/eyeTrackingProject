"""
Head-rotation compensation for the eye features: the one thing a regression cannot learn
from a calibration, because the head hardly moves during one.

When the head turns by a degree while the gaze stays on a point, the eye turns the other
way by a degree, so the iris moves inside the eye as if the gaze had moved. Measured on the
head steps of the Sweep / Points / Line model calibrations (gaze fixed on the centre dot,
the head turning, nodding, tilting): each of the 6 eye features shifts by a fixed amount
per degree of yaw, pitch and roll, and the amounts are the same for both people measured
(18 Oct 2026: 6 head steps, 1477 frames, two people; their own matrices agree within
~0.001 per degree). So one default serves anyone, and a calibration with a head step
measures its own. Corrected eye = eye - (head angles - reference angles) @ K, with the
reference = the calibration's mean head angles: every frame is turned back to the posture
of the calibration, and the regression only ever sees eye features at that posture.

Why only rotation, and only the eye features (eyetracking/testing/bench_*.py on 14 real
calibrations: train on one, test on another sitting of the same person):
  * today's model (12 raw features, a curve): 2800% of the screen. The head POSITION
    features barely move within a calibration, so the curve extrapolates wildly when the
    next sitting is 10 cm further away.
  * the 6 eye features only: 32%; with this correction: 23% (a straight line), the same
    with the default K as with the person's own.
  * a parallax term from MediaPipe's head position made it worse (32%): that position is
    relative, not metric (weak-perspective camera, MediaPipe docs).
The same idea as "data normalisation" in appearance-based gaze estimation (Zhang, Sugano
& Bulling, ETRA 2018, https://perceptualui.org/publications/zhang18_etra/): cancel the
head pose before learning, instead of asking the model to learn it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from eyetracking.core.features import EYE, HEAD, IRIS_DOWN
from eyetracking.core.models import CalibrationSample

EYE6 = list(range(EYE.start, EYE.stop)) + list(range(IRIS_DOWN.start, IRIS_DOWN.stop))   # the 6 eye features
ANGLES = [HEAD.start, HEAD.start + 1, HEAD.start + 2]                                    # yaw, pitch, roll (degrees)
MIN_HEAD_FRAMES = 60
K_RIDGE = 1.0            # a small ridge: an angle that hardly moved in the head step can't get a wild weight

# per degree of yaw, pitch, roll (rows): the change of [right x, left x, right y, left y, right down, left down]
DEFAULT_K = np.array([[-0.00341, -0.00382, -0.00140, -0.00051, -0.00097,  0.00005],
                      [-0.00059,  0.00008, -0.00006, -0.00093,  0.00455,  0.00392],
                      [-0.00207, -0.00215, -0.00336,  0.00058,  0.00074, -0.00233]])


def fit_k(head: Sequence[CalibrationSample]) -> np.ndarray | None:
    """K from one head step (gaze on the centre dot, the head moving), or None if it is too short."""
    if len(head) < MIN_HEAD_FRAMES:
        return None
    F = np.array([s.features for s in head], float)
    A = F[:, ANGLES] - F[:, ANGLES].mean(axis=0)
    E = F[:, EYE6] - F[:, EYE6].mean(axis=0)
    return np.linalg.solve(A.T @ A + K_RIDGE * np.eye(3), A.T @ E)


def reference_angles(samples: Sequence[CalibrationSample]) -> np.ndarray:
    """The posture a calibration was made in: its mean yaw, pitch, roll."""
    return np.array([s.features for s in samples], float)[:, ANGLES].mean(axis=0)


def corrected(features: Sequence[float], K: np.ndarray, ref: np.ndarray) -> tuple[float, ...]:
    """The 6 eye features of one frame, as they would be at the reference posture."""
    f = np.asarray(features, float)
    return tuple(float(v) for v in f[EYE6] - (f[ANGLES] - ref) @ K)


if __name__ == "__main__":
    # a fake head step whose eye features shift with the angles exactly by K: fit_k gets K back
    rng = np.random.default_rng(0)
    ang = rng.normal(0, 8, (200, 3))
    eye = 0.5 + ang @ DEFAULT_K + rng.normal(0, 0.002, (200, 6))
    frames = [CalibrationSample((*e[:4], *a, 0, 0, -60, *e[4:]), 0.5, 0.5) for e, a in zip(eye, ang)]
    K = fit_k(frames)
    assert K is not None and np.abs(K - DEFAULT_K).max() < 0.0003, K - DEFAULT_K
    assert fit_k(frames[:10]) is None
    # a frame at a turned head is corrected back to the reference posture's eye features
    turned = 0.5 + np.array([10.0, 0, 0]) @ DEFAULT_K
    f = (*turned[:4], 10, 0, 0, 0, 0, -60, *turned[4:])
    assert np.abs(np.array(corrected(f, DEFAULT_K, np.zeros(3))) - 0.5).max() < 1e-9
    print("ok: K recovered within", round(float(np.abs(K - DEFAULT_K).max()), 5), "per degree")
