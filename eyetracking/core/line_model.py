"""
Line model: where a line from the eyes hits the screen, in a rough 3D model
of the user and the computer. No Qt, no camera: testable with numbers.

THE 3D MODEL (camera space, cm; the same space MediaPipe's head pose is in)
    camera   at the origin, looking at the user (who is at z < 0);
             x = right in the camera image, y = up
    screen   the flat rectangle just below the camera (ScreenGeometry)
    eyes     placed with MediaPipe's head pose (angles + position) from its
             canonical face model, decoded from models/face_landmarker.task:
             eye-corner midpoints at x = -/+3.151, y = 2.625, z = 3.466 cm
             (6.30 cm apart: the average adult IPD, see core/vergence.py)

1. THE STRAIGHT LINE (no calibration)
   For each eye: how far the iris sits from the middle of the eye (sideways:
   the iris-between-corners ratio; up/down: its height vs the corners), in cm,
   / the iris's distance from the eye's centre of rotation = sin(eye angle).
   That turns the eye inside the head; the head pose turns the head. Each
   eye's line is aimed so the two lines meet on the screen at the vergence
   angle (2*atan(IPD/2 / d), ~6 degrees at 60 cm); the straight line is the
   middle one, from the point between the eyes.

2. THE CALIBRATED LINE
   During calibration we know the dot, so the TRUE line is "from between the
   eyes to the dot". For each direction (sideways, up/down), fit
       true angle = gain * straight angle + offset
   (offset ~ the eye's ~5 degree "angle kappa" between its optical axis and
   where it really looks; gain fixes the guessed eyeball size).

3. THE BLEND (option C)
       final line = w * straight line + (1 - w) * calibrated line
   w = how much to trust the plain 3D model. It is chosen by leave-one-dot-
   out: the w that best predicts dots the fit never saw. w near 1 = the 3D
   model is already right; w near 0 = the calibration does the work.

Eye sizes used (rough; the calibration's gain absorbs the error):
  * centre of rotation ~13.3-14.8 mm behind the front of the cornea
    (Donders; Fry & Hill 1962), iris ~3.5 mm behind the cornea ->
    iris-to-centre ~1.0 cm:  https://onlinelibrary.wiley.com/doi/full/10.1111/opo.12940
  * visible iris ~11.7 mm across:  https://pmc.ncbi.nlm.nih.gov/articles/PMC8639187/
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from eyetracking.core.features import HEAD, IRIS_DOWN, N_FEATURES
from eyetracking.core.models import CalibrationSample
from eyetracking.core.vergence import vergence_angle_rad

# canonical face model (cm): right eye, left eye corner midpoints
EYE_CORNER_MIDS = np.array([[-3.151, 2.625, 3.466], [3.151, 2.625, 3.466]])
EYE_WIDTH_CM = 2.590         # corner to corner, one eye (canonical model)
ROTATION_DEPTH_CM = 1.0      # eye's centre of rotation behind the corner midpoint
IRIS_RADIUS_CM = 1.0         # centre of rotation -> iris
CAMERA_ABOVE_SCREEN_CM = 1.0  # webcam centre above the screen's top edge: CHANGE if yours differs
BLEND_STEPS = np.linspace(0, 1, 11)   # w candidates tried: 0, 0.1, ... 1


@dataclass(frozen=True)
class ScreenGeometry:
    """The screen as a rectangle in camera space. Screen fractions (0..1,
    x across from the user's left, y down) <-> points in cm."""
    width_cm: float
    height_cm: float
    camera_above_cm: float = CAMERA_ABOVE_SCREEN_CM

    def point(self, fx: float, fy: float) -> np.ndarray:
        # the user's left is the camera's right (+x): the camera faces the user
        return np.array([self.width_cm / 2 - fx * self.width_cm,
                         -(self.camera_above_cm + fy * self.height_cm), 0.0])

    def fraction(self, p: np.ndarray) -> tuple[float, float]:
        return ((self.width_cm / 2 - p[0]) / self.width_cm,
                (-p[1] - self.camera_above_cm) / self.height_cm)


def head_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rebuild MediaPipe's head rotation from the 3 angles landmarker.py took
    out of it (yaw about y, pitch about x, roll about z: R = Rz @ Ry @ Rx)."""
    a, b, g = (math.radians(v) for v in (pitch_deg, yaw_deg, roll_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]])
    return rz @ ry @ rx


def angles(direction: np.ndarray) -> np.ndarray:
    """Direction (towards the screen) -> [sideways, up/down] angles, radians."""
    return np.array([math.atan2(direction[0], direction[2]), math.atan2(direction[1], direction[2])])


def direction(ang: np.ndarray) -> np.ndarray:
    d = np.array([math.tan(ang[0]), math.tan(ang[1]), 1.0])
    return d / np.linalg.norm(d)


def eye_lines(features: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """The rough 3D model for one frame. Returns (eye centres (2, 3), measured
    eye directions (2, 3), straight-line direction IN THE HEAD (3,), head
    rotation (3, 3), vergence angle used, rad).
    Each measured direction is where that eye actually points (camera space);
    the straight line is the middle line, with each eye's share of the
    vergence removed, kept in head coordinates so calibration can correct the
    EYE's angles (kappa and eyeball size are properties of the eye, not of
    the head: correcting the head turn too would bend every head movement)."""
    f = np.asarray(features, dtype=float)
    yaw, pitch, roll, hx, hy, hz = f[HEAD]
    rot = head_rotation(yaw, pitch, roll)
    centres = (EYE_CORNER_MIDS - [0, 0, ROTATION_DEPTH_CM]) @ rot.T + [hx, hy, hz]
    vergence = vergence_angle_rad(-centres[:, 2].mean())   # eyes -> screen plane distance
    measured, parallel = [], []
    # right eye: its nose side is +x, so converging turns it +x; the left eye -x
    for ratio, down, nasal in ((f[0], f[IRIS_DOWN][0], +1), (f[1], f[IRIS_DOWN][1], -1)):
        side = math.asin(np.clip((ratio - 0.5) * EYE_WIDTH_CM / IRIS_RADIUS_CM, -1, 1))
        up = math.asin(np.clip(-down / IRIS_RADIUS_CM, -1, 1))       # down in the image = negative up
        for out, s, turn in ((measured, side, rot), (parallel, side - nasal * vergence / 2, np.eye(3))):
            d_head = [math.sin(s) * math.cos(up), math.sin(up), math.cos(s) * math.cos(up)]
            out.append(turn @ d_head)
    straight = np.mean(parallel, axis=0)
    return centres, np.array(measured), straight / np.linalg.norm(straight), rot, vergence


class LineModel:
    """Same use as GazeModel: fit(samples, sample_weight) then predict(features)
    -> (x, y) screen fractions. Works on RAW features (not GazeProfile-scaled)."""

    def __init__(self, screen: ScreenGeometry) -> None:
        self.screen = screen
        self.gain = np.ones(2)      # calibrated line: true = gain * straight + offset
        self.offset = np.zeros(2)
        self.w = 1.0                # blend: final = w * straight + (1 - w) * calibrated
        self._trained = False

    def _straight_and_true(self, samples: Sequence[CalibrationSample]) -> tuple[np.ndarray, np.ndarray]:
        """Both as the EYE's angles in the head: straight (from the 3D model) and
        true (the line from between the eyes to the dot, turned into head coordinates)."""
        straight, true = [], []
        for s in samples:
            centres, _, d, rot, _ = eye_lines(s.features)
            to_dot = self.screen.point(s.target_x, s.target_y) - centres.mean(axis=0)
            straight.append(angles(d))
            true.append(angles(rot.T @ to_dot))
        return np.array(straight), np.array(true)

    @staticmethod
    def _fit_lines(s: np.ndarray, t: np.ndarray, wt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Weighted least squares per direction: t = gain * s + offset."""
        gain, offset = np.ones(2), np.zeros(2)
        sw = np.sqrt(wt)
        for k in range(2):
            a = np.c_[s[:, k], np.ones(len(s))] * sw[:, None]
            gain[k], offset[k] = np.linalg.lstsq(a, t[:, k] * sw, rcond=None)[0]
        return gain, offset

    def fit(self, samples: Sequence[CalibrationSample],
            sample_weight: Sequence[float] | None = None) -> LineModel:
        if not samples or len(samples[0].features) != N_FEATURES:
            raise ValueError(f"need samples with {N_FEATURES} raw features")
        s, t = self._straight_and_true(samples)
        wt = np.ones(len(s)) if sample_weight is None else np.asarray(sample_weight, float)
        self.gain, self.offset = self._fit_lines(s, t, wt)

        # choose w on dots the fit never saw: hold out each dot, fit on the rest
        dots = [(x.target_x, x.target_y) for x in samples]
        errors = np.zeros(len(BLEND_STEPS))
        for dot in set(dots):
            out = np.array([d == dot for d in dots])
            if out.all():
                continue                                  # only one dot: nothing to hold out
            g, o = self._fit_lines(s[~out], t[~out], wt[~out])
            for i, w in enumerate(BLEND_STEPS):
                final = w * s[out] + (1 - w) * (g * s[out] + o)
                errors[i] += (wt[out] * np.hypot(*(final - t[out]).T)).sum()
        self.w = float(BLEND_STEPS[int(np.argmin(errors))])
        self._trained = True
        return self

    def predict(self, features: Sequence[float]) -> tuple[float, float] | None:
        """Raw features -> (x, y) screen fractions, or None if the line points away from the screen."""
        centres, _, d, rot, _ = eye_lines(features)
        s = angles(d)                                         # the eye's angles, in the head
        final = self.w * s + (1 - self.w) * (self.gain * s + self.offset)
        origin, line = centres.mean(axis=0), rot @ direction(final)   # then the head turn
        if line[2] <= 1e-6:
            return None
        return self.screen.fraction(origin + (-origin[2] / line[2]) * line)

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        out = [self.predict(f) for f in X]
        return np.array([p if p is not None else (np.nan, np.nan) for p in out])

    def is_trained(self) -> bool:
        return self._trained


def features_for(screen: ScreenGeometry, target: tuple[float, float], head: Sequence[float],
                 kappa_deg: tuple[float, float] = (0.0, 0.0), radius_cm: float = IRIS_RADIUS_CM,
                 rng: np.random.Generator | None = None) -> tuple[float, ...]:
    """The OPPOSITE of eye_lines(): the raw features a face with this head pose
    would give while looking at `target`. Used to test the model: real eyes
    differ from the model by kappa (an angle offset) and eyeball size."""
    yaw, pitch, roll, hx, hy, hz = head
    rot = head_rotation(yaw, pitch, roll)
    centres = (EYE_CORNER_MIDS - [0, 0, ROTATION_DEPTH_CM]) @ rot.T + [hx, hy, hz]
    p = screen.point(*target)
    ratios, downs = [], []
    for c in centres:                                       # each eye aims at the point itself
        d_head = rot.T @ ((p - c) / np.linalg.norm(p - c))
        side = math.atan2(d_head[0], d_head[2]) + math.radians(kappa_deg[0])
        up = math.asin(d_head[1]) + math.radians(kappa_deg[1])
        ratios.append(0.5 + math.sin(side) * radius_cm / EYE_WIDTH_CM)
        downs.append(-math.sin(up) * radius_cm)
    noise = (lambda n: rng.normal(0, n)) if rng is not None else (lambda n: 0.0)
    # iris-between-the-lids barely moves (the lids follow the eye): real data 0.44-0.49
    lids = [0.47 - 0.05 * d for d in downs]
    return (ratios[0] + noise(0.004), ratios[1] + noise(0.004), lids[0] + noise(0.004), lids[1] + noise(0.004),
            yaw, pitch, roll, hx, hy, hz, downs[0] + noise(0.003), downs[1] + noise(0.003))


if __name__ == "__main__":
    # self-check: a fake person whose eyes differ from the model (kappa 2.5 / -1.5
    # degrees, eyeball 20% smaller), with some head movement and noise
    from eyetracking.core.calibration import point_grid
    rng = np.random.default_rng(0)
    screen = ScreenGeometry(27.7, 18.9)
    def head():
        return [rng.normal(0, 3), rng.normal(0, 3), rng.normal(0, 2),
                rng.normal(0, 2), rng.normal(-5, 2), rng.normal(-60, 3)]
    samples = [CalibrationSample(features_for(screen, t, head(), (2.5, -1.5), 0.8, rng), *t)
               for t in point_grid() for _ in range(20)]

    def err_px(model, test):
        return np.mean([np.hypot((p[0] - s.target_x) * 1440, (p[1] - s.target_y) * 960)
                        for s in test if (p := model.predict(s.features)) is not None])

    # 1. the two eye lines really meet on the screen at the vergence angle
    f = features_for(screen, (0.5, 0.5), [0, 0, 0, 0, -5, -60])
    centres, measured, _, _, verg = eye_lines(f)
    between = math.degrees(math.acos(np.clip(measured[0] @ measured[1], -1, 1)))
    print(f"1. eye lines meet at {between:.2f} deg; formula says {math.degrees(verg):.2f} deg")
    assert abs(between - math.degrees(verg)) < 0.2

    # 2. a perfect eye needs no calibration: the straight line alone is right
    perfect = [CalibrationSample(features_for(screen, t, head()), *t) for t in point_grid()]
    plain = LineModel(screen)
    plain._trained = True
    print(f"2. perfect eye, straight line only: {err_px(plain, perfect):.1f} px")
    assert err_px(plain, perfect) < 5

    # 3. a real-ish eye: the straight line is off, the calibration fixes it.
    #    Without noise the fix must be near-perfect; with noise, what's left is the noise.
    clean = [CalibrationSample(features_for(screen, t, head(), (2.5, -1.5), 0.8), *t) for t in point_grid()]
    fixed = LineModel(screen).fit(clean)
    print(f"3a. no noise: straight {err_px(plain, clean):.0f} px -> calibrated {err_px(fixed, clean):.1f} px")
    assert err_px(fixed, clean) < 10
    fitted = LineModel(screen).fit(samples)
    print(f"3b. with noise: straight {err_px(plain, samples):.0f} px -> calibrated {err_px(fitted, samples):.0f} px"
          f" (w = {fitted.w:.1f}, gain {fitted.gain.round(2)}, offset {np.degrees(fitted.offset).round(1)} deg)")
    assert err_px(fitted, samples) < err_px(plain, samples) / 2
    print("ok")
