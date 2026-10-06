"""
ML(Machine Learning) file
input gazing features -> output gazing coordinates on screen

Screen positions are fractions 0..1, the same as calibration. Errors are
measured in pixels, so those functions also need the screen size.

Model: Ridge regression on degree-2 polynomial features: enough bend for
the eye's non-linear mapping, few enough knobs for ~25 dots not to overfit.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eyetracking.core.models import CalibrationSample


def _xy(samples: list[CalibrationSample]) -> tuple[np.ndarray, np.ndarray]:
    # X = the model INPUT: one row per sample, one column per feature (samples x 10)
    X = np.array([s.features for s in samples])
    # y = the right ANSWERS: one row per sample, where the dot was (samples x 2)
    y = np.array([(s.target_x, s.target_y) for s in samples])
    return X, y


class GazeModel:
    def __init__(self, regressor=None) -> None:
        # any sklearn regressor works here. The default is 3 steps in a row:
        #   StandardScaler      every feature on the same scale, so iris ratios
        #                       (~0..1) and head angles don't drown each other
        #   PolynomialFeatures  adds squares and pairs (a^2, a*b) so the model
        #                       can fit a curve, not only a straight line
        #   RidgeCV             linear regression that keeps the weights small
        #                       (less overfitting); tries 13 strengths, keeps the best
        # To compare: GazeModel(LinearRegression()) and check the numbers.
        # ponytail: one global poly+ridge; try per-axis models or a small MLP only once you have 100s of dots
        self._regressor = regressor if regressor is not None else make_pipeline(
            StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        self._trained = False

    def fit(self, samples: list[CalibrationSample]) -> GazeModel:
        X, y = _xy(samples)
        # every sklearn model learns with .fit(); a 2-column y trains x and y together
        self._regressor.fit(X, y)
        self._trained = True
        return self   # so GazeModel().fit(samples) hands back the trained model

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        # sklearn predicts a whole TABLE of rows at once, so one sample goes in
        # as a table with 1 row: shape (1, 10), not (10,)
        row = np.array([features], dtype=float)
        x, y = self._regressor.predict(row)[0]   # answer is (1, 2): take row 0
        return float(x), float(y)

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        # the whole table at once: one call instead of predict() in a loop
        return self._regressor.predict(X)

    def is_trained(self) -> bool:
        return self._trained


def mean_error_px(model: GazeModel, samples: list[CalibrationSample],
                  screen_w: int, screen_h: int) -> float:
    """Average distance in pixels between where the model says you looked
    and where the dot really was."""
    X, y = _xy(samples)
    # to pixels BEFORE measuring: the screen isn't square (1440 x 960), so
    # 0.1 across and 0.1 down are different distances
    d = (model.predict_many(X) - y) * (screen_w, screen_h)
    # straight-line distance for every sample (Pythagoras), then the average
    return float(np.hypot(d[:, 0], d[:, 1]).mean())


def leave_one_dot_out(samples: list[CalibrationSample],
                      screen_w: int, screen_h: int) -> dict[tuple[float, float], float]:
    """For each dot: train on the OTHER dots only, then measure the error on
    this dot. Honest, because the model never saw the dot it's tested on.
    Edge/corner dots measure extrapolation, so expect them to be worst."""
    dots = sorted({(s.target_x, s.target_y) for s in samples})
    errors = {}
    for dot in dots:
        # hold out a WHOLE dot: samples from the same dot are near-copies, so
        # testing on any of them would let the model peek at the answer
        train = [s for s in samples if (s.target_x, s.target_y) != dot]
        test = [s for s in samples if (s.target_x, s.target_y) == dot]
        model = GazeModel().fit(train)
        errors[dot] = mean_error_px(model, test, screen_w, screen_h)
    return errors


if __name__ == "__main__":
    # self-check on fake data: a smooth non-linear eye->screen map plus noise.
    from eyetracking.core.calibration import point_grid
    rng = np.random.default_rng(0)
    samples = []
    for tx, ty in point_grid(5):
        for _ in range(40):
            hx, hy = tx - 0.5, ty - 0.5
            f = [hx + 0.3 * hx**2, hy + 0.2 * hx * hy, hx * 0.9, hy * 1.1]
            f = np.array(f + [0, 0, 0, 0.5, 0.5, 1.0]) + rng.normal(0, 0.01, 10)
            samples.append(CalibrationSample(tuple(f), tx, ty))
    w, h = 1920, 1080
    fit_err = mean_error_px(GazeModel().fit(samples), samples, w, h)
    errs = leave_one_dot_out(samples, w, h)
    inner = [e for (x, y), e in errs.items() if 0.1 < x < 0.9 and 0.1 < y < 0.9]
    print(f"train {fit_err:.0f}px, held-out interior dots {np.mean(inner):.0f}px")
    assert fit_err < 30 and np.mean(inner) < 40
    print("ok")
