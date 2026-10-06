"""
ML(Machine Learning) file
input gazing features -> output gazing coordinates on screen

Screen positions are fractions 0..1, the same as calibration. Errors are
measured in pixels, so those functions also need the screen size.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

from eyetracking.core.models import CalibrationSample


class GazeModel:
    def __init__(self, regressor=None) -> None:
        # any sklearn regressor works here; the default fits straight lines.
        # Later: GazeModel(Ridge(alpha=...)) and compare the numbers.
        self._regressor = regressor if regressor is not None else LinearRegression()
        self._trained = False

    def fit(self, samples: list[CalibrationSample]) -> None:
        # X = the model INPUT: one row per sample, one column per feature (289 x 18)
        X = np.array([s.features for s in samples])
        # y = the right ANSWERS: one row per sample, where the dot was (289 x 2)
        y = np.array([(s.target_x, s.target_y) for s in samples])
        # every sklearn model learns with .fit(); a 2-column y trains x and y together
        self._regressor.fit(X, y)
        self._trained = True

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        # sklearn predicts a whole TABLE of rows at once, so one sample goes in
        # as a table with 1 row: shape (1, 18), not (18,)
        row = np.array([features])
        x, y = self._regressor.predict(row)[0]   # answer is (1, 2): take row 0
        return float(x), float(y)

    def is_trained(self) -> bool:
        return self._trained


def mean_error_px(model: GazeModel, samples: list[CalibrationSample],
                  screen_w: int, screen_h: int) -> float:
    """Average distance in pixels between where the model says you looked
    and where the dot really was."""
    distances = []
    for s in samples:
        px, py = model.predict(s.features)
        # to pixels BEFORE measuring: the screen isn't square (1440 x 960), so
        # 0.1 across and 0.1 down are different distances
        guess = (px * screen_w, py * screen_h)
        truth = (s.target_x * screen_w, s.target_y * screen_h)
        distances.append(math.dist(guess, truth))
    return sum(distances) / len(distances)


def leave_one_dot_out(samples: list[CalibrationSample],
                      screen_w: int, screen_h: int) -> dict[tuple[float, float], float]:
    """For each dot: train on the OTHER dots only, then measure the error on
    this dot. Honest, because the model never saw the dot it's tested on."""
    dots = sorted({(s.target_x, s.target_y) for s in samples})
    errors = {}
    for dot in dots:
        # hold out a WHOLE dot: samples from the same dot are near-copies, so
        # testing on any of them would let the model peek at the answer
        train = [s for s in samples if (s.target_x, s.target_y) != dot]
        test = [s for s in samples if (s.target_x, s.target_y) == dot]
        model = GazeModel()
        model.fit(train)
        errors[dot] = mean_error_px(model, test, screen_w, screen_h)
    return errors
