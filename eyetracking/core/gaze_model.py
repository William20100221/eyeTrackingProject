"""
ML(Machine Learning) file
input gazing features -> output gazing coordinates on screen

Screen positions are fractions 0..1, the same as calibration. Errors are
measured in pixels, so those functions also need the screen size.

Model: Ridge regression, a straight line (degree 1) by default since 18 Oct
2026, on the 6 eye features turned back to the calibration's posture
(core/head_correction.py). Measured on 14 real calibrations (testing/bench_final.py):
a straight line beats the degree-2 curve both inside a sitting (interior dots held
out: 11.8% vs 13.5% of the screen) and on a new sitting (23.6% vs 26.5%); the curve
bends away faster where the features drift. degree=2 is still there to compare.
The ridge strength is chosen with whole DOTS held out (5 folds): a dot's ~25 frames are
near-copies, so leaving one FRAME out (RidgeCV's default) never really hides
a dot and picks a strength that memorises the dots (bench: 27% -> 25% error
on dots the model never saw). Block cross-validation, Roberts et al. 2017:
https://doi.org/10.1111/ecog.02881
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ALPHAS = np.logspace(-3, 3, 7)       # ridge strengths tried (13 measured the same error for twice the time)
DOT_FOLDS = 5                        # whole-dot folds in the strength search

from eyetracking.core.models import CalibrationSample


def _xy(samples: list[CalibrationSample]) -> tuple[np.ndarray, np.ndarray]:
    # X = the model INPUT: one row per sample, one column per feature
    X = np.array([s.features for s in samples])
    # y = the right ANSWERS: one row per sample, where the dot was (samples x 2)
    y = np.array([(s.target_x, s.target_y) for s in samples])
    return X, y


def dot_groups(samples: list[CalibrationSample]) -> np.ndarray:
    """Which dot each sample belongs to (0, 1, 2...): the blocks cross-validation holds out."""
    dots = sorted({(s.target_x, s.target_y) for s in samples})
    return np.array([dots.index((s.target_x, s.target_y)) for s in samples])


def leave_dots_out(groups: np.ndarray):
    """Whole dots held out, DOT_FOLDS at a time (every dot in turn when there are fewer).
    Leave-one-dot-out with 13 strengths gave the same error (27.3% vs 27.4% of the screen on
    a new sitting, 102 pairs of real files, 18 Oct 2026) for 4x the training time (1.2 s vs
    0.3 s per file), so the app trains with 5 folds and 7 strengths."""
    return GroupKFold(min(DOT_FOLDS, len(set(groups))))


class GazeModel:
    def __init__(self, regressor=None, degree: int = 1) -> None:
        # any sklearn regressor works here. The default is 3 steps in a row:
        #   StandardScaler      every feature on the same scale, so iris ratios
        #                       (~0..1) and head angles don't drown each other
        #   PolynomialFeatures  degree 2 adds squares and pairs (a^2, a*b) so the
        #                       model can fit a curve; degree 1 = a straight line
        #   Ridge               linear regression that keeps the weights small
        #                       (less overfitting); the strength is chosen in fit()
        #                       by leave-one-dot-out over 13 candidates
        # To compare: GazeModel(LinearRegression()) and check the numbers.
        # ponytail: one global poly+ridge; try per-axis models or a small MLP only once you have 100s of dots
        self._regressor = regressor if regressor is not None else make_pipeline(
            StandardScaler(), PolynomialFeatures(degree, include_bias=False), Ridge())
        self._search = regressor is None
        self._trained = False

    def fit(self, samples: list[CalibrationSample],
            sample_weight: Sequence[float] | None = None) -> GazeModel:
        X, y = _xy(samples)
        # sample_weight: how much each sample counts (newer calibrations more).
        # A pipeline must be told WHICH step gets it: the last one, the regressor
        kw = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)}
        if kw and isinstance(self._regressor, Pipeline):
            kw = {f"{self._regressor.steps[-1][0]}__sample_weight": kw["sample_weight"]}
        if self._search:
            # the strength that best predicts dots the fit never saw (whole dots held out)
            groups = dot_groups(samples)
            if len(set(groups)) >= 3:
                search = GridSearchCV(self._regressor, {"ridge__alpha": ALPHAS}, cv=leave_dots_out(groups),
                                      scoring="neg_mean_squared_error")
                search.fit(X, y, groups=groups, **kw)
                self._regressor = search.best_estimator_
                self._trained = True
                return self
        # every sklearn model learns with .fit(); a 2-column y trains x and y together
        self._regressor.fit(X, y, **kw)
        self._trained = True
        return self   # so GazeModel().fit(samples) hands back the trained model

    @property
    def alpha(self) -> float | None:
        """The ridge strength in use (None for a regressor of your own)."""
        return float(self._regressor.steps[-1][1].alpha) if isinstance(self._regressor, Pipeline) \
            and hasattr(self._regressor.steps[-1][1], "alpha") else None

    def predict(self, features: Sequence[float]) -> tuple[float, float]:
        # sklearn predicts a whole TABLE of rows at once, so one sample goes in
        # as a table with 1 row: shape (1, 4), not (4,)
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
    # nanmean: the line model gives no point (NaN) for a line that misses the screen
    return float(np.nanmean(np.hypot(d[:, 0], d[:, 1])))


def leave_one_dot_out(samples: list[CalibrationSample],
                      screen_w: int, screen_h: int,
                      weights: Sequence[float] | None = None,
                      test: list[CalibrationSample] | None = None,
                      make_model=None) -> dict[tuple[float, float], float]:
    """For each dot: train on the OTHER dots only, then measure the error on
    this dot. Honest, because the model never saw the dot it's tested on.
    Edge/corner dots measure extrapolation, so expect them to be worst.
    `test` (default: `samples`) = which samples to score, e.g. only the newest
    session, so models trained on different data are scored the same way.
    `make_model` (default GazeModel) builds a fresh model, e.g. a LineModel."""
    make_model = make_model or GazeModel
    test = samples if test is None else test
    dots = sorted({(s.target_x, s.target_y) for s in test})
    errors = {}
    for dot in dots:
        # hold out a WHOLE dot (in every session): samples from the same dot are
        # near-copies, so testing on any of them would let the model peek
        keep = [i for i, s in enumerate(samples) if (s.target_x, s.target_y) != dot]
        model = make_model().fit([samples[i] for i in keep],
                                None if weights is None else [weights[i] for i in keep])
        held_out = [s for s in test if (s.target_x, s.target_y) == dot]
        errors[dot] = mean_error_px(model, held_out, screen_w, screen_h)
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
    model = GazeModel(degree=2).fit(samples)             # the fake map is a curve: degree 2 fits it
    fit_err = mean_error_px(model, samples, w, h)
    assert model.alpha in ALPHAS
    errs = leave_one_dot_out(samples, w, h, make_model=lambda: GazeModel(degree=2))
    inner = [e for (x, y), e in errs.items() if 0.1 < x < 0.9 and 0.1 < y < 0.9]
    print(f"train {fit_err:.0f}px, held-out interior dots {np.mean(inner):.0f}px")
    assert fit_err < 30 and np.mean(inner) < 40

    # weights: an older session where the head sat differently (eye features
    # shifted). Weighting today's session higher should fit TODAY better.
    shift = np.array([0.1] * 4 + [0] * 6)
    old = [CalibrationSample(tuple(np.array(s.features) + shift), s.target_x, s.target_y)
           for s in samples]
    both, weights = samples + old, [1.0] * len(samples) + [0.5] * len(old)
    plain = mean_error_px(GazeModel(degree=2).fit(both), samples, w, h)
    weighted = mean_error_px(GazeModel(degree=2).fit(both, weights), samples, w, h)
    print(f"today's error: equal weights {plain:.0f}px, newest weighted {weighted:.0f}px")
    assert weighted < plain
    print("ok")
