"""
ML(Machine Learning) file
input gazing features -> output gazing coordinates on screen

Screen positions are fractions 0..1, the same as calibration. Errors are
measured in pixels, so those functions also need the screen size.

Model (ported from calibration-lab, "Models, step 1 + 2", 18 Oct 2026; the measurements
and the why: docs/model_improvements.md): Ridge regression, a straight line (degree 1)
by default, on the EYE features only, each frame first turned back to the calibration's
posture with the head-rotation matrix K (core/head_correction.py). The head features
themselves are no longer inputs: they barely move within a calibration, so a model
given them extrapolated wildly at the next sitting (2800% of the screen on the lab's
14 real calibrations); the corrected eye features gave 23%. On the same files a straight
line beats the degree-2 curve both inside a sitting (interior dots held out: 11.8% vs
13.5% of the screen) and on a new sitting (23.6% vs 26.5%); the curve bends away faster
where the features drift. degree=2 is still there to compare.
The ridge strength is chosen by leave-one-DOT-out: a dot's ~25 frames are
near-copies, so leaving one FRAME out (RidgeCV's default) never really hides
a dot and picks a strength that memorises the dots (bench: 27% -> 25% error
on dots the model never saw). Block cross-validation, Roberts et al. 2017:
https://doi.org/10.1111/ecog.02881
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eyetracking.core.head_correction import DEFAULT_K, corrected, reference_angles
from eyetracking.core.models import CalibrationSample

ALPHAS = np.logspace(-3, 3, 13)      # ridge strengths tried


def _xy(samples: list[CalibrationSample]) -> tuple[np.ndarray, np.ndarray]:
    # X = the model INPUT: one row per sample, one column per raw feature
    X = np.array([s.features for s in samples])
    # y = the right ANSWERS: one row per sample, where the dot was (samples x 2)
    y = np.array([(s.target_x, s.target_y) for s in samples])
    return X, y


def dot_groups(samples: list[CalibrationSample]) -> np.ndarray:
    """Which dot each sample belongs to (0, 1, 2...): the blocks cross-validation holds out."""
    dots = sorted({(s.target_x, s.target_y) for s in samples})
    return np.array([dots.index((s.target_x, s.target_y)) for s in samples])


def leave_dots_out(groups: np.ndarray):
    """Every dot held out in turn (up to 25 dots), else 10 folds of whole dots."""
    return LeaveOneGroupOut() if len(set(groups)) <= 25 else GroupKFold(10)


class GazeModel:
    def __init__(self, regressor=None, degree: int = 1, K: np.ndarray | None = DEFAULT_K) -> None:
        # any sklearn regressor works here. The default is 3 steps in a row:
        #   StandardScaler      every feature on the same scale
        #   PolynomialFeatures  degree 2 adds squares and pairs (a^2, a*b) so the
        #                       model can fit a curve; degree 1 = a straight line
        #   Ridge               linear regression that keeps the weights small
        #                       (less overfitting); the strength is chosen in fit()
        #                       by leave-one-dot-out over 13 candidates
        # K: the head-rotation matrix (core/head_correction.py) every frame's eye features
        #    are corrected with, before the fit and before each prediction: the default, or
        #    the calibration's own. None = the raw features, all of them, as they are.
        # To compare: GazeModel(LinearRegression()) and check the numbers.
        # ponytail: one global poly+ridge; try per-axis models or a small MLP only once you have 100s of dots
        self._regressor = regressor if regressor is not None else make_pipeline(
            StandardScaler(), PolynomialFeatures(degree, include_bias=False), Ridge())
        self._search = regressor is None
        self._K, self._ref = K, np.zeros(3)
        self._trained = False

    def _rows(self, X: np.ndarray) -> np.ndarray:
        """What the regressor sees: the eye features at the calibration's posture (or the raw rows)."""
        return X if self._K is None else corrected(X, self._K, self._ref)

    def fit(self, samples: list[CalibrationSample],
            sample_weight: Sequence[float] | None = None) -> GazeModel:
        X, y = _xy(samples)
        if self._K is not None:
            # the calibration's posture: its mean head angles. A constant, so the scaler absorbs it;
            # what matters is that live frames are turned to the same posture (predict)
            self._ref = reference_angles(samples)
        X = self._rows(X)
        # sample_weight: how much each sample counts (newer calibrations more).
        # A pipeline must be told WHICH step gets it: the last one, the regressor
        kw = {} if sample_weight is None else {"sample_weight": np.asarray(sample_weight)}
        if kw and isinstance(self._regressor, Pipeline):
            kw = {f"{self._regressor.steps[-1][0]}__sample_weight": kw["sample_weight"]}
        if self._search:
            # the strength that best predicts dots the fit never saw (one dot held out at a time)
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
        # as a table with 1 row: shape (1, n), not (n,)
        row = self._rows(np.array([features], dtype=float))
        x, y = self._regressor.predict(row)[0]   # answer is (1, 2): take row 0
        return float(x), float(y)

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        # the whole table at once: one call instead of predict() in a loop
        return self._regressor.predict(self._rows(np.asarray(X, dtype=float)))

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
    # nanmean: a line model gives no point (NaN) for a line that misses the screen
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
    `make_model` (default GazeModel) builds a fresh model, e.g. one with the
    calibration's own K, or a LineModel."""
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
    # self-check on fake data: a smooth non-linear eye->screen map plus noise, the head turned
    # a little differently in every frame (sd 8 degrees) and the eye features shifted by that
    # turn exactly as DEFAULT_K says, as a real head does: the model must see through that
    from eyetracking.core.calibration import point_grid
    from eyetracking.core.features import N_FEATURES
    from eyetracking.core.head_correction import ANGLES, DEG, EYE_FEATURES
    rng = np.random.default_rng(0)
    samples = []
    for tx, ty in point_grid(5):
        for _ in range(40):
            hx, hy = tx - 0.5, ty - 0.5
            angles = rng.normal(0, 8, 3)
            f = np.full(N_FEATURES, 0.5)
            f[:4] = [hx + 0.3 * hx**2, hy + 0.2 * hx * hy, hx * 0.9, hy * 1.1]
            f[EYE_FEATURES] += angles @ DEFAULT_K
            f[ANGLES] = angles / DEG
            samples.append(CalibrationSample(tuple(f + rng.normal(0, 0.01, N_FEATURES)), tx, ty))
    w, h = 1920, 1080
    model = GazeModel(degree=2).fit(samples)             # the fake map is a curve: degree 2 fits it
    fit_err = mean_error_px(model, samples, w, h)
    assert model.alpha in ALPHAS
    interior = [s for s in samples if 0.1 < s.target_x < 0.9 and 0.1 < s.target_y < 0.9]
    errs = leave_one_dot_out(samples, w, h, test=interior, make_model=lambda: GazeModel(degree=2))
    print(f"train {fit_err:.0f}px, held-out interior dots {np.mean(list(errs.values())):.0f}px")
    assert fit_err < 30 and np.mean(list(errs.values())) < 40
    assert len(GazeModel(K=None).fit(samples).predict(samples[0].features)) == 2   # the raw path still works

    # weights: an older session where the head sat differently (eye features
    # shifted). Weighting today's session higher should fit TODAY better.
    shift = np.zeros(N_FEATURES)
    shift[:4] = 0.1
    old = [CalibrationSample(tuple(np.array(s.features) + shift), s.target_x, s.target_y)
           for s in samples]
    both, weights = samples + old, [1.0] * len(samples) + [0.5] * len(old)
    plain = mean_error_px(GazeModel(degree=2).fit(both), samples, w, h)
    weighted = mean_error_px(GazeModel(degree=2).fit(both, weights), samples, w, h)
    print(f"today's error: equal weights {plain:.0f}px, newest weighted {weighted:.0f}px")
    assert weighted < plain
    print("ok")
