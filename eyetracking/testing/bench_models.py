"""
Offline benchmark of gaze-model variants on REAL calibration files (no camera):

    python -m eyetracking.testing.bench_models <data folder> [--quick]

For every person, every pair of their calibration files (train on one, test on the
other's dots) is a new-sitting test: the model never saw that sitting. Pairs are split
by how far the head moved between the two files (the head position features, cm):
"still" (< 3 cm) is the same posture, "moved" is a new posture. Plus leave-one-dot-out
inside each file (filling the gap between a file's own dots, same posture).
The error is what the gaze check reports: the mean distance from the prediction to the
dot, as a fraction of the screen, every frame of the dot counting.
"""
from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eyetracking.core.calibration import GazeProfile
from eyetracking.core.features import EYE, HEAD, IRIS_DOWN
from eyetracking.core.line_model import LineModel, ScreenGeometry, angles, direction, eye_lines
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.quality import check_recording
from eyetracking.services.storage import load_calibration, method_of, person_of

warnings.filterwarnings("ignore")
SCREEN = ScreenGeometry(34.4, 19.4)      # a 15.6" 16:9 laptop: the geometric models need a size
ALPHAS = np.logspace(-3, 3, 13)
EYE6 = list(range(EYE.start, EYE.stop)) + list(range(IRIS_DOWN.start, IRIS_DOWN.stop))
EYE6_ANGLES = EYE6 + [HEAD.start, HEAD.start + 1, HEAD.start + 2]


def xy(samples):
    return np.array([s.features for s in samples], float), np.array([(s.target_x, s.target_y) for s in samples])


def dot_groups(samples):
    dots = sorted({(s.target_x, s.target_y) for s in samples})
    return np.array([dots.index((s.target_x, s.target_y)) for s in samples])


def cv_for(groups):
    return LeaveOneGroupOut() if len(set(groups)) <= 25 else GroupKFold(10)


# ---- regressions: fit(X, y, groups) -> object with predict(X) --------------------------
def ridge_lodo(X, y, groups, degree=2, alphas=ALPHAS):
    """Ridge on polynomial features, the strength chosen by leave-one-DOT-out."""
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge())
    gs = GridSearchCV(pipe, {"ridge__alpha": alphas}, cv=cv_for(groups), scoring="neg_mean_squared_error")
    return gs.fit(X, y, groups=groups).best_estimator_


def ridge_loo(X, y, groups):
    """Today's GazeModel: the strength by leave-one-SAMPLE-out (a dot's frames are near-copies)."""
    return make_pipeline(StandardScaler(), PolynomialFeatures(2), RidgeCV(alphas=ALPHAS)).fit(X, y)


def kernel_ridge(X, y, groups):
    pipe = make_pipeline(StandardScaler(), KernelRidge(kernel="rbf"))
    gs = GridSearchCV(pipe, {"kernelridge__alpha": np.logspace(-3, 1, 5), "kernelridge__gamma": np.logspace(-2, 0.5, 4)},
                      cv=GroupKFold(min(5, len(set(groups)))), scoring="neg_mean_squared_error")
    return gs.fit(X, y, groups=groups).best_estimator_


def gpr(X, y, groups):
    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(1e-3)
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel, normalize_y=True)).fit(X, y)


def knn(X, y, groups):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(15, weights="distance")).fit(X, y)


class Columns:
    """A regression on some of the 12 features only."""

    def __init__(self, cols, fit):
        self.cols, self.fit_fn = cols, fit

    def fit(self, X, y, groups):
        self.model = self.fit_fn(X[:, self.cols], y, groups)
        return self

    def predict(self, X):
        return self.model.predict(X[:, self.cols])


class Hybrid:
    """Learn the EYE -> gaze angle mapping; let GEOMETRY place the head.
    The target is the true gaze angle from between the eyes to the dot, in the head's own
    frame (frame="head": the head rotation from MediaPipe is applied afterwards) or in the
    camera's frame (frame="camera": the head angles are regression inputs instead; only the
    head POSITION is used geometrically). Prediction = that line, from the eyes' position
    given by the head pose, hitting the screen plane. The head's effect on the picture never
    has to be learned from a sitting in which the head did not move."""

    def __init__(self, cols, frame="head", fit=ridge_lodo, screen=SCREEN):
        self.cols, self.frame, self.fit_fn, self.screen = cols, frame, fit, screen

    def _geometry(self, f):
        centres, _, _, rot, _ = eye_lines(f)
        return centres.mean(axis=0), rot

    def fit(self, X, y, groups):
        targets = []
        for f, (tx, ty) in zip(X, y):
            origin, rot = self._geometry(f)
            to_dot = self.screen.point(tx, ty) - origin
            targets.append(angles(rot.T @ to_dot if self.frame == "head" else to_dot))
        self.model = self.fit_fn(X[:, self.cols], np.array(targets), groups)
        return self

    def predict(self, X):
        ang = self.model.predict(X[:, self.cols])
        out = []
        for f, a in zip(X, ang):
            origin, rot = self._geometry(f)
            d = direction(a)
            if self.frame == "head":
                d = rot @ d
            if d[2] <= 1e-6:
                out.append((np.nan, np.nan))
                continue
            out.append(self.screen.fraction(origin + (-origin[2] / d[2]) * d))
        return np.array(out)


class Line:
    def fit(self, X, y, groups):
        from eyetracking.core.models import CalibrationSample
        self.model = LineModel(SCREEN).fit([CalibrationSample(tuple(f), *t) for f, t in zip(X, y)])
        return self

    def predict(self, X):
        return self.model.predict_many(X)


def candidates(quick: bool):
    all12 = list(range(12))
    c = {
        "today: ridge, 12 features, poly 2, alpha by LOO sample": Columns(all12, ridge_loo),
        "ridge, 12 features, poly 2, alpha by leave-one-dot-out": Columns(all12, ridge_lodo),
        "ridge, eye 6 (iris ratios + iris height), poly 2, LODO": Columns(EYE6, ridge_lodo),
        "ridge, eye 6 + head angles, poly 2, LODO": Columns(EYE6_ANGLES, ridge_lodo),
        "hybrid: eye 6 -> head-frame angles (poly 2 ridge), head by geometry": Hybrid(EYE6, "head"),
        "hybrid: eye 6 + angles -> camera-frame angles, position by geometry": Hybrid(EYE6_ANGLES, "camera"),
        "line model (today's 3D line, gain + offset, blended)": Line(),
    }
    if not quick:
        c.update({
            "hybrid head-frame, poly 1 (linear)": Hybrid(EYE6, "head", lambda X, y, g: ridge_lodo(X, y, g, degree=1)),
            "hybrid head-frame, kernel ridge": Hybrid(EYE6, "head", kernel_ridge),
            "ridge, eye 6, poly 1 (linear), LODO": Columns(EYE6, lambda X, y, g: ridge_lodo(X, y, g, degree=1)),
            "kernel ridge (RBF), eye 6": Columns(EYE6, kernel_ridge),
            "Gaussian process (RBF + noise), eye 6": Columns(EYE6, gpr),
            "kNN (15, distance), eye 6": Columns(EYE6, knn),
        })
    return c


# ---- the profile candidates (files with a sweep + head step only) -----------------------
def scale_4(profile, f):
    return profile.scale(f)


def scale_6(profile, f):
    return (*profile.scale(f), *np.asarray(f)[IRIS_DOWN])


PROFILE = {"profile: 4 scaled eye ratios (today)": scale_4, "profile: 4 scaled ratios + iris height": scale_6}


class Profiled:
    def __init__(self, profile, scale):
        self.profile, self.scale = profile, scale

    def fit(self, X, y, groups):
        self.model = ridge_lodo(np.array([self.scale(self.profile, f) for f in X]), y, groups)
        return self

    def predict(self, X):
        return self.model.predict(np.array([self.scale(self.profile, f) for f in X]))


# ---- the data ---------------------------------------------------------------------------
def load(data: Path):
    files = []
    for path in sorted(data.rglob("*.json")):
        if "trash" in path.parts or "disabled" in path.parts or method_of(path) not in METHOD_BY_ID:
            continue
        try:
            rec = load_calibration(path)
        except Exception:
            continue
        report = check_recording(rec)
        if not report.ok:
            print(f"skip {path.name}: {report.summary()}")
            continue
        try:
            profile = GazeProfile.from_recording(report.kept)
        except ValueError:
            profile = None
        files.append((person_of(path) or "data", method_of(path), path.name, report.kept, profile))
    return files


def head_shift(a, b) -> float:
    """How far the head moved between two files: the mean head position (cm) of each, cm apart."""
    pa = np.mean([s.features[HEAD.start + 3:HEAD.stop] for s in a.dots], axis=0)
    pb = np.mean([s.features[HEAD.start + 3:HEAD.stop] for s in b.dots], axis=0)
    return float(np.linalg.norm(pa - pb))


def error(pred, y):
    return float(np.nanmean(np.hypot(*(pred - y).T)))


def main() -> int:
    data = Path(sys.argv[1])
    quick = "--quick" in sys.argv
    files = load(data)
    print(f"{len(files)} usable calibrations:", ", ".join(f"{p}/{m}" for p, m, *_ in files))
    results = defaultdict(list)                    # candidate -> [(shift cm, err)]
    t0 = time.time()
    for person in sorted({f[0] for f in files}):
        mine = [f for f in files if f[0] == person]
        for ti, (_, ttype, tname, train, tprofile) in enumerate(mine):
            Xtr, ytr = xy(train.dots)
            gtr = dot_groups(train.dots)
            fitted = {}
            for name, cand in candidates(quick).items():
                try:
                    fitted[name] = cand.fit(Xtr, ytr, gtr)
                except Exception as e:
                    print(f"  {name} failed on {tname}: {e}")
            if tprofile is not None and not quick:
                for name, scale in PROFILE.items():
                    fitted[name] = Profiled(tprofile, scale).fit(Xtr, ytr, gtr)
            for j, (_, _, name2, test, _) in enumerate(mine):
                if j == ti:
                    continue
                Xte, yte = xy(test.dots)
                shift = head_shift(train, test)
                for name, model in fitted.items():
                    results[name].append((shift, error(np.asarray(model.predict(Xte), float), yte)))
            print(f"  {person}/{ttype} {tname[-11:-5]} trained ({time.time() - t0:.0f}s)")

    print("\nNEW-SITTING TEST: train on one calibration, test on another of the same person")
    print("(error = mean distance to the dot, % of the screen; still = head moved < 3 cm between the files)")
    print(f"{'model':<70} {'all':>7} {'still':>7} {'moved':>7}   pairs still/moved")
    rows = []
    for name, rs in results.items():
        a = np.nanmean([e for _, e in rs])
        still = [e for s, e in rs if s < 3]
        moved = [e for s, e in rs if s >= 3]
        rows.append((a, f"{name:<70} {a:7.1%} {np.nanmean(still) if still else float('nan'):7.1%} "
                        f"{np.nanmean(moved) if moved else float('nan'):7.1%}   {len(still)}/{len(moved)}"))
    for _, row in sorted(rows):
        print(row)

    print("\nLEAVE-ONE-DOT-OUT inside each file (same posture: filling the gap between a file's own dots)")
    lodo = defaultdict(list)
    picks = {"today: ridge, 12 features, poly 2, alpha by LOO sample": Columns(list(range(12)), ridge_loo),
             "ridge, 12 features, poly 2, alpha by leave-one-dot-out": Columns(list(range(12)), ridge_lodo),
             "ridge, eye 6, poly 2, LODO": Columns(EYE6, ridge_lodo),
             "hybrid: eye 6 -> head-frame angles, head by geometry": Hybrid(EYE6, "head"),
             "line model (today)": Line()}
    for person, mtype, name, rec, _ in files:
        X, y = xy(rec.dots)
        g = dot_groups(rec.dots)
        for cname, cand in picks.items():
            preds = np.zeros_like(y)
            for dot in set(g):
                keep = g != dot
                preds[~keep] = cand.fit(X[keep], y[keep], g[keep]).predict(X[~keep])
            lodo[cname].append(error(preds, y))
    for cname, errs in sorted(lodo.items(), key=lambda kv: np.mean(kv[1])):
        print(f"{cname:<70} {np.mean(errs):7.1%}   over {len(errs)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
