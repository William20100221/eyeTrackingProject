"""
Does head-ROTATION compensation, learned from the head step, fix the new-sitting error?

    python -m eyetracking.testing.bench_head <data folder>

K = how much each of the 6 eye features (4 iris ratios + 2 iris heights) changes per degree
of head yaw / pitch / roll, fitted on head-step frames (gaze fixed on the centre dot, the
head moving). Corrected eye = eye - (angles - angles_ref) @ K, angles_ref = the training
file's mean head angles. Then the usual eye-only regressions on the corrected features.
Three places K can come from: the training file's own head step (the person, that sitting),
the person's other files (pooled), or the OTHER person (a default for someone with no
head step at all).
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eyetracking.core.features import EYE, HEAD, IRIS_DOWN
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.quality import check_recording
from eyetracking.services.storage import load_calibration, method_of, person_of

warnings.filterwarnings("ignore")
ALPHAS = np.logspace(-3, 3, 13)
EYE6 = list(range(EYE.start, EYE.stop)) + list(range(IRIS_DOWN.start, IRIS_DOWN.stop))
ANG = [HEAD.start, HEAD.start + 1, HEAD.start + 2]          # yaw, pitch, roll (degrees)
HZ = HEAD.start + 5
K_RIDGE = 1.0


def xy(samples):
    return np.array([s.features for s in samples], float), np.array([(s.target_x, s.target_y) for s in samples])


def groups_of(samples):
    dots = sorted({(s.target_x, s.target_y) for s in samples})
    return np.array([dots.index((s.target_x, s.target_y)) for s in samples])


def fit_k(head_frames_list) -> np.ndarray:
    """(3 angles -> 6 eye features) from one or more head steps, each centred on its own means."""
    A, E = [], []
    for frames in head_frames_list:
        F = np.array([s.features for s in frames], float)
        A.append(F[:, ANG] - F[:, ANG].mean(axis=0))
        E.append(F[:, EYE6] - F[:, EYE6].mean(axis=0))
    A, E = np.vstack(A), np.vstack(E)
    return np.linalg.solve(A.T @ A + K_RIDGE * np.eye(3), A.T @ E)


def correct(X, K, ref_angles):
    return X[:, EYE6] - (X[:, ANG] - ref_angles) @ K


def ridge_lodo(X, y, groups, degree=2):
    pipe = make_pipeline(StandardScaler(), PolynomialFeatures(degree), Ridge())
    cv = LeaveOneGroupOut() if len(set(groups)) <= 25 else GroupKFold(10)
    return GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=cv, scoring="neg_mean_squared_error").fit(X, y, groups=groups).best_estimator_


def knn(X, y, groups):
    return make_pipeline(StandardScaler(), KNeighborsRegressor(15, weights="distance")).fit(X, y)


REGRESSIONS = {"poly 2 ridge": ridge_lodo, "linear ridge": lambda X, y, g: ridge_lodo(X, y, g, 1), "kNN 15": knn}


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
        if report.ok:
            files.append((person_of(path) or "data", method_of(path), path.name, report.kept))
    return files


def error(pred, y):
    return float(np.nanmean(np.hypot(*(pred - y).T)))


def main() -> int:
    files = load(Path(sys.argv[1]))
    people = sorted({f[0] for f in files})
    heads = {p: [f[3].head for f in files if f[0] == p and len(f[3].head) >= 60] for p in people}
    K = {p: fit_k(heads[p]) for p in people if heads[p]}
    all_K = fit_k([h for p in people for h in heads[p]])
    print("K per degree of head angle (rows yaw, pitch, roll; columns rx lx ry ly rdown ldown):")
    for p, k in K.items():
        print(f"  {p} ({len(heads[p])} head steps):\n" + "\n".join("    " + " ".join(f"{v:8.4f}" for v in row) for row in k))

    results = defaultdict(list)
    for person in people:
        mine = [f for f in files if f[0] == person]
        others = [p for p in people if p != person and p in K]
        for ti, (_, ttype, tname, train) in enumerate(mine):
            Xtr, ytr = xy(train.dots)
            gtr = groups_of(train.dots)
            ref = Xtr[:, ANG].mean(axis=0)
            hz_ref = Xtr[:, HZ].mean()
            sources = {"no correction": None}
            if len(train.head) >= 60:
                sources["K from this file's own head step"] = fit_k([train.head])
            if person in K:
                sources["K from the person's head steps (pooled)"] = K[person]
            if others:
                sources["K from the OTHER person (a default)"] = K[others[0]]
            sources["K pooled over everyone"] = all_K
            for j, (_, _, name2, test) in enumerate(mine):
                if j == ti:
                    continue
                Xte, yte = xy(test.dots)
                still = np.linalg.norm(Xtr[:, HEAD.start + 3:HEAD.stop].mean(0) - Xte[:, HEAD.start + 3:HEAD.stop].mean(0)) < 3
                for sname, k in sources.items():
                    ctr = Xtr[:, EYE6] if k is None else correct(Xtr, k, ref)
                    cte = Xte[:, EYE6] if k is None else correct(Xte, k, ref)
                    for rname, fit in REGRESSIONS.items():
                        m = fit(ctr, ytr, gtr)
                        results[(sname, rname, "")].append((still, error(m.predict(cte), yte)))
                    if k is not None and "pooled" in sname and "person" in sname:
                        # + distance: the eye deviation from straight ahead scales with 1/distance (MediaPipe's z is relative)
                        centre = ctr.mean(axis=0)
                        scale = lambda C, X: centre + (C - centre) * (X[:, HZ:HZ + 1] / hz_ref)
                        m = ridge_lodo(scale(ctr, Xtr), ytr, gtr)
                        results[(sname, "poly 2 ridge", "+ distance scaling")].append((still, error(m.predict(scale(cte, Xte)), yte)))

    print("\nNEW-SITTING TEST (train on one calibration, test on another of the same person)")
    print(f"{'correction':<44} {'regression':<14} {'':<20} {'all':>7} {'still':>7} {'moved':>7}   n")
    rows = []
    for (sname, rname, extra), rs in results.items():
        a = np.mean([e for _, e in rs])
        st = [e for s, e in rs if s]
        mv = [e for s, e in rs if not s]
        rows.append((a, f"{sname:<44} {rname:<14} {extra:<20} {a:7.1%} {np.mean(st) if st else float('nan'):7.1%} "
                        f"{np.mean(mv) if mv else float('nan'):7.1%}   {len(rs)}"))
    for _, row in sorted(rows):
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
