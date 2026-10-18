"""
The final choice: which regression on the rotation-corrected eye features?

    python -m eyetracking.testing.bench_final <data folder>

Same posture (leave one dot out, interior dots) AND a new sitting (every file pair), for:
linear ridge, degree-2 ridge, their average, degree-2 with the strength floored at 1.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetracking.testing.bench_head import ANG, EYE6, correct, fit_k, groups_of, load, ridge_lodo, xy
from eyetracking.testing.bench_models import error

warnings.filterwarnings("ignore")


class Avg:
    def __init__(self, *models):
        self.models = models

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models], axis=0)


CANDIDATES = {
    "linear ridge": lambda X, y, g: ridge_lodo(X, y, g, 1),
    "degree-2 ridge": lambda X, y, g: ridge_lodo(X, y, g, 2),
    "average of linear + degree-2": lambda X, y, g: Avg(ridge_lodo(X, y, g, 1), ridge_lodo(X, y, g, 2)),
}


def main() -> int:
    files = load(Path(sys.argv[1]))
    people = sorted({f[0] for f in files})
    K = fit_k([f[3].head for f in files if len(f[3].head) >= 60])
    cross = defaultdict(list)
    same = defaultdict(list)
    for person in people:
        mine = [f for f in files if f[0] == person]
        for ti, (_, ttype, tname, train) in enumerate(mine):
            Xtr, ytr = xy(train.dots)
            gtr = groups_of(train.dots)
            ref = Xtr[:, ANG].mean(axis=0)
            ctr = correct(Xtr, K, ref)
            grid = len(set(gtr))
            for cname, fit in CANDIDATES.items():
                m = fit(ctr, ytr, gtr)
                for j, (_, _, _, test) in enumerate(mine):
                    if j != ti:
                        Xte, yte = xy(test.dots)
                        cross[cname].append((grid, error(m.predict(correct(Xte, K, ref)), yte)))
                # same posture: leave one INTERIOR dot out
                preds, mask = np.zeros_like(ytr), np.zeros(len(ytr), bool)
                for dot in set(gtr):
                    out = gtr == dot
                    tx, ty = ytr[out][0]
                    if min(tx, ty) <= 0.06 or max(tx, ty) >= 0.94:
                        continue
                    preds[out] = fit(ctr[~out], ytr[~out], gtr[~out]).predict(ctr[out])
                    mask |= out
                if mask.any():
                    same[cname].append((grid, error(preds[mask], ytr[mask])))
    print(f"{'regression on corrected eye features':<34} {'same posture, interior dots':>28} {'new sitting, all pairs':>24} {'new sitting, 16+ dot files':>28}")
    for cname in CANDIDATES:
        s = np.mean([e for _, e in same[cname]])
        c = np.mean([e for _, e in cross[cname]])
        big = np.mean([e for g, e in cross[cname] if g >= 16])
        print(f"{cname:<34} {s:28.1%} {c:24.1%} {big:28.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
