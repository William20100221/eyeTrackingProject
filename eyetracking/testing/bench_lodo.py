"""
Same-posture accuracy, per grid: leave one dot out of a file, predict it from the others.

    python -m eyetracking.testing.bench_lodo <data folder>

Interior dots = filling the gap between dots (what a reader's gaze mostly needs);
edge dots = extrapolating past the outermost dots (always worse). Today's 12-feature
ridge vs the 6 eye-only features.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetracking.testing.bench_models import Columns, EYE6, dot_groups, error, load, ridge_lodo, ridge_loo, xy

warnings.filterwarnings("ignore")


def main() -> int:
    files = load(Path(sys.argv[1]))
    picks = {"today: 12 features, LOO-sample alpha": Columns(list(range(12)), ridge_loo),
             "12 features, LODO alpha": Columns(list(range(12)), ridge_lodo),
             "eye 6, LODO alpha": Columns(EYE6, ridge_lodo)}
    table = defaultdict(lambda: defaultdict(list))        # grid -> model -> [(interior err, edge err)]
    for person, mtype, name, rec, _ in files:
        X, y = xy(rec.dots)
        g = dot_groups(rec.dots)
        grid = {9: "3x3", 16: "4x4", 25: "5x5"}.get(len(set(g)), str(len(set(g))))
        for cname, cand in picks.items():
            preds = np.zeros_like(y)
            for dot in set(g):
                keep = g != dot
                preds[~keep] = cand.fit(X[keep], y[keep], g[keep]).predict(X[~keep])
            edge = (y[:, 0] <= 0.06) | (y[:, 0] >= 0.94) | (y[:, 1] <= 0.06) | (y[:, 1] >= 0.94)
            table[grid][cname].append((error(preds[~edge], y[~edge]) if (~edge).any() else np.nan, error(preds[edge], y[edge])))
    print(f"{'grid':<5} {'model':<40} {'interior':>9} {'edge':>7}   files")
    for grid in sorted(table):
        for cname, rs in table[grid].items():
            print(f"{grid:<5} {cname:<40} {np.nanmean([i for i, _ in rs]):9.1%} {np.mean([e for _, e in rs]):7.1%}   {len(rs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
