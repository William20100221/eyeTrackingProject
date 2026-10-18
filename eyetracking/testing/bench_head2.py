"""
Head compensation, step 2: rotation (K from the head step) + TRANSLATION by geometry.

    python -m eyetracking.testing.bench_head2 <data folder>

Parallax: if the eyes move by (dx, dy) cm (MediaPipe's head position, camera space: x = the
user's left, y = up), a line at the same gaze angle hits the screen (dx, dy) cm away from where
it did, i.e. the fraction across changes by -dx/W and down by -dy/H (ScreenGeometry: the
user's left is the camera's +x; screen y runs down). The regression learns the mapping at the
calibration posture; the term adds the shift of the eyes since then. The flipped sign is a
control: if physics is right, only the right sign helps. Results per TRAINING grid too: a 3x3
calibration can't support the same model as a 5x5 one.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetracking.core.features import HEAD
from eyetracking.testing.bench_head import ANG, EYE6, REGRESSIONS, correct, fit_k, groups_of, load, xy

warnings.filterwarnings("ignore")
POS = [HEAD.start + 3, HEAD.start + 4, HEAD.start + 5]        # hx, hy, hz (cm, MediaPipe)
W_CM, H_CM = 34.4, 19.4                                       # a 15.6" laptop screen (the real one is unknown here)


def error(pred, y):
    return float(np.nanmean(np.hypot(*(pred - y).T)))


def main() -> int:
    files = load(Path(sys.argv[1]))
    people = sorted({f[0] for f in files})
    heads = {p: [f[3].head for f in files if f[0] == p and len(f[3].head) >= 60] for p in people}
    all_K = fit_k([h for p in people for h in heads[p]])
    results = defaultdict(list)          # (correction, regression) -> [(train grid, still?, err)]
    for person in people:
        mine = [f for f in files if f[0] == person]
        for ti, (_, ttype, tname, train) in enumerate(mine):
            Xtr, ytr = xy(train.dots)
            gtr = groups_of(train.dots)
            grid = {9: "3x3", 16: "4x4", 25: "5x5"}.get(len(set(gtr)), "?")
            ref = Xtr[:, ANG].mean(axis=0)
            pos_ref = Xtr[:, POS].mean(axis=0)
            ctr = correct(Xtr, all_K, ref)
            fitted = {}
            for rname, fit in REGRESSIONS.items():
                fitted[("no correction", rname)] = (fit(Xtr[:, EYE6], ytr, gtr), False, 0)
                m = fit(ctr, ytr, gtr)
                fitted[("rotation K (default)", rname)] = (m, True, 0)
                fitted[("rotation K + translation (physics sign)", rname)] = (m, True, -1)
                fitted[("rotation K + translation (flipped sign)", rname)] = (m, True, +1)
            for j, (_, _, name2, test) in enumerate(mine):
                if j == ti:
                    continue
                Xte, yte = xy(test.dots)
                still = np.linalg.norm(Xte[:, POS].mean(0) - pos_ref) < 3
                shift = (Xte[:, POS][:, :2] - pos_ref[:2]) / (W_CM, H_CM)          # per frame, screen fractions
                for key, (m, corrected, sign) in fitted.items():
                    pred = m.predict(correct(Xte, all_K, ref) if corrected else Xte[:, EYE6])
                    pred = pred + sign * shift
                    results[key].append((grid, still, error(pred, yte)))
    print("NEW-SITTING TEST, every pair; then by the TRAINING file's grid")
    print(f"{'correction':<46} {'regression':<13} {'all':>7} {'still':>7} {'moved':>7} {'3x3':>7} {'4x4':>7} {'5x5':>7}")
    rows = []
    for (c, r), rs in results.items():
        pick = lambda cond: np.mean([e for g, s, e in rs if cond(g, s)]) if any(cond(g, s) for g, s, _ in rs) else float("nan")
        a = pick(lambda g, s: True)
        rows.append((a, f"{c:<46} {r:<13} {a:7.1%} {pick(lambda g, s: s):7.1%} {pick(lambda g, s: not s):7.1%} "
                        f"{pick(lambda g, s: g == '3x3'):7.1%} {pick(lambda g, s: g == '4x4'):7.1%} {pick(lambda g, s: g == '5x5'):7.1%}"))
    for _, row in sorted(rows):
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
