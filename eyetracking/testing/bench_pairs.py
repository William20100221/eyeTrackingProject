"""
The paired models (a special + a grid) against the grid alone: the new-sitting error on the real files.

    python -m eyetracking.testing.bench_pairs <data folder> [--old-specials]

For every person, each usable grid calibration trains the models of its grid with the app's own train()
(core/methods.py): the grid alone, and the grid + each of that person's usable Sweep / Points / Line
model calibrations, one special at a time (storage.calibration_problem: its step 1 really is a gliding
dot for Sweep, still dots for Points / Line model; old types' folders are not usable, as in the app).
Each model predicts the dots of every OTHER grid file of that person (another sitting), and the own 4x4
dots of old special files that recorded their own (no dots_from, no same_as); a special's own recording
is never a test of a model it trained. Error: the mean distance to the dot as a share of the screen
(TrainedModel.predict, the head correction included). A pair (training file, test file) with several
specials of a kind counts once, as their mean. Printed:
  1. the error per model kind and grid, and paired minus alone on the same pairs
  2. paired minus alone per person (the same training and test files: a fair paired comparison)
  3. the pursuit lag pursuit_samples picks on each real Sweep step 1, and Sweep + grid with the lag forced to 0
  4. EXTRA_WEIGHT 0.25 / 0.5 / 1.0 for + Sweep and + Points
--old-specials: the old types' folders (random-sweep/ ...) also give specials, when their step 1 is of
the right kind and they are not a same_as copy (a check of how much more data changes the answer).
Read only: nothing is written.
"""
from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetracking.core import methods
from eyetracking.core.line_model import ScreenGeometry
from eyetracking.core.head_correction import ANGLES, EYE6
from eyetracking.core.methods import GRIDS, LEGACY, METHOD_BY_ID, MODEL_BY_ID, model_for, pursuit_samples, train
from eyetracking.core.quality import MIN_HEAD_FRAMES, check_special, step_one_problem
from eyetracking.services import storage

warnings.filterwarnings("ignore")
SCREEN = ScreenGeometry(34.4, 19.4)          # as bench_train
NAMES = {"sweep": "+ Sweep", "points": "+ Points", "line-model": "+ Line model"}


def collect(data: Path, old: bool) -> dict:
    """person -> (grid files [(path, grid id, rec)], usable specials {special id: [(path, rec)]},
    extra test sittings [(path, rec)]: old special files' own dots)."""
    storage.BASE_DIR = data                  # person_of() names the folder under it (a module variable, nothing written)
    people = defaultdict(lambda: ([], defaultdict(list), []))
    for path in sorted(data.rglob("*.json")):
        kind = storage.method_of(path)
        if {"trash", "disabled"} & set(path.parts) or (kind not in METHOD_BY_ID and kind not in LEGACY):
            continue                         # removed files, the other branches' folders
        try:
            rec, _, dots_from, same_as = storage._load(path)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        grids, specials, extra = people[storage.person_of(path) or "data"]
        method = METHOD_BY_ID.get(kind)
        if method and storage.calibration_problem(path) is None:
            if method.kind == "grid":
                grids.append((path, kind, rec))
            else:
                specials[kind].append((path, rec))
        elif old and not method and not same_as:
            sid = kind.removeprefix("random-")
            if len(rec.head) >= MIN_HEAD_FRAMES and step_one_problem(rec.sweep, METHOD_BY_ID[sid].kind) is None:
                specials[sid].append((path, rec))
        if (not method or method.kind != "grid") and rec.dots and not dots_from and not same_as:
            extra.append((path, rec))
    return people


def dots_error(trained, rec) -> float | None:
    """methods.dots_error with every frame in one predict call: the same numbers (TrainedModel.predict's
    head correction, then the model; a line model's None is its NaN), ~200x quicker for a ridge model."""
    X = np.array([s.features for s in rec.dots], float)
    if trained.K is not None:
        X = X[:, EYE6] - (X[:, ANGLES] - trained.ref) @ trained.K
    d = np.hypot(*(trained.inner.predict_many(X) - [(s.target_x, s.target_y) for s in rec.dots]).T)
    return float(np.nanmean(d)) if not np.isnan(d).all() else None


def tests(path: Path, grids: list, extra: list) -> list:
    return [(q, r) for q, _, r in grids if q != path] + extra


def alone_errors(people: dict) -> dict:
    """(training file, test file) -> (person, grid id, error, error, key) for the grid alone."""
    out = {}
    for person, (grids, _, extra) in people.items():
        for path, gid, rec in grids:
            trained, _ = train(MODEL_BY_ID[gid], [rec], [], SCREEN)
            for q, r in tests(path, grids, extra):
                if trained and (e := dots_error(trained, r)) is not None:
                    out[path, q] = (person, gid, e, e, (path, q))
    return out


def paired(people: dict, alone: dict, kind: str, weight: float = 0.5, max_lag: int = methods.MAX_LAG) -> list:
    """[(person, grid id, error, the grid alone's error, key)], one per (training file, test file):
    the grid + each usable special of `kind` in turn, the mean over them."""
    saved = methods.EXTRA_WEIGHT, methods.MAX_LAG
    methods.EXTRA_WEIGHT, methods.MAX_LAG = weight, max_lag
    rows = []
    try:
        for person, (grids, specials, extra) in people.items():
            for path, gid, rec in grids:
                errs = defaultdict(list)
                for spath, srec in specials[kind]:
                    trained, _ = train(model_for(gid, kind), [rec], [srec], SCREEN)
                    for q, r in tests(path, grids, extra):
                        if trained and q.stem[-17:] != spath.stem[-17:] and (path, q) in alone \
                                and (e := dots_error(trained, r)) is not None:   # never the special's own recording
                            errs[q].append(e)
                rows += [(person, gid, float(np.mean(e)), alone[path, q][2], (path, q)) for q, e in errs.items()]
    finally:
        methods.EXTRA_WEIGHT, methods.MAX_LAG = saved
    return rows


def cell(values: list[float], diff: bool) -> str:
    if not values:
        return "-"
    return (f"{np.mean(values) * 100:+.1f} pp" if diff else f"{np.mean(values):.1%}") + f" ({len(values)})"


def grid_table(title: str, runs: dict, diff: bool = False) -> None:
    """Rows: the runs; columns: the 6 grids and all. Mean error (or minus the baseline), (pairs)."""
    print(f"\n{title}")
    print(f"{'':<20}" + "".join(f"{g.id:>16}" for g in GRIDS) + f"{'all':>16}")
    for label, rows in runs.items():
        value = lambda rs: [r[2] - r[3] if diff else r[2] for r in rs]
        cells = [value([r for r in rows if r[1] == g.id]) for g in GRIDS] + [value(rows)]
        print(f"{label:<20}" + "".join(f"{cell(c, diff):>16}" for c in cells))


def person_table(title: str, runs: dict) -> None:
    """Rows: the people; columns: the runs. Mean of (error - baseline) in percentage points, and in how
    many pairs the run was better (lower) than its baseline."""
    print(f"\n{title}")
    people = sorted({r[0] for rows in runs.values() for r in rows})
    print(f"{'':<16}" + "".join(f"{label:>26}" for label in runs))
    means = {label: [] for label in runs}
    for person in people:
        line = f"{person:<16}"
        for label, rows in runs.items():
            d = [r[2] - r[3] for r in rows if r[0] == person]
            if d:
                means[label].append(np.mean(d))
            line += f"{'-' if not d else f'{np.mean(d) * 100:+.1f} pp  {sum(x < 0 for x in d)}/{len(d)} better':>26}"
        print(line)
    pooled = {label: [r[2] - r[3] for r in rows] for label, rows in runs.items()}
    print(f"{'all pairs':<16}" + "".join(
        f"{'-' if not d else f'{np.mean(d) * 100:+.1f} pp  {sum(x < 0 for x in d)}/{len(d)} better':>26}"
        for d in pooled.values()))
    print(f"{'mean of people':<16}" + "".join(
        f"{'-' if not m else f'{np.mean(m) * 100:+.1f} pp  {sum(x < 0 for x in m)}/{len(m)} people':>26}"
        for m in means.values()))


def lag_table(people: dict) -> None:
    """The lag (frames) pursuit_samples picks per pass on each Sweep step 1 that trains (cleaned as train() does)."""
    print(f"\n3a. Pursuit lag picked by pursuit_samples (frames; tried 0..{methods.MAX_LAG}; ~33 ms a frame at 30 fps)")
    print(f"{'person':<12}{'file':<44}{'across: frames, lag':>22}{'down: frames, lag':>20}")
    for person, (_, specials, _) in people.items():
        for path, rec in specials["sweep"]:
            kept = check_special(rec, "sweep").kept.sweep
            parts = [[s for s in kept if (s.target_y == 0.5) == across] for across in (True, False)]
            lags = [len(p) - len(pursuit_samples(p)) for p in parts]     # each pass alone: its output is len - lag
            print(f"{person:<12}{path.parent.name + '/' + path.name:<44}"
                  + "".join(f"{f'{len(p)}, {lag}' + (' (max)' if lag == methods.MAX_LAG else ''):>21}"
                            for p, lag in zip(parts, lags)))


def main() -> int:
    start = time.perf_counter()
    data = Path(sys.argv[1]).resolve()
    people = collect(data, "--old-specials" in sys.argv)
    print(f"data: {data}{'  (old types folders give specials too)' if '--old-specials' in sys.argv else ''}")
    for person, (grids, specials, extra) in people.items():
        if grids:
            print(f"  {person}: {len(grids)} grid files, {len(extra)} old files' own dots; specials: "
                  + ", ".join(f"{NAMES[k].strip('+ ')} {len(v)}" for k, v in specials.items() if v))
    people = {p: v for p, v in people.items() if v[0]}
    alone = alone_errors(people)
    main_runs = {"grid alone": list(alone.values())} | {NAMES[k]: paired(people, alone, k) for k in NAMES}

    grid_table("1a. New-sitting error: mean distance to the dot, share of the screen (pairs: training file, test file)",
               main_runs)
    grid_table("1b. Paired minus the grid alone on the same pairs (percentage points; minus = the special helps)",
               {k: v for k, v in main_runs.items() if k != "grid alone"}, diff=True)
    person_table("2. Paired minus the grid alone, per person (pp; better = lower error than the grid alone)",
                 {k: v for k, v in main_runs.items() if k != "grid alone"})

    lag_table(people)
    chosen = {r[4]: r[2] for r in main_runs["+ Sweep"]}
    lag0 = [(*r[:3], chosen[r[4]], r[4]) for r in paired(people, alone, "sweep", max_lag=0)]
    grid_table("3b. Sweep + grid error with the lag forced to 0 (error, pairs)", {"+ Sweep, lag 0": lag0,
               "+ Sweep, lag chosen": main_runs["+ Sweep"]})
    person_table("3c. Lag 0 minus lag as chosen (pp; better = lag 0 is lower)", {"+ Sweep, lag 0": lag0})

    weights = {}
    for kind in ("sweep", "points"):
        for w in (0.25, 0.5, 1.0):
            weights[f"{NAMES[kind]}, w {w}"] = main_runs[NAMES[kind]] if w == 0.5 else paired(people, alone, kind, w)
    grid_table("4a. EXTRA_WEIGHT (the step-1 frames' weight): error (pairs)", weights)
    person_table("4b. EXTRA_WEIGHT: minus the grid alone, per person (pp)", weights)
    print(f"\n{time.perf_counter() - start:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
