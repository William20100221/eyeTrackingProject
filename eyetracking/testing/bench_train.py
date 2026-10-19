"""
The app's own train(): the new-sitting error per calibration type, on the real files.

    python -m eyetracking.testing.bench_train <data folder>

For every person and every pair of their grid calibrations: train(the grid's model of the TRAINING file,
[that file]) exactly as the app does, predict the other file's dots (TrainedModel.predict,
the head correction included), mean distance to the dot as a fraction of the screen.
new_sitting_errors() is the same as a function (report_figures uses it), with any other way to fit
(`fit`) on the same pairs.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetracking.core.line_model import ScreenGeometry
from eyetracking.core.methods import MODEL_BY_ID, train
from eyetracking.services import storage
from eyetracking.services.storage import load_calibration, method_of, person_of

warnings.filterwarnings("ignore")


def grid_files(data: Path, person: str | None = None) -> list[tuple[str, str, Path, object]]:
    """(person, grid type, path, recording) of every active grid calibration under `data` (person: only theirs)."""
    storage.BASE_DIR = Path(data).resolve()          # person_of() reads the people from here
    files = []
    for path in sorted(Path(data).resolve().rglob("*.json")):
        if "trash" in path.parts or "disabled" in path.parts or method_of(path) not in MODEL_BY_ID:   # the grids
            continue
        who = person_of(path)
        if person is not None and who != person:
            continue
        try:
            files.append((who or "data", method_of(path), path, load_calibration(path)))
        except Exception:
            pass
    return files


def app_fit(mtype: str, rec):
    """The app's own training on one file: a predictor (features -> point or None), or None."""
    model, _ = train(MODEL_BY_ID[mtype], [rec], [], ScreenGeometry(34.4, 19.4))
    return model.predict if model is not None else None


def new_sitting_errors(files, fit=app_fit) -> tuple[dict[str, list[float]], list[Path]]:
    """Per grid type, the error of every (training file, test file) pair of one person: fit on the first,
    the mean distance to the dot (screen fractions) on the other's dots. And the files that trained one."""
    by_type, used = defaultdict(list), []
    for person in sorted({f[0] for f in files}):
        mine = [f for f in files if f[0] == person]
        for i, (_, mtype, path, train_rec) in enumerate(mine):
            predict = fit(mtype, train_rec)
            if predict is None:
                print(f"  {person}/{path.name}: unusable")
                continue
            used.append(path)
            for j, (_, _, _, test_rec) in enumerate(mine):
                if j == i:
                    continue
                said = [(p, s) for s in test_rec.dots if (p := predict(s.features)) is not None]
                if said:
                    by_type[mtype].append(float(np.mean([np.hypot(p[0] - s.target_x, p[1] - s.target_y) for p, s in said])))
    return dict(by_type), used


def main() -> int:
    by_type, _ = new_sitting_errors(grid_files(Path(sys.argv[1])))
    print(f"{'trained with':<16} {'new-sitting error':>18}   pairs")
    for mtype, errs in by_type.items():
        print(f"{mtype:<16} {np.mean(errs):18.1%}   {len(errs)}")
    print(f"{'all':<16} {np.mean([e for errs in by_type.values() for e in errs]):18.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
