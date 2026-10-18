"""
The app's own train(): the new-sitting error per calibration type, on the real files.

    python -m eyetracking.testing.bench_train <data folder>

For every person and every pair of their calibrations: train(method of the TRAINING file,
[that file]) exactly as the app does, predict the other file's dots (TrainedModel.predict,
the head correction included), mean distance to the dot as a fraction of the screen.
"""
from __future__ import annotations

import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from eyetracking.core.line_model import ScreenGeometry
from eyetracking.core.methods import METHOD_BY_ID, train
from eyetracking.services.storage import load_calibration, method_of, person_of

warnings.filterwarnings("ignore")


def main() -> int:
    data = Path(sys.argv[1])
    files = []
    for path in sorted(data.rglob("*.json")):
        if "trash" in path.parts or "disabled" in path.parts or method_of(path) not in METHOD_BY_ID:
            continue
        try:
            files.append((person_of(path) or "data", method_of(path), load_calibration(path)))
        except Exception:
            pass
    by_type = defaultdict(list)
    for person in sorted({f[0] for f in files}):
        mine = [f for f in files if f[0] == person]
        for i, (_, mtype, train_rec) in enumerate(mine):
            model, n = train(METHOD_BY_ID[mtype], [train_rec], ScreenGeometry(34.4, 19.4))
            if model is None:
                print(f"  {person}/{mtype}: unusable")
                continue
            for j, (_, _, test_rec) in enumerate(mine):
                if j == i:
                    continue
                errs = [np.hypot(*(np.array(model.predict(s.features)) - (s.target_x, s.target_y)))
                        for s in test_rec.dots if model.predict(s.features) is not None]
                by_type[mtype].append(float(np.mean(errs)))
    print(f"{'trained with':<16} {'new-sitting error':>18}   pairs")
    for mtype, errs in by_type.items():
        print(f"{mtype:<16} {np.mean(errs):18.1%}   {len(errs)}")
    print(f"{'all':<16} {np.mean([e for errs in by_type.values() for e in errs]):18.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
