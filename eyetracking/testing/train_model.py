"""
Train on the saved calibrations and print how accurate the models are.

    python -m eyetracking.testing.train_model

Both models train on every calibration (the newest weighted most) and are
scored on the NEWEST session's dots, each dot held out (leave-one-dot-out):
    ridge model   a straight line on the 6 eye features, the head turn taken out with
                  the head step's matrix (core/gaze_model.py, core/head_correction.py)
    line model    the 3D line from the eyes, calibrated + blended (core/line_model.py)
Also checks the 3D model itself: the angle between the two eyes' measured
lines vs the vergence formula (core/vergence.py).
data/ is ignored by git, so every branch is measured on the same data.
"""
import math

import numpy as np
from PySide6.QtWidgets import QApplication

from eyetracking.core.calibration import training_set
from eyetracking.core.gaze_model import GazeModel, leave_one_dot_out
from eyetracking.core.line_model import LineModel, eye_lines
from eyetracking.services.storage import load_all_calibrations
from eyetracking.ui.gaze_window import screen_geometry


def main() -> None:
    raw, weights, K = training_set(load_all_calibrations())     # raw dots; K = the newest head step's matrix
    if not raw:
        print("No usable calibration yet: run calibrate first")
        return
    n = weights.count(1.0)                                      # newest first; only it weighs 1.0

    # the dots were placed using Qt's screen size, so measure with the same size
    app = QApplication([])
    size = app.primaryScreen().size()
    w, h = size.width(), size.height()
    screen = screen_geometry()
    print(f"{len(raw)} samples from all calibrations, newest {n} | screen {w} x {h} px, "
          f"{screen.width_cm:.1f} x {screen.height_cm:.1f} cm")

    # the 3D model's own check: do the two eyes' lines meet at the vergence angle?
    measured, formula = [], []
    for s in raw[:n]:
        _, lines, _, _, verg = eye_lines(s.features)
        measured.append(math.degrees(math.acos(np.clip(lines[0] @ lines[1], -1, 1))))
        formula.append(math.degrees(verg))
    print(f"angle between the two eye lines: measured {np.median(measured):.1f} deg (median), "
          f"vergence formula {np.mean(formula):.1f} deg")

    ridge = leave_one_dot_out(raw, w, h, weights, test=raw[:n], make_model=lambda: GazeModel(K=K))
    line = leave_one_dot_out(raw, w, h, weights, test=raw[:n], make_model=lambda: LineModel(screen))
    print("\nEach dot tested by a model that never saw it:   ridge model | line model")
    for dot in ridge:
        print(f"  dot ({dot[0]:.2f}, {dot[1]:.2f}): {ridge[dot]:5.0f} px | {line[dot]:5.0f} px")
    print(f"  average:            {sum(ridge.values()) / len(ridge):5.0f} px | "
          f"{sum(line.values()) / len(line):5.0f} px")

    full = LineModel(screen).fit(raw, weights)
    print(f"\nline model on all data: blend w = {full.w:.1f} (1 = plain 3D line, 0 = fully calibrated), "
          f"gain {full.gain.round(2)}, offset {np.degrees(full.offset).round(1)} deg")


if __name__ == "__main__":
    main()
