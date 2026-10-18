"""
Train on the saved calibrations and print how accurate the model is.

    python -m eyetracking.testing.train_model

Compares two models, both scored on the NEWEST session's dots, each dot
held out (leave-one-dot-out):
    newest only    trained on the newest calibration
    all, weighted  trained on every calibration, the newest weighted most
data/ is ignored by git, so every branch is measured on the same data.
"""
from PySide6.QtGui import QGuiApplication

from eyetracking.core.calibration import training_set
from eyetracking.core.gaze_model import GazeModel, leave_one_dot_out, mean_error_px
from eyetracking.services.storage import load_all_calibrations


def main() -> None:
    # raw dot samples; K = the newest calibration's own head-rotation matrix (the model takes the head turn out)
    samples, weights, K = training_set(load_all_calibrations())
    if not samples:
        print("No usable calibration yet: run calibrate first")
        return
    newest = samples[:weights.count(1.0)]     # newest first, and only the newest has weight 1.0
    print(f"{len(samples)} samples from all calibrations; "
          f"newest: {len(newest)} samples, {len(newest[0].features)} raw features each")

    # the dots were placed using Qt's screen size, so measure with the same size
    app = QGuiApplication([])
    size = app.primaryScreen().size()
    w, h = size.width(), size.height()
    print(f"screen: {w} x {h}")

    model = GazeModel(K=K).fit(samples, weights)
    print(f"\nTested on the SAME dots it trained on: {mean_error_px(model, newest, w, h):.0f} px")

    own = lambda: GazeModel(K=K)
    only = leave_one_dot_out(newest, w, h, make_model=own)
    both = leave_one_dot_out(samples, w, h, weights, test=newest, make_model=own)
    print("Each dot tested by a model that never saw it:   newest only | all, weighted")
    for dot in only:
        print(f"  dot ({dot[0]:.2f}, {dot[1]:.2f}): {only[dot]:5.0f} px | {both[dot]:5.0f} px")
    print(f"  average:            {sum(only.values()) / len(only):5.0f} px | "
          f"{sum(both.values()) / len(both):5.0f} px")


if __name__ == "__main__":
    main()
