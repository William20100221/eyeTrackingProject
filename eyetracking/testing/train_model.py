"""
Train on a saved calibration and print how accurate the model is.

    python -m eyetracking.testing.train_model                 (newest file)
    python -m eyetracking.testing.train_model data/<file>.json

Give the file when comparing versions (branches), so every version is
measured on the SAME data.
"""
import sys
from pathlib import Path

from PySide6.QtGui import QGuiApplication

from eyetracking.core.gaze_model import GazeModel, leave_one_dot_out, mean_error_px
from eyetracking.services.storage import latest_calibration, load_calibration


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_calibration()
    if path is None:
        print("No calibration saved yet: run the calibration window first")
        return
    samples = load_calibration(path)
    print(f"{path.name}: {len(samples)} samples, {len(samples[0].features)} features each")

    # the dots were placed using Qt's screen size, so measure with the same size
    app = QGuiApplication([])
    size = app.primaryScreen().size()
    w, h = size.width(), size.height()
    print(f"screen: {w} x {h}")

    model = GazeModel()
    model.fit(samples)
    print(f"\nTested on the SAME dots it trained on: {mean_error_px(model, samples, w, h):.0f} px")

    errors = leave_one_dot_out(samples, w, h)
    print("Each dot tested by a model that never saw it:")
    for (x, y), err in errors.items():
        print(f"  dot ({x:.2f}, {y:.2f}): {err:.0f} px")
    print(f"  average: {sum(errors.values()) / len(errors):.0f} px")


if __name__ == "__main__":
    main()
