"""
Train the gaze model on the saved calibrations and print how it did.

    python -m eyetracking.testing.train_model

For every calibration: does it pass the quality check (core/quality.py), and if not, why.
Then the model (core/points_model.py, Points + Random 4x4) and its mean error on the newest
calibration's own dots, in pixels. data/ is ignored by git, so every branch is measured on the same data.
"""
import numpy as np
from PySide6.QtGui import QGuiApplication

from eyetracking.core.points_model import kept_recordings, train
from eyetracking.services.storage import list_calibrations, load_all_calibrations


def main() -> None:
    recordings = load_all_calibrations()
    if not recordings:
        print("No calibration with the current features yet: run the calibration window first")
        return
    kept, skipped = kept_recordings(recordings)
    print(f"{len(list_calibrations())} calibration files, {len(recordings)} readable, {len(kept)} pass the check")
    for why in skipped:
        print(f"  not used: {why}")
    model, n = train(recordings)
    if model is None:
        return
    # the dots were placed using Qt's screen size, so measure with the same size
    size = QGuiApplication([]).primaryScreen().size()
    w, h = size.width(), size.height()
    err = [np.hypot((p[0] - s.target_x) * w, (p[1] - s.target_y) * h)
           for s in kept[0].dots if (p := model.predict(s.features)) is not None]
    print(f"Trained on {n} samples, head turns corrected with {model.k_source}")
    print(f"Mean error on the newest calibration's own dots: {np.mean(err):.0f} px (screen {w} x {h})")


if __name__ == "__main__":
    main()
