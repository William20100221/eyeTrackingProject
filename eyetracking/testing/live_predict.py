"""
Checkpoint: live gaze predictions printed in the console (no window yet).

    python -m eyetracking.testing.live_predict                  (newest calibration)
    python -m eyetracking.testing.live_predict data/<file>.json

Trains on a saved calibration, then for every camera frame:

    camera -> MediaPipe -> 10 features -> model -> (x, y) on screen

x and y are fractions of the screen (0 = left/top, 1 = right/bottom), the
same as the calibration dots. Look at different parts of the screen and
check the "where" column follows you. Ctrl+C stops it.
"""
import sys
import time
from pathlib import Path

from eyetracking.core.gaze_model import GazeModel
from eyetracking.services.storage import latest_calibration, load_calibration
from eyetracking.services.tracker import EyeTracker

PRINT_EVERY_S = 0.2   # 5 lines a second is readable; every frame (~20-30) is not


def region(x: float, y: float) -> str:
    """Screen split into a 3x3 grid, in words: easy to check by eye."""
    row = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else "middle"
    col = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "centre"
    return f"{row}-{col}"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_calibration()
    if path is None:
        print("No calibration saved yet: run the calibration window first")
        return

    # train on the saved features. Live features come from the SAME function
    # (EyeTracker.read_features), so the model sees the same kind of numbers
    model = GazeModel()
    model.fit(load_calibration(path))
    print(f"Trained on {path.name}. Look around the screen; Ctrl+C to stop.\n")

    with EyeTracker() as tracker:           # asks which camera, in the console
        frames = skipped = 0
        start = last_print = time.perf_counter()
        try:
            while True:
                features = tracker.read_features()
                frames += 1
                if features is None:        # no face, or blinking
                    skipped += 1
                    point = None
                else:
                    point = model.predict(features)

                now = time.perf_counter()
                if now - last_print >= PRINT_EVERY_S:
                    fps = frames / (now - start)
                    if point is None:
                        shown = "no face / blink"
                    else:
                        x, y = point
                        shown = f"x={x:5.2f} y={y:5.2f}  {region(x, y)}"
                    print(f"{shown:34s} {fps:4.1f} fps  ({skipped}/{frames} frames skipped)")
                    last_print = now
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
