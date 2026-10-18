"""
Checkpoint: live gaze predictions printed in the console (no window yet).

    python -m eyetracking.testing.live_predict                  (all calibrations)
    python -m eyetracking.testing.live_predict data/<file>.json (just that one)

Trains on the saved calibrations (newest weighted most), then for every camera frame:

    camera -> MediaPipe -> 10 raw features -> model (the head turn taken out) -> 1-euro filter -> (x, y)

x and y are fractions of the screen (0 = left/top, 1 = right/bottom), the
same as the calibration dots. Look at different parts of the screen and
check the "where" column follows you. Ctrl+C stops it.
"""
import sys
import time
from pathlib import Path

from eyetracking.core.calibration import training_set
from eyetracking.core.gaze_model import GazeModel
from eyetracking.core.smoothing import GazeSmoother
from eyetracking.services.storage import load_all_calibrations, load_calibration
from eyetracking.services.tracker import EyeTracker

PRINT_EVERY_S = 0.2   # 5 lines a second is readable; every frame (~20-30) is not


def region(x: float, y: float) -> str:
    """Screen split into a 3x3 grid, in words: easy to check by eye."""
    row = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else "middle"
    col = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "centre"
    return f"{row}-{col}"


def main() -> None:
    # one file if given, otherwise every saved calibration, the newest weighted most
    recordings = [load_calibration(Path(sys.argv[1]))] if len(sys.argv) > 1 else load_all_calibrations()
    samples, weights, K = training_set(recordings)
    if not samples:
        print("No usable calibration yet: run calibrate first")
        return

    # the model takes the head turn out of every frame itself (K: the newest calibration's
    # own head step), so live frames go in raw, exactly as the training frames did
    model = GazeModel(K=K).fit(samples, weights)
    smoother = GazeSmoother()               # the 1-euro filter on the live point, as in the app
    print(f"Trained on {len(samples)} samples. Look around the screen; Ctrl+C to stop.\n")

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
                    point = smoother.feed(model.predict(features))

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
