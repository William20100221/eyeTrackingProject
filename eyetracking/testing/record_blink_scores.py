"""
Record MediaPipe's blink scores in known eye states, to choose the blink
threshold from data (GitHub issue #5).

    python -m eyetracking.testing.record_blink_scores

The console asks for six states, 6 s each, with a 3 s countdown before each
(not recorded): eyes open, looking down, half closed, closed, wink left, wink
right. Sit as you normally do in front of the screen. Every frame's left and
right blink score goes to data/blink_scores_<time>.csv, and a summary per state
is printed: what a threshold has to separate (open / looking down / half closed
must stay below it, closed above it; the winks show whether one eye alone
crosses it).

Uses the camera chosen in the app (Settings > Camera); asks in the console if
none is saved.
"""
from __future__ import annotations

import csv
import statistics
import time
from datetime import datetime

from PySide6.QtCore import QSettings

from eyetracking.services.camera import camera_names
from eyetracking.services.tracker import EyeTracker
from path import DATA_DIR

STATES = [("open", "Eyes open, look at the middle of the screen"),
          ("down", "Eyes open, look at the BOTTOM edge of the screen (like reading the last line)"),
          ("half", "Eyes HALF closed (squint), look at the screen"),
          ("closed", "Eyes CLOSED"),
          ("wink_left", "Close only your LEFT eye"),
          ("wink_right", "Close only your RIGHT eye")]
SECONDS = 6
COUNTDOWN = 3


def camera_index() -> int | None:
    name = QSettings("StudyAssistant", "StudyAssistant").value("camera", "")
    names = camera_names()
    return names.index(name) if name in names else None


def main() -> None:
    tracker = EyeTracker(camera_index())
    rows = []
    try:
        for state, prompt in STATES:
            print(f"\nNext: {prompt}")
            for n in range(COUNTDOWN, 0, -1):
                print(f"  {n}...", flush=True)
                end = time.monotonic() + 1
                while time.monotonic() < end:          # keep reading, so the camera's buffer stays fresh
                    tracker.read()
            print("  recording", flush=True)
            end = time.monotonic() + SECONDS
            while time.monotonic() < end:
                r = tracker.read()
                if r is not None and r.face_found:
                    rows.append((state, r.timestamp_ms, r.blink_score_left, r.blink_score_right))
    finally:
        tracker.close()

    path = DATA_DIR / f"blink_scores_{datetime.now():%Y-%m-%d_%H%M%S}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "t_ms", "left", "right"])
        w.writerows(rows)
    print(f"\nSaved {len(rows)} frames to {path}\n")
    print(f"{'state':<11}{'frames':>7}   left  p5 / median / p95      right p5 / median / p95")
    for state, _ in STATES:
        got = [r for r in rows if r[0] == state]
        if len(got) < 2:
            print(f"{state:<11}{len(got):>7}   (too few frames: was the face in view?)")
            continue
        cols = []
        for i in (2, 3):
            q = statistics.quantiles([r[i] for r in got], n=20)    # 5 % steps
            cols.append(f"{q[0]:.2f} / {statistics.median(r[i] for r in got):.2f} / {q[-1]:.2f}")
        print(f"{state:<11}{len(got):>7}   {cols[0]:<26}{cols[1]}")


if __name__ == "__main__":
    main()
