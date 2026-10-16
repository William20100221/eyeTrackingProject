"""
Self-check for the gaze-check log (no webcam):  python -m eyetracking.testing.test_gaze_log

Each sample gets an index that is written BEFORE its coordinates, on screen
and in the csv saved when you press Esc.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from eyetracking.ui.gaze_window import GazeWindow

qt = QApplication([])
points = iter([(0.1, 0.2), None, (0.3, 0.4)])           # None = blink: the last point is kept, flagged stale
window = GazeWindow(get_point=lambda: next(points))
for _ in range(3):
    window._tick()

assert window.sample_lines() == ["#0000  x 0.100  y 0.200",
                                 "#0001  x 0.100  y 0.200  (stale)",
                                 "#0002  x 0.300  y 0.400"], window.sample_lines()

with tempfile.TemporaryDirectory() as folder:
    rows = list(csv.reader(open(window.save_log(Path(folder)))))
assert rows[0] == ["index", "x", "y", "stale"] and rows[1] == ["0", "0.1000", "0.2000", "0"] and rows[2][3] == "1", rows
print("gaze log ok")
