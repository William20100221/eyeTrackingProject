"""
Self-check for the gaze-check samples (no webcam):  python -m eyetracking.testing.test_gaze_log

A sample is recorded only on a deliberate blink (or a click): the point held
through the blink, i.e. where you looked before the lids closed. Each sample
gets an index written BEFORE its coordinates, on screen and in the csv saved on
Esc. Space pauses.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from eyetracking.ui.gaze_window import GazeWindow

qt = QApplication([])
points = iter([(0.1, 0.2), None, (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)])
blinks = iter([False, True, False, True, False])         # tick 2: eyes shut (point None) -> the held point
window = GazeWindow(get_point=lambda: next(points), get_blink=lambda: next(blinks))
for _ in range(4):
    window._tick()
window._clicked = True                                    # mouse mode: a click plays the blink
window._tick()
assert window.sample_lines() == ["#0000  x 0.100  y 0.200", "#0001  x 0.500  y 0.600",
                                 "#0002  x 0.700  y 0.800"], window.sample_lines()

for _ in range(2):                                        # Space pauses, Space carries on
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
    assert window.paused and not window._timer.isActive()
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier))
    assert not window.paused and window._timer.isActive()

with tempfile.TemporaryDirectory() as folder:
    rows = list(csv.reader(open(window.save_log(Path(folder)))))
assert rows[0] == ["index", "x", "y"] and rows[1] == ["0", "0.1000", "0.2000"] and len(rows) == 4, rows
print("gaze samples ok")
