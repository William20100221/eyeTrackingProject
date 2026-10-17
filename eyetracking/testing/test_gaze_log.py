"""
Self-check for the gaze-check samples (no webcam):  python -m eyetracking.testing.test_gaze_log

Enter records a sample (where the dot is right now); each one gets an index
written BEFORE its coordinates, on screen and in the csv saved on Esc. Space
pauses. Nothing is recorded by itself.
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


def key(window, k):
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))


points = iter([(0.1, 0.2), None, (0.3, 0.4), (0.5, 0.6)])
window = GazeWindow(get_point=lambda: next(points))
window._tick()
key(window, Qt.Key.Key_Return)                            # sample 0 at (0.1, 0.2)
window._tick()                                            # a blink: the point is held...
key(window, Qt.Key.Key_Enter)                             # ...so sample 1 is (0.1, 0.2) too
window._tick()                                            # no Enter: nothing recorded
window._tick()
key(window, Qt.Key.Key_Return)                            # sample 2 at (0.5, 0.6)
assert window.sample_lines() == ["#0000  x 0.100  y 0.200", "#0001  x 0.100  y 0.200",
                                 "#0002  x 0.500  y 0.600"], window.sample_lines()

for _ in range(2):                                        # Space pauses, Space carries on
    key(window, Qt.Key.Key_Space)
    assert window.paused and not window._timer.isActive()
    key(window, Qt.Key.Key_Space)
    assert not window.paused and window._timer.isActive()

with tempfile.TemporaryDirectory() as folder:
    rows = list(csv.reader(open(window.save_log(Path(folder)))))
assert rows[0] == ["index", "x", "y"] and rows[1] == ["0", "0.1000", "0.2000"] and len(rows) == 4, rows
print("gaze samples ok")
