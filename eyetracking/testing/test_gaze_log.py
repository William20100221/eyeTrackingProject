"""
Self-check for the gaze window (no webcam):  python -m eyetracking.testing.test_gaze_log

Enter records a sample (where the dot is right now) against the lit ring, then lights the
next one; each sample gets an index written BEFORE its coordinates, on screen and in the
csv saved on Esc. Space pauses. Nothing is recorded by itself.
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


points = iter([None, (0.1, 0.2), None, (0.3, 0.4), (0.5, 0.6)])
window = GazeWindow(get_point=lambda: next(points))
assert len(window._targets) == 16 and window._lit == 0                  # 4x4 rings, the first one lit
window._tick()
key(window, Qt.Key.Key_Return)                            # no face yet: nothing recorded
assert window.log == [] and window._lit == 0
window._tick()
key(window, Qt.Key.Key_Return)                            # sample 0 at (0.1, 0.2), against ring (1, 1)
window._tick()                                            # a blink: the point is held...
key(window, Qt.Key.Key_Enter)                             # ...so sample 1 is (0.1, 0.2) too, against ring (2, 1)
window._tick()                                            # no Enter: nothing recorded
window._tick()
key(window, Qt.Key.Key_Return)                            # sample 2 at (0.5, 0.6)
assert window.sample_lines() == ["#0000  x 0.100  y 0.200  off 0.158", "#0001  x 0.100  y 0.200  off 0.292",
                                 "#0002  x 0.500  y 0.600  off 0.570"], window.sample_lines()   # off = from the lit ring
assert window._lit == 3 and [r[3:] for r in window.log] == [(0.05, 0.05), (0.35, 0.05), (0.65, 0.05)]   # one by one

for _ in range(2):                                        # Space pauses, Space carries on
    key(window, Qt.Key.Key_Space)
    assert window.paused and not window._timer.isActive()
    key(window, Qt.Key.Key_Space)
    assert not window.paused and window._timer.isActive()

with tempfile.TemporaryDirectory() as folder:
    rows = list(csv.reader(open(window.save_log(Path(folder)))))
assert rows[0] == ["index", "x", "y", "target_x", "target_y"]
assert rows[1] == ["0", "0.1000", "0.2000", "0.0500", "0.0500"] and len(rows) == 4, rows
print("gaze samples ok")
