"""
Self-check for issue #10 (no webcam):  python -m eyetracking.testing.test_check_carry

After Esc in the gaze check, the app's gaze point carries on from the check's
last point (no jump back to an old one), and the samples recorded with Enter
are saved to the csv.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.ui.control_window as cw

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
import eyetracking.app as appmod
import eyetracking.ui.gaze_window as gw

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")
real_save = gw.GazeWindow.save_log
gw.GazeWindow.save_log = lambda self, folder=tmp: real_save(self, tmp)     # the csv goes to a temp folder
points = iter([(0.2, 0.2), (0.6, 0.4), (0.62, 0.41)])
appmod.fake_point_from_mouse = lambda: next(points)                         # where the "eyes" look

app = appmod.App(mouse=True)
app.timer.stop()
app.smoothed = (0.9, 0.9)                                                   # an old point from before the check
app.check()
win = app.fullscreen
for _ in range(3):
    win._tick()
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
for _ in range(5):
    qt.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

assert app.fullscreen is None and app.smoothed == (0.62, 0.41), app.smoothed  # carries on from the check's point
rows = list(csv.reader(open(next(tmp.glob("gaze_check_*.csv")))))
assert rows == [["index", "x", "y", "target_x", "target_y"], ["0", "0.6200", "0.4100", "0.0500", "0.0500"]], rows   # the Enter sample, with the lit ring
assert "Gaze samples saved" in app.panel.said[-1]
print("check carry-on ok")
