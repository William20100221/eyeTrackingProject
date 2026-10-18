"""
Self-check for the gaze check in the app (no webcam):  python -m eyetracking.testing.test_check_carry

The check uses the chosen method's grid for its rings; the samples recorded with
Enter are saved on Esc into that method's gaze-check folder, with the method in
the file name, and the log says so.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.core.calibration import point_grid

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
storage.DATA_DIR = storage.CALIB_DIR = storage.BLINK_DIR = tmp / "data"
storage.GAZE_CHECK_DIR = tmp / "gaze_checks"
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
points = iter([(0.2, 0.2), (0.6, 0.4), (0.62, 0.41)])
appmod.fake_point_from_mouse = lambda: next(points)                         # where the "eyes" look

app = appmod.App(mouse=True)
app.testing.set_method("original-3x3")
app.check()
win = app.fullscreen
assert win._targets == point_grid(3) and win.method_id == "original-3x3"
for _ in range(3):
    win._tick()
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
for _ in range(5):
    qt.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

assert app.fullscreen is None
saved = list((tmp / "gaze_checks" / "original-3x3").glob("gaze-check_original-3x3_*.csv"))
assert len(saved) == 1, saved
rows = list(csv.reader(open(saved[0])))
assert rows == [["index", "x", "y"], ["0", "0.6200", "0.4100"]], rows       # the Enter sample, saved
assert "gaze samples saved as gaze-check_original-3x3_" in app.testing.status.text(), app.testing.status.text()
assert app.testing.gaze_checks.files.item(0).text().endswith("  original-3x3"), app.testing.gaze_checks.files.item(0).text()
print("check carry-on ok")
