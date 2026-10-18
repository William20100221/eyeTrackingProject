"""
Self-check for the gaze check in the app (no webcam):  python -m eyetracking.testing.test_check_carry

The check tests the type picked on its page, on the same 16 rings (4x4) as every type; the
samples recorded with Enter (with the lit ring) are saved on Esc into that type's
folder, with the type in the file name, and the log says so.
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
storage.BASE_DIR = tmp                                          # never the real data/
storage.use_person("")                                          # every folder under it
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
points = iter([(0.2, 0.2), (0.6, 0.4), (0.62, 0.41)])
appmod.fake_point_from_mouse = lambda: next(points)                         # where the "eyes" look

app = appmod.App(mouse=True)
app.testing.set_check(method_id="original-3x3")
app.check()
win = app.fullscreen
assert win._targets == point_grid(4) and win.names == ["original-3x3"]      # the same 16 rings for every type
for _ in range(3):
    win._tick()
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
for _ in range(5):
    qt.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

assert app.fullscreen is None
saved = list((tmp / "original-3x3").glob("gaze-check_original-3x3_*.csv"))
assert len(saved) == 1, saved
rows = list(csv.reader(open(saved[0])))
assert rows == [["index", "x", "y", "target_x", "target_y"], ["0", "0.6200", "0.4100", "0.0500", "0.0500"]], rows
assert "No person's Original 3×3: gaze check saved as gaze-check_original-3x3_" in app.testing.status.text(), app.testing.status.text()
files = app.testing.gaze_checks.files
assert files.item(0).text() == "Original 3×3" and files.item(1).data(0x0100) == str(saved[0])   # under its type
print("check carry-on ok")
