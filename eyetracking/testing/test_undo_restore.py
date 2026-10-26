"""
Self-check for Undo and Removed > Restore (no webcam):  python -m eyetracking.testing.test_undo_restore

As in calibration-lab: Undo (the Testing sidebar's button, Ctrl+Z) takes the newest calibration or
gaze check back into its trash; the Undo button says what it would undo and hides when there is
nothing; Data > Calibrations lists removed ones under Removed, and Restore brings them back; several
picked files are disabled, removed or restored at once. Temp folders only.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.core.calibration import point_grid, sweep_points
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.models import CalibrationRecording, CalibrationSample

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
storage.CALIB_DIR, storage.DISABLED_DIR, storage.TRASH_DIR = tmp / "c", tmp / "c" / "disabled", tmp / "c" / "trash"
storage.GAZE_CHECK_DIR = tmp / "gaze_checks"
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)   # confirm removals
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")
app = appmod.App(mouse=True)
app.timer.stop()
app.mouse = False
page = app.control.testing
said = []
app.panel.reply = said.append

rng, screen = np.random.default_rng(3), ScreenGeometry(27.7, 18.9)
look = lambda t, n, sd=1.0: tuple(CalibrationSample(features_for(screen, t, [rng.normal(0, 2 * sd), rng.normal(0, 2 * sd),
                                  rng.normal(0, sd), 0, -5, -65], (2, -1), 0.9, rng), *t) for _ in range(n))
sitting = lambda: CalibrationRecording(sum((look(t, 20) for t in sweep_points()), ()), look((0.5, 0.5), 150, 6),
                                       sum((look(t, 25) for t in point_grid(4)), ()))
names = lambda lst: [lst.item(i).data(0x0100) for i in range(lst.count()) if lst.item(i).data(0x0100)]

# ---- a calibration, then Undo: into the trash, listed under Removed, the model retrained without it
assert page.undo_btn.isHidden()
app._calibrated(QWidget(), sitting())
first = storage.list_calibrations()[0]
assert not page.undo_btn.isHidden() and page.undo_btn.text().endswith("Undo the calibration")
assert "second calibration" in said[-1] and page.status.text() == said[-1]      # the sidebar says it too
app.undo()
assert storage.list_calibrations() == [] and storage.list_removed() == [storage.TRASH_DIR / first.name]
assert page.undo_btn.isHidden() and said[-1].startswith("Undid the calibration: 1 file moved to the trash")
assert names(page.removed_list) == [str(storage.TRASH_DIR / first.name)] and app.model is None

# ---- Removed > Restore brings it back; the detail page of a trash file has no Remove button
page.open_detail(storage.TRASH_DIR / first.name)
assert page.main.currentIndex() == 6 and page.remove_btn.isHidden()
page.restore_files([storage.TRASH_DIR / first.name])
assert storage.list_calibrations() == [first] and storage.list_removed() == [] and page.status.text() == f"Restored {first.name}."
assert page.main.currentIndex() == 3                                            # the detail page's file moved: back to the lists

# ---- several at once: disable two, remove two (Delete), restore two
app._calibrated(QWidget(), sitting())
app._calibrated(QWidget(), sitting())
three = storage.list_calibrations()
page.set_enabled(three[:2], False)
assert storage.list_disabled() == [storage.DISABLED_DIR / p.name for p in three[:2]] and "Disabled 2 calibrations" in page.status.text()
page.remove_files(storage.list_disabled())
assert len(storage.list_removed()) == 2 and storage.list_disabled() == [] and "Removed 2 calibrations" in page.status.text()
page.restore_files(storage.list_removed())
assert sorted(p.name for p in storage.list_calibrations()) == sorted(p.name for p in three)

# ---- a gaze check, then Undo: its csv into the trash beside it; two undos in a row, newest first
app._calibrated(QWidget(), sitting())
check = storage.save_gaze_check([(0, 0.1, 0.1, 0.05, 0.05)])
app._gaze_logged(str(check))
assert page.undo_btn.text().endswith("Undo the gaze check") and len(app.undo_stack) == 4   # every calibration since, too
app.undo()
assert not check.exists() and (storage.GAZE_CHECK_DIR / "trash" / check.name).exists()
assert page.undo_btn.text().endswith("Undo the calibration")                   # the one before is next
app.undo()
assert len(storage.list_calibrations()) == 3 and len(app.undo_stack) == 2 and not page.undo_btn.isHidden()
app.fullscreen = QWidget()                                                      # never during a full-screen page
app.undo()
assert len(storage.list_calibrations()) == 3 and len(app.undo_stack) == 2
print("undo and restore ok")
