"""
Self-check for what the app says after a calibration (no webcam):  python -m eyetracking.testing.test_calibrated

As in calibration-lab: a sitting whose grid fails the quality check is rejected and NOT saved
("Try calibrate again"); a good one is saved and the model retrained on it, with how many good
frames each step kept; a sitting whose line dots / head step can't be used is still saved (the grid
trains), the user is told why, and the head matrix then comes from an older sitting or the built-in
one, whichever the retrain says. Temp folders only.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QWidget

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.core.calibration import point_grid, sweep_points
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.models import CalibrationRecording, CalibrationSample

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
storage.CALIB_DIR = tmp / "calibrations"                   # never data/
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")
app = appmod.App(mouse=True)
app.timer.stop()
app.mouse = False                                          # from here on, as with a camera
said = []
app.panel.reply = said.append

# a simulated person (as core/points_model.py's self-check): line dots, a head step, a 4x4 grid
rng = np.random.default_rng(1)
screen = ScreenGeometry(27.7, 18.9)
head_now = lambda sd=1.0: [rng.normal(0, 2 * sd), rng.normal(0, 2 * sd), rng.normal(0, sd), rng.normal(0, 1),
                           rng.normal(-5, 1), rng.normal(-65, 1)]
frames = lambda t, n, sd=1.0: [CalibrationSample(features_for(screen, t, head_now(sd), (2, -1), 0.9, rng), *t)
                               for _ in range(n)]
line = tuple(s for t in sweep_points() for s in frames(t, 20))
head = tuple(frames((0.5, 0.5), 200, sd=6))
grid = tuple(s for t in point_grid(4) for s in frames(t, 25))
saved = lambda: sorted(storage.CALIB_DIR.glob("calibration_*.json"))

app._calibrated(QWidget(), CalibrationRecording(line, head, grid))           # a good sitting
assert len(saved()) == 1 and said[-1].startswith("Calibrated with "), said[-1]
assert "good line-dot and" in said[-1] and "head turns corrected with your own head step" in said[-1], said[-1]
assert app.model is not None and app.model.k_source == "your own head step"

app._calibrated(QWidget(), CalibrationRecording(line, head[:20], grid))      # head step too short: saved, said
assert len(saved()) == 2 and "can't be used (" in said[-1] and "only 20 head-step frames (need 60)" in said[-1], said[-1]
assert "built-in" not in said[-1].split("Gaze model")[0]                    # no guess: the retrain names the source
assert "head turns corrected with your own head step" in said[-1], said[-1]   # the first sitting's head step

storage.CALIB_DIR = tmp / "fresh"                                            # alone, a short head step: built-in matrix
app._calibrated(QWidget(), CalibrationRecording(line, head[:20], grid))
assert "head turns corrected with the built-in head matrix" in said[-1], said[-1]
storage.CALIB_DIR = tmp / "calibrations"

stare = tuple(CalibrationSample(grid[0].features, s.target_x, s.target_y) for s in grid)   # eyes never followed
app._calibrated(QWidget(), CalibrationRecording(line, head, stare))
assert len(saved()) == 2 and said[-1].startswith("Calibration rejected (") and "Try calibrate again" in said[-1], said[-1]

app._calibrated(QWidget(), CalibrationRecording(line, head, ()))             # no face in view
assert len(saved()) == 2 and said[-1].startswith("No samples")

# every outcome is kept in Testing > Data > Calibration log, newest first (the "fresh" folder's one stays there)
log = [text for _, text in storage.calibration_log()]
assert len(log) == 4 and log[0].startswith("No samples") and log[1].startswith("Calibration rejected"), log
assert log[2].startswith("Calibrated with") and "can't be used" in log[2] and log[3].startswith("Calibrated with"), log
page = app.control.testing
page.show_page(7)
assert page.main.currentIndex() == 7 and page.side_nav.button(7).isChecked()
rows = [page.log_list.item(i).text() for i in range(page.log_list.count())]
assert len(rows) == 4 and rows[0].endswith(log[0]) and rows[3].endswith(log[3]), rows
print("calibrated ok:", said[0][:60] + "…")
