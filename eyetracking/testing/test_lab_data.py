"""
Self-check (no webcam):  python -m eyetracking.testing.test_lab_data

1. Calibration: Space starts, Space pauses (timer stops, nothing recorded), Space carries on, as often
   as you like; Esc leaves at any time
2. Data per method: data/calibrations/<method>/<method>_<date>_<time>.json, each file carrying its method and
   grid; a file in words ("dot (2, 1) of the 3x3 grid"); disable (the method's disabled/), enable, remove
   samples / the file (the original kept in the method's trash/); the Testing page lists the chosen method only
3. Gaze checks per method (nearest dot of THAT method's grid), blink scores, older loose files moved to
   older-versions/
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.core.calibration import describe_target, point_grid
from eyetracking.core.features import N_FEATURES
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.ui.calibration_window import CalibrationWindow

qt = QApplication([])
tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)   # never the user's settings
storage.DATA_DIR = tmp                                                                 # never the real data/
storage.CALIB_DIR, storage.GAZE_CHECK_DIR, storage.BLINK_DIR = tmp / "calibrations", tmp / "gaze_checks", tmp / "blink_scores"


def key(window, k):
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))


# ---- 1. pause ------------------------------------------------------------
window = CalibrationWindow(METHOD_BY_ID["sweep"], get_features=lambda: (0.5,) * N_FEATURES)
key(window, Qt.Key.Key_Space)                                  # start
assert window._timer.isActive() and not window.paused and window._phase == "sweep"
for _ in range(3):                                             # pause / carry on, again and again
    key(window, Qt.Key.Key_Space)
    assert window.paused and not window._timer.isActive()
    key(window, Qt.Key.Key_Space)
    assert not window.paused and window._timer.isActive() and window._phase == "sweep"
key(window, Qt.Key.Key_Space)                                  # paused...
left = []
window.cancelled.connect(lambda: left.append(True))
key(window, Qt.Key.Key_Escape)                                 # ...and Esc still leaves
assert left and not window._timer.isActive()

# ---- 2. data per method ----------------------------------------------------
assert describe_target(*point_grid(3)[1], 3) == "dot (2, 1) of the 3x3 grid"      # 2nd dot = column 2, row 1
assert describe_target(0.43, 0.61) == "the point 43% across, 61% down"
sample = lambda x, y: CalibrationSample((0.1, 0.2) + (0.0,) * (N_FEATURES - 2), x, y)
x, y = point_grid(3)[1]
path = storage.save_calibration(CalibrationRecording((), (), (sample(x, y),) * 2), "original-3x3")
assert path.parent == tmp / "calibrations" / "original-3x3" and path.name.startswith("original-3x3_20"), path
assert storage.method_of(path) == "original-3x3" and storage.file_date(path).year >= 2026
text = storage.calibration_text(path)
assert text.startswith("Original 3×3, ") and "The 3x3 dot grid: 2 samples" in text, text
assert "dot (2, 1) of the 3x3 grid: 2 samples" in text and "#2: eyes: iris 10% across" in text, text
other = storage.save_calibration(CalibrationRecording((), (), (sample(*point_grid(5)[1]),)), "original-5x5")
assert "dot (2, 1) of the 5x5 grid" in storage.calibration_text(other)
assert storage.list_calibrations("original-3x3") == [path] and storage.list_calibrations("original-5x5") == [other]
assert storage.list_calibrations("sweep") == [] and storage.calibration_usable(path)
broken = storage.method_dir("original-3x3") / "original-3x3_2026-01-01_000000.json"
broken.write_text("{}")
assert "Can't show this file" in storage.calibration_text(broken) and not storage.calibration_usable(broken)
off = storage.set_calibration_enabled(path, False)
assert off == storage.method_dir("original-3x3") / "disabled" / path.name and storage.method_of(off) == "original-3x3"
assert storage.list_calibrations("original-3x3") == [broken] and storage.list_disabled("original-3x3") == [off]
page = cw.TestingPage("original-3x3")
counts = lambda: (page.active_list.count(), page.disabled_list.count(), page.inactive_list.count())
assert counts() == (1, 1, 1) and page.active_list.item(0).data(0x0100) is None, counts()   # active: placeholder only
assert page.lists_title.text() == "Calibrations · Original 3×3"
retrained = []
page.data_changed.connect(lambda: retrained.append(True))
page.set_enabled(off, True)                                     # Enable: back in the method's folder, the model retrains
assert storage.list_calibrations("original-3x3") == [path, broken] and storage.list_disabled("original-3x3") == [] and retrained
assert page.active_list.item(0).data(0x0100) == str(path) and page.disabled_list.item(0).data(0x0100) is None
chosen = []
page.method_changed.connect(chosen.append)
page.set_method("original-5x5")                                 # another method: its own files only
assert chosen == ["original-5x5"] and page.active_list.item(0).data(0x0100) == str(other) and counts() == (1, 1, 1)
assert storage.remove_samples(path, {0}) == 1 and "1 sample," in storage.calibration_text(path)
assert (storage.method_dir("original-3x3") / "trash" / path.name).exists()
storage.remove_calibration(path)
assert storage.list_calibrations("original-3x3") == [broken] and not path.exists()
try:                                                            # an empty calibration is never written
    storage.save_calibration(CalibrationRecording((), (), ()), "sweep")
    raise AssertionError("an empty calibration was saved")
except ValueError:
    pass

# ---- 3. gaze checks per method, blink scores, older files -------------------
loose = tmp / "gaze_check_2026-01-02_000000.csv"
loose.write_text("index,x,y\n0,0.5,0.5\n")
old_calib = tmp / "calibration_2026-01-02_000000.json"
old_calib.write_text("{}")
assert storage.migrate_layout() == 2 and not loose.exists() and not old_calib.exists()
assert (tmp / "gaze_checks" / "older-versions" / loose.name).exists() and (tmp / "calibrations" / "older-versions" / old_calib.name).exists()
assert storage.list_calibrations("older-versions") == []       # never trains anything
check = storage.save_gaze_check([(0, 0.05, 0.05), (1, 0.5, 0.52)], "original-3x3")
assert check.parent == tmp / "gaze_checks" / "original-3x3" and check.name.startswith("gaze-check_original-3x3_20"), check
words = storage.gaze_check_lines(check)
assert words[0].startswith("Gaze check with Original 3×3,") and "2 samples" in words[0], words[0]
assert words[2] == "#0: 5% across, 5% down; nearest calibration dot (1, 1), 0% of the screen away", words
assert words[3] == "#1: 50% across, 52% down; nearest calibration dot (2, 2), 2% of the screen away", words   # the 3x3's centre
assert storage.list_gaze_checks() == [check, tmp / "gaze_checks" / "older-versions" / loose.name]
rows = [("open", 0, 0.05, 0.06), ("open", 33, 0.07, 0.05), ("down", 66, 0.2, 0.21), ("closed", 99, 0.8, 0.85),
        ("closed", 132, 0.9, 0.88), ("wink_left", 165, 0.9, 0.1)]
blink = storage.save_blink_scores(rows)
assert blink.name.startswith("blink-scores_20") and storage.load_blink_scores(blink) == rows and storage.list_blink_scores() == [blink]
words = storage.blink_score_lines(blink)
assert "Closed eyes score at least 0.8" in words[-1] and "never pass 0.21" in words[-1] and "wink" in words[-1], words[-1]
assert storage.trash_file(blink) == storage.BLINK_DIR / "trash" / blink.name and storage.list_blink_scores() == []
print("lab data ok")
