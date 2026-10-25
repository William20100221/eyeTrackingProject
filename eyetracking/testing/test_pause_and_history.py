"""
Self-check (no webcam):  python -m eyetracking.testing.test_pause_and_history

1. Calibration: Space starts, Space pauses (timer stops, nothing recorded),
   Space carries on (back to "settle" for the same dot), as often as you like;
   Esc leaves at any time
2. Gaze data: usable or not (format + feature count); disable / enable (data/disabled/); a calibration in words, per dot;
   removing samples or the whole file (the original kept in data/trash), "#n came from dot (c, r) of the 4x4 grid";
   a simulated person's three-step calibration saved, read back and trained on (Points + Random 4x4)
4. One folder per kind of data, older files moved in; gaze-check and blink-score files in words
3. History: a reading's AI results (summary, questions, answers, scores) are saved and listed
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
from eyetracking.core.calibration import describe_target, point_grid
from eyetracking.core.features import N_FEATURES
from eyetracking.core.models import CalibrationRecording
from eyetracking.ui.calibration_window import CalibrationWindow

qt = QApplication([])
tmp = Path(tempfile.mkdtemp())
storage.DATA_DIR = tmp                                           # never touch the real data/
storage.CALIB_DIR, storage.DISABLED_DIR, storage.TRASH_DIR = tmp / "calibrations", tmp / "calibrations/disabled", tmp / "calibrations/trash"
storage.GAZE_CHECK_DIR, storage.BLINK_DIR, storage.HISTORY_DIR = tmp / "gaze_checks", tmp / "blink_scores", tmp / "history"


def key(window, k):
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))


# ---- 1. pause ------------------------------------------------------------
window = CalibrationWindow(get_features=lambda: (0.5, 0.5))
key(window, Qt.Key.Key_Space)                                  # start
assert window._timer.isActive() and not window.paused
for _ in range(3):                                             # pause / carry on, again and again
    key(window, Qt.Key.Key_Space)
    assert window.paused and not window._timer.isActive()
    key(window, Qt.Key.Key_Space)
    assert not window.paused and window._timer.isActive() and window._phase == "settle"
key(window, Qt.Key.Key_Space)                                  # paused...
left = []
window.cancelled.connect(lambda: left.append(True))
key(window, Qt.Key.Key_Escape)                                 # ...and Esc still leaves
assert left and not window._timer.isActive()

# ---- 2. gaze data in words ----------------------------------------------
assert describe_target(*point_grid()[1]) == "dot (2, 1) of the 5x5 grid"      # 2nd dot = column 2, row 1
assert describe_target(0.43, 0.61) == "the point 43% across, 61% down"
from eyetracking.core.models import CalibrationSample
x, y = point_grid(4)[1]
path = storage.save_calibration(CalibrationRecording((), (), (CalibrationSample((0.1, 0.2) + (0.0,) * (N_FEATURES - 2), x, y),) * 2))
text = storage.calibration_text(path)
assert "dot (2, 1) of the 4x4 grid: 2 samples" in text and "#2: eyes: iris 10% across" in text, text
assert storage.list_calibrations() == [path] and storage.calibration_usable(path)
broken = storage.CALIB_DIR / "calibration_2026-01-01_000000.json"
broken.write_text("{}")
assert "Can't show this file" in storage.calibration_text(broken) and not storage.calibration_usable(broken)
off = storage.set_calibration_enabled(path, False)
assert storage.list_calibrations() == [broken] and storage.list_disabled() == [off] and storage.calibration_usable(off)
from eyetracking.ui.control_window import TestingPage
page = TestingPage()
page.refresh()
counts = lambda: (page.active_list.count(), page.disabled_list.count(), page.inactive_list.count())
assert counts() == (1, 1, 1) and page.active_list.item(0).data(0x0100) is None, counts()   # active: placeholder only
retrained = []
page.data_changed.connect(lambda: retrained.append(True))
page.set_enabled(off, True)                                     # Enable: back in data/, the model retrains
assert storage.list_calibrations() == [path, broken] and storage.list_disabled() == [] and retrained
assert page.active_list.item(0).data(0x0100) == str(path) and page.disabled_list.item(0).data(0x0100) is None
assert storage.remove_samples(path, {0}) == 1 and "1 sample," in storage.calibration_text(path)
assert (storage.TRASH_DIR / path.name).exists()
storage.remove_calibration(path)
assert storage.list_calibrations() == [broken] and not path.exists()

# ---- 3. history ----------------------------------------------------------
reading = storage.new_reading_path()
storage.save_reading(reading, {"title": "The water cycle moves heat", "when": "2026-10-17T10:00:00", "summary": "Heat moves.",
                               "questions": [{"question": "Why?", "model_answer": "Because.", "answer": "sun", "score": 7,
                                              "explanation": "Close."}, {"question": "How?", "model_answer": "Rain."}]})
assert storage.list_readings() == [reading] and storage.reading_title(reading) == "The water cycle moves heat"
assert storage.reading_messages(reading) == [("Heat moves.", False), ("Question 1: Why?", False), ("sun", True),
                                             ("Score 7/10. Close.\nModel answer: Because.", False),
                                             ("Question 2: How?", False)], storage.reading_messages(reading)
try:                                                            # an empty calibration is never written
    storage.save_calibration(CalibrationRecording((), (), ()))
    raise AssertionError("an empty calibration was saved")
except ValueError:
    pass
assert storage.list_calibrations() == [broken]
# a simulated person (core/line_model.features_for) through the three steps: saved, in words, trained on
import numpy as np
from eyetracking.core.calibration import sweep_points
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.points_model import train
rng, screen = np.random.default_rng(2), ScreenGeometry(27.7, 18.9)
look = lambda t, n, sd=1.0: tuple(CalibrationSample(features_for(screen, t, [rng.normal(0, 2 * sd), rng.normal(0, 2 * sd),
                                  rng.normal(0, sd), 0, -5, -65], (2, -1), 0.9, rng), *t) for _ in range(n))
rec = CalibrationRecording(sum((look(t, 20) for t in sweep_points()), ()), look((0.5, 0.5), 150, 6),
                           sum((look(t, 25) for t in point_grid(4)), ()))
person = storage.save_calibration(rec)
assert storage.load_calibration(person) == rec and storage.calibration_usable(person)
text = storage.calibration_text(person)
assert all(step in text for step in storage.STEP_NAMES.values()) and "of the 4x4 grid: 25 samples" in text
model, n = train(storage.load_all_calibrations())                              # the old v1 'broken' file is skipped
assert model is not None and model.k_source == "your own head step" and n > 400, (model, n)
assert storage.remove_samples(person, set(range(len(rec.sweep)))) == len(rec.sweep)     # step 1 out, the rest kept
assert storage.load_calibration(person).sweep == () and storage.load_calibration(person).dots == rec.dots
v1 = storage.CALIB_DIR / "calibration_2026-01-02_000000.json"                 # this app's files before the port
v1.write_text('{"format": "gaze-calibration-v1", "samples": [{"features": [0.5, 0.5], "target_x": 0.5, "target_y": 0.5}]}')
assert not storage.calibration_usable(v1) and "Can't show this file" in storage.calibration_text(v1)
for p in (person, v1):
    storage.remove_calibration(p)
# ---- 4. one folder per kind of data; older loose files are moved in; csv files in words
loose = tmp / "gaze_check_2026-01-02_000000.csv"
loose.write_text("index,x,y\n0,0.5,0.5\n")
assert storage.migrate_layout() == 1 and storage.list_gaze_checks() == [storage.GAZE_CHECK_DIR / loose.name] and not loose.exists()
assert storage.file_date(loose).day == 2
check = storage.save_gaze_check([(0, 0.05, 0.05), (1, 0.5, 0.52)])
words = storage.gaze_check_lines(check)
assert words[0].startswith("Gaze check of") and "2 samples" in words[0], words[0]
assert words[2] == "#0: 5% across, 5% down; nearest calibration dot (1, 1), 0% of the screen away", words
assert words[3] == "#1: 50% across, 52% down; nearest calibration dot (3, 3), 2% of the screen away", words
rows = [("open", 0, 0.05, 0.06), ("open", 33, 0.07, 0.05), ("down", 66, 0.2, 0.21), ("closed", 99, 0.8, 0.85),
        ("closed", 132, 0.9, 0.88), ("wink_left", 165, 0.9, 0.1)]
blink = storage.save_blink_scores(rows)
assert storage.load_blink_scores(blink) == rows and storage.list_blink_scores() == [blink]
words = storage.blink_score_lines(blink)
assert "Closed eyes score at least 0.8" in words[-1] and "never pass 0.21" in words[-1] and "wink" in words[-1], words[-1]
assert storage.trash_file(blink) == storage.BLINK_DIR / "trash" / blink.name and storage.list_blink_scores() == []
print("pause, gaze data and history ok")
