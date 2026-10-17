"""
Self-check (no webcam):  python -m eyetracking.testing.test_pause_and_history

1. Calibration: Space starts, Space pauses (timer stops, nothing recorded),
   Space carries on (a timed step starts over, a dot goes back to "settle"), as often as you like;
   Esc leaves at any time
2. Gaze data: usable or not (format + feature count); disable / enable (data/disabled/); a calibration in words, per dot;
   removing samples or the whole file (the original kept in data/trash), "#n came from dot (c, r) of the 5x5 grid"
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
from eyetracking.ui.calibration_window import CalibrationWindow

qt = QApplication([])
tmp = Path(tempfile.mkdtemp())
storage.DATA_DIR, storage.HISTORY_DIR = tmp, tmp / "history"     # never touch the real data/


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
    assert not window.paused and window._timer.isActive() and window._phase != "collect"
key(window, Qt.Key.Key_Space)                                  # paused...
left = []
window.cancelled.connect(lambda: left.append(True))
key(window, Qt.Key.Key_Escape)                                 # ...and Esc still leaves
assert left and not window._timer.isActive()

# ---- 2. gaze data in words ----------------------------------------------
assert describe_target(*point_grid()[1]) == "dot (2, 1) of the 5x5 grid"      # 2nd dot = column 2, row 1
assert describe_target(0.43, 0.61) == "the point 43% across, 61% down"
from eyetracking.core.models import CalibrationRecording, CalibrationSample
x, y = point_grid()[1]
path = storage.save_calibration(CalibrationRecording((), (), (CalibrationSample((0.1, 0.2) + (0.0,) * (N_FEATURES - 2), x, y),) * 2))
text = storage.calibration_text(path)
assert "dot (2, 1) of the 5x5 grid: 2 samples" in text and "#2: eyes: iris 10% across" in text, text
assert storage.list_calibrations() == [path] and storage.calibration_usable(path)
broken = tmp / "calibration_2026-01-01_000000.json"
broken.write_text("{}")
assert "Can't show this file" in storage.calibration_text(broken) and not storage.calibration_usable(broken)
storage.DISABLED_DIR = tmp / "disabled"                         # disabling: kept, but out of training
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
storage.TRASH_DIR = tmp / "trash"                               # removing: the original goes to trash first
assert storage.remove_samples(path, {0}) == 1 and "1 sample," in storage.calibration_text(path)
assert (tmp / "trash" / path.name).exists()
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
print("pause, gaze data and history ok")
