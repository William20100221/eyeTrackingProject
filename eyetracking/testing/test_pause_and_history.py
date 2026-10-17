"""
Self-check (no webcam):  python -m eyetracking.testing.test_pause_and_history

1. Calibration: Space starts, Space pauses (timer stops, nothing recorded),
   Space carries on (back to "settle" for the same dot), as often as you like;
   Esc leaves at any time
2. Gaze data: usable or not (format + feature count); a calibration file in words, "#n came from dot (c, r) of the 5x5 grid"
3. History: every chat / quiz message is saved and listed, newest file first
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
x, y = point_grid()[1]
path = storage.save_calibration([CalibrationSample((0.1, 0.2) + (0.0,) * (N_FEATURES - 2), x, y)] * 2)
text = storage.calibration_text(path)
assert "#2 came from dot (2, 1) of the 5x5 grid: 0.100, 0.200" in text, text
assert storage.list_calibrations() == [path] and storage.calibration_usable(path)
broken = tmp / "calibration_2026-01-01_000000.json"
broken.write_text("{}")
assert "Can't show this file" in storage.calibration_text(broken) and not storage.calibration_usable(broken)

# ---- 3. history ----------------------------------------------------------
chat_path = storage.new_chat_path()
storage.save_chat(chat_path, [{"who": "assistant", "text": "Question: why?", "at": "2026-10-17T10:00:00"},
                              {"who": "you", "text": "because", "at": "2026-10-17T10:00:09"}])
assert storage.list_chats() == [chat_path]
assert storage.chat_text(chat_path) == "Assistant (10:00):\nQuestion: why?\n\nYou (10:00):\nbecause"
print("pause, gaze data and history ok")
