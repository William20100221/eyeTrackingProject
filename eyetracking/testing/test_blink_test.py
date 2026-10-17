"""
Self-check for the blink test (no webcam):  python -m eyetracking.testing.test_blink_test

The window walks through every state (countdown, then recording), hands back a
row per recorded frame tagged with its state, and Esc keeps the states that
finished.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.core.blink_scores as bs
import eyetracking.ui.blink_window as bw

bs.SECONDS, bs.COUNTDOWN, bw.TICK_MS = 0.12, 0.04, 10         # quick
qt = QApplication.instance() or QApplication([])
scores = iter([(0.1, 0.2)] * 10 ** 6)
window = bw.BlinkWindow(get_scores=lambda: next(scores))
out = []
window.finished.connect(lambda rows: (out.append(rows), qt.quit()))
window.show()
window.start()
QTimer.singleShot(10000, qt.quit)
qt.exec()
rows = out[0]
assert {r[0] for r in rows} == {s for s, _ in bs.STATES} and all(r[2:] == (0.1, 0.2) for r in rows), rows[:3]
order = [s for s, _ in bs.STATES]
states = [r[0] for r in rows]
assert states == sorted(states, key=order.index), states          # recorded in the order the screen asked

early = bw.BlinkWindow(get_scores=lambda: (0.5, 0.5))         # Esc before any state finished: cancelled
got = {"finished": 0, "cancelled": 0}
early.finished.connect(lambda rows: got.__setitem__("finished", got["finished"] + 1))
early.cancelled.connect(lambda: got.__setitem__("cancelled", got["cancelled"] + 1))
early.start()
QApplication.sendEvent(early, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
assert got == {"finished": 0, "cancelled": 1}, got
print("blink test ok:", len(rows), "frames over", len(bs.STATES), "states")
