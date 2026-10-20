"""
Self-check for the gaze window (no webcam):  python -m eyetracking.testing.test_gaze_log

Enter records a sample (where the dot is right now) against the lit ring, then lights the
next one; each sample gets an index written BEFORE its coordinates, on screen and in the
csv saved on Esc. Space pauses. Nothing is recorded by itself. With several models, every
one records its own point at the same press (6x6 rings, no dot); the window saves and closes
by itself after the last ring, and Esc before that saves nothing.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from eyetracking.ui.gaze_window import GazeWindow

qt = QApplication([])


def key(window, k):
    QApplication.sendEvent(window, QKeyEvent(QEvent.Type.KeyPress, k, Qt.KeyboardModifier.NoModifier))


# ---- one model: its dot, the lit ring, the csv ----------------------------------
points = iter([(0.1, 0.2), None, (0.3, 0.4), (0.5, 0.6)])
window = GazeWindow.single(lambda: next(points))
window._tick()
key(window, Qt.Key.Key_Return)                            # sample 0 at (0.1, 0.2)
window._tick()                                            # a blink: the point is held...
key(window, Qt.Key.Key_Enter)                             # ...so sample 1 is (0.1, 0.2) too
window._tick()                                            # no Enter: nothing recorded
window._tick()
key(window, Qt.Key.Key_Return)                            # sample 2 at (0.5, 0.6)
assert window.sample_lines() == ["#0000  x 0.100  y 0.200  off 0.158", "#0001  x 0.100  y 0.200  off 0.292",
                                 "#0002  x 0.500  y 0.600  off 0.570"], window.sample_lines()   # off = from the lit ring
assert window._lit == 3 and [r[3:] for r in window.log] == [(0.05, 0.05), (0.35, 0.05), (0.65, 0.05)]   # 4x4 rings, one by one
assert window.show_dot and window.names == ["model"]

for _ in range(2):                                        # Space pauses, Space carries on
    key(window, Qt.Key.Key_Space)
    assert window.paused and not window._timer.isActive()
    key(window, Qt.Key.Key_Space)
    assert not window.paused and window._timer.isActive()

with tempfile.TemporaryDirectory() as folder:
    saved = window.save_log(Path(folder))
    rows = list(csv.reader(open(saved[0])))
assert len(saved) == 1 and rows[0] == ["index", "x", "y", "target_x", "target_y"]
assert rows[1] == ["0", "0.1000", "0.2000", "0.0500", "0.0500"] and len(rows) == 4, rows

# ---- several models at once: one press, a sample per model, 6x6 rings, no dot ----
a = iter([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)])
b = iter([None, (0.8, 0.8), None])                        # b has no point for the first press
with tempfile.TemporaryDirectory() as folder:
    folder = Path(folder)
    multi = GazeWindow(lambda: {"a": next(a), "b": next(b)}, {"a": folder / "a", "b": folder / "b"},
                       {"a": 2, "b": None}, grid=6, show_dot=False, tag="multi")
    assert not multi.show_dot and len(multi._targets) == 36 and multi.names == ["a", "b"]
    multi._tick()
    key(multi, Qt.Key.Key_Return)                         # press 0: a only (b has nothing yet)
    multi._tick()
    key(multi, Qt.Key.Key_Return)                         # press 1: both
    multi._tick()                                         # b blinks: held
    key(multi, Qt.Key.Key_Return)                         # press 2: both (b's held point)
    assert [r[:3] for r in multi.logs["a"]] == [(0, 0.1, 0.1), (1, 0.2, 0.2), (2, 0.3, 0.3)]
    assert [r[:3] for r in multi.logs["b"]] == [(1, 0.8, 0.8), (2, 0.8, 0.8)]      # the index is the press, not b's count
    assert multi.logs["a"][1][3:] == (0.23, 0.05) and multi.presses == 3            # the second 6x6 ring
    assert multi.sample_lines() == [] and multi.status_text().startswith("Ring 4 of 36     Enter: record")
    got = []
    multi.logged.connect(got.extend)
    key(multi, Qt.Key.Key_Escape)                         # a round not finished: nothing saved
    assert got == [] and not any(folder.iterdir())

    # one round of the 36 rings: it saves and closes by itself, no Esc
    whole = GazeWindow(lambda: {"a": (0.5, 0.5), "b": (0.4, 0.4)}, {"a": folder / "a", "b": folder / "b"},
                       {"a": 2, "b": None}, grid=6, show_dot=False, tag="multi")
    done = []
    whole.logged.connect(done.extend)
    whole._tick()
    for _ in range(35):
        key(whole, Qt.Key.Key_Return)
    assert not done and whole.status_text().startswith("Ring 36 of 36")
    key(whole, Qt.Key.Key_Return)
    assert len(whole.logs["a"]) == 36 and not whole._timer.isActive()
    assert [p.parent.name for p in done] == ["a", "b"] and done[0].name.startswith("gaze-check_a_multi_2-calibrations_")
    assert done[1].name.startswith("gaze-check_b_multi_20"), done[1].name
print("gaze samples ok")
