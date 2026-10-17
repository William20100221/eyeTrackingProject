"""
Self-check for issue #12 (no webcam):  python -m eyetracking.testing.test_ball_eye

The ball's eye opens while the main window is up and closes when it goes away;
while a reading is recorded (locked ball) the recording decides.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.ui.control_window as cw

cw.QSettings = lambda *_: QSettings(str(Path(tempfile.mkdtemp()) / "settings.ini"), QSettings.IniFormat)
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")
appmod.save_reading = lambda path, record: None
app = appmod.App(mouse=True)
app.timer.stop()
ball = app.ball


def settle():
    """Run the ball's spring for ~1.5 s of animation time."""
    for _ in range(40):
        ball._last_ms = ball._clock.elapsed() - 40
        ball._tick()
    return ball._open


assert ball.window_open and settle() > 0.9, ball._open        # window up at start: eye open
app.go_reading()                                               # window hidden, ball locked, not recording
assert not ball.window_open and settle() < 0.1, ball._open
app.ball_clicked()                                             # recording: open
assert settle() > 0.9
app.ball_clicked()                                             # stopped: the window comes back with the summary
assert ball.window_open and not ball.locked and settle() > 0.9
app.control.hide()
assert settle() < 0.1
print("ball eye ok")
