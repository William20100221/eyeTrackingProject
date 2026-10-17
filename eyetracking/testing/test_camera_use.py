"""
Self-check for issues #11 and #2 (no webcam):  python -m eyetracking.testing.test_camera_use

The app starts without opening the camera (the window is up at once, issue #2);
the camera opens when a feature needs the gaze and closes when the last one is
done (issue #11): calibrate -> idle -> Go Reading -> idle. A camera that fails
to open leaves the app running, with the reason in the chat.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.ui.control_window as cw

cw.QSettings = lambda *_: QSettings(str(Path(tempfile.mkdtemp()) / "settings.ini"), QSettings.IniFormat)
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
log = []


class FakeTracker:
    face_found = True

    def __init__(self, index):
        if fail:
            raise RuntimeError("camera busy")
        log.append("open")

    def read_features(self):
        return None

    def read(self):
        return None

    def close(self):
        log.append("close")


fail = False
appmod.EyeTracker = FakeTracker
appmod.camera_names = lambda: ["Cam A"]
appmod.pick_camera = lambda names, current="": names[0]
appmod.save_reading = lambda path, record: None
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")
app = appmod.App(mouse=False)
app.timer.stop()


def pump():
    for _ in range(5):
        qt.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


assert log == [] and app.tracker is None and app.control.isVisible(), log      # start: window up, camera off
app.calibrate()                                                                # calibration: on
assert log == ["open"] and app.camera_users == {"calibration"}
app.fullscreen.close()
pump()
assert log == ["open", "close"] and app.tracker is None                        # done: off
app.tick()                                                                     # idle: no frames read
app.go_reading()
app.ball_clicked()                                                             # the ball's eyes open: on
assert log[-1] == "open" and app.camera_users == {"screen"}
app.ball_clicked()                                                             # eyes closed: off
assert log[-1] == "close" and app.tracker is None and not app.camera_users

fail = True                                                                    # a camera that can't open
app.go_reading()
app.ball_clicked()
assert app.tracker is None and not app.camera_users and not app.ball.locked
assert "didn't start" in app.panel.said[-1], app.panel.said[-1]
print("camera use ok:", log)
