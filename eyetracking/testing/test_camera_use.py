"""
Self-check for issues #11 and #2 (no webcam):  python -m eyetracking.testing.test_camera_use

The app starts without opening the camera (the window is up at once, issue #2);
the camera opens when a full-screen page needs the gaze and closes when it is
done (issue #11): calibrate -> idle -> blink test -> idle. A camera that fails
to open leaves the app running, with the reason in the log.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
storage.BASE_DIR = tmp / "data"                                 # never the real data/
storage.use_person("")                                          # every folder under it
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
app = appmod.App(mouse=False)


def pump():
    for _ in range(5):
        qt.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


assert log == [] and app.tracker is None and app.control.isVisible(), log      # start: window up, camera off
assert app.models == {} and "no calibration yet" in app.testing.status.text()
app.calibrate()                                                                # calibration: on
assert log == ["open"] and app.camera_users == {"calibration"}
assert app.fullscreen.method is app.method and app.control.pages.currentWidget() is app.fullscreen
app.fullscreen.close()
pump()
assert log == ["open", "close"] and app.tracker is None                        # done: off
app.check()                                                                    # no model yet: refused, camera stays off
assert log[-1] == "close" and "calibrate first" in app.testing.status.text()
app.blink_test()                                                               # the blink test: on
assert log[-1] == "open" and app.camera_users == {"blink"}
app.fullscreen.close()
pump()
assert log[-1] == "close" and app.tracker is None and not app.camera_users

fail = True                                                                    # a camera that can't open
app.calibrate()
assert app.tracker is None and not app.camera_users and app.fullscreen is None
assert "didn't start" in app.testing.status.text() and "didn't start" in app.testing.log.item(0).text()
print("camera use ok:", log)
