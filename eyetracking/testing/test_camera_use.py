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
assert log[-1] == "close" and "no usable calibration yet" in app.testing.status.text(), app.testing.status.text()
app.blink_test()                                                               # the blink test: on
assert log[-1] == "open" and app.camera_users == {"blink"}
app.fullscreen.close()
pump()
assert log[-1] == "close" and app.tracker is None and not app.camera_users

# a gaze check of ANOTHER person's 3x3 (not the person in use): trained from that person's files only
import numpy as np
from eyetracking.core.calibration import point_grid
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.models import CalibrationRecording, CalibrationSample
rng = np.random.default_rng(3)
frames = lambda t: [CalibrationSample(features_for(ScreenGeometry(27.7, 18.9), t, list(rng.normal([0, 0, 0, 0, -5, -65], 1)),
                                                  (2, -1), 0.9, rng), *t) for _ in range(20)]
storage.save_calibration(CalibrationRecording((), (), tuple(s for t in point_grid(3) for s in frames(t))), "original-3x3",
                         tmp / "data" / "William" / "original-3x3" / "original-3x3_2026-10-18_120000.json")
app.testing.set_check(person="William", method_id="original-3x3")
page = app.testing
assert (page.check_count, page.check_most) == (1, 1) and page.check_count_btn.isEnabled()   # most = what's recorded
storage.save_calibration(CalibrationRecording((), (), tuple(s for t in point_grid(3) for s in frames(t))), "original-3x3",
                         tmp / "data" / "William" / "original-3x3" / "original-3x3_2026-10-18_130000.json")
app.testing.refresh()
assert (page.check_count, page.check_most) == (1, 2)                             # a choice is kept while it fits
page._fill_check_count_menu()
assert [a.text() for a in page.check_count_menu.actions()] == ["1", "2 (all)"]   # 1 up to what's recorded
page.check_count_menu.actions()[1].trigger()
assert page.check_count == 2 and page.check_count_btn.text().startswith("2 (all)")
app.testing.set_check(method_id="original-4x4")                                 # none of this type: None
assert page.check_count_btn.text() == "None" and not page.check_count_btn.isEnabled() and page.check_most == 0
app.testing.set_check(method_id="original-3x3")                                 # another type: back to all of them
assert page.check_count == 2
page.set_check_count(1)
app.check()
assert app.fullscreen.method_id == "original-3x3" and app.fullscreen.folder == tmp / "data" / "William" / "original-3x3"
assert app.fullscreen.calibrations == 1
said = app.testing.log.item(0).text()
assert "William's Original 3×3: model trained on " in said and "from the newest 1 calibration." in said, said
assert storage.DATA_DIR == tmp / "data"                                        # the person in use didn't change
app.fullscreen.close()
pump()
storage.save_calibration(CalibrationRecording((), (), tuple(s for t in point_grid(3) for s in frames(t))), "original-3x3",
                         tmp / "data" / "William" / "original-3x3" / "original-3x3_2026-10-18_140000.json")   # made outside
assert page.check_most == 2                                     # not seen yet...
from PySide6.QtWidgets import QPushButton
next(b for b in page.findChildren(QPushButton) if b.text().endswith("Refresh")).click()
assert page.check_most == 3 and page.check_count == 1            # ...until Refresh; the choice of 1 is kept

fail = True                                                                    # a camera that can't open
app.calibrate()
assert app.tracker is None and not app.camera_users and app.fullscreen is None
assert "didn't start" in app.testing.status.text() and "didn't start" in app.testing.log.item(0).text()
print("camera use ok:", log)
