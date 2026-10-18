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

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
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


class FakeCamera:
    def read_bgr(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)          # a black frame (no face)


class FakeTracker:
    face_found = True
    camera = FakeCamera()

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
assert app.fullscreen.names == ["original-3x3"] and app.fullscreen.folders == {"original-3x3": tmp / "data" / "William" / "original-3x3"}
assert app.fullscreen.calibrations == {"original-3x3": 1} and app.fullscreen.show_dot
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

# the multiple-model check: the popup picks the types and how many calibrations each; 6x6 rings, no dot
storage.save_calibration(CalibrationRecording((), (), tuple(s for t in point_grid(4) for s in frames(t))), "original-4x4",
                         tmp / "data" / "William" / "original-4x4" / "original-4x4_2026-10-18_120000.json")
dlg = cw.MultiCheckDialog("William", {"original-3x3": 3, "original-4x4": 1})
assert dlg.chosen() == {"original-3x3": 3, "original-4x4": 1}                   # all ticked, all calibrations
dlg.set_count("original-3x3", 1)
dlg.rows["original-4x4"][0].setChecked(False)                                   # 4x4 left out
assert dlg.chosen() == {"original-3x3": 1} and dlg.rows["original-3x3"][1].text().startswith("1  ")
again = cw.MultiCheckDialog("William", {"original-3x3": 3, "original-4x4": 1}, {"original-3x3": 1, "original-4x4": 0})
assert again.chosen() == {"original-3x3": 1}                                     # remembered: the count and the tick
page.multi_counts = {"original-3x3": 1, "original-4x4": 0}                       # as if the popup was answered so
app.check_all()
win = app.fullscreen
assert win.names == ["original-3x3"] and win.grid == 6 and not win.show_dot and win.tag == "multi"
assert win.calibrations == {"original-3x3": 1} and win.folders["original-3x3"] == tmp / "data" / "William" / "original-3x3"
win.close()
pump()

# record mode: the camera is recorded with the presses; Esc runs the models over the video
page.record_switch.setChecked(True)
app.check()
win = app.fullscreen
assert win.recorder is not None and win.names == [] and win.grid == 4
for _ in range(3):
    win._tick()                                                      # three frames written
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier))
assert win.recorder.presses[0]["target_x"] == 0.05 and len(win.recorder.frames_ms) == 3
QApplication.sendEvent(win, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
pump()
videos = tmp / "data" / "William" / "gaze-videos"
assert len(list(videos.glob("gaze-video_*.json"))) == 1 and len(list(videos.glob("gaze-video_*.mp4"))) + len(list(videos.glob("gaze-video_*.avi"))) == 1
assert "no face was found in the recording" in app.testing.status.text(), app.testing.status.text()   # black frames: the real tracker ran
assert page.recordings_btn.text().startswith("Recordings (1)")
page.record_switch.setChecked(False)

# Sweep with the dots reused from the person in use's newest Original 4x4
storage.save_calibration(CalibrationRecording((), (), tuple(s for t in point_grid(4) for s in frames(t))), "original-4x4")
app.testing.set_method("sweep")
assert app.testing.reuse_row.isVisibleTo(app.testing) and app.testing.reuse_dots and "reused from the newest one" in app.testing.reuse_note.text()
app.calibrate()
assert app.fullscreen._borrowed is not None and len(app.fullscreen._borrowed) == 16 * 20 and app.fullscreen._dots_from.startswith("original-4x4_")
app.fullscreen.close()
pump()
app.testing.set_method("random-sweep")                                           # a random twin wants Random 4x4
assert not app.testing.reuse_switch.isEnabled() and "No Random 4×4" in app.testing.reuse_note.text()
app.testing.set_method("original-3x3")
assert not app.testing.reuse_row.isVisibleTo(app.testing)

fail = True                                                                    # a camera that can't open
app.calibrate()
assert app.tracker is None and not app.camera_users and app.fullscreen is None
assert "didn't start" in app.testing.status.text() and "didn't start" in app.testing.log.item(0).text()
print("camera use ok:", log)
