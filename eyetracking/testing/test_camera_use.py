"""
Self-check of the tester app (no webcam, no network):  python -m eyetracking.testing.test_camera_use

The app starts without opening the camera; the camera opens when a full-screen page needs the gaze and
closes when it is done. Nothing starts without a name. A calibration is saved in data/<name>/<type>/
(Points also as Line model) and emailed, and the next one to do is picked. The gaze check runs every
model the calibrations make, on 6x6 rings, no dot. A file not sent goes at the next start. A camera
that fails to open leaves the app running.
"""
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.services import mailer

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)   # never the real settings
storage.BASE_DIR = tmp / "data"                                 # never the real data/
storage.use_person("")
sent = []
mailer.smtp_send = sent.append                                  # never a real email
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


def pump():
    for _ in range(5):
        qt.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def mailed(app):
    """The emails on their way have gone, and what they said is on screen."""
    for t in app.senders:
        t.join(5)
    pump()


def files(msg):
    return msg.get_body(("plain",)).get_content().split("\n\n")[1].splitlines()


fail = False
appmod.EyeTracker = FakeTracker
appmod.camera_names = lambda: ["Cam A"]
appmod.pick_camera = lambda names, current="": names[0]
app = appmod.App(mouse=False)
page = app.testing

assert log == [] and app.tracker is None and app.control.isVisible(), log      # start: window up, camera off
app.calibrate()                                                                # no name: nothing starts
assert log == [] and page.status.text() == "Type your name first." and page.main.currentIndex() == 0
page.name_edit.setText("Ann")
assert storage.DATA_DIR == tmp / "data" / "Ann" and app.control.store.value("person") == "Ann"
assert page.method_id == "original-3x3" and page.todo_label.text().startswith("Still to do: Original 3×3, Original 4×4")
app.calibrate()                                                                # calibration: on
assert log == ["open"] and app.camera_users == {"calibration"} and app.camera == "Cam A"
assert app.fullscreen.method is app.method and app.control.pages.currentWidget() is app.fullscreen
app.fullscreen.close()
pump()
assert log == ["open", "close"] and app.tracker is None                        # done: off
app.check()                                                                    # no calibration: refused, camera stays off
assert log[-1] == "close" and page.status.text() == "No calibration yet: calibrate first."

from eyetracking.core.calibration import point_grid, sweep_points
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.models import CalibrationRecording, CalibrationSample
rng = np.random.default_rng(3)
frames = lambda t: [CalibrationSample(features_for(ScreenGeometry(27.7, 18.9), t, list(rng.normal([0, 0, 0, 0, -5, -65], 1)),
                                                  (2, -1), 0.9, rng), *t) for _ in range(20)]


class Done:                                                       # the finished calibration window (it closes later)
    def close(self):
        pass


grid = CalibrationRecording((), (), tuple(s for t in point_grid(3) for s in frames(t)))
app._calibrated(Done(), METHOD_BY_ID["original-3x3"], grid)
assert page.status.text() == "Original 3×3: saved.", page.status.text()
mailed(app)
assert len(storage.list_calibrations("original-3x3")) == 1 and page.method_id == "original-4x4"   # the next one picked
assert files(sent[0]) == [f"Ann/original-3x3/{storage.list_calibrations('original-3x3')[0].name}"]
assert sent[0]["Subject"] == "Eye tracking test: Ann, 1 file" and page.status.text() == f"Results sent to {mailer.TO}."

head = [s for _ in range(4) for s in frames((0.5, 0.5))]           # 80 head-step frames
points = CalibrationRecording(tuple(s for t in sweep_points() for s in frames(t)), tuple(head), ())
app._calibrated(Done(), METHOD_BY_ID["points"], points)            # saved as Points and as Line model
mailed(app)
assert [f.split("/")[1] for f in files(sent[1])] == ["points", "line-model"], files(sent[1])
app._calibrated(Done(), METHOD_BY_ID["sweep"], points)             # still dots are not a Sweep: not saved, not sent
assert "not saved (its step 1 is still dots" in page.status.text() and storage.list_calibrations("sweep") == []
mailed(app)
assert len(sent) == 2 and "Sweep" in page.todo_label.text() and "Points" not in page.todo_label.text()

# the gaze check: every model the calibrations make, from the same frames, 6x6 rings, no dot
app.check()
win = app.fullscreen
assert win.names[0] == "original-3x3" and set(win.names) <= {"original-3x3", "points-original-3x3", "line-model-original-3x3"}
assert win.grid == 6 and not win.show_dot and win.tag == "multi" and win.calibrations["original-3x3"] == 1
assert win.folders["original-3x3"] == tmp / "data" / "Ann" / "original-3x3" and log[-1] == "open"
win.close()
pump()
checks = [storage.save_gaze_check([(0, 0.5, 0.5, 0.5, 0.5)], m, tmp / "data" / "Ann" / m, 1, "multi")
          for m in ("original-3x3", "points-original-3x3")]
app._checked(checks)
mailed(app)
assert len(files(sent[2])) == 2 and page.status.text() == f"Results sent to {mailer.TO}."

# no internet: the file waits, and goes at the next start
mailer.smtp_send = lambda msg: (_ for _ in ()).throw(OSError("no internet"))
app._checked(checks[:1])
mailed(app)
assert page.status.text() == "Results not sent yet (no internet). They go with the next one."
mailer.smtp_send = sent.append
again = appmod.App(mouse=False)
mailed(again)
assert len(sent) == 4 and files(sent[3]) == [f"Ann/original-3x3/{checks[0].name}"] and not mailer.outbox().exists()
assert again.testing.name_edit.text() == "Ann" and again.camera == "Cam A"     # remembered

fail = True                                                                    # a camera that can't open
app.calibrate()
assert app.tracker is None and not app.camera_users and app.fullscreen is None
assert "didn't start" in page.status.text()
print("tester app ok:", log)
