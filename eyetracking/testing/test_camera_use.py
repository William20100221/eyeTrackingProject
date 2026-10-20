"""
Self-check of the tester app (no webcam, no network):  python -m eyetracking.testing.test_camera_use

The app starts without opening the camera; the camera opens when a full-screen page needs the gaze and
closes when it is done. Nothing starts without a name. Each of the 8 types is calibrated twice (two
sets, "n/16"), the next one picked after each; Points is saved as Line model too. The gaze check stays
locked until all 16 are done, then runs twice ("n/2"): every model trained on the newest set, then on
both. Only after the second is everything emailed, in one email; a send that fails goes at the next
start. A camera that fails to open leaves the app running.
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
mailer.FALLBACK_DIR = tmp                                       # never the real Desktop
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


nav = lambda i: page.side_nav.button(i).text().split("   ")[-1]
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
assert page.method_id == "original-3x3" and page.todo_label.text().startswith("Set 1 of 2. Still to do: Original 3×3, Original 4×4")
assert (nav(1), nav(2)) == ("0/16", "0/2") and not page.check_go.isEnabled()
assert page.check_note.text() == "Finish the calibrations first (0/16)."
app.calibrate()                                                                # calibration: on
assert log == ["open"] and app.camera_users == {"calibration"} and app.camera == "Cam A"
assert app.fullscreen.method is app.method and app.control.pages.currentWidget() is app.fullscreen
app.fullscreen.close()
pump()
assert log == ["open", "close"] and app.tracker is None                        # done: off
app.check()                                                                    # locked: refused, camera stays off
assert log[-1] == "close" and page.status.text() == "Finish the calibrations first."

# ---- the two sets of calibrations ---------------------------------------------------------
from eyetracking.core.calibration import point_grid, sweep_points
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.models import CalibrationRecording, CalibrationSample
rng = np.random.default_rng(3)


def frames(t, n=20, spread=(2, 2, 1, 1, 1, 1)):
    return [CalibrationSample(features_for(ScreenGeometry(27.7, 18.9), t, list(rng.normal([0, 0, 0, 0, -5, -65], spread)),
                                           (2, -1), 0.9, rng), *t) for _ in range(n)]


def recording(m):
    if m.kind == "grid":
        return CalibrationRecording((), (), tuple(s for t in point_grid(m.grid) for s in frames(t)))
    head = tuple(frames((0.5, 0.5), 150, spread=(8, 6, 3, 2, 2, 4)))                # the head really moves
    if m.kind == "sweep":                                                         # a gliding dot: one frame a place
        line = np.round(0.05 + 0.9 * (1 - np.cos(np.linspace(0, 4 * np.pi, 160))) / 2, 4)
        return CalibrationRecording(tuple(s for x in line for s in frames((float(x), 0.5), 1)) +
                                    tuple(s for y in line for s in frames((0.5, float(y)), 1)), head, ())
    return CalibrationRecording(tuple(s for t in sweep_points() for s in frames(t, 15)), head, ())


class Done:                                                       # the finished calibration window (it closes later)
    def close(self):
        pass


app._calibrated(Done(), METHOD_BY_ID["original-3x3"], recording(METHOD_BY_ID["original-3x3"]))
assert page.status.text() == "Original 3×3: saved." and nav(1) == "1/16", page.status.text()
assert page.method_id == "original-4x4" and sent == []                  # the next one picked; nothing emailed yet
app._calibrated(Done(), METHOD_BY_ID["sweep"], recording(METHOD_BY_ID["points"]))   # still dots are not a Sweep
assert "not saved (its step 1 is still dots" in page.status.text() and storage.list_calibrations("sweep") == []
for k, m in enumerate(cw.TESTER_METHODS[1:]):                                   # the rest of set 1, as if done
    path = storage.save_calibration(recording(m), m.id, storage.method_dir(m.id) / f"{m.id}_2026-10-21_1000{k:02d}.json")
    if m.id == "points":
        storage.copy_as(path, "line-model")
page.refresh()
assert nav(1) == "8/16" and page.todo_label.text().startswith("Set 2 of 2. Still to do: Original 3×3, Original 4×4")
assert page.method_id == "original-3x3" and not page.check_go.isEnabled()
for k, m in enumerate(cw.TESTER_METHODS):                                       # set 2
    path = storage.save_calibration(recording(m), m.id, storage.method_dir(m.id) / f"{m.id}_2026-10-21_1100{k:02d}.json")
    if m.id == "points":
        storage.copy_as(path, "line-model")
page.refresh()
assert nav(1) == "16/16" and page.todo_label.text() == "Both sets done. Next: the gaze check." and page.check_go.isEnabled()
assert page.check_note.text() == "Gaze check 1 of 2: the models trained on one set of your calibrations."


# ---- the two gaze checks: on the newest set, then on both; one email after the second -------
def gaze_check(newest):
    app.check()
    win = app.fullscreen
    assert win.grid == 6 and not win.show_dot and win.tag == "multi" and len(win.names) == 24, win.names
    assert set(win.calibrations.values()) == {newest} and win.folders["original-3x3"] == tmp / "data" / "Ann" / "original-3x3"
    win._points = {m: (0.5, 0.5) for m in win.names}
    for _ in range(36):                                          # a whole round: it saves and closes by itself
        win.record()
    pump()
    assert app.fullscreen is None


gaze_check(1)
assert nav(2) == "1/2" and page.status.text() == "Gaze check 1/2 saved. One more to do." and sent == []
assert page.check_note.text() == "Gaze check 2 of 2: the models trained on both sets of your calibrations."
gaze_check(2)
assert nav(2) == "2/2" and not page.check_go.isEnabled() and page.check_note.text() == "Both done. Thank you!"
mailed(app)
assert len(sent) == 1 and page.status.text() == f"Results emailed to {mailer.TO}. Thank you!", page.status.text()
names = files(sent[0])
assert sum(n.endswith(".json") for n in names) == 18 and sum("_multi_1-calibrations_" in n for n in names) == 24
assert sum("_multi_2-calibrations_" in n for n in names) == 24 and all(n.startswith("Ann/") for n in names)
app.check()
assert page.status.text() == "Both gaze checks are done."

# no internet: a copy on the Desktop to send by hand, and the email goes at the next start
mailer.smtp_send = lambda msg: (_ for _ in ()).throw(OSError("no internet"))
app._mail(storage.result_files()[:1])
mailed(app)
assert page.status.text() == f"Not emailed (no internet). Please email {tmp / 'Eye tracking results - Ann.zip'} to {mailer.TO}."
mailer.smtp_send = sent.append
again = appmod.App(mouse=False)
mailed(again)
assert len(sent) == 2 and len(files(sent[1])) == 1 and not mailer.outbox().exists()
assert again.testing.name_edit.text() == "Ann" and again.camera == "Cam A" and again.testing.check_note.text() == "Both done. Thank you!"

fail = True                                                                    # a camera that can't open
app.calibrate()
assert app.tracker is None and not app.camera_users and app.fullscreen is None
assert "didn't start" in page.status.text()
print("tester app ok:", log)
