"""
Self-check for person mode (no webcam):  python -m eyetracking.testing.test_person_mode

Settings > Person mode: off = files straight under data/<method>/; on with a person =
under data/<name>/<method>/. Saving the setting moves the app there at once (lists, "Saved as"
path, models), and a restart comes back to the same person.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.core.features import N_FEATURES
from eyetracking.core.models import CalibrationRecording, CalibrationSample

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
storage.BASE_DIR = tmp / "data"                                 # never the real data/
storage.use_person("")                                          # every folder under it
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
app = appmod.App(mouse=True)
w, page = app.control, app.testing
page.set_method("original-3x3")
assert storage.DATA_DIR == tmp / "data" and not w.person_switch.isChecked()
assert "Saved as data/original-3x3/original-3x3_<date>_<time>.json" in page.method_steps.text()

rec = CalibrationRecording((), (), (CalibrationSample((0.5,) * N_FEATURES, 0.5, 0.5),))
mine = storage.save_calibration(rec, "original-3x3")            # someone's file straight under data/
page.refresh()
assert page.active_list.item(0).data(0x0100) == str(mine)

w.set_person("Ann")                                              # Settings: person Ann, mode on, Save
w.person_switch.setChecked(True)
w.save()
assert storage.DATA_DIR == tmp / "data" / "Ann" and (tmp / "data" / "Ann" / "sweep").is_dir()
assert page.active_list.item(0).data(0x0100) is None             # Ann has nothing yet: data/ files not shown
assert "Saved as data/Ann/original-3x3/" in page.method_steps.text()
assert page.status.text() == "Saving for Ann in data/Ann/.", page.status.text()
assert storage.list_people() == ["Ann"]

w.person_switch.setChecked(False)                                # mode off: back to data/, the name is kept
w.save()
assert storage.DATA_DIR == tmp / "data" and page.active_list.item(0).data(0x0100) == str(mine)
assert w.person == "Ann" and page.status.text() == "Saving straight under data/."

w.person_switch.setChecked(True)
w.save()
w.store.sync()
storage.use_person("")
app2 = appmod.App(mouse=True)                                    # a restart: still Ann
assert storage.DATA_DIR == tmp / "data" / "Ann" and app2.control.person_switch.isChecked()
print("person mode ok")
