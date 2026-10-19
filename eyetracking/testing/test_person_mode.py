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
from PySide6.QtCore import QSettings, Qt
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
assert page.active_list.item(0).text() == "Original 3×3" and page.active_list.item(1).data(0x0100) == str(mine)   # no people: types only

w.set_person("Ann")                                              # Settings: person Ann, mode on, Save
w.person_switch.setChecked(True)
w.save()
assert storage.DATA_DIR == tmp / "data" / "Ann" and (tmp / "data" / "Ann" / "sweep").is_dir()
texts = lambda lst: [lst.item(i).text() for i in range(lst.count())]
assert texts(page.active_list)[:2] == ["No person (data/)", "   Original 3×3"] and len(texts(page.active_list)) == 3   # name, then type
assert page.active_list.item(2).data(0x0100) == str(mine)
assert page.active_list.item(0).flags() == page.active_list.item(1).flags() == Qt.NoItemFlags   # headings: inert
ann = storage.save_calibration(rec, "original-3x3")              # Ann's first calibration
page.refresh()
assert texts(page.active_list)[3:5] == ["Ann   · in use", "   Original 3×3"], texts(page.active_list)
assert page.active_list.item(5).data(0x0100) == str(ann)
assert storage.set_calibration_enabled(mine, False).parent == tmp / "data" / "original-3x3" / "disabled"   # in ITS folder
page.refresh()
assert texts(page.disabled_list)[0] == "No person (data/)" and texts(page.active_list)[0] == "Ann   · in use"
storage.set_calibration_enabled(storage.list_disabled("original-3x3", tmp / "data")[0], True)
assert "Saved as data/Ann/original-3x3/" in page.method_steps.text()
assert page.status.text() == "Saving for Ann in data/Ann/.", page.status.text()
assert storage.list_people() == ["Ann"]

w.person_switch.setChecked(False)                                # mode off: back to data/, the name is kept
w.save()
assert storage.DATA_DIR == tmp / "data" and texts(page.active_list)[0] == "No person (data/)   · in use"
assert w.person == "Ann" and page.status.text() == "Saving straight under data/."

w.person_switch.setChecked(True)
w.save()
w.store.sync()
storage.use_person("")
app2 = appmod.App(mouse=True)                                    # a restart: still Ann
assert storage.DATA_DIR == tmp / "data" / "Ann" and app2.control.person_switch.isChecked()

# picking or adding a person saves at once (20 Oct), no Save changes needed
w2 = app2.control
w2.person_switch.setChecked(False)
w2.save()
assert storage.DATA_DIR == tmp / "data"
cw.QInputDialog.getText = lambda *a, **k: ("Bo", True)          # Add a person… > "Bo" > OK
w2.add_person()
assert storage.DATA_DIR == tmp / "data" / "Bo" and w2.store.value("person") == "Bo" and w2.person_switch.isChecked()
assert w2.store.value("person_mode", type=bool) and "Saving for Bo" in app2.testing.status.text()
w2._fill_person_menu()
next(a for a in w2.person_menu.actions() if a.text() == "Ann").trigger()   # the menu: Ann, saved at once
assert storage.DATA_DIR == tmp / "data" / "Ann" and w2.store.value("person") == "Ann"
print("person mode ok")
