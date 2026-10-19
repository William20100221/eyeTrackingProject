"""
Self-check (no webcam):  python -m eyetracking.testing.test_window_memory

The window reopens on the page it was left on (the same Testing entry, and the
same method), also after a restart.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
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
app = appmod.App(mouse=True)
w = app.control
assert w.pages.currentWidget() is w.testing and w.testing.main.currentIndex() == 0     # first run: Calibration
assert app.method.id == "original-5x5"

w.testing.side_nav.button(5).click()                         # Data > Blink scores
w.testing.method_chips.button(6).click()                     # Sweep: the third row's first chip
assert app.method.id == "sweep" and "Sweep" in w.testing.status.text()
w.hide()
w.show_last()
assert w.isVisible() and w.pages.currentWidget() is w.testing and w.testing.main.currentIndex() == 5

w.nav.button(2).click()                                      # the Settings tab
w.hide()
w.show_last()
assert w.pages.currentWidget() is w.settings_page             # ...where it was left

w.store.sync()
w2 = cw.SettingsWindow()                                     # a restart: the same settings file
w2.show_last()
assert w2.pages.currentWidget() is w2.settings_page and w2.testing.main.currentIndex() == 5
assert w2.testing.method_id == "sweep" and w2.testing.method_chips.checkedId() == 6
print("window memory ok")
