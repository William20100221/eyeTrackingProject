"""
Self-check (no webcam):  python -m eyetracking.testing.test_window_memory

The window reopens on the page it was left on (and the same Testing entry), also
after a restart; an AI answer still opens the chat.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.ui.control_window as cw

ini = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(ini), QSettings.IniFormat)
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")
app = appmod.App(mouse=True)
app.timer.stop()
w = app.control
assert w.pages.currentWidget() is w.chat                      # first run: the chat

w.show_page(w.testing)
w.testing.side_nav.button(5).click()                         # Data > Blink scores
w.hide()
app.ball.clicked.emit()                                      # the ball brings it back...
assert w.isVisible() and w.pages.currentWidget() is w.testing and w.testing.main.currentIndex() == 5

w.show_page(w.settings_page)
w.hide()
app._show_window(None)
assert w.pages.currentWidget() is w.settings_page             # ...where it was left

w.store.sync()
w2 = cw.SettingsWindow()                                     # a restart: the same settings file
w2.show_last()
assert w2.pages.currentWidget() is w2.settings_page and w2.testing.main.currentIndex() == 5

w.hide()
w.show_page(w.chat)                                          # what an AI answer does
assert w.pages.currentWidget() is w.chat
print("window memory ok")
