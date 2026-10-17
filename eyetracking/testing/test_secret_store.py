"""
Self-check for issue #13 (Windows):  python -m eyetracking.testing.test_secret_store

The AI key never lands in the settings as plain text: it round-trips through
DPAPI, an old plain-text key is encrypted the first time it is read, and the key
field's eye button shows and hides it.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QLineEdit

from eyetracking.services.secret_store import PREFIX, protect, unprotect

key = "sk-ant-test-0123456789"
locked = protect(key)
assert locked.startswith(PREFIX) and key not in locked and unprotect(locked) == key
assert unprotect(key) == key and protect("") == "" and unprotect(PREFIX + "bm90IGRwYXBp") == ""

import eyetracking.ui.control_window as cw
ini = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(ini), QSettings.IniFormat)
qt = QApplication([])
QSettings(str(ini), QSettings.IniFormat).setValue("api_key", key)        # as an older version saved it
w = cw.SettingsWindow()
assert w.read_store()["api_key"] == key and w.key.text() == key           # still works...
w.store.sync()
assert key not in ini.read_text() and PREFIX in ini.read_text()           # ...and is now encrypted on disk
w.key.setText("AIza-another-key")
w.save()
w.store.sync()
assert "AIza-another-key" not in ini.read_text() and w.read_store()["api_key"] == "AIza-another-key"
assert w.key.echoMode() == QLineEdit.Password
w.key_eye.trigger()
assert w.key.echoMode() == QLineEdit.Normal
w.key_eye.trigger()
assert w.key.echoMode() == QLineEdit.Password
print("secret store ok")
