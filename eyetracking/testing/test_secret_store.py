"""
Self-check for issue #13 (Windows):  python -m eyetracking.testing.test_secret_store

AI keys never land in the settings as plain text: they round-trip through DPAPI,
an old single key (plain or encrypted) becomes Key 1 the first time the window
opens, and the new-key field's eye button shows and hides what is typed.
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
QSettings(str(ini), QSettings.IniFormat).setValue("api_key", key)        # one key, as an older version saved it
w = cw.SettingsWindow()
w.store.sync()
keys = w.stored_keys()
assert [k["name"] for k in keys] == ["Key 1"] and w.entry_secret(keys[0]) == key   # it became Key 1...
assert key not in ini.read_text() and "api_key" not in ini.read_text()            # ...encrypted, the old entry gone
assert w.key.echoMode() == QLineEdit.Password
w.key_eye.trigger()
assert w.key.echoMode() == QLineEdit.Normal
w.key_eye.trigger()
assert w.key.echoMode() == QLineEdit.Password
print("secret store ok")
