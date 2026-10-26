"""
Self-check for the chat's AI text and the reading language (no webcam, no network):
    python -m eyetracking.testing.test_chat_format

The AI's Markdown is shown formatted (bold, bullet lists), the user's own answers as typed; the
language picker beside the AI picker lists the languages Windows can read, remembers the choice, and
the app reads the screen (OCR) in it. Temp settings only.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication

import eyetracking.services.screen_text as screen_text
import eyetracking.ui.control_window as cw

ini = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(ini), QSettings.IniFormat)
cw.ocr_languages = lambda: [("en-US", "English (United States)"), ("zh-Hans-CN", "中文(中华人民共和国)")]
import eyetracking.app as appmod

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)
appmod.key_from_environment = lambda: ("claude", "sk-ant-test")

# ---- Markdown: the AI's text formatted, yours as typed
_, ai_label = cw.message_widget("Here is what you missed:\n\n*   the **oldest** foragers\n*   35 °C", mine=False)
_, my_label = cw.message_widget("a **star** answer", mine=True)
assert ai_label.textFormat() == Qt.MarkdownText and my_label.textFormat() == Qt.PlainText
doc = ai_label.text()
from PySide6.QtGui import QTextDocument
d = QTextDocument()
d.setMarkdown(doc)
assert "**" not in d.toPlainText() and "oldest" in d.toPlainText()          # bold rendered, the stars gone

# ---- the language picker: listed, picked, remembered; the app's OCR follows
app = appmod.App(mouse=True)
app.timer.stop()
w = app.control
assert w.reading_language() == ("en", "English") and screen_text.OCR_LANG == "en" and "English" in w.chat.lang_btn.text()
w._fill_language_menu()
acts = [a.text() for a in w.chat.lang_btn.menu().actions() if a.text()]
assert acts == ["English (United States)   (en-US)", "中文(中华人民共和国)   (zh-Hans-CN)",
                "Add a language… (Windows settings)"], acts
next(a for a in w.chat.lang_btn.menu().actions() if "zh-Hans-CN" in a.text()).trigger()
assert screen_text.OCR_LANG == "zh-Hans-CN" and "中文" in w.chat.lang_btn.text()
w2 = cw.SettingsWindow()                                                         # a restart: remembered
assert w2.reading_language() == ("zh-Hans-CN", "中文(中华人民共和国)")
w._fill_language_menu()
assert next(a for a in w.chat.lang_btn.menu().actions() if "zh-Hans-CN" in a.text()).isChecked()
cw.ocr_languages = lambda: []                                                    # no Windows OCR: said, not empty
w._fill_language_menu()
assert "isn't available" in w.chat.lang_btn.menu().actions()[0].text()
print("chat format and language ok")
