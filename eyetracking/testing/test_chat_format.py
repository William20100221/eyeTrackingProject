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
import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw

ini = Path(tempfile.mkdtemp()) / "settings.ini"
storage.HISTORY_DIR = ini.parent / "history"                                     # never the real data/history
cw.QSettings = lambda *_: QSettings(str(ini), QSettings.IniFormat)
LANGS = [("en-NZ", "English (New Zealand)", True), ("zh-Hans-CN", "中文(中华人民共和国)", True),
         ("fr-FR", "Français (France)", False)]
cw.reading_languages = lambda: LANGS
installed = []
cw.install_ocr = installed.append
cw.QMessageBox.question = staticmethod(lambda *a, **k: cw.QMessageBox.Yes)
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
assert acts == ["Ready on this PC", "English (New Zealand)   (en-NZ)", "中文(中华人民共和国)   (zh-Hans-CN)",
                "Windows needs its reading pack (offered when picked)", "Français (France)   (fr-FR)",
                "Add a language… (Windows settings)"], acts
next(a for a in w.chat.lang_btn.menu().actions() if "zh-Hans-CN" in a.text()).trigger()
assert screen_text.OCR_LANG == "zh-Hans-CN" and "中文" in w.chat.lang_btn.text() and installed == []   # ready: no install
w2 = cw.SettingsWindow()                                                         # a restart: remembered
assert w2.reading_language() == ("zh-Hans-CN", "中文(中华人民共和国)")
w._fill_language_menu()
assert next(a for a in w.chat.lang_btn.menu().actions() if "zh-Hans-CN" in a.text()).isChecked()
next(a for a in w.chat.lang_btn.menu().actions() if "fr-FR" in a.text()).trigger()   # not ready: picked, install offered
assert screen_text.OCR_LANG == "fr-FR" and installed == ["fr-FR"]
cw.reading_languages = lambda: []                                                # no Windows OCR: said, not empty
w._fill_language_menu()
assert "isn't available" in w.chat.lang_btn.menu().actions()[0].text()

# ---- the real OCR side on this PC: a language without its pack falls back to an installed one of it, or any
if screen_text.ocr_tag("en") is not None:
    assert screen_text.ocr_tag("en-US") in ("en-US", "en-GB") and screen_text.ocr_tag("en-NZ").startswith("en")
    assert screen_text.ocr_tag("xx-XX") is not None                               # nothing alike: any installed one
    offered = [t for t, _, _ in screen_text.reading_languages()]
    assert all(t in offered for t in screen_text.COMMON_LANGUAGES)               # the 8 common ones always offered
assert "-".join(p for p in "zh-Hans-CN".split("-") if len(p) != 4) == "zh-CN"   # install_ocr's capability name

# ---- deleting chats: past readings to the trash (several at once), Today stays; the one being written stays gone
paths = []
for i, title in enumerate(("Bees choose a home", "The water cycle", "Saccades")):
    p = storage.HISTORY_DIR / f"reading_2026-10-2{i}_100000.json"
    storage.save_reading(p, {"title": title, "when": "", "summary": "Bees **swarm**.", "questions": []})
    paths.append(p)
chat = w.chat
chat.refresh()
titles = lambda: [chat.chat_list.item(i).text() for i in range(chat.chat_list.count())]
assert len(titles()) == 4 and titles()[0].endswith("Today")
chat.chat_list.setCurrentRow(1)                                                 # the newest reading, shown on the right
assert chat.view.currentIndex() == 1
app.record_path = paths[2]                                                      # its quiz still running
chat.chat_list.clearSelection()
for row in (1, 2):
    chat.chat_list.item(row).setSelected(True)
assert cw.picked(chat.chat_list) == [paths[2], paths[1]]
chat.delete_chats(cw.picked(chat.chat_list))
assert storage.list_readings() == [paths[0]] and (storage.HISTORY_DIR / "trash" / paths[2].name).exists()
assert len(titles()) == 2 and chat.view.currentIndex() == 0                     # back to Today
assert app.record_path is None
app.record = {"title": "x", "summary": "late answer", "questions": []}
app._save_record()                                                              # the next quiz answer: not written back
assert storage.list_readings() == [paths[0]]
chat.delete_chats([])                                                           # "Today" picked: nothing to delete
assert storage.list_readings() == [paths[0]]
assert "·" not in chat.model_btn.text()                                          # the model only, not its provider
print("chat format and language ok")
