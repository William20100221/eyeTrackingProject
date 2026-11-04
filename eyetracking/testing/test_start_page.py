"""
Self-check for the sidebar and the start page (no webcam):  python -m eyetracking.testing.test_start_page

The window is a sidebar (New reading, Calibrate, Tests, Data, recent readings, Settings) beside the
pages. The chat's start page has three looks, Settings > Start page, Clean by default: each holds the
input box (Go Reading), and the box moves under the messages once something is said. New reading brings
the start page back; this run's messages stay under Today. Temp settings and data only.
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
storage.HISTORY_DIR, storage.CALIB_DIR, storage.GAZE_CHECK_DIR = tmp / "history", tmp / "calibrations", tmp / "checks"
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)
qt = QApplication([])
storage.save_reading(storage.HISTORY_DIR / "reading_2026-11-04_100000.json",
                     {"title": "Bees choose a home", "when": "", "summary": "", "questions": [],
                      "full_text": " ".join(f"word{i}" for i in range(40)),
                      "read_text": " ".join(f"word{i}" for i in range(20))})

w = cw.SettingsWindow()
w.resize(1200, 760)
w.show()
qt.processEvents()
chat = w.chat
assert w.home_style == "clean" and chat.home_style == "clean"                    # the default: Clean
assert chat.home.isVisible() and not chat.scroll.isVisible()
assert chat.composer.parentWidget() is not None and chat.home.isAncestorOf(chat.composer)   # the box on the start page
assert w.nav.checkedId() == 0                                                    # New reading marked
assert chat.chat_list.item(0).isHidden()                                         # Today: nothing said yet

# ---- the three looks: each holds the input box; Hero card's own button stands for Go Reading
for style in ("hero", "dashboard", "clean"):
    w.home_group.button(list(cw.HOME_STYLES).index(style)).click()
    qt.processEvents()
    assert chat.home_style == style and chat.home.isAncestorOf(chat.composer), style
    assert chat.go.isVisibleTo(chat) == (style != "hero"), style
w.set_home_style("dashboard")
w.save()
assert w.store.value("home_style") == "dashboard"
w.load(cw.DEFAULTS)
assert chat.home_style == "clean"                                                # Reset to defaults: Clean
assert cw.reading_facts(storage.list_readings()[0])[2] == 0.5                     # half of it read

# ---- a test from the start page goes where the Testing page's buttons go
asked = []
w.testing.command.connect(asked.append)
chat.test_requested.emit("check")
assert asked == ["check"]

# ---- something said: the messages, the box under them, Today in the sidebar; New reading: the start page again
chat.reply("Calibrated.")                                                     # as the app speaks
assert chat.scroll.isVisibleTo(chat) and not chat.home.isVisibleTo(chat) and not chat.home.isAncestorOf(chat.composer)
assert not chat.chat_list.item(0).isHidden() and w.nav.checkedId() == -1          # Today, not New reading
w.nav.button(0).click()
qt.processEvents()
assert chat.home.isVisibleTo(chat) and chat.home.isAncestorOf(chat.composer) and w.nav.checkedId() == 0
chat._clicked(chat.chat_list.item(0))                                            # Today: the messages are still there
assert chat.scroll.isVisibleTo(chat) and chat.msgs.count() == 2

# ---- the sidebar's places, and its search over the readings
w.nav.button(4).click()
assert w.pages.currentWidget() is w.testing and w.testing.section() == "data" and w.nav.checkedId() == 4
w.nav.button(1).click()
assert w.testing.section() == "tests" and w.nav.checkedId() == 1
w.side_find.setText("water")
assert chat.chat_list.item(1).isHidden()
w.side_find.setText("bees")
assert not chat.chat_list.item(1).isHidden()

# ---- the look: neutral greys, the theme colour as a readable accent in both themes
for dark in (False, True):
    c = cw.make_scheme("#E5483B", dark)
    assert cw.contrast(c["onPrimary"], c["primary"]) >= 4.5 and cw.contrast(c["accentText"], c["surfaceContainerHigh"]) >= 4.5
assert cw.make_scheme("#E8B400", False)["primary"] != "#E8B400"                   # a light yellow is darkened for white text
print("start page ok")
