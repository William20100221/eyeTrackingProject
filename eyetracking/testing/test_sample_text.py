"""
Self-check for the sample text test (no webcam):  python -m eyetracking.testing.test_sample_text

1. The reading page: a paragraph counts as read once the gaze stayed on it 0.1 s a word, the
   others don't; a look into the gap between paragraphs counts for none; the whole text fits on
   a small screen too; Esc hands back one (words, seconds, read?) per paragraph
2. The lab: Tests > Sample text opens the page full screen with the model picked on the Gaze check
   page (mouse mode here), and its result lands in the status line
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication, QEvent, QSettings, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

import eyetracking.services.storage as storage
import eyetracking.ui.control_window as cw
from eyetracking.ui.reading_window import SECONDS_PER_WORD, ReadingWindow, summary

qt = QApplication.instance() or QApplication([])
tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)   # never the user's settings
storage.BASE_DIR = tmp / "data"                                                      # never the real data/
storage.use_person("")


def esc(w):
    QApplication.sendEvent(w, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))


# ---- 1. the page ------------------------------------------------------------------------
where = [None]
w = ReadingWindow(lambda: where[0])
w.resize(1440, 900)
font, boxes = w.boxes()
assert len(boxes) == 4 and boxes[-1].bottom() <= 900 and font.pixelSize() <= 22    # (offscreen fonts run tall)
second = boxes[1].center()
where[0] = (second.x() / 1440, second.y() / 900)                  # looking at paragraph 2
needed = SECONDS_PER_WORD * len(w.paras[1].split())
for _ in range(int(needed / 0.5) + 1):
    w._tick(0.5)
assert [w.is_read(i) for i in range(4)] == [False, True, False, False], w.seen
gap = (boxes[1].bottom() + boxes[2].top()) / 2                    # between paragraphs 2 and 3: counts for none
where[0] = (0.5, gap / 900)
w._tick(10.0)
assert w.seen[2] == 0 and w.seen[1] < needed + 1
where[0] = None                                                   # a blink: nothing counts
w._tick(10.0)
small = ReadingWindow(lambda: None)
small.resize(1280, 720)
assert small.boxes()[1][-1].bottom() <= 720 and small.boxes()[0].pixelSize() < font.pixelSize()   # a small screen: smaller text, all of it
got = []
w.finished.connect(got.append)
esc(w)
assert len(got) == 1 and [r for _, _, r in got[0]] == [False, True, False, False] and got[0][1][0] == len(w.paras[1].split())
assert summary(got[0]).startswith("1 of 4 paragraphs counted as read")
print("1. page ok:", summary(got[0]))

# ---- 2. the lab ---------------------------------------------------------------------------
import eyetracking.app as appmod
appmod.fake_point_from_mouse = lambda: (0.5, 0.5)
app = appmod.App(mouse=True)
page = app.testing
page.show_page(6)
assert page.main.currentIndex() == 6 and page.side_nav.checkedButton().text().endswith("Sample text")
app.testing.command.emit("sample")
win = app.fullscreen
assert isinstance(win, ReadingWindow) and app.control.pages.currentWidget() is win
win._tick(0.2)
esc(win)
for _ in range(5):
    qt.processEvents()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
assert app.fullscreen is None and "paragraphs counted as read" in page.status.text(), page.status.text()
print("sample text ok")
