"""
Self-check for Go Reading (no camera, no real screen, no AI):
    python -m eyetracking.testing.test_go_reading

Go Reading locks the ball. Click: eyes open, the page under the gaze is captured
and the gaze is tracked over it. Click: eyes closed, the chat comes back with
"you read X of Y words" and the AI's summary of what was missed. "Open assistant"
from the ball's menu ends a reading the same way.
"""
import os
import tempfile
import time
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

import eyetracking.ui.control_window as cw

cw.QSettings = lambda *_: QSettings(str(Path(tempfile.mkdtemp()) / "settings.ini"), QSettings.IniFormat)
import eyetracking.app as appmod
appmod.save_reading = lambda path, record: None          # no history files from a test
from eyetracking.core.reading import FIXATION_S, Line

qt = QApplication([])
qt.setQuitOnLastWindowClosed(False)

# ---- fakes: the screen, the window under the gaze, the AI, the clock
LINES = [Line("The water cycle moves heat around the planet and drives the weather.", 50, 300, 600, 30),
         Line("Evaporation is fastest over warm oceans near the equator, where the sun is strongest.", 50, 340, 600, 30)]
appmod.lines_on_screen = lambda hwnd: LINES
appmod.window_at = lambda x, y: 4242
appmod.window_title = lambda hwnd: "Water cycle - Notepad"
asked = []
appmod.summarize = lambda full, read, mode, provider, words: (asked.append((full, read, mode)), "SUMMARY")[1]
clock = [0.0]
appmod.time = types.SimpleNamespace(monotonic=lambda: clock[0])

app = appmod.App(mouse=True, provider="claude")
app.timer.stop()


def pump(n=10):
    for _ in range(n):
        qt.processEvents()


def wait(cond, seconds=5.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < seconds:
        qt.processEvents()
        time.sleep(0.01)
    assert cond()


def look(px: float, py: float) -> None:
    """Put the gaze at a physical screen point, the way the gaze model would."""
    g = QApplication.primaryScreen().geometry()
    ratio = QApplication.primaryScreen().devicePixelRatio()
    app.smoothed = ((px / ratio - g.x()) / (g.width() - 1), (py / ratio - g.y()) / (g.height() - 1))


# 1. Go Reading: window hidden, ball locked with its eyes closed, not recording
app.panel.go.click()
pump()
assert not app.control.isVisible() and app.ball.locked and not app.ball.eyes_open and app.reader is None

# 2. click: eyes open, recording; the page under the gaze is captured (on the background thread)
app.ball.clicked.emit()
pump()
assert app.ball.eyes_open and app.reader is not None and app.debug.isVisible()
look(300, 315)
app._capture()
wait(lambda: app.reader.full_text() != "")
assert app.reader_title == "Water cycle - Notepad"
assert app.reader.full_text().split("\n") == [ln.text for ln in LINES]

# 3. read line 1 at a normal pace (a fixation every 8 letters); line 2 never looked at
letter = LINES[0].letter_px()
look(LINES[0].x + 3 * letter, LINES[0].y + 15)
app._feed_reader()                                        # the first sample of a reading carries no time
for pos in range(3, len(LINES[0].text), 8):
    look(LINES[0].x + pos * letter, LINES[0].y + 15)
    for _ in range(int(FIXATION_S * 30) + 1):             # 8 samples = 0.27 s, one fixation
        clock[0] += 1 / 30
        app._feed_reader()
assert all(app.reader.read_flags(LINES[0].text)), app.reader.read_flags(LINES[0].text)
assert not any(app.reader.read_flags(LINES[1].text))
assert "words read" in app.debug.text() and "Water cycle" in app.debug.text()

# 4. click: eyes closed -> the chat is back, the ball unlocked, "you read X of Y words", then the summary
app.ball.clicked.emit()
pump()
assert not app.ball.eyes_open and not app.ball.locked and app.control.isVisible() and app.reader is None
assert not app.debug.isVisible()
n = len(LINES[0].text.split())
assert any(f"you read {n} of" in m for m in app.panel.said), app.panel.said
wait(lambda: asked)
assert asked[0][2] == "unread" and asked[0][1].split() == LINES[0].text.split()
wait(lambda: app.panel.said[-1] == "SUMMARY")
assert app.record["summary"] == "SUMMARY" and app.record["title"] == "The water cycle moves heat around"

# 5. a reading where nothing was read, ended by "Open assistant" from the menu
app.panel.go.click()
app.ball.clicked.emit()
pump()
assert app.reader is not None and app.ball.locked
app.ball.open_requested.emit()
pump()
assert app.reader is None and not app.ball.locked and app.control.isVisible()
assert "didn't see you read" in app.panel.said[-1]
# 6. a double-click on the ball records without Go Reading, and a second one stops
app.ball.double_clicked.emit()
pump()
assert app.reader is not None and app.ball.locked and not app.control.isVisible()
app.ball.double_clicked.emit()
pump()
assert app.reader is None and not app.ball.locked and app.control.isVisible()
print("go reading ok")
