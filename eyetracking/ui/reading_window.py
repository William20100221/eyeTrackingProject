"""
The sample text test, full screen: read a short text while a gaze model follows you.

    python -m eyetracking.ui.reading_window --mouse      (no camera: the mouse plays the eyes)

The gaze error (a few % of the screen, ~50-150 px) is far bigger than a line of text (~30 px),
so reading is tracked per paragraph, as in the app's reading window (master branch): a paragraph
counts as read once the gaze has stayed on it SECONDS_PER_WORD per word, and turns green. The
paragraphs are far apart (GAP), so a gaze error lands between them, not in the next one. A faint
dot shows where the model thinks you look. Space pauses, Esc finishes: finished(results), one
(words, seconds looked at, read?) per paragraph, then the window closes.

Like the gaze check, this window does NOT know about the camera or the model: it calls
`get_point()` every tick (0..1 screen fractions, or None while the face is lost or a blink).
"""
from __future__ import annotations

import sys
from typing import Callable

from PySide6.QtCore import QElapsedTimer, QPointF, QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import QApplication, QWidget

SAMPLE = """\
When you read, your eyes do not glide smoothly along a line. They move in quick jumps called saccades, each lasting only a few hundredths of a second. During a saccade you take in almost no new information; the brain hides the blur so you do not notice it.

Between saccades the eyes stop on a word for about a quarter of a second. These stops are called fixations, and they are when the actual reading happens. Short, common words like "the" are often skipped entirely, while long or unfamiliar words get longer fixations, or even two.

About one saccade in ten goes backwards. These regressions usually happen when a reader misunderstands a sentence or skipped a word too quickly. Skilled readers make fewer regressions, but no reader avoids them completely.

At the end of each line the eyes make a long jump back to the start of the next line, called a return sweep. Because an eye tracker can see all of these movements, it can estimate which parts of a text a person actually read and which parts they skipped."""

SECONDS_PER_WORD = 0.1   # gaze time a paragraph needs per word to count as read (the app's own rule)
TICK_MS = 33
GAP = 48                 # px between paragraphs: a gaze error lands in the gap, not in the next paragraph
PAD = 18                 # px around a paragraph's text: its box, where a look counts
FONT_PX = (22, 14)       # the text's size: 22 px, smaller on a short screen until it all fits
PAPER, UNREAD, READ, INK = QColor("#F4F5F7"), QColor("white"), QColor("#DDF3E4"), QColor("#1F2328")
DOT = QColor(61, 108, 246, 110)
Point = tuple[float, float]
Result = tuple[int, float, bool]   # words, seconds looked at, read?


class ReadingWindow(QWidget):
    finished = Signal(object)      # list[Result], one per paragraph, just before the window closes

    def __init__(self, get_point: Callable[[], Point | None], text: str = SAMPLE, parent: QWidget | None = None):
        super().__init__(parent)
        self._get_point = get_point
        self.paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        self.seen = [0.0] * len(self.paras)        # seconds of gaze per paragraph
        self.point: Point | None = None
        self.paused = False
        self._layout: tuple[tuple[int, int], QFont, list[QRectF]] | None = None   # (size it was made for, font, boxes)
        self._clock = QElapsedTimer()
        self._clock.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)
        self.setWindowTitle("Sample text")
        self.setCursor(Qt.CursorShape.BlankCursor)  # a visible mouse pulls the eye

    def is_read(self, i: int) -> bool:
        # ponytail: one fixed reading speed for everyone, as in the app; measure it per user if it misjudges
        return self.seen[i] >= SECONDS_PER_WORD * len(self.paras[i].split())

    def results(self) -> list[Result]:
        return [(len(p.split()), round(s, 2), self.is_read(i)) for i, (p, s) in enumerate(zip(self.paras, self.seen))]

    # ---- layout: one box per paragraph, the whole text on screen at once (made again only on a resize)
    def boxes(self) -> tuple[QFont, list[QRectF]]:
        size = (self.width(), self.height())
        if self._layout is None or self._layout[0] != size:
            width = int(self.width() * 0.72)
            left = (self.width() - width) // 2
            for px in range(FONT_PX[0], FONT_PX[1] - 1, -1):
                font = QFont(self.font())
                font.setPixelSize(px)
                fm = QFontMetrics(font)
                y, boxes = 90, []
                for p in self.paras:
                    h = fm.boundingRect(QRect(0, 0, width - 2 * PAD, 100000), Qt.TextWordWrap, p).height()
                    boxes.append(QRectF(left, y, width, h + 2 * PAD))
                    y += h + 2 * PAD + GAP
                if y - GAP <= self.height() - 40:
                    break
            self._layout = (size, font, boxes)
        return self._layout[1], self._layout[2]

    def paragraph_at(self, x: float, y: float) -> int | None:
        """Which paragraph's box (window px) holds the point, or None (a gap, the margins)."""
        return next((i for i, box in enumerate(self.boxes()[1]) if box.contains(QPointF(x, y))), None)

    # ---- every TICK_MS
    def _tick(self, dt: float | None = None) -> None:
        dt = self._clock.restart() / 1000 if dt is None else dt
        self.point = self._get_point()
        if self.point is not None:
            i = self.paragraph_at(self.point[0] * self.width(), self.point[1] * self.height())
            if i is not None:
                self.seen[i] += dt
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.paused = not self.paused
            (self._timer.stop if self.paused else self._timer.start)()
            self._clock.restart()                   # the pause doesn't count as reading
            self.update()
        elif event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.finished.emit(self.results())
            self.close()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), PAPER)
        font, boxes = self.boxes()
        read = sum(self.is_read(i) for i in range(len(self.paras)))
        p.setPen(QColor(110, 110, 110))
        p.drawText(QRectF(0, 28, self.width(), 30), Qt.AlignCenter,
                   ("Paused: Space to carry on.   " if self.paused else "") +
                   f"Read the text at your own pace.   {read} / {len(self.paras)} paragraphs read   ·   "
                   "Space: pause   ·   Esc: finish")
        p.setFont(font)
        for i, (text, box) in enumerate(zip(self.paras, boxes)):
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(READ if self.is_read(i) else UNREAD)
            p.drawRoundedRect(box, 12, 12)
            p.setPen(INK)
            p.drawText(box.adjusted(PAD, PAD, -PAD, -PAD), Qt.TextWordWrap, text)
        if self.point is not None and not self.paused:
            x = min(max(self.point[0], 0.0), 1.0) * self.width()
            y = min(max(self.point[1], 0.0), 1.0) * self.height()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(DOT)
            p.drawEllipse(QPointF(x, y), 14, 14)


def summary(results: list[Result]) -> str:
    """The test's result in words, for the status line."""
    read = sum(r for _, _, r in results)
    return (f"{read} of {len(results)} paragraphs counted as read (each needs {SECONDS_PER_WORD:.1f} s a word); "
            "looked at for " + ", ".join(f"{s:.1f} s" for _, s, _ in results) + ".")


if __name__ == "__main__":
    if "--mouse" not in sys.argv:
        print("Run it from the lab (Tests > Sample text), or with --mouse to try the page alone.")
        sys.exit(1)
    from eyetracking.ui.gaze_window import fake_point_from_mouse
    app = QApplication(sys.argv)
    w = ReadingWindow(fake_point_from_mouse)
    w.finished.connect(lambda r: print(summary(r)))
    w.showFullScreen()
    sys.exit(app.exec())
