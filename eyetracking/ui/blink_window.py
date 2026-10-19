"""
Blink test, full screen: hold six eye states while the blink scores are recorded.

    python -m eyetracking.ui.blink_window            (your webcam)
    python -m eyetracking.ui.blink_window --mouse    (fake scores: the mouse's height on the screen)

Like the calibration page it knows nothing about the camera: get_scores() is
called every tick and returns (left, right) blink scores, None (no face), or
NO_FRAME (no new camera frame since the last tick: nothing to record).
Each state: a short "get ready" (not recorded), then the recording. Two bars
show the live scores against today's cut-off, so you can see what a blink
looks like to the app. Space starts, Esc leaves (what was recorded so far is
still handed back once at least one state finished).

The states, timings and the reading of the result: core/blink_scores.py.
"""
from __future__ import annotations

import sys
from typing import Callable

from PySide6.QtCore import QElapsedTimer, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

import eyetracking.core.blink_scores as bs
from eyetracking.core.analysis import BLINK_THRESHOLD
from eyetracking.core.models import NO_FRAME

TICK_MS = 33
Scores = tuple[float, float]


class BlinkWindow(QWidget):
    finished = Signal(list)   # the rows (state, ms, left, right) once the last state is done, or Esc after one or more
    cancelled = Signal()      # Esc before any state finished

    def __init__(self, get_scores: Callable[[], Scores | None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._get = get_scores
        self.rows: list[bs.Row] = []
        self._state = 0                 # index into bs.STATES
        self._phase = "intro"           # intro -> (ready -> record) x states -> done
        self._scores: Scores | None = None
        self._clock = QElapsedTimer()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setWindowTitle("Blink test")
        self.setCursor(Qt.CursorShape.BlankCursor)

    # ---- control ---------------------------------------------------
    def start(self) -> None:
        if self._phase != "intro":
            return
        self._phase = "ready"
        self._clock.start()
        self._timer.start(TICK_MS)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.start()
        elif event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            if self._phase == "done":
                pass                                  # already handed back
            elif self._state > 0:
                self.finished.emit(list(self.rows))   # the states that finished are worth keeping
            else:
                self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    def _tick(self) -> None:
        scores = self._get()
        fresh = scores is not NO_FRAME               # NO_FRAME: no camera frame since the last tick
        if fresh:
            self._scores = scores
        elapsed = self._clock.elapsed()
        if self._phase == "ready" and elapsed >= bs.COUNTDOWN * 1000:
            self._phase = "record"
            self._clock.restart()
        elif self._phase == "record":
            if fresh and self._scores is not None:
                self.rows.append((bs.STATES[self._state][0], elapsed, *self._scores))
            if elapsed >= bs.SECONDS * 1000:
                self._state += 1
                if self._state >= len(bs.STATES):
                    self._finish()
                    return
                self._phase = "ready"
                self._clock.restart()
        self.update()

    def _finish(self) -> None:
        self._timer.stop()
        self._phase = "done"
        self.update()
        self.finished.emit(list(self.rows))

    # ---- drawing ---------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("black"))
        big, small = p.font(), p.font()
        big.setPointSize(20)
        small.setPointSize(12)

        if self._phase in ("intro", "done"):
            p.setFont(big)
            p.setPen(QColor("white"))
            text = (f"Blink test: {len(bs.STATES)} eye states, about {len(bs.STATES) * (bs.SECONDS + bs.COUNTDOWN):.0f} s.\n"
                    "Hold each one as the screen says; a countdown comes first.\n\n"
                    "Press Space to start, Esc to leave."
                    if self._phase == "intro" else "Blink test done.")
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            return

        state, instruction = bs.STATES[self._state]
        cx, cy = self.width() / 2, self.height() / 2
        p.setFont(big)
        p.setPen(QColor("white"))
        p.drawText(QRectF(0, cy - 160, self.width(), 60), Qt.AlignmentFlag.AlignCenter, instruction)
        p.setFont(small)
        if self._phase == "ready":
            left = bs.COUNTDOWN - self._clock.elapsed() / 1000
            p.setPen(QColor(200, 200, 200))
            p.drawText(QRectF(0, cy - 90, self.width(), 30), Qt.AlignmentFlag.AlignCenter, f"get ready: {left:.0f}")
        else:
            done = min(1.0, self._clock.elapsed() / (bs.SECONDS * 1000))
            p.setPen(QColor(80, 220, 120))
            p.drawText(QRectF(0, cy - 90, self.width(), 30), Qt.AlignmentFlag.AlignCenter, f"recording  {done:.0%}")

        # the live scores against today's cut-off: this is what a blink looks like to the app
        w, h = 360, 22
        for i, name in enumerate(("left eye", "right eye")):
            y = cy + i * 44
            x0 = cx - w / 2
            p.setPen(QPen(QColor(90, 90, 90), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x0, y, w, h))
            if self._scores is not None:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(80, 220, 120) if self._scores[i] < BLINK_THRESHOLD else QColor(235, 149, 46))
                p.drawRect(QRectF(x0, y, w * max(0.0, min(1.0, self._scores[i])), h))
            tx = x0 + w * BLINK_THRESHOLD
            p.setPen(QPen(QColor("white"), 2))
            p.drawLine(int(tx), int(y - 4), int(tx), int(y + h + 4))     # the cut-off
            p.setPen(QColor(160, 160, 160))
            p.drawText(QRectF(x0 - 110, y, 100, h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
        p.setPen(QColor(120, 120, 120))
        p.drawText(20, 34, f"State {self._state + 1} / {len(bs.STATES)}     "
                           f"{'no face' if self._scores is None else 'face found'}     Esc: leave")


# =====================================================================
# RUNNER: real camera by default, or `--mouse` to test the page alone
# =====================================================================

def fake_scores_from_mouse() -> Scores:
    """Stand-in for blink scores: the mouse's height on the screen, 0 at the top, 1 at the bottom."""
    screen = QApplication.primaryScreen().geometry()
    y = QCursor.pos().y() / screen.height()
    return y, y


def main() -> int:
    from eyetracking.services.storage import save_blink_scores
    from eyetracking.services.tracker import EyeTracker

    app = QApplication(sys.argv)
    if "--mouse" in sys.argv:
        tracker, get = None, fake_scores_from_mouse
    else:
        tracker = EyeTracker()

        def get() -> Scores | None:
            r = tracker.read()
            return (r.blink_score_left, r.blink_score_right) if r is not None and r.face_found else None

    window = BlinkWindow(get_scores=get)

    def on_finished(rows: list) -> None:
        print(f"Saved to {save_blink_scores(rows)}")
        print("\n".join(bs.lines(rows)))
        QTimer.singleShot(800, window.close)

    window.finished.connect(on_finished)
    window.showFullScreen()
    try:
        return app.exec()
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    sys.exit(main())
