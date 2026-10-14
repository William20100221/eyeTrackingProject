"""
Calibration page, full screen (PySide6). Three steps, ~1 minute:

  1. sweep  follow a dot gliding edge to edge (head still)       -> eye range
  2. head   keep looking at the centre dot while moving the head -> head effect
  3. dots   look at each of the 25 dots in turn                  -> training data

Only drawing and timing live here. What each step means, and the timing of
steps 1-2, is in core/calibration.py; the camera is in services/tracker.py;
a finished camera run is saved to data/ by services/storage.py.

Run from the project root, with your webcam:

    python -m eyetracking.ui.calibration_window

or with FAKE features from your mouse position, so no camera is needed:

    python -m eyetracking.ui.calibration_window --mouse

Space starts, Esc quits. Screen positions are stored as fractions 0..1
(x across, y down), so they work on any screen size.
"""
from __future__ import annotations

import sys
from typing import Callable, Sequence

from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.calibration import CalibrationSession, head_prompt, point_grid, sweep_target
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.services.storage import save_calibration
from eyetracking.services.tracker import EyeTracker


# =====================================================================
# Draws the current dot and runs the timing. It does NOT know about the
# camera: it just calls `get_features()` and stores whatever comes back.
# =====================================================================

SETTLE_MS = 1000   # dot appears, eyes travel to it: DON'T record yet
COLLECT_MS = 1500  # eyes resting on the dot: record features
TICK_MS = 33       # check about 30 times a second, like a webcam

RING_START = 40    # px: ring radius when a dot first appears
RING_END = 12      # px: ring radius once it has shrunk onto the dot


class CalibrationWindow(QWidget):
    finished = Signal(object)   # a CalibrationRecording, once all three steps are done
    cancelled = Signal()      # user pressed Esc

    def __init__(self,
                 get_features: Callable[[], Sequence[float] | None],
                 targets: Sequence[tuple[float, float]] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._get_features = get_features
        self._session = CalibrationSession(targets or point_grid())
        self._sweep: list[CalibrationSample] = []   # step 1 frames
        self._head: list[CalibrationSample] = []    # step 2 frames
        self._target = (0.5, 0.5)                   # where the dot is drawn in steps 1-2
        self._prompt = ""                           # step 2 instruction
        # intro -> sweep -> head -> settle -> collect -> ... -> done
        self._phase = "intro"
        self._clock = QElapsedTimer()  # time spent in the current phase
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self.setWindowTitle("Calibration")
        self.setCursor(Qt.CursorShape.BlankCursor)  # a visible mouse pulls the eye

    # ---- control ---------------------------------------------------
    def start(self) -> None:
        if self._phase != "intro":
            return
        self._phase = "sweep"
        self._clock.start()
        self._timer.start(TICK_MS)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.start()
        elif event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    # ---- timing: runs every TICK_MS --------------------------------
    def _tick(self) -> None:
        # Read on EVERY tick, settle included, and throw settle frames away.
        # A webcam keeps a few frames buffered: if we stop reading while the
        # eye travels, the first "collect" frames could be those stale ones.
        features = self._get_features()
        elapsed = self._clock.elapsed()

        if self._phase == "sweep":                    # step 1: follow the moving dot
            step = sweep_target(elapsed)
            if step is None:
                self._phase = "head"
                self._clock.restart()
            else:
                x, y, record = step
                self._target = (x, y)
                if record and features is not None:
                    self._sweep.append(CalibrationSample(tuple(map(float, features)), x, y))

        elif self._phase == "head":                   # step 2: centre dot, head moves
            step = head_prompt(elapsed)
            if step is None:
                self._phase = "settle"                # on to step 3, the dot grid
                self._clock.restart()
            else:
                self._target = (0.5, 0.5)
                self._prompt, record = step
                if record and features is not None:
                    self._head.append(CalibrationSample(tuple(map(float, features)), 0.5, 0.5))

        elif self._phase == "settle" and elapsed >= SETTLE_MS:
            self._phase = "collect"
            self._clock.restart()

        elif self._phase == "collect":
            if features is not None:   # None = no face or a blink: skip that frame
                self._session.add_sample(features)

            if elapsed >= COLLECT_MS:
                self._session.next_target()
                if self._session.is_complete():
                    self._finish()
                    return
                self._phase = "settle"
                self._clock.restart()

        self.update()   # ask Qt to repaint (it calls paintEvent)

    def _finish(self) -> None:
        self._timer.stop()
        self._phase = "done"
        self.update()
        # the window doesn't close itself: whoever listens to `finished` decides
        self.finished.emit(CalibrationRecording(tuple(self._sweep), tuple(self._head),
                                                tuple(self._session.samples())))

    # ---- drawing ---------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("black"))

        if self._phase in ("intro", "done"):
            text = ("Three short steps, about a minute:\n"
                    "1. Follow the moving dot with your eyes, head still\n"
                    "2. Keep looking at the centre dot while you move your head\n"
                    "3. Look at each dot until it moves on, head still\n\n"
                    "Press Space to start, Esc to quit."
                    if self._phase == "intro" else "Calibration done.")
            font = p.font()
            font.setPointSize(18)
            p.setFont(font)
            p.setPen(QColor("white"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            return

        if self._phase in ("sweep", "head"):          # steps 1-2: one dot + an instruction
            x, y = self._target
            centre = QPointF(x * self.width(), y * self.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(80, 220, 120))
            p.drawEllipse(centre, 10, 10)
            text = ("Follow the dot with your eyes. Keep your head still."
                    if self._phase == "sweep" else self._prompt)
            font = p.font()
            font.setPointSize(14)
            p.setFont(font)
            p.setPen(QColor(200, 200, 200))
            # step 1: text at the top, out of the dot's way; step 2: just under
            # the centre dot, so it can be read without looking away from it
            top = 20 if self._phase == "sweep" else int(centre.y()) + 30
            p.drawText(self.rect().adjusted(0, top, 0, 0),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)
            return

        x, y = self._session.current_target()
        centre = QPointF(x * self.width(), y * self.height())

        if self._phase == "settle":
            # ring shrinks onto the dot: pulls the eye to its exact centre
            t = min(1.0, self._clock.elapsed() / SETTLE_MS)
            radius = RING_START - (RING_START - RING_END) * t
            colour = QColor("white")
        else:
            radius = RING_END
            colour = QColor(80, 220, 120)   # green = recording

        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(colour, 3))
        p.drawEllipse(centre, radius, radius)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        p.drawEllipse(centre, 4, 4)

        n, total = self._session.progress()
        font = p.font()
        font.setPointSize(12)
        p.setFont(font)
        p.setPen(QColor(120, 120, 120))
        p.drawText(20, 34, f"Dot {n} / {total}")


# =====================================================================
# RUNNER: real camera by default, or `--mouse` to test the page alone
# =====================================================================

def fake_features_from_mouse() -> tuple[float, float]:
    """Stand-in for real eye features: the mouse position as 0..1 fractions."""
    screen = QApplication.primaryScreen().geometry()
    pos = QCursor.pos()
    return pos.x() / screen.width(), pos.y() / screen.height()


def main() -> int:
    app = QApplication(sys.argv)

    if "--mouse" in sys.argv:
        tracker = None
        get_features = fake_features_from_mouse
    else:
        tracker = EyeTracker()             # asks which camera, in the console
        get_features = tracker.read_features

    window = CalibrationWindow(get_features=get_features)

    def on_finished(rec: CalibrationRecording) -> None:
        print(f"Collected {len(rec.sweep)} sweep, {len(rec.head)} head, {len(rec.dots)} dot frames")
        for tx, ty in point_grid():
            count = sum(1 for s in rec.dots if (s.target_x, s.target_y) == (tx, ty))
            print(f"  dot ({tx:.2f}, {ty:.2f}): {count} samples")

        if tracker is None:
            print("--mouse mode: fake data, not saved")
        elif rec.dots:
            print(f"Saved to {save_calibration(rec)}")
        else:
            print("No samples (was your face in view?), nothing saved")
        QTimer.singleShot(800, window.close)   # show "done" briefly, then close

    window.finished.connect(on_finished)
    window.showFullScreen()
    try:
        return app.exec()
    finally:
        if tracker is not None:
            tracker.close()                # camera light off, MediaPipe freed


if __name__ == "__main__":
    sys.exit(main())
