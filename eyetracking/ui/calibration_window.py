"""
Calibration page, full screen (PySide6). Three steps, ~2 minutes:

  1. line   14 still dots across + down the middle, head still   -> eye range
  2. head   keep looking at the centre dot while moving the head -> head effect
  3. dots   the 25-dot grid, in random order, head still         -> training data

Steps 1 and 3 run the same settle -> collect loop per dot (two
CalibrationSessions); step 2 is time-driven. Only drawing and timing live
here. What each step means is in core/calibration.py; the camera is in services/tracker.py;
a finished camera run is saved to data/ by services/storage.py.

Run from the project root, with your webcam:

    python -m eyetracking.ui.calibration_window

or with FAKE features from your mouse position, so no camera is needed:

    python -m eyetracking.ui.calibration_window --mouse

Space starts, Space again pauses / carries on (as often as you like),
Esc leaves at any time. Screen positions are stored as fractions 0..1
(x across, y down), so they work on any screen size.
"""
from __future__ import annotations

import random
import sys
from typing import Callable, Sequence

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.calibration import CalibrationSession, head_prompt, point_grid, sweep_points
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.services.storage import save_calibration
from eyetracking.services.tracker import EyeTracker


# =====================================================================
# Draws the current dot and runs the timing. It does NOT know about the
# camera: it just calls `get_features()` and stores whatever comes back.
# =====================================================================

SETTLE_MS = 1000   # dot appears, eyes travel to it: DON'T record yet
COLLECT_MS = 1500  # eyes resting on the dot: record features
DOT_SAMPLES = 25   # good frames each dot needs; a blink costs frames, so the dot waits longer
MAX_COLLECT_MS = 4000  # ...but never longer than this (face lost, eyes closed for ages)
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
        # step 1: still dots along the old sweep path, in path order
        self._line = CalibrationSession(sweep_points())
        # step 3: the grid in random order, so the eye can't guess the next dot
        grid = list(targets or point_grid())
        self._session = CalibrationSession(random.sample(grid, len(grid)))
        self._active = self._line                   # whose dots settle/collect runs on now
        # step 2 also goes through a CalibrationSession, for its blink handling
        # (frames just before a blink are dropped): one long "dot" at the centre
        self._head = CalibrationSession([(0.5, 0.5)])
        self._prompt = ""                           # step 2 instruction
        # intro -> (settle -> collect) x line dots -> head -> (settle -> collect) x grid -> done
        self._phase = "intro"
        self.paused = False
        self._clock = QElapsedTimer()  # time spent in the current phase
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self.setWindowTitle("Calibration")
        self.setCursor(Qt.CursorShape.BlankCursor)  # a visible mouse pulls the eye

    # ---- control ---------------------------------------------------
    def start(self) -> None:
        if self._phase != "intro":
            return
        self._phase = "settle"                        # first line dot
        self._clock.start()
        self._timer.start(TICK_MS)
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            if self._phase == "intro":
                self.start()
            elif self._phase != "done":
                self.toggle_pause()
        elif event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self._timer.stop()
        else:
            # the eyes wandered off while paused. Step 2 follows a timed path, so
            # it starts over; any other dot goes back to settle (its frames are kept)
            if self._phase == "head":
                self._head = CalibrationSession([(0.5, 0.5)])
            else:
                self._phase = "settle"
            self._clock.restart()
            self._timer.start(TICK_MS)
        self.update()

    # ---- timing: runs every TICK_MS --------------------------------
    def _tick(self) -> None:
        # Read on EVERY tick, settle included, and throw settle frames away.
        # A webcam keeps a few frames buffered: if we stop reading while the
        # eye travels, the first "collect" frames could be those stale ones.
        features = self._get_features()
        elapsed = self._clock.elapsed()

        if self._phase == "head":                     # step 2: centre dot, head moves
            step = head_prompt(elapsed)
            if step is None:
                self._head.next_target()              # step over: flush the session
                self._active = self._session          # on to step 3, the dot grid
                self._phase = "settle"
                self._clock.restart()
            else:
                self._prompt, record = step
                if record:
                    self._head.add_sample(features)

        elif self._phase == "settle" and elapsed >= SETTLE_MS:
            self._phase = "collect"
            self._clock.restart()

        elif self._phase == "collect":                # steps 1 and 3: record the current dot
            # None (no face / blink / eyes reopening) also drops the frames held
            # back just before it: see CalibrationSession.add_sample
            self._active.add_sample(features)

            # move on after COLLECT_MS once the dot has enough good frames;
            # blinks extend the dot, up to MAX_COLLECT_MS
            enough = self._active.current_count() >= DOT_SAMPLES
            if (elapsed >= COLLECT_MS and enough) or elapsed >= MAX_COLLECT_MS:
                self._active.next_target()
                if self._active.is_complete():
                    if self._active is self._session: # grid done: all three steps done
                        self._finish()
                        return
                    self._phase = "head"              # line done: on to step 2
                else:
                    self._phase = "settle"
                self._clock.restart()

        self.update()   # ask Qt to repaint (it calls paintEvent)

    def _finish(self) -> None:
        self._timer.stop()
        self._phase = "done"
        self.update()
        # the window doesn't close itself: whoever listens to `finished` decides
        self.finished.emit(CalibrationRecording(tuple(self._line.samples()), tuple(self._head.samples()),
                                                tuple(self._session.samples())))

    # ---- drawing ---------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("black"))

        if self._phase in ("intro", "done"):
            text = ("Three short steps, about two minutes:\n"
                    "1. Look at each dot on a line until it moves on, head still\n"
                    "2. Keep looking at the centre dot while you move your head\n"
                    "3. Look at each dot until it moves on, head still\n\n"
                    "Press Space to start, Space again to pause,\nEsc to leave at any time."
                    if self._phase == "intro" else "Calibration done.")
            font = p.font()
            font.setPointSize(18)
            p.setFont(font)
            p.setPen(QColor("white"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            return

        if self.paused:
            x, y = (0.5, 0.5) if self._phase == "head" else self._active.current_target()
            font = p.font()
            font.setPointSize(18)
            p.setFont(font)
            p.setPen(QColor("white"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Paused.\n\nSpace to carry on, Esc to leave.")
            p.setPen(QPen(QColor(90, 90, 90), 3))     # the dot you'll go back to
            p.drawEllipse(QPointF(x * self.width(), y * self.height()), RING_END, RING_END)
            return

        if self._phase == "head":                     # step 2: centre dot + an instruction
            centre = QPointF(0.5 * self.width(), 0.5 * self.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(80, 220, 120))
            p.drawEllipse(centre, 10, 10)
            font = p.font()
            font.setPointSize(14)
            p.setFont(font)
            p.setPen(QColor(200, 200, 200))
            # just under the dot, so it can be read without looking away from it
            p.drawText(self.rect().adjusted(0, int(centre.y()) + 30, 0, 0),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, self._prompt)
            return

        x, y = self._active.current_target()
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

        if self._phase == "collect" and self._clock.elapsed() > COLLECT_MS:
            # extended because of blinks: tell the user why the dot is staying
            p.setPen(QColor(80, 220, 120))
            p.drawText(QRectF(centre.x() - 100, centre.y() + 22, 200, 24),
                       Qt.AlignmentFlag.AlignCenter, "keep looking...")

        n, total = self._active.progress()
        step = "Step 1: line dot" if self._active is self._line else "Step 3: dot"
        font = p.font()
        font.setPointSize(12)
        p.setFont(font)
        p.setPen(QColor(120, 120, 120))
        p.drawText(20, 34, f"{step} {n} / {total}   (keep your head still)     Space: pause     Esc: leave")


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
