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

Step 2 shows a bar: move your head until both marks turn green, so every
user turns a known amount. In steps 1 and 3 a frame with the head turned more
than 3 degrees from the dot's first frame is left out. Space starts, Space again pauses / carries on (as often as you like),
Esc leaves at any time. Screen positions are stored as fractions 0..1
(x across, y down), so they work on any screen size.
"""
from __future__ import annotations

import sys
from typing import Callable, Sequence

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.calibration import CalibrationSession, HeadStep, HeadStill, point_grid, sweep_target
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
        self._session = CalibrationSession(targets or point_grid())
        # steps 1-2 also go through a CalibrationSession, for its blink handling
        # (frames just before a blink are dropped); each is one long "dot"
        self._sweep = CalibrationSession([(0.5, 0.5)])   # step 1 frames (position per frame)
        self._head = CalibrationSession([(0.5, 0.5)])    # step 2 frames (centre dot)
        self._head_step = HeadStep()                # step 2: which move, how far the head is (the gauge)
        self._still = HeadStill()                   # steps 1 and 3: the head must not move
        self._head_moved = False
        self._target = (0.5, 0.5)                   # where the dot is drawn in steps 1-2
        self._prompt = ""                           # step 2 instruction
        # intro -> sweep -> head -> settle -> collect -> ... -> done
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
        self._phase = "sweep"
        self._still.reset()                           # the head's place for the whole sweep
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
            # the eyes wandered off while paused. Steps 1-2 follow a timed path,
            # so that step starts over; a grid dot goes back to settle (its frames are kept)
            if self._phase == "sweep":
                self._sweep = CalibrationSession([(0.5, 0.5)])
            elif self._phase == "head":
                self._head = CalibrationSession([(0.5, 0.5)])
                self._head_step = HeadStep()
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

        if self._phase == "sweep":                    # step 1: follow the moving dot
            step = sweep_target(elapsed)
            if step is None:
                self._sweep.next_target()             # step over: flush the session
                self._phase = "head"
                self._clock.restart()
            else:
                x, y, record = step
                self._target = (x, y)
                self._head_moved = features is not None and self._still.moved(features)
                if record and not self._head_moved:   # the head turned: leave the frame out
                    self._sweep.add_sample(features, at=(x, y))

        elif self._phase == "head":                   # step 2: centre dot, head moves
            step = self._head_step.feed(features, elapsed)
            if step is None:
                self._head.next_target()              # step over: flush the session
                self._phase = "settle"                # on to step 3, the dot grid
                self._clock.restart()
            else:
                self._target = (0.5, 0.5)
                self._prompt, record = step
                if record:
                    self._head.add_sample(features)

        elif self._phase == "settle" and elapsed >= SETTLE_MS:
            self._phase = "collect"
            self._clock.restart()
            self._still.reset()                       # the head's place for this dot: its first frame

        elif self._phase == "collect":
            # None (no face / blink / eyes reopening) also drops the frames held
            # back just before it: see CalibrationSession.add_sample
            # a head that turned since the dot's first frame: the frame is left out and
            # the dot waits (like for a blink), with "keep your head still" under it
            self._head_moved = features is not None and self._still.moved(features)
            if not self._head_moved:
                self._session.add_sample(features)

            # move on after COLLECT_MS once the dot has enough good frames;
            # blinks extend the dot, up to MAX_COLLECT_MS
            enough = self._session.current_count() >= DOT_SAMPLES
            if (elapsed >= COLLECT_MS and enough) or elapsed >= MAX_COLLECT_MS:
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
        self.finished.emit(CalibrationRecording(tuple(self._sweep.samples()), tuple(self._head.samples()),
                                                tuple(self._session.samples())))

    def _draw_gauge(self, p: QPainter, centre: QPointF) -> None:
        """Step 2: a bar under the dot showing how far the head has moved, the two
        targets (green once reached) and where the head is now."""
        g = self._head_step.gauge()
        if self._phase != "head" or g is None:
            return
        value, target, too_far, reached = g
        w, y0 = 360, int(centre.y()) + 70
        x0 = int(centre.x()) - w // 2
        p.setPen(QPen(QColor(90, 90, 90), 2))
        p.drawLine(x0, y0, x0 + w, y0)                       # the bar: -too far .. +too far
        for sign, hit in ((-1, reached[0]), (1, reached[1])):
            tx = int(centre.x() + sign * target / too_far * w / 2)
            p.setPen(QPen(QColor(80, 220, 120) if hit else QColor(160, 160, 160), 3))
            p.drawLine(tx, y0 - 10, tx, y0 + 10)             # a target mark
        vx = int(centre.x() + max(-1.0, min(1.0, value / too_far)) * w / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(vx, y0), 6, 6)                 # the head now
        p.setPen(QColor(160, 160, 160))
        p.drawText(QRectF(x0, y0 + 14, w, 20), Qt.AlignmentFlag.AlignCenter,
                   "move until both marks turn green" if not all(reached) else "good, keep going")

    # ---- drawing ---------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("black"))

        if self._phase in ("intro", "done"):
            text = ("Three short steps, about a minute:\n"
                    "1. Follow the moving dot with your eyes, head still\n"
                    "2. Keep looking at the centre dot and move your head as far as the bar asks\n"
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
            x, y = self._target if self._phase in ("sweep", "head") else self._session.current_target()
            font = p.font()
            font.setPointSize(18)
            p.setFont(font)
            p.setPen(QColor("white"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Paused.\n\nSpace to carry on, Esc to leave.")
            p.setPen(QPen(QColor(90, 90, 90), 3))     # the dot you'll go back to
            p.drawEllipse(QPointF(x * self.width(), y * self.height()), RING_END, RING_END)
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
            p.setPen(QColor(235, 149, 46) if self._phase == "sweep" and self._head_moved else QColor(200, 200, 200))
            # step 1: text at the top, out of the dot's way; step 2: just under
            # the centre dot, so it can be read without looking away from it
            top = 20 if self._phase == "sweep" else int(centre.y()) + 30
            p.drawText(self.rect().adjusted(0, top, 0, 0),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)
            self._draw_gauge(p, centre)
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

        if self._phase == "collect" and self._head_moved:
            p.setPen(QColor(235, 149, 46))
            p.drawText(QRectF(centre.x() - 120, centre.y() + 22, 240, 24),
                       Qt.AlignmentFlag.AlignCenter, "keep your head still")
        elif self._phase == "collect" and self._clock.elapsed() > COLLECT_MS:
            # extended because of blinks: tell the user why the dot is staying
            p.setPen(QColor(80, 220, 120))
            p.drawText(QRectF(centre.x() - 100, centre.y() + 22, 200, 24),
                       Qt.AlignmentFlag.AlignCenter, "keep looking...")

        n, total = self._session.progress()
        font = p.font()
        font.setPointSize(12)
        p.setFont(font)
        p.setPen(QColor(120, 120, 120))
        p.drawText(20, 34, f"Dot {n} / {total}     Space: pause     Esc: leave")


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
