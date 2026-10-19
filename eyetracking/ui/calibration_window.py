"""
Calibration page, full screen (PySide6), for every type in core/methods.py:

  Original 3x3 / 4x4 / 5x5   intro -> the dots, row by row -> done
  Random 3x3 / 4x4 / 5x5     intro -> the same dots, shuffled anew each time -> done
  Sweep                      intro -> gliding dot (across, down) -> head step -> done
  Points, Line model         intro -> 14 still dots along a line -> head step -> done
A special (Sweep, Points, Line model) records no dots: a grid calibration brings them (core/methods.py).

Only drawing and timing live here. What each step means is in core/calibration.py
(the dot bookkeeping, the sweep motion, the head step with its gauge, the head-still
rule); the camera is in services/tracker.py; the app saves the recording through
services/storage.py, in the person's folder of that type.

    python -m eyetracking.ui.calibration_window sweep           (your webcam)
    python -m eyetracking.ui.calibration_window random-3x3 --mouse

Space starts, Space again pauses / carries on (as often as you like). Esc leaves at any
time; once samples are collected it pauses and asks first, because nothing is saved if
you leave (No is the default). In the head-still steps a frame whose head turned more
than 3 degrees from the dot's first frame is left out. Screen positions are fractions
0..1 (x across, y down), so they work on any screen size.
"""
from __future__ import annotations

import random
import sys
from typing import Callable, Sequence

from PySide6.QtCore import QElapsedTimer, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from eyetracking.core.calibration import CalibrationSession, HeadStep, HeadStill, point_grid, sweep_points, sweep_target
from eyetracking.core.methods import DEFAULT_METHOD, METHOD_BY_ID, Method
from eyetracking.core.models import NO_FRAME, CalibrationRecording

SETTLE_MS = 1000       # dot appears, eyes travel to it: DON'T record yet
COLLECT_MS = 1500      # eyes resting on the dot: record features
DOT_SAMPLES = 25       # good frames each dot needs; a blink costs frames, so the dot waits longer
MAX_COLLECT_MS = 4000  # ...but never longer than this (face lost, eyes closed for ages)
TICK_MS = 16           # the dot moves at the screen's pace; each camera frame (~30 a second) is recorded once

RING_START = 40        # px: ring radius when a dot first appears
RING_END = 12          # px: ring radius once it has shrunk onto the dot
GREEN, AMBER, GREY = QColor(80, 220, 120), QColor(235, 149, 46), QColor(120, 120, 120)


class CalibrationWindow(QWidget):
    finished = Signal(object)   # a CalibrationRecording, once every step is done
    cancelled = Signal()        # Esc

    def __init__(self, method: Method, get_features: Callable[[], Sequence[float] | None],
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.method = method
        self._get_features = get_features
        grid = point_grid(method.grid) if method.kind == "grid" else []          # row by row...
        self._session = CalibrationSession(random.sample(grid, len(grid)) if method.random_order else grid) \
            if grid else None                                                     # ...or shuffled; none for a special
        # steps 1-2 also go through a CalibrationSession, for its blink handling
        # (frames just before a blink are dropped): one long "dot" for the sweep and the head step
        self._line = CalibrationSession(sweep_points()) if method.kind == "points" else None
        self._sweep = CalibrationSession([(0.5, 0.5)]) if method.kind == "sweep" else None
        self._head = CalibrationSession([(0.5, 0.5)]) if method.kind != "grid" else None
        self._head_step = HeadStep()                # step 2: which move, how far the head is (the gauge)
        self._still = HeadStill()                   # the head must not move while the eyes work
        self._head_moved = False
        self._active = self._line or self._session  # whose dots settle / collect runs on now
        self._target = (0.5, 0.5)                   # where the dot is drawn in the sweep and head steps
        self._prompt = ""                           # the head step's instruction
        # a grid: intro -> (settle -> collect) x dots -> done
        # a special: intro -> sweep | (settle -> collect) x line dots -> head -> done
        self._phase = "intro"
        self.paused = False
        self._clock = QElapsedTimer()               # time spent in the current phase
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self.setWindowTitle(f"Calibration: {method.name}")
        self.setCursor(Qt.CursorShape.BlankCursor)  # a visible mouse pulls the eye

    # ---- control ---------------------------------------------------
    def start(self) -> None:
        if self._phase != "intro":
            return
        self._phase = "sweep" if self.method.kind == "sweep" else "settle"
        self._still.reset()
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
            n = self.collected()
            if n and self._phase != "done":       # samples would be lost: ask first, paused meanwhile
                if not self.paused:
                    self.toggle_pause()
                if not self.confirm_leave(n):
                    return                        # stays paused: Space carries on
            self._timer.stop()
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    def collected(self) -> int:
        """How many samples this calibration has kept so far (every step)."""
        return sum(len(s.samples()) for s in (self._sweep, self._line, self._head, self._session) if s)

    def confirm_leave(self, n: int) -> bool:
        """Esc with samples collected: are you sure? Nothing is saved if you leave."""
        box = QMessageBox(QMessageBox.Icon.Question, "Leave the calibration?",
                          f"{n} samples have been collected. If you leave now they won't be saved.\n\n"
                          "Leave anyway?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, self)
        box.setDefaultButton(QMessageBox.StandardButton.No)   # Enter / a second Esc keeps you in
        box.setCursor(Qt.CursorShape.ArrowCursor)              # the page hides the mouse; the box needs it
        return box.exec() == QMessageBox.StandardButton.Yes

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.paused:
            self._timer.stop()
        else:
            # the eyes wandered off while paused. A timed step (sweep, head) starts over;
            # a dot goes back to settle (its frames are kept)
            if self._phase == "sweep":
                self._sweep = CalibrationSession([(0.5, 0.5)])
                self._still.reset()
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
        # From the camera's thread (app.py) most ticks come before the next frame: NO_FRAME. The
        # dot moves on with the clock, but nothing is recorded, and it isn't a blink: `fresh`.
        features = self._get_features()
        fresh = features is not NO_FRAME
        if not fresh:
            features = None
        elapsed = self._clock.elapsed()

        if self._phase == "sweep":                    # Sweep step 1: follow the gliding dot
            step = sweep_target(elapsed)
            if step is None:
                self._sweep.next_target()             # step over: flush the session
                self._phase = "head"
                self._clock.restart()
            else:
                x, y, record = step
                self._target = (x, y)
                if fresh:
                    self._head_moved = features is not None and self._still.moved(features)
                if fresh and record and not self._head_moved:   # the head turned: leave the frame out
                    self._sweep.add_sample(features, at=(x, y))

        elif self._phase == "head":                   # step 2: centre dot, the head moves
            step = self._head_step.feed(features, elapsed)
            if step is None:
                self._head.next_target()              # step over: flush the session; a special ends here
                self._finish()
                return
            else:
                self._target = (0.5, 0.5)
                self._prompt, record = step
                if record and fresh:
                    self._head.add_sample(features)

        elif self._phase == "settle" and elapsed >= SETTLE_MS:
            self._phase = "collect"
            self._clock.restart()
            self._still.reset()                       # the head's place for this dot: its first frame

        elif self._phase == "collect":                # a still dot: record it
            # a head that turned since the dot's first frame: the frame is left out and
            # the dot waits (like for a blink), with "keep your head still" under it.
            # None (no face / blink / eyes reopening) also drops the frames held
            # back just before it: see CalibrationSession.add_sample
            if fresh:
                self._head_moved = features is not None and self._still.moved(features)
                if not self._head_moved:
                    self._active.add_sample(features)
            # move on after COLLECT_MS once the dot has enough good frames;
            # blinks extend the dot, up to MAX_COLLECT_MS
            enough = self._active.current_count() >= DOT_SAMPLES
            if (elapsed >= COLLECT_MS and enough) or elapsed >= MAX_COLLECT_MS:
                self._active.next_target()
                if self._active.is_complete():
                    if self._active is self._session:   # the grid is done: every step is
                        self._finish()
                        return
                    self._phase = "head"              # the line dots are done: on to the head step
                else:
                    self._phase = "settle"
                self._clock.restart()

        self.update()

    def _finish(self) -> None:
        self._timer.stop()
        self._phase = "done"
        self.update()
        first = self._sweep or self._line             # the sweep's frames, or the line dots'
        # the window doesn't close itself: whoever listens to `finished` decides
        self.finished.emit(CalibrationRecording(tuple(first.samples()) if first else (),
                                                tuple(self._head.samples()) if self._head else (),
                                                tuple(self._session.samples()) if self._session else ()))

    # ---- drawing ---------------------------------------------------
    def _text(self, p: QPainter, size: int, colour: QColor = QColor("white")) -> None:
        font = p.font()
        font.setPointSize(size)
        p.setFont(font)
        p.setPen(colour)

    def _draw_gauge(self, p: QPainter, centre: QPointF) -> None:
        """Head step: a bar under the dot showing how far the head has moved, the two
        targets (green once reached) and where the head is now."""
        g = self._head_step.gauge()
        if g is None:
            return
        value, target, too_far, reached = g
        w, y0 = 360, int(centre.y()) + 70
        x0 = int(centre.x()) - w // 2
        p.setPen(QPen(QColor(90, 90, 90), 2))
        p.drawLine(x0, y0, x0 + w, y0)                       # the bar: -too far .. +too far
        for sign, hit in ((-1, reached[0]), (1, reached[1])):
            tx = int(centre.x() + sign * target / too_far * w / 2)
            p.setPen(QPen(GREEN if hit else QColor(160, 160, 160), 3))
            p.drawLine(tx, y0 - 10, tx, y0 + 10)             # a target mark
        vx = int(centre.x() + max(-1.0, min(1.0, value / too_far)) * w / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("white"))
        p.drawEllipse(QPointF(vx, y0), 6, 6)                 # the head now
        self._text(p, 12, QColor(160, 160, 160))
        p.drawText(QRectF(x0, y0 + 12, w, 30), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip,
                   "move until both marks turn green" if not all(reached) else "good, keep going")

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("black"))

        if self._phase in ("intro", "done"):
            self._text(p, 18)
            if self._phase == "done":
                text = "Calibration done."
            else:
                steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.method.steps))
                text = (f"{self.method.name}\n\n{steps}\n\n"
                        "Press Space to start, Space again to pause,\nEsc to leave at any time.")
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            return

        if self.paused:
            if self._phase in ("sweep", "head"):
                x, y = self._target
            else:
                x, y = self._active.current_target()
            self._text(p, 18)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Paused.\n\nSpace to carry on, Esc to leave.")
            p.setPen(QPen(QColor(90, 90, 90), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(x * self.width(), y * self.height()), RING_END, RING_END)   # the dot you'll go back to
            return

        if self._phase in ("sweep", "head"):          # one dot + an instruction (+ the gauge)
            x, y = self._target
            centre = QPointF(x * self.width(), y * self.height())
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(GREEN)
            p.drawEllipse(centre, 10, 10)
            moved = self._phase == "sweep" and self._head_moved
            self._text(p, 14, AMBER if moved else QColor(200, 200, 200))
            text = ("Follow the dot with your eyes. Keep your head still." if self._phase == "sweep" else self._prompt)
            # the sweep's text at the top, out of the dot's way; the head step's just under the centre dot
            top = 20 if self._phase == "sweep" else int(centre.y()) + 30
            p.drawText(self.rect().adjusted(0, top, 0, 0),
                       Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text)
            if self._phase == "head":
                self._draw_gauge(p, centre)
            self._text(p, 12, GREY)
            p.drawText(20, 34, f"{self.method.name}: {'step 1, the sweep' if self._phase == 'sweep' else 'step 2, the head'}"
                               "     Space: pause     Esc: leave")
            return

        x, y = self._active.current_target()
        centre = QPointF(x * self.width(), y * self.height())
        if self._phase == "settle":
            # ring shrinks onto the dot: pulls the eye to its exact centre
            t = min(1.0, self._clock.elapsed() / SETTLE_MS)
            radius, colour = RING_START - (RING_START - RING_END) * t, QColor("white")
        else:
            radius, colour = RING_END, GREEN           # green = recording
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(colour, 3))
        p.drawEllipse(centre, radius, radius)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(colour)
        p.drawEllipse(centre, 4, 4)

        if self._phase == "collect" and self._head_moved:
            self._text(p, 12, AMBER)
            p.drawText(QRectF(centre.x() - 120, centre.y() + 22, 240, 24), Qt.AlignmentFlag.AlignCenter,
                       "keep your head still")
        elif self._phase == "collect" and self._clock.elapsed() > COLLECT_MS:
            self._text(p, 12, GREEN)                      # extended because of blinks: say why the dot stays
            p.drawText(QRectF(centre.x() - 100, centre.y() + 22, 200, 24), Qt.AlignmentFlag.AlignCenter,
                       "keep looking...")

        n, total = self._active.progress()
        step = "Dot" if self.method.kind == "grid" else "Step 1: line dot"
        self._text(p, 12, GREY)
        p.drawText(20, 34, f"{self.method.name}     {step} {n} / {total}     Space: pause     Esc: leave")


# =====================================================================
# RUNNER: real camera by default, or `--mouse` to test the page alone
# =====================================================================

def fake_features_from_mouse() -> tuple[float, float]:
    """Stand-in for real eye features: the mouse position as 0..1 fractions."""
    screen = QApplication.primaryScreen().geometry()
    pos = QCursor.pos()
    return pos.x() / screen.width(), pos.y() / screen.height()


def main() -> int:
    from eyetracking.services.storage import save_calibration
    from eyetracking.services.tracker import EyeTracker

    app = QApplication(sys.argv)
    method = METHOD_BY_ID[next((a for a in sys.argv[1:] if a in METHOD_BY_ID), DEFAULT_METHOD)]
    if "--mouse" in sys.argv:
        tracker, get_features = None, fake_features_from_mouse
    else:
        tracker = EyeTracker()
        get_features = tracker.read_features
    window = CalibrationWindow(method, get_features)

    def on_finished(rec: CalibrationRecording) -> None:
        print(f"{method.name}: {len(rec.sweep)} step-1 frames, {len(rec.head)} head frames, {len(rec.dots)} dot samples")
        if tracker is None:
            print("--mouse mode: fake data, not saved")
        elif rec.dots or rec.head:
            print(f"Saved to {save_calibration(rec, method.id)}")
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
