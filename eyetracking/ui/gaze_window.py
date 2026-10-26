"""
The gaze check, full screen (as calibration-lab's): test rings, one lit at a time, and the model's dot.

    python -m eyetracking.ui.gaze_window                    (all calibrations, newest weighted most)
    python -m eyetracking.ui.gaze_window data/<file>.json   (just that one)
    python -m eyetracking.ui.gaze_window --mouse            (no camera: dot follows the mouse)

A green dot is drawn where the model thinks you look. One of the 16 rings (4x4) is lit:
look at it and press Enter, which records where the model says you look against that
ring and lights the next one (row by row, round again after the last). A sample's error
is its distance to the lit ring. Samples are listed as "#index  x  y  off" on screen and
saved to data/gaze_checks/gaze_check_<time>.csv on Esc; nothing is saved without one.
Space pauses.

Like CalibrationWindow, this window does NOT know about the camera or the
model: it calls `get_point()` every tick and draws whatever comes back
(NO_FRAME: no new camera frame since the last tick, nothing to draw).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.calibration import point_grid
from eyetracking.core.points_model import train
from eyetracking.core.models import NO_FRAME
from eyetracking.core.smoothing import GazeSmoother
from eyetracking.services.storage import CHECK_GRID, Check, load_all_calibrations, load_calibration, save_gaze_check
from eyetracking.services.tracker import EyeTracker

TICK_MS = 16         # asks for the newest camera frame (read on its own thread) at the screen's pace
DOT_RADIUS = 18      # px
RING_RADIUS = 12     # px, the faint test rings
SHOWN_SAMPLES = 12   # how many of the latest "#index x y off" lines are drawn

LIVE = QColor(80, 220, 120)      # green: a prediction from this frame
STALE = QColor(90, 90, 90)       # grey: face lost / blink, showing the last one
OFF_SCREEN = QColor(235, 149, 46)  # orange ring: the guess is past the edge


class GazeWindow(QWidget):
    logged = Signal(str)    # the csv file's path, just before the window closes

    def __init__(self,
                 get_point: Callable[[], tuple[float, float] | None],
                 targets: Sequence[tuple[float, float]] | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._get_point = get_point
        self.paused = False
        self._targets = list(targets or point_grid(CHECK_GRID))
        self._lit = 0                                     # the ring to look at now: Enter records it and lights the next
        self._point: tuple[float, float] | None = None   # last prediction, 0..1
        self._stale = True                                # no prediction this tick
        self.log: list[Check] = []                        # (index, x, y, ring x, ring y), one per Enter

        # frames per second, worked out once a second: the number to compare
        # when testing speed changes (e.g. a threaded camera)
        self._frames = 0
        self._fps = 0.0
        self._fps_clock = QElapsedTimer()
        self._fps_clock.start()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

        self.setWindowTitle("Gaze")
        self.setCursor(Qt.CursorShape.BlankCursor)  # a visible mouse pulls the eye

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.paused = not self.paused
            (self._timer.stop if self.paused else self._timer.start)()
            self.update()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.record()
        elif event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            if self.log:
                self.logged.emit(str(self.save_log()))
            self.close()
        else:
            super().keyPressEvent(event)

    # ---- runs every TICK_MS ----------------------------------------
    def _tick(self) -> None:
        point = self._get_point()
        if point is NO_FRAME:       # the camera's next frame hasn't come: nothing new, no repaint
            return
        self._stale = point is None
        if point is not None:
            self._point = point     # while the face is lost, keep the last one

        self._frames += 1
        if self._fps_clock.elapsed() >= 1000:
            self._fps = self._frames * 1000 / self._fps_clock.elapsed()
            self._frames = 0
            self._fps_clock.restart()

        self.update()   # ask Qt to repaint (it calls paintEvent)

    def record(self) -> None:
        """Enter: one sample of where the dot is now (index first, then x, y) and the lit ring; then the
        next ring lights up (row by row, round again after the last)."""
        if self._point is None:
            return                                        # no face yet: nothing to record
        tx, ty = self._targets[self._lit]
        self.log.append((len(self.log), *self._point, tx, ty))
        self._lit = (self._lit + 1) % len(self._targets)
        self.update()

    def save_log(self, folder: Path | None = None) -> Path:
        """Every sample as index,x,y,target_x,target_y, to data/gaze_checks (or `folder`)."""
        return save_gaze_check(self.log, folder)

    def sample_lines(self) -> list[str]:
        """The latest samples, newest last: "#0042  x 0.430  y 0.610  off 0.032" (index first; off = the
        distance from the lit ring)."""
        return [f"#{i:04d}  x {x:.3f}  y {y:.3f}  off {math.hypot(x - tx, y - ty):.3f}"
                for i, x, y, tx, ty in self.log[-SHOWN_SAMPLES:]]

    def dot_position(self) -> QPointF | None:
        """Where the dot is drawn, in window pixels. A guess past the edge is
        pinned to the edge, so it stays visible."""
        if self._point is None:
            return None
        x, y = self._point
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        return QPointF(x * self.width(), y * self.height())

    # ---- drawing ---------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor("black"))

        if self.paused:
            font = p.font()
            font.setPointSize(18)
            p.setFont(font)
            p.setPen(QColor("white"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Paused.\n\nSpace to carry on, Esc to leave.")
            return

        # the test rings: look at the lit one
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(70, 70, 70), 2))
        for tx, ty in self._targets:
            p.drawEllipse(QPointF(tx * self.width(), ty * self.height()),
                          RING_RADIUS, RING_RADIUS)
        tx, ty = self._targets[self._lit]                  # the ring to look at: bright, larger
        p.setPen(QPen(QColor("white"), 3))
        p.drawEllipse(QPointF(tx * self.width(), ty * self.height()), RING_RADIUS + 6, RING_RADIUS + 6)

        centre = self.dot_position()
        if centre is not None:
            x, y = self._point
            if not (0 <= x <= 1 and 0 <= y <= 1):
                p.setPen(QPen(OFF_SCREEN, 3))
                p.drawEllipse(centre, DOT_RADIUS + 6, DOT_RADIUS + 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(STALE if self._stale else LIVE)
            p.drawEllipse(centre, DOT_RADIUS, DOT_RADIUS)

        font = p.font()
        font.setPointSize(12)
        p.setFont(font)
        p.setPen(QColor(120, 120, 120))
        p.drawText(20, 34, self.status_text())
        for n, line in enumerate(self.sample_lines()):
            p.drawText(20, 64 + 22 * n, line)

    def status_text(self) -> str:
        if self._point is None:
            gaze = "waiting for a face..."
        else:
            x, y = self._point
            gaze = f"x {x:.2f}  y {y:.2f}"
            if self._stale:
                gaze += "   (no face / blink: showing the last point)"
        return (f"{gaze}     {self._fps:.0f} fps     {len(self.log)} samples (look at the lit ring, Enter records one)"
                "     Space: pause     Esc: save and quit")


# =====================================================================
# RUNNER: real camera + model by default, or `--mouse` to test the window
# =====================================================================

def fake_point_from_mouse() -> tuple[float, float]:
    """Stand-in for a gaze prediction: the mouse position as 0..1 fractions."""
    screen = QApplication.primaryScreen().geometry()
    pos = QCursor.pos()
    return pos.x() / screen.width(), pos.y() / screen.height()


def main() -> int:
    app = QApplication(sys.argv)

    if "--mouse" in sys.argv:
        tracker = None
        get_point = fake_point_from_mouse
    else:
        # one file if given, otherwise every saved calibration, the newest weighted most
        files = [a for a in sys.argv[1:] if not a.startswith("--")]
        model, n = train([load_calibration(Path(files[0]))] if files else load_all_calibrations())
        if model is None:
            print("No usable calibration saved yet: run the calibration window first")
            return 1
        print(f"Trained on {n} samples, head turns corrected with {model.k_source}")
        tracker = EyeTracker()             # asks which camera, in the console
        smoother = GazeSmoother()          # the 1-euro filter on the live point, as in the app

        def get_point() -> tuple[float, float] | None:
            # the same features the model was trained on, then a prediction, then the filter
            features = tracker.read_features()
            return smoother.feed(None if features is None else model.predict(features))

    window = GazeWindow(get_point=get_point)
    window.showFullScreen()   # full screen, like calibration: same 0..1 -> pixels
    try:
        return app.exec()
    finally:
        if tracker is not None:
            tracker.close()                # camera light off, MediaPipe freed


if __name__ == "__main__":
    sys.exit(main())
