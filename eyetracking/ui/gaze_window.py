"""
The gaze check, full screen: test rings, one lit at a time, and what the models say.

    python -m eyetracking.ui.gaze_window sweep              (that type's calibrations, newest weighted most)
    python -m eyetracking.ui.gaze_window --mouse            (no camera: dot follows the mouse)

Three ways to run it (app.py picks):
  single     one model: its dot is drawn where it thinks you look; 4x4 rings (CHECK_GRID)
  multiple   every model of a person at once, from the same camera frames: no dot (it would
             show one model only); 6x6 rings (MULTI_GRID); one file per model
  record     no model: the camera is recorded (services/recording.py) with the time of each
             Enter; afterwards any model can be run over the video as if it had been live
One ring is lit: look at it and press Enter, which records where each model says you look
(or the moment, in record mode) and lights the next ring. A sample's error is its distance to
the lit ring. Esc saves: one csv per model in the person's folder of that type (the file name
says how many calibrations trained it); nothing is saved without a sample. Space pauses.

This window does NOT know about the camera or the models: it calls `get_points()` every tick
(a point per model name, None while the face is lost) and draws whatever comes back.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.calibration import point_grid
from eyetracking.core.line_model import ScreenGeometry
from eyetracking.core.methods import DEFAULT_METHOD, METHOD_BY_ID, live_clock, train
from eyetracking.core.models import NO_FRAME
from eyetracking.services.storage import CHECK_GRID, Check, load_all_calibrations, save_gaze_check
from eyetracking.services.tracker import EyeTracker

TICK_MS = 16         # asks for the newest camera frame (read on its own thread) at the screen's pace
DOT_RADIUS = 18      # px
RING_RADIUS = 12     # px, the faint test rings
SHOWN_SAMPLES = 12   # how many of the latest "#index x y" lines are drawn

LIVE = QColor(80, 220, 120)      # green: a prediction from this frame
STALE = QColor(90, 90, 90)       # grey: face lost / blink, showing the last one
OFF_SCREEN = QColor(235, 149, 46)  # orange ring: the guess is past the edge
Point = tuple[float, float]


class GazeWindow(QWidget):
    logged = Signal(object)     # list[Path]: the csv files saved, just before the window closes
    recorded = Signal(object)   # Path: the recording's json (record mode), just before the window closes

    def __init__(self, get_points: Callable[[], dict[str, Point | None]], folders: dict[str, Path | None],
                 calibrations: dict[str, int | None] | None = None, grid: int = CHECK_GRID,
                 show_dot: bool = True, tag: str = "", recorder=None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._get_points = get_points
        self.folders = dict(folders)                 # model name -> where its csv goes (None: the type's folder)
        self.calibrations = calibrations or {}       # model name -> how many calibrations trained it
        self.grid = grid
        self.show_dot = show_dot and len(folders) == 1   # the dot is one model's view: only when there is one
        self.tag = tag                               # goes in the file names ("multi")
        self.recorder = recorder                     # record mode: gets every press, finishes on Esc
        self._targets = point_grid(grid)
        self._lit = 0                                # the ring to look at now: Enter records it and lights the next
        self.presses = 0                             # Enter presses so far: the sample index, the same for every model
        self._pressed: list[tuple[int, int]] = []    # (press index, lit ring), for the lines on screen
        self.paused = False
        self._points: dict[str, Point | None] = {name: None for name in folders}   # the last prediction per model
        self._stale = True                                                        # no prediction this tick
        self.logs: dict[str, list[Check]] = {name: [] for name in folders}       # (index, x, y, ring x, ring y)

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

    @classmethod
    def single(cls, get_point: Callable[[], Point | None], method_id: str = "model", folder: Path | None = None,
               calibrations: int | None = None, grid: int = CHECK_GRID) -> GazeWindow:
        """One model, its dot drawn."""
        def points():
            p = get_point()
            return NO_FRAME if p is NO_FRAME else {method_id: p}
        return cls(points, {method_id: folder}, {method_id: calibrations}, grid)

    @property
    def names(self) -> list[str]:
        return list(self.folders)

    @property
    def log(self) -> list[Check]:
        """The first model's samples (the one whose dot is drawn)."""
        return self.logs[self.names[0]] if self.names else []

    @property
    def point(self) -> Point | None:
        return self._points[self.names[0]] if self.names else None

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self.paused = not self.paused
            (self._timer.stop if self.paused else self._timer.start)()
            self.update()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.record()
        elif event.key() == Qt.Key.Key_Escape:
            self._timer.stop()
            if self.recorder is not None:
                meta = self.recorder.finish()
                if meta is not None:
                    self.recorded.emit(meta)
            elif any(self.logs.values()):
                self.logged.emit(self.save_log())
            self.close()
        else:
            super().keyPressEvent(event)

    # ---- runs every TICK_MS ----------------------------------------
    def _tick(self) -> None:
        points = self._get_points()
        if points is NO_FRAME:                       # the camera's next frame hasn't come: nothing new
            return
        self._stale = all(p is None for p in points.values()) if points else True
        for name, p in points.items():
            if p is not None:
                self._points[name] = p     # while the face is lost, keep the last one

        self._frames += 1
        if self._fps_clock.elapsed() >= 1000:
            self._fps = self._frames * 1000 / self._fps_clock.elapsed()
            self._frames = 0
            self._fps_clock.restart()

        self.update()   # ask Qt to repaint (it calls paintEvent)

    def record(self) -> None:
        """Enter: one sample per model of where its dot is now (index first, then x, y) and the
        lit ring, or the moment in record mode; then the next ring lights up (row by row, round
        again after the last)."""
        tx, ty = self._targets[self._lit]
        if self.recorder is not None:
            self.recorder.press(self.presses, tx, ty)
        else:
            got = [name for name, p in self._points.items() if p is not None]
            if not got:
                return                                   # no face yet: nothing to record
            for name in got:
                self.logs[name].append((self.presses, *self._points[name], tx, ty))
        self._pressed.append((self.presses, self._lit))
        self.presses += 1
        self._lit = (self._lit + 1) % len(self._targets)
        self.update()

    def save_log(self, folder: Path | None = None) -> list[Path]:
        """Every model's samples as index,x,y,target_x,target_y, to its folder (or `folder`)."""
        return [save_gaze_check(log, name, folder or self.folders[name], self.calibrations.get(name), self.tag)
                for name, log in self.logs.items() if log]

    def sample_lines(self) -> list[str]:
        """The latest samples, newest last: "#0042  x 0.430  y 0.610  off 0.032" (index first; off = from
        the ring) for the drawn model; without a dot, the ring each press was on."""
        if self.show_dot:
            return [f"#{i:04d}  x {x:.3f}  y {y:.3f}  off {math.hypot(x - tx, y - ty):.3f}"
                    for i, x, y, tx, ty in self.log[-SHOWN_SAMPLES:]]
        return [f"#{i:04d}  ring ({lit % self.grid + 1}, {lit // self.grid + 1})" for i, lit in self._pressed[-SHOWN_SAMPLES:]]

    def dot_position(self) -> QPointF | None:
        """Where the dot is drawn, in window pixels. A guess past the edge is
        pinned to the edge, so it stays visible."""
        if self.point is None:
            return None
        x, y = self.point
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

        centre = self.dot_position() if self.show_dot else None
        if centre is not None:
            x, y = self.point
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
        if self.recorder is not None:
            gaze = f"recording the camera: {len(self.recorder.frames_ms)} frames"
        elif not self.show_dot:
            gaze = f"{len(self.names)} models predicting" + ("   (no face / blink: holding the last points)" if self._stale else "")
        elif self.point is None:
            gaze = "waiting for a face..."
        else:
            x, y = self.point
            gaze = f"x {x:.2f}  y {y:.2f}" + ("   (no face / blink: showing the last point)" if self._stale else "")
        return (f"{gaze}     {self._fps:.0f} fps     {self.presses} samples (Enter records one)"
                "     Space: pause     Esc: save and quit")


# =====================================================================
# RUNNER: real camera + model by default, or `--mouse` to test the window
# =====================================================================

def fake_point_from_mouse() -> tuple[float, float]:
    """Stand-in for a gaze prediction: the mouse position as 0..1 fractions."""
    screen = QApplication.primaryScreen().geometry()
    pos = QCursor.pos()
    return pos.x() / screen.width(), pos.y() / screen.height()


def screen_geometry() -> ScreenGeometry:
    """The real screen size in cm, from Windows (via Qt). Needs a running Qt app.
    The webcam is assumed to sit CAMERA_ABOVE_SCREEN_CM above the screen's top
    edge, centred: change that constant in core/line_model.py if yours isn't."""
    mm = QApplication.primaryScreen().physicalSize()
    return ScreenGeometry(mm.width() / 10, mm.height() / 10)


def main() -> int:
    app = QApplication(sys.argv)

    method = METHOD_BY_ID[next((a for a in sys.argv[1:] if a in METHOD_BY_ID), DEFAULT_METHOD)]
    if "--mouse" in sys.argv:
        tracker = None
        get_point = fake_point_from_mouse
    else:
        model, n = train(method, load_all_calibrations(method.id), screen_geometry())
        if model is None:
            print(f"No usable {method.name} calibration yet: calibrate first")
            return 1
        print(f"{method.name}: trained on {n} samples")
        tracker = EyeTracker()             # asks which camera, in the console
        track = model.tracker()            # the smoothed live point

        def get_point() -> tuple[float, float] | None:
            return track(tracker.read_features(), live_clock())

    window = GazeWindow.single(get_point, method.id)
    window.showFullScreen()   # full screen, like calibration: same 0..1 -> pixels
    try:
        return app.exec()
    finally:
        if tracker is not None:
            tracker.close()                # camera light off, MediaPipe freed


if __name__ == "__main__":
    sys.exit(main())
