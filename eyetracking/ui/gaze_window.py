"""
Live gaze demo: a dot on a full-screen window where the model thinks you look.

    python -m eyetracking.ui.gaze_window                    (all calibrations, newest weighted most)
    python -m eyetracking.ui.gaze_window data/<file>.json   (just that one)
    python -m eyetracking.ui.gaze_window --mouse            (no camera: dot follows the mouse)

The faint rings are the calibration positions (5x5 grid). Look at one and see where
the dot lands: a quick visual accuracy check. Press Enter to record a sample
(where the dot is right now). Samples are listed as "#index  x  y" on screen
and saved to data/gaze_checks/gaze_check_<time>.csv on Esc; nothing is saved without one.
Space pauses.

Like CalibrationWindow, this window does NOT know about the camera or the
model: it calls `get_point()` every tick and draws whatever comes back.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.calibration import point_grid
from eyetracking.core.features import N_FEATURES
from eyetracking.core.gaze_model import GazeModel
from eyetracking.services.storage import load_all_calibrations, load_calibration, save_gaze_check
from eyetracking.services.tracker import EyeTracker

TICK_MS = 33         # ~30 checks a second; the camera read limits it anyway
DOT_RADIUS = 18      # px
RING_RADIUS = 12     # px, the faint calibration positions
SHOWN_SAMPLES = 12   # how many of the latest "#index x y" lines are drawn

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
        self._targets = list(targets or point_grid())
        self._point: tuple[float, float] | None = None   # last prediction, 0..1
        self._stale = True                                # no prediction this tick
        self.log: list[tuple[int, float, float]] = []   # (index, x, y), one per Enter

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
        """Enter: one sample of where the dot is now (index first, then x, y)."""
        if self._point is not None:
            self.log.append((len(self.log), *self._point))
            self.update()

    def save_log(self, folder: Path | None = None) -> Path:
        """Every sample as index,x,y, to data/gaze_checks (or `folder`): to compare after a test."""
        return save_gaze_check(self.log, folder)

    def sample_lines(self) -> list[str]:
        """The latest samples, newest last: "#0042  x 0.430  y 0.610" (index first)."""
        return [f"#{i:04d}  x {x:.3f}  y {y:.3f}" for i, x, y in self.log[-SHOWN_SAMPLES:]]

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

        # where the calibration dots were: look at one to check the accuracy
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(70, 70, 70), 2))
        for tx, ty in self._targets:
            p.drawEllipse(QPointF(tx * self.width(), ty * self.height()),
                          RING_RADIUS, RING_RADIUS)

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
        return (f"{gaze}     {self._fps:.0f} fps     {len(self.log)} samples (Enter records one)"
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
        if files:
            samples, weights = load_calibration(Path(files[0])), None
        else:
            samples, weights = load_all_calibrations(N_FEATURES)
        if not samples:
            print("No calibration saved yet: run the calibration window first")
            return 1

        model = GazeModel().fit(samples, weights)
        print(f"Trained on {len(samples)} samples")
        tracker = EyeTracker()             # asks which camera, in the console

        def get_point() -> tuple[float, float] | None:
            # the same features the model was trained on, then a prediction
            features = tracker.read_features()
            return None if features is None else model.predict(features)

    window = GazeWindow(get_point=get_point)
    window.showFullScreen()   # full screen, like calibration: same 0..1 -> pixels
    try:
        return app.exec()
    finally:
        if tracker is not None:
            tracker.close()                # camera light off, MediaPipe freed


if __name__ == "__main__":
    sys.exit(main())
