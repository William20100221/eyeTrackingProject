"""
Floating assistant overlay (PySide6) - desktop-assistant style.

  * FloatingBall    - avatar puck that snaps half-hidden to a screen edge,
                      drag to move, click to open the main window, right-click for a menu.
  * GazeCursor      - click-through dot with a dwell ring.
  * DwellController - stare at a button long enough and it clicks.

The chat lives in the main window now (ui/control_window.py, ChatPage).
The ball and the main window swap: only one of them is ever on screen.

=============================================================================
 PORTS - the complete interface to the rest of the project
=============================================================================
IN  (call these methods)            OUT (connect these signals)
---------------------------------   ---------------------------------------
ball.set_tracking_state(found)      ball.clicked()
dwell.feed(QPoint)   <- main one    dwell.activated(QWidget)

Find each one with:   grep -n "EYE TRACKING HOOK" assistant_overlay.py

  HOOK 1  FloatingBall.set_tracking_state(face_found)  <- FrameResult.face_found
  HOOK 2  DwellController.feed(QPoint(x, y))           <- your gaze point
  HOOK 5  blink-to-click, using FrameResult.blink_score_left / _right
  HOOK 6  main() - worker-thread wiring, camera to widgets, done safely

Run:  python assistant_overlay.py        (draws the ball; ports sit idle)
"""
from __future__ import annotations

import sys

from PySide6.QtCore import (
    QElapsedTimer, QEasingCurve, QObject, QPoint, QPointF, QPropertyAnimation,
    QRectF, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QMenu, QTextEdit, QWidget,
)

# --------------------------------------------------------------------------- theme
ACCENT   = "#3D6CF6"
CARD_BG  = "#FFFFFF"
BORDER   = "#E4E8F4"
TEXT     = "#1F2328"
HOVER    = "#ECEEF3"

# Halo colours around the floating ball - see HOOK 1 (set_tracking_state).
TRACK_IDLE  = QColor(61, 108, 246)   # blue  - tracker not running yet
TRACK_LIVE  = QColor(34, 177, 106)   # green - face found this frame
TRACK_LOST  = QColor(235, 149, 46)   # amber - face lost, gaze is stale


# --------------------------------------------------------------------------- icons
def draw_icon(p: QPainter, name: str, r: QRectF, color: QColor, w: float = 1.7) -> None:
    """Vector icons drawn in normalised 0..1 space - no icon files needed."""

    def P(fx: float, fy: float) -> QPointF:
        return QPointF(r.x() + fx * r.width(), r.y() + fy * r.height())

    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    pen = QPen(color, w)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.BrushStyle.NoBrush)

    if name == "eye":
        path = QPainterPath(P(0.04, 0.50))
        path.cubicTo(P(0.30, 0.08), P(0.70, 0.08), P(0.96, 0.50))
        path.cubicTo(P(0.70, 0.92), P(0.30, 0.92), P(0.04, 0.50))
        p.drawPath(path)
        p.drawEllipse(P(0.50, 0.50), 0.16 * r.width(), 0.16 * r.width())
    p.restore()


# --------------------------------------------------------------------------- ball
class FloatingBall(QWidget):
    """Avatar puck: drag anywhere, snaps half-hidden to the nearest screen edge."""

    # PORT (inbound): set_tracking_state(face_found) - see HOOK 1 below.

    BALL = 56
    PAD = 12  # room for the shadow

    clicked = Signal()             # PORT (outbound): the ball was clicked, open the main window

    def __init__(self, avatar: str | None = None):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(self.BALL + 2 * self.PAD, self.BALL + 2 * self.PAD)

        self.avatar = QPixmap(avatar) if avatar else QPixmap()
        self._press: QPoint | None = None
        self._moved = False
        self._edge = "right"
        self._halo = TRACK_IDLE          # see set_tracking_state() / HOOK 1
        self.menu_actions: list = []     # (text, slot) pairs the app adds to the right-click menu

        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(180)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._pulse = 0.0
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._tick)
        self._ticker.start(40)

    def _tick(self) -> None:
        self._pulse = (self._pulse + 0.02) % 1.0
        self.update()

    # ======================= EYE TRACKING HOOK 1 =========================
    # Call this once per frame with FrameResult.face_found. The halo around
    # the ball turns green while the tracker sees a face and amber when it
    # loses one, so you can tell at a glance whether gaze data is trustworthy.
    #
    #   from eyetracking.core.models import FrameResult
    #
    #   def on_frame(result: FrameResult) -> None:
    #       ball.set_tracking_state(result.face_found)
    #
    # MUST be called on the GUI thread - if your landmarker runs in a worker
    # thread, emit a Qt Signal and connect it to this method (see HOOK 6).
    # =====================================================================
    def set_tracking_state(self, face_found: bool | None) -> None:
        """True = face found, False = face lost, None = tracker not running."""
        if face_found is None:
            self._halo = TRACK_IDLE
        else:
            self._halo = TRACK_LIVE if face_found else TRACK_LOST
        self.update()

    # ---- painting ------------------------------------------------------
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        circle = QRectF(self.PAD, self.PAD, self.BALL, self.BALL)

        # hand-painted shadow (graphics effects are unreliable on translucent windows)
        glow = QRadialGradient(circle.center(), self.BALL / 2 + self.PAD)
        glow.setColorAt(0.70, QColor(15, 23, 42, 55))
        glow.setColorAt(1.00, QColor(15, 23, 42, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(circle.adjusted(-self.PAD, -self.PAD, self.PAD, self.PAD))

        p.setBrush(QColor("white"))          # white ring
        p.drawEllipse(circle)

        inner = circle.adjusted(4, 4, -4, -4)
        if not self.avatar.isNull():
            clip = QPainterPath()
            clip.addEllipse(inner)
            p.setClipPath(clip)
            p.drawPixmap(inner.toRect(), self.avatar.scaled(
                inner.size().toSize(), Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation))
            p.setClipping(False)
        else:
            grad = QLinearGradient(inner.topLeft(), inner.bottomRight())
            grad.setColorAt(0.0, QColor("#7FA6FF"))
            grad.setColorAt(1.0, QColor(ACCENT))
            p.setBrush(grad)
            p.drawEllipse(inner)
            draw_icon(p, "eye", inner.adjusted(10, 10, -10, -10), QColor("white"), 2.2)

        # Breathing halo. Its colour is the tracker status set by HOOK 1.
        alpha = int(70 * (1 - abs(self._pulse * 2 - 1)))
        grow = 3 + self._pulse * 5
        halo = QColor(self._halo)
        halo.setAlpha(alpha)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(halo, 2))
        p.drawEllipse(circle.adjusted(-grow, -grow, grow, grow))

    # ---- interaction ---------------------------------------------------
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.globalPosition().toPoint() - self.pos()
            self._moved = False

    def mouseMoveEvent(self, e):
        if self._press is not None:
            target = e.globalPosition().toPoint() - self._press
            if (target - self.pos()).manhattanLength() > 3:
                self._moved = True
            self.move(target)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = None
            if self._moved:
                self.snap_to_edge()
            else:
                self.clicked.emit()

    def enterEvent(self, _e):
        self.slide(peek=False)

    def leaveEvent(self, _e):
        self.slide(peek=True)

    def contextMenuEvent(self, e):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:10px;
                     padding:6px; color:{TEXT}; font-size:13px; }}
            QMenu::item {{ padding:7px 26px 7px 14px; border-radius:6px; }}
            QMenu::item:selected {{ background:{HOVER}; }}
            QMenu::separator {{ height:1px; background:{BORDER}; margin:5px 8px; }}
        """)
        # To add your own entry (e.g. "Calibrate"), append a QAction here and
        # connect it to a real slot - do not add one that does nothing:
        #
        #   act = QAction("Calibrate", menu)
        #   act.triggered.connect(self.window().start_calibration)
        #   menu.addAction(act)
        menu.addAction("Open assistant", self.clicked.emit)
        for text, slot in self.menu_actions:
            menu.addAction(text, slot)
        menu.addSeparator()
        quit_act = QAction("Quit", menu)
        quit_act.triggered.connect(QApplication.quit)
        menu.addAction(quit_act)
        menu.exec(e.globalPos())

    # ---- edge docking ---------------------------------------------------
    def snap_to_edge(self) -> None:
        area = self._screen_area()
        c = self.frameGeometry().center()
        distance = {"left": c.x() - area.left(), "right": area.right() - c.x(),
                    "top": c.y() - area.top(), "bottom": area.bottom() - c.y()}
        self._edge = min(distance, key=distance.get)
        self.slide(peek=True)

    def slide(self, peek: bool) -> None:
        area = self._screen_area()
        hidden = self.BALL // 2 if peek else 0
        x, y = self.x(), self.y()
        if self._edge == "right":
            x = area.right() - self.width() + self.PAD + hidden
        elif self._edge == "left":
            x = area.left() - self.PAD - hidden
        elif self._edge == "top":
            y = area.top() - self.PAD - hidden
        else:
            y = area.bottom() - self.height() + self.PAD + hidden
        self._slide.stop()
        self._slide.setStartValue(self.pos())
        self._slide.setEndValue(QPoint(x, y))
        self._slide.start()

    def dock_right(self) -> None:
        area = self._screen_area()
        self._edge = "right"
        self.move(area.right() - self.width() + self.PAD + self.BALL // 2,
                  area.center().y() - self.height() // 2)

    def _screen_area(self):
        screen = QApplication.screenAt(self.frameGeometry().center()) or QApplication.primaryScreen()
        return screen.availableGeometry()


# ------------------------------------------------------- eye-tracking glue
class GazeCursor(QWidget):
    """Click-through dot with a dwell-progress ring."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedSize(52, 52)
        self.progress = 0.0

    def move_to(self, global_pos: QPoint) -> None:
        self.move(global_pos - QPoint(self.width() // 2, self.height() // 2))

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(6, 6, self.width() - 12, self.height() - 12)
        p.setPen(QPen(QColor(61, 108, 246, 70), 2))
        p.drawEllipse(r)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(61, 108, 246, 140))
        p.drawEllipse(r.center(), 5, 5)
        if self.progress > 0:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(QColor(ACCENT), 3))
            p.drawArc(r, 90 * 16, int(-self.progress * 360 * 16))


class DwellController(QObject):
    """Feed gaze points in; it clicks whatever is stared at long enough."""

    # ---------------------------- PORTS ----------------------------------
    activated = Signal(QWidget)    # fired just before the dwell click lands
    # Inbound: feed(QPoint) - see HOOK 2 below. This is the main entry point
    # for your tracker.
    # ---------------------------------------------------------------------

    def __init__(self, cursor: GazeCursor, dwell_ms: int = 800, radius: int = 40):
        super().__init__()
        self.cursor, self.dwell_ms, self.radius = cursor, dwell_ms, radius
        self._target: QWidget | None = None
        self._anchor = QPoint()
        self._clock = QElapsedTimer()
        self._clock.start()

    # ======================= EYE TRACKING HOOK 2 =========================
    # THE MAIN ONE. Call this every time you have a new gaze estimate:
    #
    #   dwell.feed(QPoint(gaze_x, gaze_y))
    #
    # `gaze_x/gaze_y` must be SCREEN pixels, not camera pixels and not the
    # 0..1 normalised coords MediaPipe gives you. Converting is the job of
    # eyetracking/core/gaze_model.py + calibration.py. Until those exist you
    # can prove the UI works by feeding it the mouse from a QTimer:
    #
    #   from PySide6.QtGui import QCursor
    #   t = QTimer(); t.timeout.connect(lambda: dwell.feed(QCursor.pos()))
    #   t.start(33)
    #
    # Smooth the point BEFORE it gets here (app.py: exponential average) -
    # raw gaze jitters several degrees and the dwell ring will never fill.
    # GUI thread only; from the camera thread, emit a Signal (see HOOK 6).
    # =====================================================================
    def feed(self, global_pos: QPoint) -> None:
        self.cursor.move_to(global_pos)
        if not self.cursor.isVisible():
            self.cursor.show()

        target = QApplication.widgetAt(global_pos)
        while target is not None and not isinstance(target, (QAbstractButton, QTextEdit)):
            target = target.parentWidget()

        drifted = (global_pos - self._anchor).manhattanLength() > self.radius
        if target is not self._target or drifted:
            self._target, self._anchor = target, global_pos
            self._clock.restart()
            self.cursor.progress = 0.0
        elif target is not None:
            self.cursor.progress = min(1.0, self._clock.elapsed() / self.dwell_ms)
            if self.cursor.progress >= 1.0:
                self.activated.emit(target)
                self._activate(target)
                self._target = None
                self.cursor.progress = 0.0
        self.cursor.update()

    # ======================= EYE TRACKING HOOK 5 =========================
    # Blink-to-click instead of (or alongside) dwell-to-click. Dwell is safer
    # to start with - blinks fire by accident - but the wiring is this small:
    #
    #   def on_frame(self, result: FrameResult) -> None:
    #       blink = max(result.blink_score_left, result.blink_score_right)
    #       if blink > 0.5 and not self._was_blinking:
    #           if self._target is not None:
    #               self._activate(self._target)     # click what they look at
    #       self._was_blinking = blink > 0.5         # edge-detect, not level
    #
    # Require the blink to last 2-3 frames so normal blinks are ignored.
    # =====================================================================
    @staticmethod
    def _activate(w: QWidget) -> None:
        if isinstance(w, QAbstractButton):
            w.click()
        elif isinstance(w, QTextEdit):
            w.setFocus()


# --------------------------------------------------------------------------- main
def main() -> int:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    ball = FloatingBall()                   # FloatingBall(avatar="avatar.png")
    ball.dock_right()
    ball.show()

    # The gaze pointer and its dwell timer. Nothing feeds them yet - that is
    # your tracker's job, see HOOK 2 / HOOK 6.
    cursor = GazeCursor()
    dwell = DwellController(cursor)

    keep = [ball, cursor, dwell]   # stop Python garbage-collecting them

    # ======================= EYE TRACKING HOOK 6 =========================
    # How to wire the real tracker in. The camera and MediaPipe must NOT run
    # on the GUI thread or the whole overlay freezes; and Qt widgets must NOT
    # be touched from the worker thread. A Signal is the bridge across.
    #
    # app.py reads the camera on the GUI thread for now. If that stutters,
    # a worker object on a QThread would look like this:
    #
    #   from PySide6.QtCore import QObject, QPoint, Signal, QThread
    #   from eyetracking.core.models import FrameResult
    #   from eyetracking.services.camera import ...        # your camera loop
    #   from eyetracking.services.landmarker import FaceLandmarker
    #
    #   class GazeWorker(QObject):
    #       frame_ready = Signal(object)      # carries a FrameResult
    #       gaze_ready  = Signal(QPoint)      # carries a SCREEN-pixel point
    #
    #       def run(self) -> None:
    #           with FaceLandmarker() as landmarker:
    #               for frame, ts in camera_frames():
    #                   result = landmarker.detect(frame, ts)
    #                   self.frame_ready.emit(result)
    #                   if result.face_found:
    #                       x, y = gaze_model.to_screen(result)   # your mapping
    #                       self.gaze_ready.emit(QPoint(int(x), int(y)))
    #
    # Then here in main(), after creating the widgets:
    #
    #   thread = QThread()
    #   worker = GazeWorker()
    #   worker.moveToThread(thread)
    #   thread.started.connect(worker.run)
    #
    #   worker.frame_ready.connect(lambda r: ball.set_tracking_state(r.face_found))
    #   worker.gaze_ready.connect(dwell.feed)
    #
    #   thread.start()
    #   keep += [thread, worker]     # or they get garbage collected mid-run
    # =====================================================================

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
