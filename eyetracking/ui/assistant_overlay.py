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

import math
import random
import sys
from pathlib import Path

from PySide6.QtCore import (
    QElapsedTimer, QEasingCurve, QObject, QPoint, QPointF, QPropertyAnimation,
    QRectF, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QCursor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QMenu, QTextEdit, QWidget,
)

from PySide6.QtSvg import QSvgRenderer

from eyetracking.ui.water_ball import WaterBallRenderer

MARK_SVG = Path(__file__).with_name("app_mark.svg")   # the logo: the still ball's default face

# --------------------------------------------------------------------------- theme
ACCENT   = "#3D6CF6"
CARD_BG  = "#FFFFFF"
BORDER   = "#E4E8F4"
TEXT     = "#1F2328"
HOVER    = "#ECEEF3"

# Halo colours around the floating ball - see HOOK 1 (set_tracking_state).
TRACK_IDLE  = QColor(150, 150, 150)  # grey  - camera off (nothing needs the gaze right now)
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

    clicked = Signal()             # PORT (outbound): left click (the app decides: open the window, or eyes open/closed)
    open_requested = Signal()      # PORT (outbound): "Open assistant" in the right-click menu
    double_clicked = Signal()      # PORT (outbound): double-click (the app starts / stops recording what you read)

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
        self.locked = False              # Go Reading: no dragging, no sliding away; click and right-click only
        self.eyes_open = False           # while locked: open = recording what you read, closed = paused
        self.window_open = False         # the main window is up: the eye is open too (issue #12)

        # a click is reported only once the double-click interval has passed without a second one
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(QApplication.doubleClickInterval())
        self._click_timer.timeout.connect(self.clicked.emit)

        self._slide = QPropertyAnimation(self, b"pos", self)
        self._slide.setDuration(180)
        self._slide.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._pulse = 0.0
        self._ticker = QTimer(self)
        self._ticker.timeout.connect(self._tick)
        self._ticker.start(40)

        # the look (Settings > Floating ball): "logo" or "image" = still, inside a thin ring of `ring`
        # colour; "water" = the moving liquid eye (ui/water_ball.py, OpenGL; still if there is none)
        self.look, self.ring = "logo", QColor(ACCENT)
        self._water: WaterBallRenderer | None = None
        self.set_look("logo", "", ACCENT)
        self._clock = QElapsedTimer()
        self._clock.start()
        self._last_ms = 0
        self._open = self._open_vel = 0.0         # 0 = drop, 1 = eye (springs, overshoots a little)
        self._hover = self._hover_target = 0.0    # liquid ripples while the mouse is over it
        self._gaze = [0.0, 0.0]                   # where the pupil looks, -1..1, y up
        self._look_at: QPoint | None = None       # gaze point fed by the app (global px)
        self._look_ms = -10_000                   # when it was last fed
        self._blink = -1.0                        # -1 = not blinking, else 0..1 through a blink
        self._next_blink = 4.0                    # seconds until the next idle blink

    def set_look(self, look: str, image: str = "", ring: str = "") -> None:
        """Settings > Floating ball: "water" (moving), "logo" or "image" (still, in a thin ring of `ring`).
        An image that can't be read, or no OpenGL for the water, falls back to the logo."""
        self.look = look
        self.ring = QColor(ring) if ring and QColor(ring).isValid() else QColor(ACCENT)
        if look == "water" and self._water is None:
            try:
                self._water = WaterBallRenderer()
            except RuntimeError as e:
                print(f"Floating ball: still look ({e})")
                self.look = "logo"
        self.avatar = QPixmap(image) if look == "image" and image else QPixmap()
        if self.look != "water" and self.avatar.isNull():
            px = self.BALL * 2                        # the logo, drawn sharp from its SVG
            self.avatar = QPixmap(px, px)
            self.avatar.fill(Qt.GlobalColor.transparent)
            p = QPainter(self.avatar)
            QSvgRenderer(str(MARK_SVG)).render(p)
            p.end()
        self.update()

    # where the eye looks: the app's gaze point if fed lately, else the mouse
    def look_at(self, global_pos: QPoint) -> None:
        self._look_at, self._look_ms = global_pos, self._clock.elapsed()

    def _tick(self) -> None:
        self._pulse = (self._pulse + 0.02) % 1.0
        now = self._clock.elapsed()
        dt = min((now - self._last_ms) / 1000, 0.05)
        self._last_ms = now

        # spring open while hovered or while the main window is up (locked: while recording),
        # closed otherwise: slight overshoot = liquid feel
        target = float(self.eyes_open) if self.locked else max(self._hover_target, float(self.window_open))
        self._open_vel += (170 * (target - self._open) - 18 * self._open_vel) * dt
        self._open += self._open_vel * dt
        self._hover += (self._hover_target - self._hover) * min(1.0, dt * 6)

        point = self._look_at if now - self._look_ms < 500 else QCursor.pos()
        c = self.frameGeometry().center()
        x, y = (point.x() - c.x()) / 400, -(point.y() - c.y()) / 400   # 400 px away = looking fully that way
        length = max(1.0, (x * x + y * y) ** 0.5)
        for i, v in enumerate((x / length, y / length)):
            self._gaze[i] += (v - self._gaze[i]) * min(1.0, dt * 8)

        # an idle blink every 4-8 s while open
        if target and self._open > 0.95:
            self._next_blink -= dt
            if self._next_blink <= 0 and self._blink < 0:
                self._blink = 0.0
        if self._blink >= 0:
            self._blink += dt / 0.22
            if self._blink >= 1:
                self._blink, self._next_blink = -1.0, 4 + random.random() * 4
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

        if self.look == "water" and self._water is not None:
            # the shader's drop fills ~80% of its image; size it so the drop = the ball
            side = self.BALL / 0.8
            lid = 1 - math.sin(math.pi * min(self._blink, 1)) if self._blink >= 0 else 1.0
            px = int(side * self.devicePixelRatioF())
            image = self._water.render(px, self._clock.elapsed() / 1000, max(0.0, self._open) * lid,
                                       self._hover, 1.0, tuple(self._gaze))
            image.setDevicePixelRatio(self.devicePixelRatioF())
            p.drawImage(QPointF(circle.center().x() - side / 2, circle.center().y() - side / 2), image)
            self._draw_halo(p, circle)
            return

        # hand-painted shadow (graphics effects are unreliable on translucent windows)
        glow = QRadialGradient(circle.center(), self.BALL / 2 + self.PAD)
        glow.setColorAt(0.70, QColor(15, 23, 42, 55))
        glow.setColorAt(1.00, QColor(15, 23, 42, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(glow)
        p.drawEllipse(circle.adjusted(-self.PAD, -self.PAD, self.PAD, self.PAD))

        p.setBrush(self.ring)                # the thin ring (Settings > Floating ball > Ring colour)
        p.drawEllipse(circle)

        inner = circle.adjusted(3, 3, -3, -3)
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

        self._draw_halo(p, circle)

    def _draw_halo(self, p: QPainter, circle: QRectF) -> None:
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

    def set_locked(self, on: bool) -> None:
        """Go Reading locks the ball in place with its eyes closed; the window coming back unlocks it."""
        self.locked, self.eyes_open = on, False
        self.slide(peek=not on)

    def mouseMoveEvent(self, e):
        if self._press is not None and not self.locked:
            target = e.globalPosition().toPoint() - self._press
            if (target - self.pos()).manhattanLength() > 3:
                self._moved = True
            self.move(target)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._press is not None:
            self._press = None
            if self._moved:
                self.snap_to_edge()
            else:
                self._click_timer.start()     # a single click, unless a second one follows

    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._click_timer.stop()          # not a single click after all
            self._press = None
            self.double_clicked.emit()

    def enterEvent(self, _e):
        self._hover_target = 1.0
        if not self.locked:
            self.slide(peek=False)

    def leaveEvent(self, _e):
        self._hover_target = 0.0
        if not self.locked:
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
        menu.addAction("Open assistant", self.open_requested.emit)
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
