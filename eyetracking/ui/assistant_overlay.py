"""
Floating assistant overlay (PySide6) - desktop-assistant style.

Two frameless, translucent, always-on-top windows:
  * FloatingBall   - avatar puck that snaps half-hidden to a screen edge,
                     drag to move, click to toggle the panel, right-click for a menu.
  * AssistantPanel - rounded card: icon top bar, message list, composer card
                     with pill buttons and a mic circle.

This file is UI ONLY. It has no tracker, no model, no fake replies. Every
widget left in it does something when you interact with it; the decorative
icons from the original design have been stripped out.

What works right now, with nothing connected:
  * drag the ball, it docks half-hidden to the nearest screen edge
  * hover the ball, it slides out; click it, the panel opens
  * right-click the ball for a menu; minimize dash hides the panel
  * type in the box, press Enter, your text appears as a bubble

=============================================================================
 PORTS - the complete interface to the rest of the project
=============================================================================
IN  (call these methods)            OUT (connect these signals)
---------------------------------   ---------------------------------------
panel.add_message(text, mine)       panel.message_sent(str)
panel.reply(text)                   dwell.activated(QWidget)
ball.set_tracking_state(found)
dwell.feed(QPoint)   <- main one

Each one is explained at the point it is defined. Find them with:

    grep -n "EYE TRACKING HOOK" assistant_overlay.py

  HOOK 1  FloatingBall.set_tracking_state(face_found)  <- FrameResult.face_found
  HOOK 2  DwellController.feed(QPoint(x, y))           <- your gaze point
  HOOK 3  AssistantPanel.on_submit / message_sent      -> your command handler
  HOOK 4  AssistantPanel.reply(text)                   <- your answer
  HOOK 5  blink-to-click, using FrameResult.blink_score_left / _right
  HOOK 6  main() - worker-thread wiring, camera to widgets, done safely

Run:  python assistant_overlay.py        (draws the UI; ports sit idle)
"""
from __future__ import annotations

import sys

from PySide6.QtCore import (
    QElapsedTimer, QEasingCurve, QEvent, QObject, QPoint, QPointF, QPropertyAnimation,
    QRectF, QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QAction, QColor, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient,
)
from PySide6.QtWidgets import (
    QAbstractButton, QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
    QMenu, QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

# --------------------------------------------------------------------------- theme
ACCENT   = "#3D6CF6"
PANEL_BG = "#F7F8FA"
CARD_BG  = "#FFFFFF"
BORDER   = "#E4E8F4"
TEXT     = "#1F2328"
HOVER    = "#ECEEF3"
RADIUS   = 20

# Halo colours around the floating ball - see HOOK 1 (set_tracking_state).
TRACK_IDLE  = QColor(61, 108, 246)   # blue  - tracker not running yet
TRACK_LIVE  = QColor(34, 177, 106)   # green - face found this frame
TRACK_LOST  = QColor(235, 149, 46)   # amber - face lost, gaze is stale

PLACEHOLDER = "Message"


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

    if name == "minimize":
        p.drawLine(P(0.14, 0.52), P(0.86, 0.52))
    elif name == "eye":
        path = QPainterPath(P(0.04, 0.50))
        path.cubicTo(P(0.30, 0.08), P(0.70, 0.08), P(0.96, 0.50))
        path.cubicTo(P(0.70, 0.92), P(0.30, 0.92), P(0.04, 0.50))
        p.drawPath(path)
        p.drawEllipse(P(0.50, 0.50), 0.16 * r.width(), 0.16 * r.width())
    p.restore()


class IconButton(QAbstractButton):
    """Flat icon button with a round hover highlight."""

    def __init__(self, name: str, *, box: int = 32, icon: int = 18,
                 color: str = TEXT, parent: QWidget | None = None):
        super().__init__(parent)
        self._name = name
        self._box, self._icon = box, icon
        self._color = QColor(color)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def sizeHint(self) -> QSize:
        return QSize(self._box, self._box)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect())

        if self.underMouse():
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(HOVER))
            p.drawRoundedRect(rect, rect.height() / 2, rect.height() / 2)

        ix = (self.width() - self._icon) / 2
        iy = (self.height() - self._icon) / 2
        draw_icon(p, self._name, QRectF(ix, iy, self._icon, self._icon), self._color)


# --------------------------------------------------------------------------- panel
class Bubble(QFrame):
    def __init__(self, text: str, mine: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(self.label)
        self.setObjectName("mine" if mine else "theirs")
        self.setStyleSheet(f"""
            QFrame#mine   {{ background:{ACCENT}; border-radius:16px; }}
            QFrame#mine QLabel  {{ color:white; font-size:13px; background:transparent; }}
            QFrame#theirs {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:16px; }}
            QFrame#theirs QLabel {{ color:{TEXT}; font-size:13px; background:transparent; }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)


class Composer(QFrame):
    """The white input card at the bottom. Enter sends, Shift+Enter newlines."""

    submitted = Signal(str)        # PORT: user pressed Enter with text

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("composer")
        self.setStyleSheet(f"""
            QFrame#composer {{ background:{CARD_BG}; border:1px solid #DCE3F7; border-radius:18px; }}
            QTextEdit {{ background:transparent; border:none; color:{TEXT}; font-size:13px; }}
        """)

        self.edit = QTextEdit()
        self.edit.setPlaceholderText(PLACEHOLDER)
        self.edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.edit.setFixedHeight(34)
        self.edit.textChanged.connect(self._grow)
        self.edit.installEventFilter(self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.addWidget(self.edit)

    def _grow(self) -> None:
        h = int(self.edit.document().size().height()) + 8
        self.edit.setFixedHeight(max(34, min(110, h)))

    def eventFilter(self, obj, ev):
        if obj is self.edit and ev.type() == QEvent.Type.KeyPress:
            enter = ev.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            if enter and not (ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
                self.send()
                return True
        return super().eventFilter(obj, ev)

    def send(self) -> None:
        text = self.edit.toPlainText().strip()
        if text:
            self.edit.clear()
            self.submitted.emit(text)


class AssistantPanel(QWidget):
    """Frameless rounded card that floats above everything."""

    # ---------------------------- PORTS ----------------------------------
    message_sent = Signal(str)     # user sent text          -> your handler
    # Inbound direction is not a signal but a method call: add_message() to
    # drop a bubble in instantly, reply() to stream one in.
    # ---------------------------------------------------------------------

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.resize(400, 620)
        self._drag: QPoint | None = None

        # the outer margin is empty space for the drop shadow to paint into
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)

        self.card = QFrame()
        self.card.setObjectName("card")
        self.card.setStyleSheet(f"QFrame#card {{ background:{PANEL_BG}; border-radius:{RADIUS}px; }}")
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(38)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(15, 23, 42, 60))
        self.card.setGraphicsEffect(shadow)
        outer.addWidget(self.card)

        col = QVBoxLayout(self.card)
        col.setContentsMargins(14, 12, 14, 14)
        col.setSpacing(10)

        # ---- top bar: just the minimize dash, which hides the panel
        bar = QHBoxLayout()
        bar.setSpacing(4)
        bar.addStretch(1)
        mini = IconButton("minimize")
        mini.clicked.connect(self.hide_animated)
        bar.addWidget(mini)
        col.addLayout(bar)

        # ---- message list
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setStyleSheet(
            "QScrollArea{background:transparent;}"
            "QScrollBar:vertical{width:6px;background:transparent;margin:0;}"
            "QScrollBar::handle:vertical{background:#D8DCE4;border-radius:3px;min-height:28px;}"
            "QScrollBar::add-line,QScrollBar::sub-line{height:0;}"
        )
        host = QWidget()
        host.setStyleSheet("background:transparent;")
        self.msgs = QVBoxLayout(host)
        self.msgs.setContentsMargins(2, 2, 2, 2)
        self.msgs.setSpacing(10)
        self.msgs.addStretch(1)
        self.scroll.setWidget(host)
        col.addWidget(self.scroll, 1)

        # ---- composer
        self.composer = Composer()
        self.composer.submitted.connect(self.on_submit)
        col.addWidget(self.composer)
        # The panel starts empty. Put the first bubble up yourself with
        # panel.reply("...") once your side is ready.

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(160)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.finished.connect(self._fade_done)

    # ---- messages ------------------------------------------------------
    def add_message(self, text: str, mine: bool) -> Bubble:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        bubble = Bubble(text, mine)
        bubble.setMaximumWidth(268)
        if mine:
            row.addStretch(1)
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch(1)
        self.msgs.insertLayout(self.msgs.count() - 1, row)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return bubble

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # ======================= EYE TRACKING HOOK 3 =========================
    # OUTBOUND PORT. Shows the user's text as a bubble, then hands it off via
    # the message_sent signal. This class never answers anything itself:
    #
    #   panel.message_sent.connect(my_handler)
    #
    #   def my_handler(text: str) -> None:
    #       if text.startswith("/calibrate"):
    #           start_calibration()
    #       panel.reply(answer_for(text))
    # =====================================================================
    def on_submit(self, text: str) -> None:
        self.add_message(text, mine=True)
        self.message_sent.emit(text)

    # ======================= EYE TRACKING HOOK 4 =========================
    # INBOUND PORT: call panel.reply("...") from anywhere to make the
    # assistant speak. Useful for reporting tracker events to the user, e.g.
    #
    #   panel.reply("Calibration finished - average error 1.2 degrees.")
    #   panel.reply("I lost your face. Can you move into the light?")
    #
    # GUI thread only - from a worker thread, emit a Signal connected to it.
    # =====================================================================
    def reply(self, text: str) -> None:
        """Stream a reply in character by character, the way the real one does."""
        bubble = self.add_message("", mine=False)
        state = {"i": 0}
        timer = QTimer(self)

        def tick() -> None:
            state["i"] += 2
            bubble.label.setText(text[:state["i"]])
            self._scroll_to_bottom()
            if state["i"] >= len(text):
                timer.stop()
                timer.deleteLater()

        timer.timeout.connect(tick)
        timer.start(18)

    # ---- window behaviour ---------------------------------------------
    def show_at(self, anchor_global: QPoint) -> None:
        screen = QApplication.screenAt(anchor_global) or QApplication.primaryScreen()
        area = screen.availableGeometry()
        x = anchor_global.x() - self.width() + 20
        if x < area.left():
            x = anchor_global.x() - 20
        y = anchor_global.y() - self.height() // 2
        x = max(area.left(), min(x, area.right() - self.width()))
        y = max(area.top(), min(y, area.bottom() - self.height()))
        self.move(x, y)
        self.setWindowOpacity(0.0)
        self.show()
        self.raise_()
        self.activateWindow()
        self.composer.edit.setFocus()
        self._animate_to(1.0)

    def hide_animated(self) -> None:
        self._animate_to(0.0)

    def _animate_to(self, end: float) -> None:
        self._fade.stop()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(end)
        self._fade.start()

    def _fade_done(self) -> None:
        if self.windowOpacity() < 0.05:
            self.hide()

    # drag the card around by any empty spot
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag = e.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            self.move(e.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, _e):
        self._drag = None


# --------------------------------------------------------------------------- ball
class FloatingBall(QWidget):
    """Avatar puck: drag anywhere, snaps half-hidden to the nearest screen edge."""

    # PORT (inbound): set_tracking_state(face_found) - see HOOK 1 below.

    BALL = 56
    PAD = 12  # room for the shadow

    def __init__(self, panel: AssistantPanel, avatar: str | None = None):
        super().__init__()
        self.panel = panel
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
                self.toggle_panel()

    def enterEvent(self, _e):
        self.slide(peek=False)

    def leaveEvent(self, _e):
        if not self.panel.isVisible():
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
        show_hide = QAction("Hide assistant" if self.panel.isVisible()
                            else "Open assistant", menu)
        show_hide.triggered.connect(self.toggle_panel)
        menu.addAction(show_hide)
        menu.addSeparator()
        quit_act = QAction("Quit", menu)
        quit_act.triggered.connect(QApplication.quit)
        menu.addAction(quit_act)
        menu.exec(e.globalPos())

    def toggle_panel(self) -> None:
        if self.panel.isVisible():
            self.panel.hide_animated()
        else:
            self.slide(peek=False)
            self.panel.show_at(self.frameGeometry().center())

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
    # Smooth the point BEFORE it gets here (eyetracking/core/smoothing.py) -
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

    panel = AssistantPanel()
    ball = FloatingBall(panel)              # FloatingBall(panel, avatar="avatar.png")
    ball.dock_right()
    ball.show()

    # The gaze pointer and its dwell timer. Nothing feeds them yet - that is
    # your tracker's job, see HOOK 2 / HOOK 6.
    cursor = GazeCursor()
    dwell = DwellController(cursor)

    keep = [panel, ball, cursor, dwell]   # stop Python garbage-collecting them

    # ======================= EYE TRACKING HOOK 6 =========================
    # How to wire the real tracker in. The camera and MediaPipe must NOT run
    # on the GUI thread or the whole overlay freezes; and Qt widgets must NOT
    # be touched from the worker thread. A Signal is the bridge across.
    #
    # In eyetracking/ui/worker.py (currently empty):
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
