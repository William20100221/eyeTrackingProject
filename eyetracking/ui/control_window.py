"""Study Assistant main window: direction 3 ("Toolbar") in PySide6.

Two tabs in the title bar: Settings, and Calibration. Calibration (and check)
pages fill the screen with no title bar or buttons, then the window goes back
to how it was. pick_camera() is the small camera popup used at first start.

Every colour comes from one seed colour through the Material 3 Fidelity
scheme, in light and dark. Pick a colour in the side sheet and the whole
window re-themes live.

    pip install PySide6 materialyoucolor
    python settings_window.py
"""
import ctypes
import sys
from string import Template

from materialyoucolor.dynamiccolor.material_dynamic_colors import MaterialDynamicColors
from materialyoucolor.hct import Hct
from materialyoucolor.scheme.scheme_fidelity import SchemeFidelity
from PySide6.QtCore import (QEasingCurve, QEvent, QPointF, QRectF, QRegularExpression, QSettings, Qt, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QRegularExpressionValidator)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QMenu, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from eyetracking.services.camera import camera_names

DEFAULTS = {
    "seed": "#3D6CF6",
    "theme": "system",          # system | light | dark
    "show_ball": True,
    "ball_on_top": True,
    "hide_fullscreen": False,
    "model": "Claude Haiku 4.5",
    "summary_words": 150,
    "camera": "",               # webcam NAME (Windows can renumber cameras between restarts)
}
SWATCHES = ["#3D6CF6", "#00897B", "#2E8B57", "#E8B400", "#F4511E", "#D32F2F", "#D81B60", "#7E57C2"]
# name shown -> API model id ("Haiku 5.5" / "Sonnet 5.5" don't exist: the API would refuse them)
MODEL_IDS = {"Claude Haiku 4.5": "claude-haiku-4-5", "Claude Sonnet 5": "claude-sonnet-5",
             "Claude Opus 5.5": "claude-opus-5-5"}
MODELS = list(MODEL_IDS)

TEXT_FONTS = ["Segoe UI Variable Text", "Segoe UI", "Segoe Fluent Icons", "Segoe MDL2 Assets"]
ICON_FONTS = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]   # built into Windows 11 / Windows 10

# Segoe Fluent Icons code points
I_SETTINGS, I_SEARCH = "", ""
I_MIN, I_MAX, I_RESTORE, I_CLOSE = "", "", "", ""
I_CHECK, I_DISMISS, I_RESET, I_CHEVRON = "", "", "", ""
I_THEME, I_PALETTE, I_BALL, I_LAYERS = "", "", "", ""
I_FULLSCREEN, I_SPARKLE, I_LINES = "", "", ""
I_EYE, I_CAMERA = "", ""   # View (eye), Camera


# ---------------------------------------------------------------- colour

ROLES = ["primary", "onPrimary", "primaryContainer", "onPrimaryContainer", "secondary", "secondaryContainer",
         "onSecondaryContainer", "tertiary", "surface", "onSurface", "onSurfaceVariant", "surfaceContainerLowest",
         "surfaceContainerLow", "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest", "outline",
         "outlineVariant", "inverseSurface", "inverseOnSurface"]
_MDC = MaterialDynamicColors(spec="2021")


def make_scheme(seed: str, dark: bool) -> dict:
    """Seed hex -> {M3 role: hex}. Same spec as the HTML mockup (material-color-utilities 0.3)."""
    s = SchemeFidelity(Hct.from_int(0xFF000000 | int(seed[1:], 16)), dark, 0.0, spec_version="2021")
    return {r: "#%06X" % (getattr(_MDC, r).get_argb(s) & 0xFFFFFF) for r in ROLES}


def mix(fg: str, bg: str, a: float) -> str:
    """fg laid over bg at opacity a (an M3 state layer on a solid colour)."""
    f, b = QColor(fg), QColor(bg)
    return QColor.fromRgbF(*(x * (1 - a) + y * a for x, y in
                             zip((b.redF(), b.greenF(), b.blueF()), (f.redF(), f.greenF(), f.blueF())))).name()


def rgba(hex_: str, a: float) -> str:
    c = QColor(hex_)
    return f"rgba({c.red()},{c.green()},{c.blue()},{a})"


def glyph_icon(ch: str, color: str, px: int = 16) -> QIcon:
    pm = QPixmap(px * 2, px * 2)
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    f = QFont()
    f.setFamilies(ICON_FONTS)
    f.setPixelSize(px)
    p.setFont(f)
    p.setPen(QColor(color))
    p.drawText(QRectF(0, 0, px, px), Qt.AlignCenter, ch)
    p.end()
    return QIcon(pm)


QSS = Template("""
#root { background: $surface; }
QWidget { color: $onSurface; font-size: 14px; }
QToolTip { background: $inverseSurface; color: $inverseOnSurface; border: none; padding: 4px 8px; font-size: 12px; }

/* title bar */
#titlebar { background: $surfaceContainer; }
#appname { font-size: 12px; }
#caps QPushButton { font-family: "Segoe Fluent Icons"; font-size: 10px; font-weight: 400; border: none; border-radius: 0;
    min-width: 46px; max-width: 46px; min-height: 48px; max-height: 48px; padding: 0; background: transparent; }
#caps QPushButton:hover { background: $hover; }
#caps QPushButton#close:hover { background: #C42B1C; color: white; }
QPushButton#settingsBtn:checked { background: $secondaryContainer; color: $onSecondaryContainer; }
QPushButton#settingsBtn { font-family: "Segoe Fluent Icons"; font-size: 16px; font-weight: 400; min-width: 40px; max-width: 40px;
    min-height: 32px; max-height: 32px; padding: 0; border-radius: 8px; background: transparent; }
QPushButton#settingsBtn:hover { background: $hover; }
QPushButton#settingsBtn:checked:hover { background: $secContainerHover; }

/* toolbar */
#toolbar { background: $surface; border-bottom: 1px solid $outlineVariant; }
#h1 { font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 20px; font-weight: 600; }
QLineEdit#find { min-height: 34px; max-height: 34px; min-width: 280px; max-width: 280px; padding: 0 6px; border: 1px solid $outlineVariant;
    border-radius: 4px; background: $surfaceContainerLow; selection-background-color: $primary; selection-color: $onPrimary; }
QLineEdit#find:focus { border-bottom: 2px solid $primary; }

/* buttons */
QPushButton { border: none; border-radius: 12px; min-height: 36px; max-height: 36px; padding: 0 16px; font-size: 13px; font-weight: 600; background: transparent; }
QPushButton:focus { border: 2px solid $secondary; }
QPushButton[kind="filled"] { background: $primary; color: $onPrimary; }
QPushButton[kind="filled"]:hover { background: $primaryHover; }
QPushButton[kind="filled"]:pressed { background: $primaryPress; }
QPushButton[kind="outlined"] { border: 1px solid $outlineVariant; padding-left: 12px; }
QPushButton[kind="text"] { color: $primary; padding: 0 12px; }
QPushButton[kind="outlined"]:hover, QPushButton[kind="text"]:hover, QPushButton[kind="icon"]:hover { background: $hover; }
QPushButton[kind="outlined"]:pressed, QPushButton[kind="text"]:pressed, QPushButton[kind="icon"]:pressed { background: $press; }
QPushButton[kind="icon"] { font-family: "Segoe Fluent Icons"; font-size: 14px; font-weight: 400; min-width: 40px; max-width: 40px;
    min-height: 40px; max-height: 40px; padding: 0; border-radius: 20px; color: $onSurfaceVariant; }

/* chips */
QPushButton[kind="chip"], QPushButton[kind="assist"] { min-height: 30px; max-height: 30px; border-radius: 8px;
    border: 1px solid $outlineVariant; padding: 0 14px; color: $onSurfaceVariant; }
QPushButton[kind="assist"] { color: $onSurface; padding: 0 10px 0 8px; }
QPushButton[kind="assist"]::menu-indicator { image: none; width: 0; }
QPushButton[kind="chip"]:hover, QPushButton[kind="assist"]:hover { background: $hover; }
QPushButton[kind="chip"]:checked { background: $secondaryContainer; color: $onSecondaryContainer; border-color: $secondaryContainer; padding-left: 8px; }
QPushButton[kind="chip"]:checked:hover { background: $secContainerHover; }

/* setting groups */
#content { background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; margin: 4px 2px; }
QScrollBar::handle:vertical { background: $outlineVariant; border-radius: 3px; min-height: 32px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
#gh { font-size: 14px; font-weight: 600; }
#group { background: $surfaceContainerLowest; border: 1px solid $outlineVariant; border-radius: 8px; }
#divider { background: $outlineVariant; border: none; min-height: 1px; max-height: 1px; }
#rowicon { font-family: "Segoe Fluent Icons"; font-size: 16px; color: $onSurfaceVariant; }
#rowsub, #hex, #unit { font-size: 12px; color: $onSurfaceVariant; }
#hex, #unit { font-size: 13px; }
QSpinBox { min-height: 30px; max-height: 30px; min-width: 72px; max-width: 72px; padding: 0 10px; border: 1px solid $outlineVariant;
    border-bottom: 1px solid $onSurfaceVariant; border-radius: 4px; background: $surfaceContainerLow; }
QSpinBox:focus { border-bottom: 2px solid $primary; }
QMenu { background: $surfaceContainer; border: 1px solid $outlineVariant; padding: 4px; }
QMenu::item { padding: 8px 32px 8px 12px; border-radius: 4px; }
QMenu::item:selected { background: $hover; }

/* side sheet */
#sheet { background: $surfaceContainerLow; border-left: 1px solid $outlineVariant; border-top-left-radius: 12px; }
#sheetTitle { font-size: 16px; font-weight: 600; }
#sheetText { font-size: 13px; color: $onSurfaceVariant; }
QLineEdit#outlined { padding: 0 15px; border: 1px solid $outline; border-radius: 4px;
    background: transparent; font-size: 15px; selection-background-color: $primary; selection-color: $onPrimary; }
QLineEdit#outlined:hover { border-color: $onSurface; }
QLineEdit#outlined:focus { border: 2px solid $primary; padding: 0 14px; }
#floatLabel { font-size: 12px; color: $onSurfaceVariant; background: $surfaceContainerLow; padding: 0 4px; }
#floatLabel[focused="true"] { color: $primary; }
""")


def build_qss(c: dict) -> str:
    return QSS.substitute(
        c,
        hover=rgba(c["onSurface"], .08),
        press=rgba(c["onSurface"], .12),
        primaryHover=mix(c["onPrimary"], c["primary"], .08),
        primaryPress=mix(c["onPrimary"], c["primary"], .12),
        secContainerHover=mix(c["onSecondaryContainer"], c["secondaryContainer"], .08),
    )


# ---------------------------------------------------------------- small widgets

def label(text: str, name: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName(name)
    return lbl


def button(text: str, kind: str) -> QPushButton:
    b = QPushButton(text)
    b.setProperty("kind", kind)
    b.setCursor(Qt.PointingHandCursor)
    b.setFocusPolicy(Qt.TabFocus)   # focus ring only for keyboard users
    return b


def draw_check(p: QPainter, cx: float, cy: float, s: float, color: QColor, width: float = 2):
    path = QPainterPath(QPointF(cx - .42 * s, cy))
    path.lineTo(cx - .12 * s, cy + .3 * s)
    path.lineTo(cx + .42 * s, cy - .3 * s)
    p.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)


class CheckButton(QPushButton):
    """Chip that shows a check mark while selected."""

    def __init__(self, text: str, kind: str | None = None):
        super().__init__(text)
        self.plain = text
        if kind:
            self.setProperty("kind", kind)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.toggled.connect(lambda on: self.setText(f"{I_CHECK}  {self.plain}" if on else self.plain))


class Switch(QAbstractButton):
    """M3 switch, drawn at 84% of the 52x32 spec to suit the denser layout."""

    def __init__(self, name: str):
        super().__init__()
        self.setCheckable(True)
        self.setAccessibleName(name)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setAttribute(Qt.WA_Hover)
        self.setFixedSize(50, 34)
        self._t = 0.0
        self._anim = QVariantAnimation(self, duration=250, easingCurve=QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._step)
        self.toggled.connect(self._animate)

    def set_checked_now(self, on: bool):
        self.setChecked(on)
        self._anim.stop()
        self._step(1.0 if on else 0.0)

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._t)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def _step(self, v):
        self._t = v
        self.update()

    def paintEvent(self, _):
        c, on, t = self.window().c, self.isChecked(), self._t
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.translate(3, 4)
        p.scale(.84, .84)
        if self.hasFocus():
            p.setPen(QPen(QColor(c["secondary"]), 3))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(QRectF(-3, -3, 58, 38), 19, 19)
        if on:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(c["primary"]))
        else:
            p.setPen(QPen(QColor(c["outline"]), 2))
            p.setBrush(QColor(c["surfaceContainerHighest"]))
        p.drawRoundedRect(QRectF(1, 1, 50, 30), 15, 15)
        d = 28 if self.isDown() else 16 + 8 * t
        cx = 16 + 20 * t
        hover = self.underMouse()
        thumb = (c["primaryContainer"] if hover else c["onPrimary"]) if on else (c["onSurfaceVariant"] if hover else c["outline"])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(thumb))
        p.drawEllipse(QPointF(cx, 16), d / 2, d / 2)
        if on and t > .6:
            draw_check(p, cx, 16, 10, QColor(c["onPrimaryContainer"]))


class Swatch(QAbstractButton):
    def __init__(self, color: str):
        super().__init__()
        self.color = color
        self.setCheckable(True)
        self.setToolTip(color)
        self.setAccessibleName(f"Theme colour {color}")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setAttribute(Qt.WA_Hover)
        self.setFixedSize(48, 48)

    def paintEvent(self, _):
        c = self.window().c
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        mid = QPointF(24, 24)
        ring = c["onSurface"] if self.isChecked() else c["secondary"] if self.hasFocus() else \
            c["outlineVariant"] if self.underMouse() else None
        if ring:
            p.setPen(QPen(QColor(ring), 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(mid, 23, 23)
        p.setPen(QPen(QColor(0, 0, 0, 30), 1))
        p.setBrush(QColor(self.color))
        p.drawEllipse(mid, 20, 20)
        if self.isChecked():
            light = QColor(self.color).lightnessF() > .6
            draw_check(p, 24, 24, 14, QColor("#1B1B1F" if light else "#FFFFFF"))


class ColorDot(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(20, 20)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        p.setBrush(QColor(self.window().seed))
        p.drawEllipse(QRectF(.5, .5, 19, 19))


class AppMark(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(20, 20)

    def paintEvent(self, _):
        c = self.window().c
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["primary"]))
        p.drawRoundedRect(QRectF(0, 0, 20, 20), 5, 5)
        p.setBrush(QColor(c["onPrimary"]))
        p.drawEllipse(QPointF(10, 10), 4, 4)


class RoleTile(QWidget):
    """Preview of one generated colour role, with its name underneath."""

    def __init__(self, role: str, name: str):
        super().__init__()
        self.role, self.name = role, name
        self.setFixedHeight(54)

    def paintEvent(self, _):
        c = self.window().c
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(c["outlineVariant"]), 1))
        p.setBrush(QColor(c[self.role]))
        p.drawRoundedRect(QRectF(.5, .5, self.width() - 1, 35), 6, 6)
        f = self.font()
        f.setPixelSize(10)
        p.setFont(f)
        p.setPen(QColor(c["onSurfaceVariant"]))
        p.drawText(QRectF(0, 40, self.width(), 14), Qt.AlignLeft | Qt.AlignVCenter, self.name)


class OutlinedField(QWidget):
    """M3 outlined text field: the label sits in a gap in the top border."""

    def __init__(self, label_text: str, text: str):
        super().__init__()
        self.setFixedHeight(56)
        self.edit = QLineEdit(text, self)
        self.edit.setObjectName("outlined")
        self.label = QLabel(label_text, self)
        self.label.setObjectName("floatLabel")
        self.edit.installEventFilter(self)

    def resizeEvent(self, _):
        self.edit.setGeometry(0, 8, self.width(), 48)
        self.label.adjustSize()
        self.label.move(12, 0)

    def eventFilter(self, obj, ev):
        if ev.type() in (QEvent.FocusIn, QEvent.FocusOut):
            self.label.setProperty("focused", ev.type() == QEvent.FocusIn)
            self.label.style().polish(self.label)
        return False


class TitleBar(QFrame):
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.window().windowHandle().startSystemMove()

    def mouseDoubleClickEvent(self, _):
        self.window().toggle_maximized()


class Grip(QWidget):
    """Invisible strip on the window edge for resizing a frameless window."""

    def __init__(self, parent, edges, cursor):
        super().__init__(parent)
        self.edges = edges
        self.setCursor(cursor)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.window().windowHandle().startSystemResize(self.edges)


# ---------------------------------------------------------------- window

class SettingsWindow(QWidget):
    saved = Signal(dict)              # Save changes pressed: every setting, see DEFAULTS
    calibrate_requested = Signal()    # Calibration tab clicked

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setWindowTitle("Study Assistant")
        self.resize(1100, 700)
        self.setMinimumSize(920, 560)
        self.store = QSettings("StudyAssistant", "StudyAssistant")
        self.seed = self.store.value("seed", DEFAULTS["seed"])
        self.c = make_scheme(self.seed, False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.titlebar = self._title_bar()
        root.addWidget(self.titlebar)
        # pages: 0 = settings; a calibration / check page is added while it runs
        self.settings_page = QWidget()
        sp = QVBoxLayout(self.settings_page)
        sp.setContentsMargins(0, 0, 0, 0)
        sp.setSpacing(0)
        sp.addWidget(self._toolbar())
        body = QHBoxLayout()
        body.setSpacing(0)
        body.addWidget(self._content(), 1)
        body.addWidget(self._sheet())
        sp.addLayout(body, 1)
        self.pages = QStackedWidget()
        self.pages.addWidget(self.settings_page)
        root.addWidget(self.pages, 1)
        self._before_page = None          # (was visible, was maximized) while a full-screen page runs

        G = 5
        self.grips = [Grip(self, e, cur) for e, cur in [
            (Qt.LeftEdge, Qt.SizeHorCursor), (Qt.RightEdge, Qt.SizeHorCursor),
            (Qt.TopEdge, Qt.SizeVerCursor), (Qt.BottomEdge, Qt.SizeVerCursor),
            (Qt.TopEdge | Qt.LeftEdge, Qt.SizeFDiagCursor), (Qt.BottomEdge | Qt.RightEdge, Qt.SizeFDiagCursor),
            (Qt.TopEdge | Qt.RightEdge, Qt.SizeBDiagCursor), (Qt.BottomEdge | Qt.LeftEdge, Qt.SizeBDiagCursor)]]
        self._grip = G

        self.load(self.read_store())
        QGuiApplication.styleHints().colorSchemeChanged.connect(lambda _: self.apply_theme())

    # ----- layout pieces

    def _title_bar(self):
        bar = TitleBar(objectName="titlebar")
        bar.setFixedHeight(48)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(AppMark())
        h.addSpacing(12)
        h.addWidget(label("Study Assistant", "appname"))
        h.addStretch(1)

        settings = QPushButton(I_SETTINGS, objectName="settingsBtn")
        settings.setCheckable(True)
        settings.setChecked(True)
        settings.setToolTip("Settings")
        settings.setAccessibleName("Settings")
        settings.setCursor(Qt.PointingHandCursor)
        settings.setFocusPolicy(Qt.TabFocus)
        self.nav = QButtonGroup(self)   # exclusive, so clicking the selected gear can't un-select it
        self.nav.addButton(settings, 0)
        calib = QPushButton(I_EYE, objectName="settingsBtn")   # same look as the gear
        calib.setCheckable(True)
        calib.setToolTip("Calibration")
        calib.setAccessibleName("Calibration")
        calib.setCursor(Qt.PointingHandCursor)
        calib.setFocusPolicy(Qt.TabFocus)
        self.nav.addButton(calib, 1)
        self.nav.idClicked.connect(lambda i: i == 1 and self.calibrate_requested.emit())
        h.addWidget(settings)
        h.addSpacing(4)
        h.addWidget(calib)
        h.addSpacing(8)

        caps = QWidget(objectName="caps")
        ch = QHBoxLayout(caps)
        ch.setContentsMargins(0, 0, 0, 0)
        ch.setSpacing(0)
        for glyph, name, slot in [(I_MIN, "min", self.showMinimized), (I_MAX, "max", self.toggle_maximized),
                                  (I_CLOSE, "close", self.close)]:
            b = QPushButton(glyph, objectName=name)
            b.setFocusPolicy(Qt.NoFocus)
            b.setAccessibleName({"min": "Minimize", "max": "Maximize", "close": "Close"}[name])
            b.clicked.connect(slot)
            ch.addWidget(b)
        self.max_btn = caps.findChild(QPushButton, "max")
        h.addWidget(caps)
        return bar

    def _toolbar(self):
        bar = QFrame(objectName="toolbar")
        bar.setFixedHeight(60)
        h = QHBoxLayout(bar)
        h.setContentsMargins(24, 0, 24, 0)
        h.setSpacing(12)
        h.addWidget(label("Settings", "h1"))
        h.addSpacing(12)
        self.find = QLineEdit(objectName="find", placeholderText="Find a setting")
        self.find_action = self.find.addAction(QIcon(), QLineEdit.LeadingPosition)
        self.find.textChanged.connect(self.filter_rows)
        h.addWidget(self.find)
        h.addStretch(1)
        reset = button(f"{I_RESET}  Reset to defaults", "outlined")
        reset.clicked.connect(lambda: self.load(DEFAULTS))
        self.save_btn = button("Save changes", "filled")
        self.save_btn.clicked.connect(self.save)
        h.addWidget(reset)
        h.addWidget(self.save_btn)
        return bar

    def _content(self):
        self.rows = []      # (row widget, searchable text)
        inner = QWidget(objectName="content")
        self.content = QVBoxLayout(inner)
        self.content.setContentsMargins(24, 4, 24, 24)
        self.content.setSpacing(0)

        self.theme_group = QButtonGroup(self)
        chips = QWidget()
        chl = QHBoxLayout(chips)
        chl.setContentsMargins(0, 0, 0, 0)
        chl.setSpacing(8)
        for i, text in enumerate(["System", "Light", "Dark"]):
            b = CheckButton(text, "chip")
            chl.addWidget(b)
            self.theme_group.addButton(b, i)
        self.theme_group.idClicked.connect(lambda i: self.set_mode(["system", "light", "dark"][i]))

        self.dot = ColorDot()
        self.hex_label = label("", "hex")
        self.hex_label.setMinimumWidth(62)
        change = button("Change", "text")
        change.clicked.connect(self.open_sheet)
        self.group("Appearance", [
            self.row(I_THEME, "Theme", "Follow Windows, or always use one", chips),
            self.row(I_PALETTE, "Theme colour", "Every colour in the app is made from this one", self.dot, self.hex_label, change),
        ])

        self.switches = {k: Switch(t) for k, t in [("show_ball", "Show floating ball"),
                                                   ("ball_on_top", "Keep it above other windows"),
                                                   ("hide_fullscreen", "Hide during full-screen apps")]}
        self.group("Floating ball", [
            self.row(I_BALL, "Show floating ball", "Opens the assistant from any app", self.switches["show_ball"]),
            self.row(I_LAYERS, "Keep it above other windows", "Turn off to let it hide behind full windows",
                     self.switches["ball_on_top"]),
            self.row(I_FULLSCREEN, "Hide during full-screen apps", "Games and videos stay clear",
                     self.switches["hide_fullscreen"]),
        ])

        self.model_btn = button("", "assist")
        menu = QMenu(self.model_btn)
        for m in MODELS:
            menu.addAction(m, lambda m=m: self.set_model(m))
        self.model_btn.setMenu(menu)
        self.words = QSpinBox(minimum=50, maximum=500, singleStep=25, alignment=Qt.AlignRight)
        self.words.setButtonSymbols(QSpinBox.NoButtons)
        self.words.setAccessibleName("Summary length in words")
        words = QWidget()
        wl = QHBoxLayout(words)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(8)
        wl.addWidget(self.words)
        wl.addWidget(label("words", "unit"))
        self.group("Summaries and quizzes", [
            self.row(I_SPARKLE, "AI model", "Writes summaries and marks quiz answers", self.model_btn),
            self.row(I_LINES, "Summary length", "About this many words", words),
        ])
        self.camera_btn = button("", "assist")
        self.camera_menu = QMenu(self.camera_btn)
        self.camera_menu.aboutToShow.connect(self._fill_camera_menu)   # list cameras when opened
        self.camera_btn.setMenu(self.camera_menu)
        self.group("Camera", [
            self.row(I_CAMERA, "Webcam", "Used for eye tracking. Changes apply when you save", self.camera_btn),
        ])
        self.content.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.viewport().setAutoFillBackground(False)
        return scroll

    def row(self, icon, title, sub, *trailing):
        w = QWidget()
        w.setMinimumHeight(56)
        h = QHBoxLayout(w)
        h.setContentsMargins(16, 6, 16, 6)
        h.setSpacing(16)
        ic = label(icon, "rowicon")
        ic.setFixedWidth(20)
        h.addWidget(ic)
        text = QVBoxLayout()
        text.setSpacing(0)
        text.addWidget(label(title, "rowtitle"))
        text.addWidget(label(sub, "rowsub"))
        h.addLayout(text, 1)
        for t in trailing:
            h.addWidget(t)
        self.rows.append((w, f"{title} {sub}".lower()))
        return w

    def group(self, title, rows):
        heading = label(title, "gh")
        heading.setContentsMargins(0, 16, 0, 8)
        box = QFrame(objectName="group")
        v = QVBoxLayout(box)
        v.setContentsMargins(1, 1, 1, 1)
        v.setSpacing(0)
        for i, r in enumerate(rows):
            if i:
                v.addWidget(QFrame(objectName="divider"))
            v.addWidget(r)
        self.content.addWidget(heading)
        self.content.addWidget(box)

    def _sheet(self):
        self.sheet = QFrame(objectName="sheet")
        self.sheet.setFixedWidth(320)
        v = QVBoxLayout(self.sheet)
        v.setContentsMargins(24, 20, 20, 20)
        v.setSpacing(16)

        head = QHBoxLayout()
        head.addWidget(label("Theme colour", "sheetTitle"), 1)
        close = button(I_DISMISS, "icon")
        close.setAccessibleName("Close")
        close.clicked.connect(self.cancel_sheet)
        head.addWidget(close)
        v.addLayout(head)
        text = label("Pick one colour. The light and dark themes are made from it.", "sheetText")
        text.setWordWrap(True)
        v.addWidget(text)

        grid = QGridLayout()
        grid.setSpacing(6)
        self.swatches = QButtonGroup(self)
        for i, col in enumerate(SWATCHES):
            s = Swatch(col)
            self.swatches.addButton(s)
            grid.addWidget(s, i // 4, i % 4)
        self.swatches.buttonClicked.connect(lambda b: self.set_seed(b.color))
        v.addLayout(grid)

        self.hex_field = OutlinedField("Hex code", self.seed)
        self.hex_field.edit.setValidator(QRegularExpressionValidator(QRegularExpression("#?[0-9A-Fa-f]{0,6}")))
        self.hex_field.edit.textEdited.connect(self._hex_edited)
        self.hex_field.edit.editingFinished.connect(lambda: self.hex_field.edit.setText(self.seed))
        v.addWidget(self.hex_field)

        roles = QHBoxLayout()
        roles.setSpacing(6)
        for role, name in [("primary", "Primary"), ("secondary", "Secondary"), ("tertiary", "Tertiary"),
                           ("primaryContainer", "Container"), ("surfaceContainerHigh", "Surface")]:
            roles.addWidget(RoleTile(role, name))
        v.addLayout(roles)
        v.addStretch(1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addStretch(1)
        cancel = button("Cancel", "text")
        cancel.clicked.connect(self.cancel_sheet)
        done = button("Done", "filled")
        done.clicked.connect(self.sheet.hide)
        actions.addWidget(cancel)
        actions.addWidget(done)
        v.addLayout(actions)
        self.seed_before_sheet = self.seed
        return self.sheet

    # ----- behaviour

    def read_store(self) -> dict:
        return {k: self.store.value(k, d, type=type(d)) for k, d in DEFAULTS.items()}

    def load(self, values: dict):
        for k, sw in self.switches.items():
            sw.set_checked_now(values[k])
        self.set_model(values["model"])
        self.set_camera(values["camera"])
        self.words.setValue(values["summary_words"])
        self.mode = values["theme"]
        self.theme_group.button(["system", "light", "dark"].index(self.mode)).setChecked(True)
        self.seed_before_sheet = values["seed"]
        self.set_seed(values["seed"])

    def save(self):
        values = {k: sw.isChecked() for k, sw in self.switches.items()}
        values.update(seed=self.seed, theme=self.mode, model=self.model, summary_words=self.words.value(),
                      camera=self.camera)
        for k, v in values.items():
            self.store.setValue(k, v)
        self.saved.emit(values)
        self.save_btn.setText("Saved")
        QTimer.singleShot(1500, lambda: self.save_btn.setText("Save changes"))

    def set_model(self, name: str):
        self.model = name
        self.model_btn.setText(f"{name}  {I_CHEVRON}")

    def set_camera(self, name: str):
        self.camera = name
        self.camera_btn.setText(f"{name or 'Not chosen'}  {I_CHEVRON}")

    def _fill_camera_menu(self):
        self.camera_menu.clear()
        for name in camera_names() or ["No camera found"]:
            self.camera_menu.addAction(name, lambda n=name: self.set_camera(n))

    # ----- full-screen pages (calibration, check)

    def show_fullscreen_page(self, page: QWidget):
        """Show `page` alone on the whole screen: no title bar, toolbar or buttons.
        When the page closes (it's deleted), the window goes back to how it was."""
        self._before_page = (self.isVisible(), self.isMaximized())
        self.titlebar.hide()
        for g in self.grips:
            g.hide()
        self.pages.addWidget(page)
        self.pages.setCurrentWidget(page)
        page.destroyed.connect(self._page_closed)
        self.showFullScreen()
        page.setFocus()                   # so Space / Esc reach it

    def _page_closed(self):
        was_visible, was_max = self._before_page or (False, False)
        self._before_page = None
        self.titlebar.show()
        self.pages.setCurrentWidget(self.settings_page)
        self.nav.button(0).setChecked(True)
        if not was_visible:
            self.hide()
        elif was_max:
            self.showMaximized()
        else:
            self.showNormal()

    def set_mode(self, mode: str):
        self.mode = mode
        self.apply_theme()

    def set_seed(self, hex_: str):
        self.seed = hex_.upper()
        self.hex_label.setText(self.seed)
        if not self.hex_field.edit.hasFocus():
            self.hex_field.edit.setText(self.seed)
        self.swatches.setExclusive(False)    # so a custom colour can leave every swatch unselected
        for b in self.swatches.buttons():
            b.setChecked(b.color == self.seed)
        self.swatches.setExclusive(True)
        self.apply_theme()

    def _hex_edited(self, text: str):
        text = "#" + text.lstrip("#")
        if len(text) == 7:
            self.set_seed(text)

    def open_sheet(self):
        self.seed_before_sheet = self.seed
        self.sheet.show()

    def cancel_sheet(self):
        self.set_seed(self.seed_before_sheet)
        self.sheet.hide()

    def filter_rows(self, text: str):
        text = text.strip().lower()
        for w, words in self.rows:
            w.setVisible(text in words)

    def apply_theme(self):
        self.c = c = make_scheme(self.seed, is_dark(self.mode))
        self.setStyleSheet(build_qss(c))
        for edit in self.findChildren(QLineEdit):
            pal = edit.palette()
            pal.setColor(QPalette.PlaceholderText, QColor(c["onSurfaceVariant"]))
            edit.setPalette(pal)
        self.find_action.setIcon(glyph_icon(I_SEARCH, c["onSurfaceVariant"]))
        self.model_btn.setIcon(glyph_icon(I_SPARKLE, c["primary"]))
        self._dwm(34, int(c["outlineVariant"][5:7] + c["outlineVariant"][3:5] + c["outlineVariant"][1:3], 16))
        self.update()

    def toggle_maximized(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def changeEvent(self, e):
        if e.type() == QEvent.WindowStateChange:
            self.max_btn.setText(I_RESTORE if self.isMaximized() else I_MAX)
            self.max_btn.setAccessibleName("Restore" if self.isMaximized() else "Maximize")
            for g in self.grips:
                g.setVisible(not (self.isMaximized() or self.isFullScreen()))
        super().changeEvent(e)

    def resizeEvent(self, e):
        G, w, h = self._grip, self.width(), self.height()
        for g, rect in zip(self.grips, [(0, G, G, h - 2 * G), (w - G, G, G, h - 2 * G), (G, 0, w - 2 * G, G),
                                        (G, h - G, w - 2 * G, G), (0, 0, G, G), (w - G, h - G, G, G),
                                        (w - G, 0, G, G), (0, h - G, G, G)]):
            g.setGeometry(*rect)
            g.raise_()
        super().resizeEvent(e)

    def showEvent(self, e):
        self._dwm(33, 2)    # Windows 11: round the corners of the frameless window
        super().showEvent(e)

    def _dwm(self, attribute: int, value: int):
        """Set a DWM window attribute (33 = corner preference, 34 = border colour). No-op off Windows 11."""
        if sys.platform == "win32":
            v = ctypes.c_uint(value)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(int(self.winId()), attribute, ctypes.byref(v), 4)


def is_dark(mode: str) -> bool:
    return mode == "dark" or (mode == "system" and QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark)


def pick_camera(names: list[str], current: str = "") -> str | None:
    """Small popup: choose a webcam. Returns its name, or None if closed."""
    store = QSettings("StudyAssistant", "StudyAssistant")
    seed, mode = store.value("seed", DEFAULTS["seed"]), store.value("theme", DEFAULTS["theme"])
    dlg = QDialog(None, Qt.Dialog | Qt.WindowCloseButtonHint)
    dlg.setObjectName("root")
    dlg.setAttribute(Qt.WA_StyledBackground)
    dlg.setWindowTitle("Choose a camera")
    dlg.setStyleSheet(build_qss(make_scheme(seed, is_dark(mode))))
    v = QVBoxLayout(dlg)
    v.setContentsMargins(24, 20, 24, 20)
    v.setSpacing(12)
    v.addWidget(label("Which camera should track your eyes?", "sheetTitle"))
    hint = label("It is remembered. Change it any time in Settings > Camera.", "sheetText")
    hint.setWordWrap(True)
    v.addWidget(hint)
    group = QButtonGroup(dlg)
    for i, name in enumerate(names):
        b = CheckButton(name, "chip")
        group.addButton(b, i)
        v.addWidget(b)
        b.setChecked(name == current or (i == 0 and current not in names))
    ok = button("Use this camera", "filled")
    ok.clicked.connect(dlg.accept)
    v.addSpacing(4)
    v.addWidget(ok, 0, Qt.AlignRight)
    if dlg.exec() != QDialog.Accepted or group.checkedId() < 0:
        return None
    return names[group.checkedId()]


def main():
    app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(TEXT_FONTS)   # icon font last, so glyphs like the check mark work inside text
    app.setFont(font)
    win = SettingsWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
