"""Study Assistant main window: direction 3 ("Toolbar") in PySide6.

Four tabs in the title bar: Chat, Testing, Calibration, Settings. Calibration (and
check) pages fill the screen with no title bar or buttons, then the window goes
back to the Chat tab. pick_camera() is the small camera popup used at first start.

ChatPage looks like ChatGPT, but with a "Go Reading" button instead of an input
bar: it closes the window; the floating ball stays on the whole time.

Every colour comes from one seed colour through the Material 3 Fidelity
scheme, in light and dark. Pick a colour in the side sheet and the whole
window re-themes live.

    pip install PySide6 materialyoucolor
    python settings_window.py
"""
import ctypes
import sys
from pathlib import Path
from string import Template

from materialyoucolor.dynamiccolor.material_dynamic_colors import MaterialDynamicColors
from materialyoucolor.hct import Hct
from materialyoucolor.scheme.scheme_fidelity import SchemeFidelity
from PySide6.QtCore import (QEasingCurve, QEvent, QPointF, QRectF, QRegularExpression, QSettings, Qt, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap,
                           QRegularExpressionValidator)
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from eyetracking.services.camera import camera_names
from eyetracking.services.secret_store import PREFIX, protect, unprotect
from eyetracking.services.storage import (calibration_lines, calibration_usable, file_date, list_calibrations,
                                          list_readings, reading_messages, reading_title, remove_calibration,
                                          remove_samples)

DEFAULTS = {
    "seed": "#3D6CF6",
    "theme": "system",          # system | light | dark
    "show_ball": True,
    "ball_on_top": True,
    "hide_fullscreen": False,
    "summary_words": 150,
    "camera": "",               # webcam NAME (Windows can renumber cameras between restarts)
    "api_key": "",              # Claude / Gemini / OpenAI key (its prefix says which); stored encrypted (DPAPI)
}
SWATCHES = ["#3D6CF6", "#00897B", "#2E8B57", "#E8B400", "#F4511E", "#D32F2F", "#D81B60", "#7E57C2"]

PAGE_OF_TAB = {0: 0, 1: 1, 3: 2}          # title-bar tab -> page in the stack (tab 2, Calibration, is full screen)
TAB_OF_PAGE = {v: k for k, v in PAGE_OF_TAB.items()}

TEXT_FONTS = ["Segoe UI Variable Text", "Segoe UI", "Segoe Fluent Icons", "Segoe MDL2 Assets"]
ICON_FONTS = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]   # built into Windows 11 / Windows 10

# Segoe Fluent Icons code points
I_SETTINGS, I_SEARCH = "", ""
I_MIN, I_MAX, I_RESTORE, I_CLOSE = "", "", "", ""
I_CHECK, I_DISMISS, I_RESET, I_CHEVRON = "", "", "", ""
I_THEME, I_PALETTE, I_BALL, I_LAYERS = "", "", "", ""
I_FULLSCREEN, I_SPARKLE, I_LINES = "", "", ""
I_EYE, I_CAMERA = "", ""   # View (eye), Camera
I_KEY = "\uE8D7"   # Permissions (key)
I_TESTING = "\uE9D9"   # Diagnostic
I_BACK, I_DELETE = "\uE72B", "\uE74D"
I_CHAT, I_READ, I_SKIP = "\uE8BD", "\uE736", "\uE893"   # Message, ReadingMode, Next


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

/* chat page */
#side { background: $surfaceContainerLow; border-right: 1px solid $outlineVariant; }
#sideTitle { font-size: 12px; font-weight: 600; color: $onSurfaceVariant; padding: 8px 12px 4px 12px; }
QPushButton[kind="nav"] { text-align: left; padding: 0 12px; border-radius: 8px; font-weight: 400; font-size: 14px; }
QPushButton[kind="nav"]:hover { background: $hover; }
QPushButton[kind="nav"]:checked, QPushButton[kind="nav"]:checked:hover { background: $selected; color: $onSurface; }
#lead { font-size: 15px; }
#sideLabel { font-size: 14px; padding: 6px 12px 2px 12px; }
QListWidget#sideList { background: transparent; border: none; outline: none; font-size: 13px; }
QListWidget#sideList::item { padding: 6px 12px 6px 36px; border-radius: 8px; color: $onSurfaceVariant; }
QListWidget#sideList::item:hover { background: $hover; }
QListWidget#sideList::item:selected { background: $selected; color: $onSurface; }
#hero { font-family: "Segoe UI Variable Display", "Segoe UI"; font-size: 32px; font-weight: 600; }
#heroSub { font-size: 14px; color: $onSurfaceVariant; }
QLabel#theirs { font-size: 15px; }
#mine { background: $surfaceContainerHigh; border-radius: 18px; }
#mine QLabel { font-size: 15px; background: transparent; }
QPushButton[kind="go"] { background: $primary; color: $onPrimary; min-height: 56px; max-height: 56px;
    border-radius: 28px; font-size: 16px; }
QPushButton[kind="go"]:hover { background: $primaryHover; }
QPushButton[kind="go"]:pressed { background: $primaryPress; }
#answerBox { background: $surfaceContainerHigh; border-radius: 28px; }
QLineEdit#answer { border: none; background: transparent; font-size: 15px; min-height: 40px;
    selection-background-color: $primary; selection-color: $onPrimary; }

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
        selected=rgba(c["onSurface"], .14),     # the chosen sidebar entry: a darker neutral, not a tint
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


# ---------------------------------------------------------------- dated file lists

def side_list(on_open) -> QListWidget:
    """A list of dated files; double-click (or Enter) calls on_open(path)."""
    lst = QListWidget(objectName="sideList")
    lst.setToolTip("Double-click to open")
    lst.itemActivated.connect(lambda it: it.data(Qt.UserRole) and on_open(Path(it.data(Qt.UserRole))))
    return lst


def fill_list(lst: QListWidget, paths: list[Path], empty: str, name=None):
    """One item per dated file; name(path), if given, is shown after the date (full name in the tooltip)."""
    lst.clear()
    for path in paths:
        when = f"{file_date(path):%d %b %Y   %H:%M}"
        item = QListWidgetItem(f"{file_date(path):%d %b %H:%M}  {name(path)[:22]}" if name else when)
        item.setToolTip(f"{when}\n{name(path)}" if name else when)
        item.setData(Qt.UserRole, str(path))
        lst.addItem(item)
    if not paths and empty:
        lst.addItem(QListWidgetItem(empty))   # no path: clicking does nothing


def message_widget(text: str, mine: bool) -> tuple[QWidget, QLabel]:
    """One chat message: (the widget to add, its text label). Yours is a bubble pushed
    to the right by a stretch (a row, so the wrapped text keeps its height); the
    assistant's is plain text."""
    text_label = QLabel(text, objectName="theirs")
    text_label.setWordWrap(True)
    text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if not mine:
        return text_label, text_label
    bubble = QFrame(objectName="mine")
    QVBoxLayout(bubble).addWidget(text_label)
    bubble.setMaximumWidth(520)
    row = QWidget()
    rh = QHBoxLayout(row)
    rh.setContentsMargins(0, 0, 0, 0)
    rh.addStretch(1)
    rh.addWidget(bubble)
    return row, text_label


# ---------------------------------------------------------------- testing page

class TestingPage(QWidget):
    """Testing tab. The sidebar picks a page on the right: what the sample text and
    the gaze check do (with the button that starts them), or the saved gaze data
    (calibrations by date, split into the files this version can train on and the rest)."""

    command = Signal(str)          # a start button: "read" | "check"
    data_changed = Signal()        # samples or a file removed: the app retrains the gaze model

    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QFrame(objectName="side")
        side.setFixedWidth(260)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 12, 12, 12)
        sv.setSpacing(2)
        sv.addWidget(label("Testing", "sideTitle"))
        self.side_nav = QButtonGroup(self)               # the chosen entry stays highlighted: it is the page on the right
        for i, (glyph, text) in enumerate([(I_READ, "Sample text"), (I_EYE, "Gaze check"), (I_LINES, "Gaze data")]):
            b = button(f"{glyph}   {text}", "nav")
            b.setCheckable(True)
            self.side_nav.addButton(b, i)
            sv.addWidget(b)
        sv.addStretch(1)
        h.addWidget(side)

        # 0: sample text, 1: gaze check (what they do + a start button), 2: the data lists, 3: one calibration's detail
        self.main = QStackedWidget()
        self.main.addWidget(self._intro(
            I_READ, "Sample text", "Read a short text while the tracker watches where you look.",
            ["Each paragraph turns green once your eyes have rested on it long enough: about a tenth of a second per word.",
             "Press Done, or stare at it, when you finish. The AI then summarises what you skipped and asks you about it, "
             "in the Chat tab.",
             "Open .txt loads a text of your own. Esc leaves without the AI part."],
            "Open the sample text", "read"))
        self.main.addWidget(self._intro(
            I_EYE, "Gaze check", "A full-screen test of the gaze model: how accurate is it today?",
            ["A green dot follows where the model thinks you look. The faint rings are the calibration dots: look at one "
             "and see where the dot lands.",
             "Press Enter to record a sample of where you are looking. Samples are numbered on screen (#index  x  y) and "
             "saved to data/gaze_check_<time>.csv when you leave.",
             "Space pauses. Esc saves and leaves. It needs a calibration first."],
            "Start the gaze check", "check"))
        lists = QWidget()
        mv = QVBoxLayout(lists)
        mv.setContentsMargins(24, 20, 24, 24)
        mv.setSpacing(8)
        mv.addWidget(label("Gaze data", "sheetTitle"))
        hint = label("Saved calibrations by date. Double-click one to read it in words; right-click to remove it. "
                     "Only the active ones train the gaze model.", "sheetText")
        hint.setWordWrap(True)
        mv.addWidget(hint)
        cols = QHBoxLayout()
        cols.setSpacing(24)
        self.active_list = side_list(self.open_detail)
        self.inactive_list = side_list(self.open_detail)
        for title, sub, lst in [("Active", "Samples this version can use", self.active_list),
                                ("Not usable", "Another calibration version, or the file can't be read", self.inactive_list)]:
            lst.setContextMenuPolicy(Qt.CustomContextMenu)
            lst.customContextMenuRequested.connect(lambda pos, lst=lst: self._list_menu(lst, pos))
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(label(title, "gh"))
            col.addWidget(label(sub, "rowsub"))
            col.addWidget(lst, 1)
            cols.addLayout(col, 1)
        mv.addLayout(cols, 1)
        self.main.addWidget(lists)

        detail = QWidget()                               # one calibration, every sample in words
        dv = QVBoxLayout(detail)
        dv.setContentsMargins(24, 20, 24, 24)
        dv.setSpacing(8)
        head = QHBoxLayout()
        back = button(f"{I_BACK}  Back", "text")
        back.clicked.connect(lambda: self.main.setCurrentIndex(2))
        self.detail_title = label("", "sheetTitle")
        head.addWidget(back)
        head.addWidget(self.detail_title, 1)
        dv.addLayout(head)
        hint = label("Select samples, or a dot's line for all of its samples, then Remove: the file is rewritten and "
                     "the gaze model retrained. The untouched original stays in data/trash.", "sheetText")
        hint.setWordWrap(True)
        dv.addWidget(hint)
        self.detail = QListWidget(objectName="sideList")
        self.detail.setSelectionMode(QListWidget.ExtendedSelection)
        self.detail.setWordWrap(True)                    # a sample's words are long
        dv.addWidget(self.detail, 1)
        self.remove_btn = button(f"{I_DELETE}  Remove selected", "outlined")
        self.remove_btn.clicked.connect(self.remove_selected)
        dv.addWidget(self.remove_btn, 0, Qt.AlignLeft)
        self.main.addWidget(detail)
        self.detail_path: Path | None = None
        h.addWidget(self.main, 1)
        self.side_nav.idClicked.connect(self.main.setCurrentIndex)
        self.side_nav.button(2).setChecked(True)         # Gaze data first: the overview of what the model learns from
        self.main.setCurrentIndex(2)

    def _intro(self, glyph: str, title: str, lead: str, steps: list[str], action: str, cmd: str) -> QWidget:
        """An instruction page: what the test does, then the button that starts it."""
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 20, 24, 24)
        v.setSpacing(12)
        v.addWidget(label(f"{glyph}   {title}", "sheetTitle"))
        for i, text in enumerate([lead] + [f"•   {s}" for s in steps]):
            w = label(text, "lead" if i == 0 else "sheetText")
            w.setWordWrap(True)
            v.addWidget(w)
        go = button(action, "filled")
        go.clicked.connect(lambda: self.command.emit(cmd))
        v.addSpacing(8)
        v.addWidget(go, 0, Qt.AlignLeft)
        v.addStretch(1)
        return page

    def open_detail(self, path: Path):
        """One calibration in words, in place of the lists."""
        self.detail_path = path
        self.detail_title.setText(f"Calibration of {file_date(path):%d %b %Y, %H:%M}")
        self.detail.clear()
        removable = False
        for text, index, dot in calibration_lines(path):
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, index)
            item.setData(Qt.UserRole + 1, dot)
            if dot is None:                              # a heading: nothing to remove
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            removable |= dot is not None
            self.detail.addItem(item)
        self.remove_btn.setVisible(removable)
        self.main.setCurrentIndex(3)

    def remove_selected(self):
        chosen = set()
        for item in self.detail.selectedItems():
            index, dot = item.data(Qt.UserRole), item.data(Qt.UserRole + 1)
            if index is not None:
                chosen.add(index)
            else:                                        # a dot's summary line: every sample of that dot
                chosen |= {self.detail.item(i).data(Qt.UserRole) for i in range(self.detail.count())
                           if self.detail.item(i).data(Qt.UserRole + 1) == dot
                           and self.detail.item(i).data(Qt.UserRole) is not None}
        if chosen:
            remove_samples(self.detail_path, chosen)
            if self.detail_path.exists():
                self.open_detail(self.detail_path)
            else:                                        # the last samples went: so did the file
                self.main.setCurrentIndex(2)
            self.refresh()
            self.data_changed.emit()

    def _list_menu(self, lst: QListWidget, pos):
        item = lst.itemAt(pos)
        if item is None or not item.data(Qt.UserRole):
            return
        path = Path(item.data(Qt.UserRole))
        menu = QMenu(lst)
        menu.addAction(f"{I_DELETE}  Remove this calibration", lambda: self.remove_file(path))
        menu.exec(lst.mapToGlobal(pos))

    def remove_file(self, path: Path):
        """Right-click > Remove: the file moves to data/trash (move it back to undo); the model retrains."""
        if QMessageBox.question(self, "Remove calibration", f"Move {path.name} to data/trash?\n"
                                "The gaze model is retrained without it.") != QMessageBox.Yes:
            return
        remove_calibration(path)
        self.refresh()
        self.data_changed.emit()

    def refresh(self):
        """Re-read data/ (new calibrations). Runs whenever the page shows."""
        usable = {p: calibration_usable(p) for p in list_calibrations()}
        fill_list(self.active_list, [p for p, ok in usable.items() if ok], "No usable calibration yet")
        fill_list(self.inactive_list, [p for p, ok in usable.items() if not ok], "None")

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)


# ---------------------------------------------------------------- chat page

class ChatPage(QWidget):
    """ChatGPT-style page. The app talks to it like the old chat panel:
    reply(text) to speak, message_sent(str) for quiz answers."""

    message_sent = Signal(str)     # a quiz answer (or "skip")
    go_reading = Signal()          # the big button: close the window (the ball is still there)

    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QFrame(objectName="side")
        side.setFixedWidth(260)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 12, 12, 12)
        sv.setSpacing(2)
        sv.addWidget(label("Chats", "sideTitle"))
        # the live chat first, then past readings (summary + quiz): one click shows it on the right
        self.chat_list = QListWidget(objectName="sideList")
        self.chat_list.currentItemChanged.connect(lambda it, _: self._show_chat(it))
        sv.addWidget(self.chat_list, 1)
        h.addWidget(side)
        self.said: list[str] = []                # every message shown this run, newest last

        col = QWidget()
        self.v = v = QVBoxLayout(col)
        v.setContentsMargins(24, 24, 24, 32)
        v.setSpacing(16)
        v.addStretch(1)                                  # 0: centres the hero while empty

        self.scroll = QScrollArea()                      # 1: messages, once there are some
        self.scroll.setWidgetResizable(True)
        self.scroll.viewport().setAutoFillBackground(False)
        host = QWidget(objectName="content")
        self.msgs = QVBoxLayout(host)
        self.msgs.setContentsMargins(0, 0, 8, 0)
        self.msgs.setSpacing(20)
        self.msgs.addStretch(1)
        self.scroll.setWidget(host)
        self.scroll.hide()
        v.addWidget(self.scroll)

        self.hero = QWidget()                            # 2: "What should we read?"
        hv = QVBoxLayout(self.hero)
        hv.setContentsMargins(0, 0, 0, 8)
        title = label("What should we read?", "hero")
        title.setAlignment(Qt.AlignCenter)
        self.status = label("", "heroSub")              # the app puts the gaze model status here
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        hv.addWidget(title)
        hv.addWidget(self.status)
        v.addWidget(self.hero)

        self.answer_box = QFrame(objectName="answerBox")   # 3: only while a quiz question waits
        ah = QHBoxLayout(self.answer_box)
        ah.setContentsMargins(20, 8, 8, 8)
        self.answer = QLineEdit(objectName="answer", placeholderText="Type your answer, then Enter")
        self.answer.returnPressed.connect(self._send_answer)
        skip = button(f"{I_SKIP}  Skip", "text")
        skip.clicked.connect(lambda: self._send("skip"))
        ah.addWidget(self.answer, 1)
        ah.addWidget(skip)
        self.answer_box.hide()
        v.addWidget(self.answer_box)

        self.go = button(f"{I_READ}   Go Reading", "go")   # 4: instead of an input bar
        self.go.clicked.connect(self.go_reading)
        v.addWidget(self.go)
        v.addStretch(1)                                  # 5

        self.archive = QScrollArea()                     # a past reading, read-only, in place of the live chat
        self.archive.setWidgetResizable(True)
        self.archive.viewport().setAutoFillBackground(False)
        self.view = QStackedWidget()                     # 0: the live chat, 1: a past reading
        self.view.setMaximumWidth(760)
        self.view.addWidget(col)
        self.view.addWidget(self.archive)
        h.addStretch(1)
        h.addWidget(self.view, 20)
        h.addStretch(1)

    def refresh(self):
        """Re-read data/history (a reading is written once the AI answers). Runs whenever the page shows."""
        current = self.chat_list.currentItem()
        chosen = current.data(Qt.UserRole) if current else None
        self.chat_list.blockSignals(True)
        fill_list(self.chat_list, list_readings(), "", reading_title)
        live = QListWidgetItem(f"{I_CHAT}   Today")           # the live chat: no path
        self.chat_list.insertItem(0, live)
        rows = [self.chat_list.item(i) for i in range(self.chat_list.count())]
        self.chat_list.setCurrentItem(next((r for r in rows if r.data(Qt.UserRole) == chosen), live))
        self.chat_list.blockSignals(False)
        self._show_chat(self.chat_list.currentItem())

    def _show_chat(self, item):
        """One click in the sidebar: the live chat, or a past reading on the right."""
        path = item.data(Qt.UserRole) if item else None
        if not path:
            self.view.setCurrentIndex(0)
            return
        path = Path(path)
        host = QWidget(objectName="content")
        col = QVBoxLayout(host)
        col.setContentsMargins(24, 24, 32, 32)
        col.setSpacing(20)
        col.addWidget(label(reading_title(path), "sheetTitle"))
        col.addWidget(label(f"{file_date(path):%d %b %Y, %H:%M}", "sheetText"))
        for text, mine in reading_messages(path):
            col.addWidget(message_widget(text, mine)[0])
        col.addStretch(1)
        self.archive.setWidget(host)                     # replaces (and frees) the previous one
        self.view.setCurrentIndex(1)

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)

    def set_answering(self, on: bool):
        """Swap Go Reading for the answer box while the quiz waits for an answer."""
        self.answer_box.setVisible(on)
        self.go.setVisible(not on)
        if on:
            self.answer.setFocus()

    def _send_answer(self):
        if self.answer.text().strip():
            self._send(self.answer.text().strip())
            self.answer.clear()

    def _send(self, text: str):
        self.said.append(text)
        self.add_message(text, mine=True)
        self.message_sent.emit(text)

    def add_message(self, text: str, mine: bool) -> QLabel:
        if not self.hero.isHidden():   # first message: hero out, messages fill (not isVisible: the window may be hidden)
            self.hero.hide()
            self.scroll.show()
            self.v.setStretch(0, 0)
            self.v.setStretch(1, 1)
            self.v.setStretch(5, 0)
        w, text_label = message_widget(text, mine)       # you: a bubble on the right; the assistant: plain text
        self.msgs.insertWidget(self.msgs.count() - 1, w)
        QTimer.singleShot(0, self._scroll_to_bottom)
        return text_label

    def _scroll_to_bottom(self):
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def reply(self, text: str):
        """Stream the assistant's text in, two characters a tick."""
        self.said.append(text)
        text_label = self.add_message("", mine=False)
        state = {"i": 0}
        timer = QTimer(self)

        def tick():
            state["i"] += 2
            text_label.setText(text[:state["i"]])
            self._scroll_to_bottom()
            if state["i"] >= len(text):
                timer.stop()
                timer.deleteLater()

        timer.timeout.connect(tick)
        timer.start(18)


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
        # pages: 0 = chat, 1 = testing, 2 = settings; a calibration / check page is added while it runs
        self.chat = ChatPage()
        self.testing = TestingPage()
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
        self.pages.addWidget(self.chat)
        self.pages.addWidget(self.testing)
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

        self.nav = QButtonGroup(self)   # exclusive, so clicking the selected tab can't un-select it
        for i, (glyph, name) in enumerate([(I_CHAT, "Chat"), (I_TESTING, "Testing"), (I_EYE, "Calibration"),
                                           (I_SETTINGS, "Settings")]):
            b = QPushButton(glyph, objectName="settingsBtn")
            b.setCheckable(True)
            b.setToolTip(name)
            b.setAccessibleName(name)
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.TabFocus)
            self.nav.addButton(b, i)
            h.addWidget(b)
            h.addSpacing(4)
        self.nav.button(0).setChecked(True)
        # Calibration (tab 2) runs full screen; the other tabs are pages in the stack
        self.nav.idClicked.connect(lambda i: self.calibrate_requested.emit() if i == 2
                                   else self.pages.setCurrentIndex(PAGE_OF_TAB[i]))
        h.addSpacing(4)

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
        reset.clicked.connect(lambda: self.load({**DEFAULTS, "api_key": self.key.text()}))   # keeps your key
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

        self.words = QSpinBox(minimum=50, maximum=500, singleStep=25, alignment=Qt.AlignRight)
        self.words.setButtonSymbols(QSpinBox.NoButtons)
        self.words.setAccessibleName("Summary length in words")
        words = QWidget()
        wl = QHBoxLayout(words)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.setSpacing(8)
        wl.addWidget(self.words)
        wl.addWidget(label("words", "unit"))
        self.key = QLineEdit(objectName="find", placeholderText="Paste your key", echoMode=QLineEdit.Password)
        self.key.setAccessibleName("API key")
        self.key_eye = self.key.addAction(QIcon(), QLineEdit.TrailingPosition)   # show / hide the key
        self.key_eye.setToolTip("Show the key")
        self.key_eye.triggered.connect(self._toggle_key)
        self.group("Summaries and quizzes", [
            self.row(I_KEY, "AI key", "Claude (sk-ant-…), Gemini (AIza…) or OpenAI (sk-…): the app tells them apart. "
                     "Empty = Ollama on this PC. Stored encrypted for your Windows login", self.key),
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # the content always fits the width
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
        subtitle = label(sub, "rowsub")
        subtitle.setWordWrap(True)        # wraps instead of holding the page wide (issue #14: clipped rows)
        text.addWidget(subtitle)
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
        self.sheet.hide()                 # opens on Theme colour > Change
        return self.sheet

    # ----- behaviour

    def read_store(self) -> dict:
        values = {k: self.store.value(k, d, type=type(d)) for k, d in DEFAULTS.items()}
        stored = values["api_key"]
        values["api_key"] = unprotect(stored)
        if stored and not stored.startswith(PREFIX):
            self.store_key(values["api_key"])     # an old plain-text key: encrypt it now (issue #13)
        return values

    def _toggle_key(self):
        shown = self.key.echoMode() == QLineEdit.Normal
        self.key.setEchoMode(QLineEdit.Password if shown else QLineEdit.Normal)
        self.key_eye.setToolTip("Show the key" if shown else "Hide the key")

    def store_key(self, key: str):
        """The AI key goes into QSettings encrypted for this Windows user, never as plain text."""
        self.store.setValue("api_key", protect(key))

    def load(self, values: dict):
        for k, sw in self.switches.items():
            sw.set_checked_now(values[k])
        self.set_camera(values["camera"])
        self.key.setText(values["api_key"])
        self.words.setValue(values["summary_words"])
        self.mode = values["theme"]
        self.theme_group.button(["system", "light", "dark"].index(self.mode)).setChecked(True)
        self.seed_before_sheet = values["seed"]
        self.set_seed(values["seed"])

    def save(self):
        values = {k: sw.isChecked() for k, sw in self.switches.items()}
        values.update(seed=self.seed, theme=self.mode, summary_words=self.words.value(),
                      camera=self.camera, api_key=self.key.text().strip())
        for k, v in values.items():
            if k == "api_key":
                self.store_key(v)
            else:
                self.store.setValue(k, v)
        self.saved.emit(values)
        self.save_btn.setText("Saved")
        QTimer.singleShot(1500, lambda: self.save_btn.setText("Save changes"))

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
        self.pages.setCurrentWidget(self.chat)    # results (e.g. calibration) land in the chat
        self.nav.button(0).setChecked(True)
        if not was_visible:
            self.hide()
        elif was_max:
            self.showMaximized()
        else:
            self.showNormal()

    def show_page(self, page: QWidget):
        """Bring the window up on `page` (self.chat, self.testing or self.settings_page)."""
        self.pages.setCurrentWidget(page)
        self.nav.button(TAB_OF_PAGE[self.pages.currentIndex()]).setChecked(True)
        if not self.isVisible() or self.isMinimized():
            self.showNormal()
        self.raise_()
        self.activateWindow()

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
        self.key_eye.setIcon(glyph_icon(I_EYE, c["onSurfaceVariant"]))
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


def pick_ai() -> str | None:
    """First-run popup: paste an AI key. Returns it, or None (not now)."""
    store = QSettings("StudyAssistant", "StudyAssistant")
    seed, mode = store.value("seed", DEFAULTS["seed"]), store.value("theme", DEFAULTS["theme"])
    dlg = QDialog(None, Qt.Dialog | Qt.WindowCloseButtonHint)
    dlg.setObjectName("root")
    dlg.setAttribute(Qt.WA_StyledBackground)
    dlg.setWindowTitle("Choose an AI")
    dlg.setStyleSheet(build_qss(make_scheme(seed, is_dark(mode))))
    dlg.setFixedWidth(520)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(24, 20, 24, 20)
    v.setSpacing(12)
    v.addWidget(label("Which AI should write your summaries?", "sheetTitle"))
    hint = label("Paste a key from Claude (sk-ant-…), Gemini (AIza…) or OpenAI (sk-…): the app tells them apart "
                 "and remembers it (Settings > AI key). No key? Install Ollama from ollama.com and start the app "
                 "again: it is free and runs on this PC.", "sheetText")
    hint.setWordWrap(True)
    v.addWidget(hint)
    key = QLineEdit(objectName="find", placeholderText="Paste your key", echoMode=QLineEdit.Password)
    key.setAccessibleName("AI key")
    v.addWidget(key)
    actions = QHBoxLayout()
    actions.addStretch(1)
    later = button("Not now", "text")
    later.clicked.connect(dlg.reject)
    ok = button("Use this key", "filled")
    ok.clicked.connect(dlg.accept)
    actions.addWidget(later)
    actions.addWidget(ok)
    v.addLayout(actions)
    return key.text().strip() or None if dlg.exec() == QDialog.Accepted else None


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
