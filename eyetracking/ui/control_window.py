"""The calibration lab's main window (PySide6, Material 3 colours).

Three tabs in the title bar: Testing, Calibration, Settings. The calibration, gaze-check
and blink pages fill the screen with no title bar or buttons, then the window comes back
where it was. The window reopens on the page, Testing entry and type it was left on.

    Testing     a sidebar of pages:
      Calibrate   Calibration  the 12 types as chips, a row per collection (method_picker),
                               what the chosen one does, where it saves, the reuse-4x4-dots
                               switch for Sweep / Points / Line model, Start
      Tests       Gaze check   record mode (switch), person, type, how many calibrations (the
                               newest N, None when there are none), Refresh, a table comparing
                               every type's error for that person; start one type, every type
                               at once, or run the models over a recording
                  Blink test   what the blink score is, Start
      Data        Calibrations by person > type > date, in three columns (active /
                                Clear data…: everything of one person, or of everyone, to data/trash/
                                several files at once (Ctrl / Shift-click, Delete), Move to a person,
                                Removed: restore from the trash folders
                               non-active / not usable); double-click to read one in words and
                               remove samples, right-click to disable / enable / remove
                  Gaze checks  by person > type > date, each sample in words
                  Blink scores each test in words, with the cut-off it suggests
                  Log          what happened this run, newest first
    Settings    theme and colour, the webcam, person mode (on: data/<person>/, off: data/)

pick_camera() is the small camera popup used at first start. Every colour comes from
one seed colour through the Material 3 Fidelity scheme, in light and dark.

    python -m eyetracking.ui.control_window      (the window alone, no camera or app)
"""
import ctypes
import threading
import sys
from pathlib import Path
from string import Template

from materialyoucolor.dynamiccolor.material_dynamic_colors import MaterialDynamicColors
from materialyoucolor.hct import Hct
from materialyoucolor.scheme.scheme_fidelity import SchemeFidelity
from PySide6.QtCore import (QEasingCurve, QEvent, QObject, QPointF, QRectF, QRegularExpression, QSettings, Qt, QTimer,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QFont, QGuiApplication, QIcon, QKeySequence, QPainter, QPainterPath, QPalette, QPen,
                           QPixmap, QShortcut,
                           QRegularExpressionValidator)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QInputDialog,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from eyetracking.services.camera import camera_names
import eyetracking.services.storage as storage
from eyetracking.services.storage import (blink_score_lines, calibration_lines, calibration_usable, file_date,
                                          gaze_check_lines, list_blink_scores, list_calibrations, list_disabled,
                                          remove_calibration, remove_samples,
                                          set_calibration_enabled, trash_file)
from eyetracking.core.methods import (DEFAULT_METHOD, DEFAULT_MODEL, GRIDS, LEGACY, METHOD_BY_ID, METHODS, MODEL_BY_ID,
                                      MODELS, model_for, model_name, partners)

DEFAULTS = {
    "seed": "#E5483B",          # the logo's red
    "theme": "system",          # system | light | dark
    "camera": "",               # webcam NAME (Windows can renumber cameras between restarts)
    "person_mode": False,       # on: data under data/<person>/; off: straight under data/
    "person": "",
}
SWATCHES = ["#3D6CF6", "#00897B", "#2E8B57", "#E8B400", "#E5483B", "#D32F2F", "#D81B60", "#7E57C2"]


TEXT_FONTS = ["Segoe UI Variable Text", "Segoe UI", "Segoe Fluent Icons", "Segoe MDL2 Assets"]
ICON_FONTS = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]   # built into Windows 11 / Windows 10

# Segoe Fluent Icons code points
I_SETTINGS, I_SEARCH = "", ""
I_MIN, I_MAX, I_RESTORE, I_CLOSE = "", "", "", ""
I_CHECK, I_DISMISS, I_RESET, I_CHEVRON = "", "", "", ""
I_THEME, I_PALETTE, I_BALL, I_LAYERS = "", "", "", ""
I_FULLSCREEN, I_SPARKLE, I_LINES = "", "", ""
I_EYE, I_CAMERA, I_PERSON = "", "", ""   # View (eye), Camera
I_KEY = "\uE8D7"   # Permissions (key)
I_TESTING = "\uE9D9"   # Diagnostic
I_BLINK, I_CHART = "\uE7B3", "\uE9D2"   # RedEye, AreaChart
I_BACK, I_DELETE = "\uE72B", "\uE74D"
I_TABLE, I_TRAIN, I_UNDO, I_DOWNLOAD = "\uE80A", "\uE9F5", "\uE7A7", "\uE896"   # a grid table, gears, Undo, Download
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

/* keyboard focus: a 2px ring on every button. The ring is a border, which eats into the text's room,
   so each kind's padding gives that width back: focused text never moves or clips (19 Oct: "Start the
   blink test" lost its last letter while focused) */
QPushButton:focus { border: 2px solid $secondary; padding: 0 14px; }
QPushButton[kind="text"]:focus { padding: 0 10px; }
QPushButton[kind="outlined"]:focus { border: 2px solid $secondary; padding: 0 15px 0 11px; }
QPushButton[kind="icon"]:focus { padding: 0; }
QPushButton[kind="nav"]:focus { padding: 0 10px; }
QPushButton[kind="go"]:focus { padding: 0 14px; }
QPushButton[kind="chip"]:focus { border: 2px solid $secondary; padding: 0 13px; }
QPushButton[kind="chip"]:checked:focus { padding: 0 13px 0 7px; }
QPushButton[kind="assist"]:focus { border: 2px solid $secondary; padding: 0 9px 0 7px; }
#caps QPushButton:focus, QPushButton#settingsBtn:focus { padding: 0; }

/* setting groups */
#content { background: transparent; }
QScrollArea { border: none; background: transparent; }
QScrollArea > QWidget > QWidget { background: transparent; }   /* a scroll area's page: the surface, not Windows' grey */
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
QListWidget#sideList[dense="true"] { padding: 0 8px 0 12px; }   /* the Calibrations columns: no icon gutter; the */
QListWidget#sideList[dense="true"]::item { padding: 6px 0; }      /* side padding on the list, so wrapping measures right */
QListWidget#linesList { background: transparent; border: none; outline: none; font-size: 14px; }
QListWidget#linesList::item { padding: 4px 12px; color: $onSurface; }
QListWidget#linesList::item:selected { background: $selected; }
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

/* tables (Data > Train, Data > Gaze check table): an Excel-like grid in the theme */
QTableWidget#table { background: $surfaceContainerLowest; border: 1px solid $outlineVariant; border-radius: 8px;
    gridline-color: $outlineVariant; selection-background-color: $selected; selection-color: $onSurface; font-size: 13px; }
QHeaderView::section { background: $surfaceContainer; color: $onSurfaceVariant; border: none; padding: 4px 8px;
    border-right: 1px solid $outlineVariant; border-bottom: 1px solid $outlineVariant; font-size: 12px; font-weight: 600; }
QTableCornerButton::section { background: $surfaceContainer; border: none; }

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


class SpinButton(QPushButton):
    """A button whose glyph turns once when clicked (Refresh, Reset to defaults): the click is seen to
    land, like the turning arrows in Windows' and Android's refresh buttons. The glyph is drawn as an
    icon so it can turn; set_color() follows the theme. It turns anticlockwise, the way its arrow
    (Segoe's Reset, U+E777) points: turning clockwise, it ran against its own arrowhead (20 Oct).
    The turn starts once the click's work is done: Refresh blocks ~200 ms, and a turn started before
    it was first painted ~270 degrees round, which looked like a quarter turn clockwise (20 Oct)."""

    def __init__(self, glyph: str, text: str, kind: str):
        super().__init__(text)
        self.glyph, self.angle, self.color = glyph, 0.0, "#000000"
        self.setProperty("kind", kind)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self._turn = QVariantAnimation(self, startValue=0.0, endValue=-360.0, duration=550,
                                       easingCurve=QEasingCurve.OutCubic)
        self._turn.valueChanged.connect(self._draw)
        self.clicked.connect(lambda: QTimer.singleShot(0, self._turn.start))

    def set_color(self, color: str):
        self.color = color
        self._draw(self.angle)

    def _draw(self, angle: float):
        self.angle = angle
        px = 16
        pm = QPixmap(px * 2, px * 2)
        pm.setDevicePixelRatio(2)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        f = QFont()
        f.setFamilies(ICON_FONTS)
        f.setPixelSize(px - 2)
        p.setFont(f)
        p.setPen(QColor(self.color))
        p.translate(px / 2, px / 2)
        p.rotate(angle)
        p.drawText(QRectF(-px / 2, -px / 2, px, px), Qt.AlignCenter, self.glyph)
        p.end()
        self.setIcon(QIcon(pm))


class _FadeOut(QWidget):
    """A picture of the page being left, laid over the new one and faded out: see cross_fade()."""

    def __init__(self, stack: QWidget, picture: QPixmap, ms: int):
        super().__init__(stack)
        self.picture, self.opacity = picture, 1.0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)   # the new page answers clicks at once
        self.setGeometry(stack.rect())
        fade = QVariantAnimation(self, startValue=1.0, endValue=0.0, duration=ms, easingCurve=QEasingCurve.OutCubic)
        fade.valueChanged.connect(lambda v: (setattr(self, "opacity", v), self.update()))
        fade.finished.connect(self.deleteLater)
        self.show()
        self.raise_()
        fade.start()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setOpacity(self.opacity)
        p.drawPixmap(0, 0, self.picture)


def cross_fade(stack, index: int, ms: int = 180):
    """Switch a QStackedWidget's page with a short cross-fade instead of a jump: the old page's picture
    fades out over the new one. Nothing is done to the pages themselves (a QGraphicsOpacityEffect on a
    page with a scroll area painted it black on 19 Oct), so it can't leave a page drawn wrongly."""
    if stack.currentIndex() == index or not stack.isVisible():
        stack.setCurrentIndex(index)
        return
    picture = stack.currentWidget().grab()
    stack.setCurrentIndex(index)
    _FadeOut(stack, picture, ms)


class BusyBar(QWidget):
    """A thin bar along the top of the window that runs while the app works (opening the camera,
    training a model, checking and saving a calibration, running a recording): M3's indeterminate
    linear progress indicator, as IDEs show background work. It sits over the page, 3 px high, and
    takes no clicks. It only moves while events are processed: App._busy keeps them going."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.color, self.track = QColor("#E5483B"), QColor(229, 72, 59, 50)
        self._t = 0.0
        self._run = QVariantAnimation(self, startValue=0.0, endValue=1.0, duration=1300, loopCount=-1)
        self._run.valueChanged.connect(lambda v: (setattr(self, "_t", v), self.update()))
        self.hide()

    def start(self):
        self.raise_()
        self.show()
        self._run.start()

    def stop(self):
        self._run.stop()
        self.hide()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), self.track)
        w = self.width()
        x = (self._t * 1.5 - 0.4) * w                    # a 40%-wide segment sweeping left to right
        p.fillRect(QRectF(x, 0, 0.4 * w, self.height()), self.color)


_running = [False]          # run_busy is waiting: a call made meanwhile (a click it answered) runs straight away


def busy() -> bool:
    """True while run_busy waits for slow work: a second click on the same button can check it."""
    return _running[0]


def run_busy(bar: BusyBar, work, tick=None):
    """work() on a worker thread while this, the GUI thread, keeps the window alive: the bar runs, the
    pointer shows busy, clicks and paints are answered (tick(), if given, is called every 50 ms, e.g. to
    show progress). Windows marks a window "Not Responding" when its thread stops answering for 5 s: a
    cold camera driver, a first `import anthropic` (5 s here), training a dozen models or saving a big
    calibration could each do that on this PC. Returns work()'s result; raises its exception. Called
    again while already running (a click answered meanwhile), it just runs work() here: never two pumps."""
    if _running[0]:
        return work()
    box: dict = {}

    def job() -> None:
        try:
            box["result"] = work()
        except BaseException as e:                # handed back to the caller below
            box["error"] = e
    worker = threading.Thread(target=job, daemon=True)
    _running[0] = True
    bar.start()
    QApplication.setOverrideCursor(Qt.BusyCursor)
    worker.start()
    try:
        while worker.is_alive():
            QApplication.processEvents()
            if tick is not None:
                tick()
            worker.join(0.05)
    finally:
        _running[0] = False
        bar.stop()
        QApplication.restoreOverrideCursor()
    if "error" in box:
        raise box["error"]
    return box.get("result")


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


MARK_SVG = Path(__file__).with_name("app_mark.svg")   # the logo: an eye whose pages are a screen ("1b · Screen pages")


def app_icon() -> QIcon:
    """The logo as the window / taskbar icon, crisp at every size Windows asks for."""
    renderer = QSvgRenderer(str(MARK_SVG))
    icon = QIcon()
    for px in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(px, px)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        renderer.render(p)
        p.end()
        icon.addPixmap(pm)
    return icon


class AppMark(QWidget):
    """The logo in the title bar, drawn from the SVG (its own colours: red, cream, navy)."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(22, 22)
        self._svg = QSvgRenderer(str(MARK_SVG), self)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        self._svg.render(p, QRectF(0, 0, 22, 22))


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

def method_picker(group: QButtonGroup, methods=METHODS) -> QWidget:
    """Calibration types as chips, a labelled line per row ("In order", "Random order", "Special
    steps"): three chips a line. Six (~770 px) didn't fit beside the sidebar in a 920 px window, so the
    layout squeezed the chips below their text and cut it (19 Oct). A chip's id in `group` is its index
    in METHODS; `methods` picks which types (the gaze check's dots: the grids only)."""
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(8)
    grid.setVerticalSpacing(8)
    row = 0
    for name in dict.fromkeys(m.row for m in methods):
        mine = [(METHODS.index(m), m) for m in methods if m.row == name]
        grid.addWidget(label(name, "rowsub"), row, 0, Qt.AlignTop)
        for k, (i, m) in enumerate(mine):
            b = CheckButton(m.name, "chip")
            group.addButton(b, i)
            grid.addWidget(b, row + k // 3, 1 + k % 3)
        row += (len(mine) + 2) // 3
    grid.setColumnStretch(4, 1)
    return box


WITH = (None, "sweep", "points", "line-model")      # the gaze check's "With" chips: the grid alone, or a special


def results_table(results, chosen: str, colors: tuple[str, str]) -> str:
    """The gaze-check comparison as rich text (storage.model_results): per model what it trains on, its
    checks, latest and mean error, a bar for the mean (the worst so far the longest) and, once two or
    more models have been checked, the lowest mean marked in colors[0] (the others' bars in colors[1]):
    which model is the most accurate for this person, at a glance."""
    pct = lambda e: "–" if e is None else f"{e:.1%}"
    means = {m: sum(errors) / len(errors) for m, _, errors in results if errors}
    best = min(means, key=means.get) if len(means) >= 2 else None
    worst = max(means.values(), default=0) or 1
    rows = []
    for m, used, errors in results:
        name = f"<b>{model_name(m)}</b>" if m == chosen else model_name(m)
        tag = f"&nbsp; <span style='color:{colors[0]}'><b>lowest</b></span>" if m == best else ""
        bar = (f"<table cellspacing=0 cellpadding=0><tr><td bgcolor='{colors[0] if m == best else colors[1]}' "
               f"width={max(3, round(120 * means[m] / worst))} style='font-size:6px'>&nbsp;</td></tr></table>"
               if m in means else "")
        rows.append(f"<tr><td>{name}{tag}</td><td align=right>{used}</td><td align=right>{len(errors)}</td>"
                    f"<td align=right>{pct(errors[0] if errors else None)}</td><td align=right>{pct(means.get(m))}</td>"
                    f"<td valign=middle>{bar}</td></tr>")
    if not rows:
        return "No calibrations or gaze checks for this person yet."
    return ("<table cellspacing=0 cellpadding=4><tr><th align=left>Model</th><th>Calibrations</th>"
            "<th>Gaze checks</th><th>Latest error</th><th>Mean error</th><th></th></tr>" + "".join(rows) + "</table>")


class ClickAway(QObject):
    """Several files picked with Ctrl- / Shift-click stay picked while the list scrolls (the wheel, its
    scroll bar), and are let go by a click anywhere else in the window: another column, the page, a
    heading. A plain click on a file in the list makes it the only one picked (Qt's ExtendedSelection
    does that itself). Not let go: a click on a button (it may act on the picked files, as Remove
    selected does), in a popup (the right-click menu, a question box: another window), or on the title
    bar or a window edge (moving or resizing the window is not a click on the page). `views`: lists, or
    tuples of a list and the panes that belong to it (a Data page's words of the picked file: reading or
    scrolling them is not "elsewhere")."""

    def __init__(self, parent: QWidget, views):
        super().__init__(parent)
        self.views = [v if isinstance(v, tuple) else (v,) for v in views]
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseButtonPress:
            target = QApplication.widgetAt(ev.globalPosition().toPoint())
            frame = target
            while frame is not None and not isinstance(frame, (TitleBar, Grip)):
                frame = frame.parentWidget()
            if target is not None and frame is None and not isinstance(target, QAbstractButton):
                for view, *panes in self.views:
                    if (view.selectionModel() is not None and view.selectionModel().hasSelection()
                            and view.isVisible() and target.window() is view.window()
                            and not any(w is target or w.isAncestorOf(target) for w in (view, *panes))):
                        view.clearSelection()
        return False


def side_list(on_open) -> QListWidget:
    """A list of dated files; double-click (or Enter) calls on_open(path). Ctrl / Shift-click picks several."""
    lst = QListWidget(objectName="sideList")
    lst.setToolTip("Double-click to open. Ctrl- or Shift-click to pick several, then right-click or Delete")
    lst.setSelectionMode(QListWidget.ExtendedSelection)
    lst.itemActivated.connect(lambda it: it.data(Qt.UserRole) and on_open(Path(it.data(Qt.UserRole))))
    return lst


def picked(lst: QListWidget, pos=None) -> list[Path]:
    """The files picked in a list. A right-click on a file that isn't picked picks it alone first."""
    item = lst.itemAt(pos) if pos is not None else None
    if item is not None and item.data(Qt.UserRole) and not item.isSelected():
        lst.clearSelection()
        item.setSelected(True)
    return [Path(it.data(Qt.UserRole)) for it in lst.selectedItems() if it.data(Qt.UserRole)]


def count(paths: list[Path], what: str) -> str:
    return f"{len(paths)} {what}s" if len(paths) != 1 else paths[0].name


def add_move_menu(menu: QMenu, paths: list[Path], move) -> None:
    """Move to > every other person, No person (data/), or a new name: move(person name)."""
    sub = menu.addMenu(f"{I_PERSON}  Move to")
    mine = {storage.person_of(p) for p in paths}
    for name, _ in storage.person_roots():
        if mine != {name}:
            sub.addAction(name or "No person (data/)", lambda n=name: move(n))
    sub.addAction("A new person…", lambda: (n := storage.person_name(QInputDialog.getText(
        menu.parentWidget(), "Move to a new person", "Name (letters, digits, spaces, - and _):")[0])) and move(n))


def move_all(paths: list[Path], person: str) -> str:
    """Move files to a person; a file of the same name already there stays put. What happened, in words."""
    clash = []
    for p in paths:
        try:
            storage.move_to_person(p, person)
        except FileExistsError:
            clash.append(p.name)
    done = f"Moved {len(paths) - len(clash)} file{'s' if len(paths) - len(clash) != 1 else ''} to {person or 'No person (data/)'}."
    return done + (f" Already there, left in place: {', '.join(clash)}." if clash else "")


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


def fill_tree(lst: QListWidget, rows: list, empty: str, why=None):
    """Headings and dated files in order. A row is ("person", name), ("type", method name)
    or a Path. Headings can't be clicked or opened; a person's is larger than a type's,
    types sit under names and dates under types (indented, so the levels read at a glance).
    why(path), if given, adds a line to the file's tooltip (why a calibration is not usable)."""
    lst.clear()
    indent = "   " if any(not isinstance(r, Path) and r[0] == "person" for r in rows) else ""
    for row in rows:
        if isinstance(row, Path):
            when = f"{file_date(row):%d %b  %H:%M}" if lst.property("dense") else f"{file_date(row):%d %b %Y  %H:%M}"
            item = QListWidgetItem(f"{indent}   {when}")
            reason = why(row) if why else None
            item.setToolTip(f"{file_date(row):%d %b %Y, %H:%M}\n{row.name}" + (f"\nNot usable: {reason}" if reason else ""))
            item.setData(Qt.UserRole, str(row))
        else:
            kind, text = row
            item = QListWidgetItem(text if kind == "person" else indent + text)
            item.setFlags(Qt.NoItemFlags)
            font = item.font()
            font.setBold(True)
            if kind == "person" and not lst.property("dense"):   # dense columns: bold only, so a long name wraps
                font.setPointSizeF(font.pointSizeF() * 1.15)
            item.setFont(font)
        lst.addItem(item)
    if not rows and empty:
        lst.addItem(QListWidgetItem(empty))


CHECK_IDS = [m.id for m in MODELS] + list(LEGACY)       # every model a gaze check can be of, the old types last


def person_type_rows(list_for, ids=CHECK_IDS) -> list:
    """fill_tree rows of every person's files: list_for(model id, person folder) -> paths.
    Person headings only once people exist; a model only when it has files."""
    roots = storage.person_roots()
    rows = []
    for name, root in roots:
        mine = []
        for m in ids:
            paths = list_for(m, root)
            if paths:
                mine += [("type", model_name(m)), *paths]
        if mine:
            heading = (name or "No person (data/)") + ("   · in use" if root == storage.DATA_DIR else "")
            rows += ([("person", heading)] if len(roots) > 1 else []) + mine
    return rows


class DataPage(QWidget):
    """One kind of recorded data: the files by date on the left, the chosen one in
    words on the right. Right-click a file to remove it (it moves to a trash folder
    beside it)."""

    def __init__(self, title: str, hint: str, list_files, to_lines, empty: str, name=None, tree=False):
        super().__init__()
        self._list, self._lines, self._empty, self._name, self._tree = list_files, to_lines, empty, name, tree
        h = QHBoxLayout(self)
        h.setContentsMargins(24, 20, 24, 24)
        h.setSpacing(24)
        self.left = left = QVBoxLayout()                 # the title, the words, the files (a page may add a button)
        left.setSpacing(8)
        left.addWidget(label(title, "sheetTitle"))
        text = label(hint, "sheetText")
        text.setWordWrap(True)
        left.addWidget(text)
        self.files = QListWidget(objectName="sideList")
        self.files.setSelectionMode(QListWidget.ExtendedSelection)    # Ctrl / Shift-click: several at once
        self.files.currentItemChanged.connect(lambda it, _: self._show(it))
        self.files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.files.customContextMenuRequested.connect(self._menu)
        QShortcut(QKeySequence.Delete, self.files, lambda: self.remove(picked(self.files)),
                  context=Qt.WidgetShortcut)
        left.addWidget(self.files, 1)
        col = QWidget()
        col.setLayout(left)
        col.setFixedWidth(320)
        h.addWidget(col)
        self.lines = QListWidget(objectName="linesList")
        self.lines.setWordWrap(True)
        self.lines.setSelectionMode(QListWidget.NoSelection)
        self.lines.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # wrap to the width, never clip
        self.lines.setResizeMode(QListWidget.Adjust)                    # re-wrap when the scrollbar appears
        h.addWidget(self.lines, 1)

    def refresh(self):
        self.files.blockSignals(True)
        if self._tree:
            fill_tree(self.files, self._list(), self._empty)
        else:
            fill_list(self.files, self._list(), self._empty, self._name)
        self.files.blockSignals(False)
        first = next((i for i in range(self.files.count()) if self.files.item(i).data(Qt.UserRole)), 0)
        self.files.setCurrentRow(first)                  # the first file, past the headings
        self._show(self.files.currentItem())

    def _show(self, item):
        self.lines.clear()
        if item is not None and item.data(Qt.UserRole):
            for line in self._lines(Path(item.data(Qt.UserRole))):
                self.lines.addItem(QListWidgetItem(line))

    def _menu(self, pos):
        paths = picked(self.files, pos)
        if not paths:
            return
        menu = QMenu(self.files)
        add_move_menu(menu, paths, lambda person: (move_all(paths, person), self.refresh()))
        menu.addAction(f"{I_DELETE}  Remove {count(paths, 'file')}", lambda: self.remove(paths))
        menu.exec(self.files.mapToGlobal(pos))

    def remove(self, paths: list[Path]):
        """Into the trash folder beside each file (move it back to undo); several only after a yes."""
        if not paths or len(paths) > 1 and QMessageBox.question(
                self, "Remove", f"Move {len(paths)} files to their trash folders?") != QMessageBox.Yes:
            return
        for p in paths:
            trash_file(p)
        self.refresh()


# ---------------------------------------------------------------- testing page

class TestingPage(QWidget):
    """The lab's working page. The sidebar picks what the right side shows:
    Calibration (the method chooser and its start button), the tests (gaze check,
    blink test) and the data (the chosen method's calibrations, every gaze check,
    the blink scores, the log of what happened)."""

    command = Signal(str)          # a start button: "calibrate" | "check" | "check-all" | "blink"
    replay_requested = Signal(str) # Recordings > one of them: run the models over it (its json path)
    method_changed = Signal(str)   # another method chosen: its id
    data_changed = Signal()        # samples or a file removed / disabled / enabled: the app retrains
    train_requested = Signal(str, list)   # Data > Train: (person, model ids) to train now
    undo_requested = Signal()      # the Undo button (or Ctrl+Z): the last calibration / gaze check goes

    def __init__(self, method_id: str = DEFAULT_METHOD):
        super().__init__()
        self.method_id = method_id
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QFrame(objectName="side")
        side.setFixedWidth(260)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 12, 12, 12)
        sv.setSpacing(2)
        self.side_nav = QButtonGroup(self)               # the chosen entry stays highlighted: it is the page on the right
        sections = [("Calibrate", [(I_EYE, "Calibration", 0)]),
                    ("Tests", [(I_CHART, "Gaze check", 1), (I_READ, "Sample text", 6), (I_BLINK, "Blink test", 2)]),
                    ("Data", [(I_LINES, "Calibrations", 3), (I_TRAIN, "Train", 9), (I_CHART, "Gaze checks", 4),
                              (I_TABLE, "Gaze check table", 8), (I_BLINK, "Blink scores", 5)])]
        for section, items in sections:
            sv.addWidget(label(section, "sideTitle"))
            for glyph, text, i in items:
                b = button(f"{glyph}   {text}", "nav")
                b.setCheckable(True)
                self.side_nav.addButton(b, i)                # its id = the page index on the right
                sv.addWidget(b)
            sv.addSpacing(8)
        sv.addStretch(1)
        self.status = label("", "rowsub")                # the latest word from the app (the model, the camera)
        self.status.setWordWrap(True)
        sv.addWidget(self.status)
        # Undo the last calibration or gaze check (a snackbar's action, M3): shown while there is one to undo
        self.undo_btn = button(f"{I_UNDO}  Undo", "text")
        self.undo_btn.clicked.connect(self.undo_requested.emit)
        self.undo_btn.hide()
        sv.addWidget(self.undo_btn, 0, Qt.AlignLeft)
        QShortcut(QKeySequence.Undo, self, lambda: self.undo_btn.isVisible() and self.undo_requested.emit(),
                  context=Qt.WidgetWithChildrenShortcut)
        h.addWidget(side)

        # 0: calibration (the method chooser); 1: gaze check; 2: blink test; 3: calibrations; 4: gaze checks;
        # 5: blink scores; 6: sample text (where the log was, so a saved page index still fits);
        # 7: one calibration's detail (from 3); 8: the gaze check table; 9: train
        self.main = QStackedWidget()
        self.main.addWidget(self._calibrate_page())
        self.check_person = ""                   # whose model the check tests (the app sets the person in use)
        self.results: list = []                  # storage.model_results of that person: the comparison table
        self.bar_colors = (DEFAULTS["seed"], mix(DEFAULTS["seed"], "#FFFFFF", .35))   # (lowest, others): apply_theme
        self.check_model = DEFAULT_MODEL                                  # ...and which model
        check = QScrollArea()                     # nine types + the table: taller than a small window
        check.setWidget(self._check_page())
        check.setWidgetResizable(True)
        check.setFrameShape(QFrame.NoFrame)
        check.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        check.viewport().setAutoFillBackground(False)
        self.main.addWidget(check)
        self.main.addWidget(self._intro(
            I_BLINK, "Blink test", "Record your blink scores in six eye states, so the blink cut-off can be set from "
            "your own numbers instead of a guess.",
            ["A blink score is a number MediaPipe gives for each eye every frame: 0 = wide open, 1 = shut. The app uses "
             "it to skip frames where the eyes are closing or closed, so calibration doesn't record them.",
             "Today the cut-off is a guess: 0.50 on both eyes. The test asks you to keep your eyes open, look down, "
             "squint, close them, and wink each eye, 6 seconds each after a 3-second countdown, and records the scores.",
             "The result is listed under Data > Blink scores, in words, with the cut-off it suggests. Space starts, "
             "Esc leaves."],
            "Start the blink test", "blink"))
        lists = QWidget()
        mv = QVBoxLayout(lists)
        mv.setContentsMargins(24, 20, 24, 24)
        mv.setSpacing(8)
        self.lists_title = label("Calibrations", "sheetTitle")
        mv.addWidget(self.lists_title)
        hint = label("Every calibration, by person, then by calibration type, then by date (data/<person>/<type>/). "
                     "Double-click one to read it in words. Right-click to disable it (kept, but it stops training), enable "
                     "it again, move it to another person, or remove it; Ctrl- or Shift-click picks several, Delete removes "
                     "them. Removed files wait under Removed: right-click to restore. The person in use (Settings > Person "
                     "mode) is the one whose active files train the models.", "sheetText")
        hint.setWordWrap(True)
        mv.addWidget(hint)
        self.clear_btn = button(f"{I_DELETE}  Clear data…", "outlined")   # everything of one person, or of everyone
        self.clear_menu = QMenu(self.clear_btn)
        self.clear_menu.aboutToShow.connect(self._fill_clear_menu)
        self.clear_btn.setMenu(self.clear_menu)
        mv.addWidget(self.clear_btn, 0, Qt.AlignLeft)
        cols = QHBoxLayout()
        cols.setSpacing(24)
        self.active_list = side_list(self.open_detail)
        self.disabled_list = side_list(self.open_detail)
        self.inactive_list = side_list(self.open_detail)
        self.removed_list = side_list(self.open_detail)
        for title, sub, lst in [("Active", "Trains the models", self.active_list),
                                ("Non-active", "Works, but you turned it off", self.disabled_list),
                                ("Not usable", "Point at one to see why (another version, a copy of another type, an old type)",
                                 self.inactive_list),
                                ("Removed", "In the trash folders: right-click to restore", self.removed_list)]:
            lst.setContextMenuPolicy(Qt.CustomContextMenu)
            lst.customContextMenuRequested.connect(lambda pos, lst=lst: self._list_menu(lst, pos))
            lst.setProperty("dense", True)         # four columns beside the sidebar: short rows (the year in the
            lst.setWordWrap(True)                  # tooltip), and a long name wraps rather than running off the edge
            lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            lst.setResizeMode(QListWidget.Adjust)
            if lst is not self.removed_list:
                QShortcut(QKeySequence.Delete, lst, lambda lst=lst: self.remove_files(picked(lst)),
                          context=Qt.WidgetShortcut)
            col = QVBoxLayout()
            col.setSpacing(2)
            col.addWidget(label(title, "gh"))
            note = label(sub, "rowsub")
            note.setWordWrap(True)                 # unwrapped, the longest subtitle took its column's width from the others
            col.addWidget(note)
            col.addWidget(lst, 1)
            cols.addLayout(col, 1)
        mv.addLayout(cols, 1)
        self.main.addWidget(lists)
        self.gaze_checks = DataPage(
            "Gaze checks", "By person, then by model, then by date. Each sample with the ring that was lit and how "
            "far the model was from it. Right-click a file (Ctrl- or Shift-click for several) to move it to another "
            "person or remove it. \"(old)\": a type from before 20 Oct.",
            lambda: person_type_rows(storage.list_gaze_checks), gaze_check_lines, "No gaze check yet", tree=True)
        self.clear_checks_btn = button(f"{I_DELETE}  Clear gaze checks…", "outlined")   # only the checks: one person, or all
        self.clear_checks_menu = QMenu(self.clear_checks_btn)
        self.clear_checks_menu.aboutToShow.connect(self._fill_clear_checks_menu)
        self.clear_checks_btn.setMenu(self.clear_checks_menu)
        self.gaze_checks.left.insertWidget(2, self.clear_checks_btn, 0, Qt.AlignLeft)   # under the page's words
        self.blink_scores = DataPage(
            "Blink scores", "Each blink test: your scores per eye state, and the cut-off they suggest. Right-click a "
            "file to remove it.", list_blink_scores, blink_score_lines, "No blink test yet")
        self.main.addWidget(self.gaze_checks)
        self.main.addWidget(self.blink_scores)
        self.main.addWidget(self._intro(
            I_READ, "Sample text", "Read a short text while a gaze model follows you: how well does it know which "
            "paragraph you are on?",
            ["The model is the one picked on the Gaze check page: the person, the calibration type and how many of "
             "their newest calibrations train it.",
             "A paragraph counts as read once your gaze has stayed on it for 0.1 s a word (the app's own rule), and "
             "turns green. A faint dot shows where the model thinks you look.",
             "Read at your own pace, top to bottom. Space pauses, Esc finishes: the result (paragraphs read, seconds "
             "on each) shows in the status line."],
            "Start the sample text", "sample"))

        detail = QWidget()                               # one calibration, every sample in words
        dv = QVBoxLayout(detail)
        dv.setContentsMargins(24, 20, 24, 24)
        dv.setSpacing(8)
        head = QHBoxLayout()
        back = button(f"{I_BACK}  Back", "text")
        back.clicked.connect(lambda: self.main.setCurrentIndex(3))
        self.detail_title = label("", "sheetTitle")
        head.addWidget(back)
        head.addWidget(self.detail_title, 1)
        dv.addLayout(head)
        hint = label("Select samples, or a dot's line for all of its samples, then Remove: the file is rewritten and "
                     "the model retrained. The untouched original stays in the method's trash folder.", "sheetText")
        hint.setWordWrap(True)
        dv.addWidget(hint)
        self.detail = QListWidget(objectName="linesList")
        self.detail.setSelectionMode(QListWidget.ExtendedSelection)
        self.detail.setWordWrap(True)
        dv.addWidget(self.detail, 1)
        self.remove_btn = button(f"{I_DELETE}  Remove selected", "outlined")
        self.remove_btn.clicked.connect(self.remove_selected)
        dv.addWidget(self.remove_btn, 0, Qt.AlignLeft)
        self.main.addWidget(detail)
        self.detail_path: Path | None = None
        from eyetracking.ui.check_table import CheckTablePage   # here: it uses this module's widgets
        self.check_table = CheckTablePage()
        self.check_table.said.connect(self.say)
        self.main.addWidget(self.check_table)
        self.main.addWidget(self._train_page())
        h.addWidget(self.main, 1)
        self.side_nav.idClicked.connect(lambda i: cross_fade(self.main, i))
        self.click_away = ClickAway(self, [self.active_list, self.disabled_list, self.inactive_list, self.removed_list,
                                           (self.gaze_checks.files, self.gaze_checks.lines),
                                           (self.blink_scores.files, self.blink_scores.lines), self.detail,
                                           self.train_table, self.check_table.table])
        self.set_method(method_id if method_id in METHOD_BY_ID else DEFAULT_METHOD, announce=False)
        self.refresh()                                   # the lists, read once now; again on show and after changes
        self.show_page(0)

    def show_page(self, i: int):
        """Open sidebar entry `i` (0-6, 8, 9; the calibration detail, 7, opens as its list)."""
        i = 3 if i == 7 else i if 0 <= i <= 9 else 0
        self.side_nav.button(i).setChecked(True)
        self.main.setCurrentIndex(i)

    def say(self, text: str):
        """A line from the app, in the sidebar's status."""
        self.status.setText(text)

    # ----- the method chooser

    def _calibrate_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 20, 24, 24)
        v.setSpacing(12)
        v.addWidget(label(f"{I_EYE}   Calibration", "sheetTitle"))
        lead = label("Pick a calibration. A grid (In order, Random order) records dots; a special (Sweep, Points, Line "
                     "model) records its own first step and the head step, and pairs with any grid of the same person: "
                     "every pair is a model of its own (Data > Train). All of them record the same 12 numbers per frame.",
                     "lead")
        lead.setWordWrap(True)
        v.addWidget(lead)
        self.method_chips = QButtonGroup(self)
        v.addWidget(method_picker(self.method_chips))
        self.method_chips.idClicked.connect(lambda i: self.set_method(METHODS[i].id))
        self.method_about = label("", "sheetText")
        self.method_about.setWordWrap(True)
        v.addWidget(self.method_about)
        self.method_steps = label("", "sheetText")
        self.method_steps.setWordWrap(True)
        v.addWidget(self.method_steps)
        self.share_row = QWidget()                       # Points / Line model: one recording saved as both
        share = QHBoxLayout(self.share_row)
        share.setContentsMargins(0, 0, 0, 0)
        share.setSpacing(12)
        self.share_switch = Switch("Save it as both")
        self.share_switch.set_checked_now(True)
        self.share_note = label("", "sheetText")
        self.share_note.setWordWrap(True)
        share.addWidget(self.share_switch)
        share.addWidget(self.share_note, 1)
        v.addWidget(self.share_row)
        go = button("Start the calibration", "filled")
        go.clicked.connect(lambda: self.command.emit("calibrate"))
        v.addSpacing(8)
        v.addWidget(go, 0, Qt.AlignLeft)
        v.addStretch(1)
        return page

    def set_method(self, method_id: str, announce: bool = True):
        """Choose the method the calibration, the gaze check and the Calibrations list work with."""
        m = METHOD_BY_ID[method_id]
        self.method_id = method_id
        self.method_chips.button(METHODS.index(m)).setChecked(True)
        self.method_about.setText(m.about)
        self.method_steps.setText("\n".join(f"{i + 1}.  {s}" for i, s in enumerate(m.steps))
                                  + f"\n\nSaved as {storage.method_dir(m.id).relative_to(storage.BASE_DIR.parent).as_posix()}"
                                    f"/{m.id}_<date>_<time>.json")
        self._update_share()                     # the lists don't depend on the type: no need to re-read data/
        if announce:
            self.method_changed.emit(method_id)

    @property
    def share_partner(self) -> bool:
        """Points / Line model: save the calibration as the other one too."""
        return self.share_switch.isChecked()

    def _update_share(self):
        m = METHOD_BY_ID[self.method_id]
        others = partners(m)
        self.share_row.setVisible(bool(others))
        if others:
            other = others[0].name
            self.share_note.setText(f"{m.name} and {other} record exactly the same steps, so this calibration is "
                                    f"saved as {other} too: both models train on the same recording, and only the "
                                    f"model differs between them. Off: {m.name} only.")

    # ----- the gaze check: one person's model (a grid, alone or paired with a special)

    def _person_picker(self, on_pick) -> tuple[QPushButton, QHBoxLayout]:
        """"Person" and its menu (every person, and No person): on_pick(name)."""
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(label("Person", "rowtitle"))
        btn = button("", "assist")
        menu = QMenu(btn)
        menu.aboutToShow.connect(lambda: (menu.clear(), [menu.addAction(n or "No person (data/)", lambda n=n: on_pick(n))
                                                         for n, _ in storage.person_roots()]))
        btn.setMenu(menu)
        row.addWidget(btn)
        row.addStretch(1)
        return btn, row

    def _check_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 20, 24, 24)
        v.setSpacing(12)
        v.addWidget(label(f"{I_CHART}   Gaze check", "sheetTitle"))
        lead = label("Test one person's model on 16 rings (4×4), or several models at once on 36 (6×6): the same "
                     "rings for all, so the errors can be compared. A model is a grid's dots, alone or with a "
                     "special's steps. Which is the most accurate for this person?", "lead")
        lead.setWordWrap(True)
        v.addWidget(lead)
        self.check_person_btn, pick = self._person_picker(lambda n: self.set_check(person=n))
        v.addLayout(pick)
        v.addWidget(label("Dots from", "rowtitle"))
        self.check_chips = QButtonGroup(self)               # the grid: its id is its index in METHODS
        v.addWidget(method_picker(self.check_chips, GRIDS))
        self.check_chips.idClicked.connect(lambda i: self.set_check(model_id=model_for(
            METHODS[i].id, MODEL_BY_ID[self.check_model].special and MODEL_BY_ID[self.check_model].special.id).id))
        with_row = QHBoxLayout()
        with_row.setSpacing(8)
        with_row.addWidget(label("With", "rowtitle"))
        self.with_chips = QButtonGroup(self)                # nothing, or a special: its id is its index in WITH
        for i, s in enumerate(WITH):
            b = CheckButton("Nothing (the dots alone)" if s is None else METHOD_BY_ID[s].name, "chip")
            self.with_chips.addButton(b, i)
            with_row.addWidget(b)
        with_row.addStretch(1)
        self.with_chips.idClicked.connect(lambda i: self.set_check(
            model_id=model_for(MODEL_BY_ID[self.check_model].grid.id, WITH[i]).id))
        v.addLayout(with_row)
        use = QHBoxLayout()
        use.setSpacing(8)
        use.addWidget(label("Calibrations to use", "rowtitle"))
        self.check_count, self.check_most = 0, 0                  # use the newest check_count of check_most
        self.check_count_btn = button("", "assist")
        self.check_count_menu = QMenu(self.check_count_btn)
        self.check_count_menu.aboutToShow.connect(self._fill_check_count_menu)
        self.check_count_btn.setMenu(self.check_count_menu)
        use.addWidget(self.check_count_btn)
        self.check_count_note = label("", "rowsub")
        use.addWidget(self.check_count_note)
        use.addStretch(1)
        fresh = SpinButton(I_RESET, "Refresh", "text")            # re-read data/: files added, moved or removed outside
        fresh.setToolTip("Read the data folder again: the people, the calibrations and the results")
        fresh.clicked.connect(self.refresh)
        use.addWidget(fresh)
        v.addLayout(use)
        # the start buttons straight after the choices (below the table they were out of view)
        go = button("Start the gaze check", "filled")
        go.clicked.connect(lambda: self.command.emit("check"))
        many = button("Multiple model gaze check", "outlined")
        many.setToolTip("Several of this person's models at once (a popup asks which), on 36 rings (6×6): one file per model")
        many.clicked.connect(lambda: self.pick_multi_counts() and self.command.emit("check-all"))
        self.multi_counts: dict[str, int] = {}             # the popup's choice: model -> grid calibrations (app reads it)
        self.recordings_btn = button("", "assist")        # run every model over an earlier recording
        self.recordings_menu = QMenu(self.recordings_btn)
        self.recordings_menu.aboutToShow.connect(self._fill_recordings_menu)
        self.recordings_btn.setMenu(self.recordings_menu)
        row = QHBoxLayout()
        row.setSpacing(8)
        for b in (go, many, self.recordings_btn):
            row.addWidget(b)
        row.addStretch(1)
        v.addSpacing(4)
        v.addLayout(row)
        mode = QHBoxLayout()                               # an option of the start: next to it
        mode.setSpacing(12)
        self.record_switch = Switch("Record mode")      # off at every start: a video of your face is not made by accident
        mode.addWidget(self.record_switch)
        note = label("Record mode: the webcam is recorded with the moment of each Enter, and the models run over "
                     "it afterwards as if live (any model, even one made later: Recordings). The video of your "
                     "face stays in the person's gaze-videos folder; delete it any time.", "sheetText")
        note.setWordWrap(True)
        mode.addWidget(note, 1)
        v.addLayout(mode)
        v.addSpacing(8)
        v.addWidget(label("How each model did", "rowtitle"))
        self.check_results = label("", "sheetText")              # the comparison: every model with data, this person
        self.check_results.setTextFormat(Qt.RichText)
        v.addWidget(self.check_results)
        note = label("Error: the mean distance from where the model said to the lit ring, as a share of the "
                     "screen. Lower is better; with a few checks only, a difference of a few points may be chance.",
                     "sheetText")
        note.setWordWrap(True)                                   # (in the table's label it set the page's width)
        v.addWidget(note)
        for s in ["One ring is lit (white). Look at it and press Enter: that records where the model says you look, "
                  "and the next ring lights up. A sample's error is its distance to the lit ring.",
                  "Multiple model gaze check: the models you tick in its popup, each on the newest N of its grid's "
                  "calibrations, all predicting from the same camera frames: 36 rings (6×6), no dot, one file per "
                  "model. One sitting, the fairest comparison. The whole picture: Data > Gaze check table.",
                  "Do a full round (16, or 36) for a fair number. Space pauses. Esc saves and leaves; the files go in "
                  "that person's folder of each model. Undo (under the status line, or Ctrl+Z) takes them back."]:
            w = label(f"•   {s}", "sheetText")
            w.setWordWrap(True)
            v.addWidget(w)
        v.addStretch(1)
        return page

    @property
    def record_mode(self) -> bool:
        return self.record_switch.isChecked()

    def _fill_recordings_menu(self):
        self.recordings_menu.clear()
        for path in storage.list_recordings(storage.person_root(self.check_person)):
            self.recordings_menu.addAction(f"Run the models over {storage.recording_label(path)}",
                                           lambda p=path: self.pick_multi_counts() and self.replay_requested.emit(str(p)))

    def pick_multi_counts(self) -> bool:
        """The popup before a multiple check (or a replay): which models, how many of each grid's
        calibrations. False if there is nothing to pick or it was cancelled."""
        root = storage.person_root(self.check_person)
        available = {m.id: len(storage.model_files(m, root)[0]) for m in MODELS if storage.model_ready(m, root) is None}
        if not available:
            self.say(f"{self.check_person or 'No person'}: no usable calibration of any type yet. Calibrate first.")
            return False
        dlg = MultiCheckDialog(self.check_person, available, self.multi_counts, self.window())
        if dlg.exec() != QDialog.Accepted or not dlg.chosen():
            return False
        chosen = dlg.chosen()
        self.multi_counts = {m: chosen.get(m, 0) for m in available}   # 0: left out, remembered
        return True

    def set_check_count(self, n: int):
        """Use the newest n calibrations (0 only when there's none: shown as None)."""
        self.check_count = n
        self.check_count_btn.setText(f"{n}{' (all)' if n == self.check_most else ''}  {I_CHEVRON}" if n else "None")
        self.check_count_btn.setEnabled(n > 0)

    def _fill_check_count_menu(self):
        self.check_count_menu.clear()
        for n in range(1, self.check_most + 1):
            self.check_count_menu.addAction(f"{n}{' (all)' if n == self.check_most else ''}",
                                            lambda n=n: self.set_check_count(n))

    def set_check(self, person: str | None = None, model_id: str | None = None):
        """Whose model and which model the gaze check tests; the comparison follows the person. How many
        of the grid's calibrations it uses goes back to all of them when the person or model changes."""
        changed = (person is not None and person != self.check_person) or \
                  (model_id is not None and model_id != self.check_model)
        if person is not None:
            self.check_person = person
        if model_id is not None:
            self.check_model = model_id
        model, root = MODEL_BY_ID[self.check_model], storage.person_root(self.check_person)
        missing = storage.model_ready(model, root)
        most = 0 if missing else len(storage.model_files(model, root)[0])
        keep = self.check_count
        self.check_most = most
        self.set_check_count(most if changed or keep == 0 or keep > most else keep)   # a choice is kept while it fits
        self.check_count_note.setText(f"of {most} {model.grid.name}, the newest first" if most else missing)
        self.check_person_btn.setText(f"{self.check_person or 'No person (data/)'}  {I_CHEVRON}")
        self.check_chips.button(METHODS.index(model.grid)).setChecked(True)
        self.with_chips.button(WITH.index(model.special and model.special.id)).setChecked(True)
        self.results = storage.model_results(root)
        self.show_results()
        n = len(storage.list_recordings(root))
        self.recordings_btn.setText(f"Recordings ({n})  {I_CHEVRON}" if n else "No recordings")
        self.recordings_btn.setEnabled(n > 0)

    def show_results(self):
        self.check_results.setText(results_table(self.results, self.check_model, self.bar_colors))

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

    # ----- the calibrations of the chosen method

    def open_detail(self, path: Path):
        """One calibration in words, in place of the lists."""
        self.detail_path = path
        self.detail_title.setText(path.name)
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
        self.remove_btn.setVisible(removable and path.parent.name != "trash")   # a removed file: restore it first
        self.main.setCurrentIndex(7)

    def remove_selected(self):
        if not (self.detail_path and self.detail_path.exists()):
            self.refresh()                               # gone meanwhile: back to the lists
            return
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
                self.main.setCurrentIndex(3)
            self.refresh()
            self.data_changed.emit()

    def _fill_clear_menu(self):
        self.clear_menu.clear()
        for name, _ in storage.person_roots():
            self.clear_menu.addAction(f"Clear {name or 'No person (data/)'}", lambda n=name: self.clear_data(n))
        self.clear_menu.addAction("Clear everyone", lambda: self.clear_data(None))

    def _fill_clear_checks_menu(self):
        self.clear_checks_menu.clear()
        for name, _ in storage.person_roots():
            self.clear_checks_menu.addAction(f"Clear {name or 'No person (data/)'}'s gaze checks",
                                             lambda n=name: self.clear_gaze_checks(n))
        self.clear_checks_menu.addAction("Clear everyone's gaze checks", lambda: self.clear_gaze_checks(None))

    def clear_gaze_checks(self, person: str | None):
        """Only the gaze checks of one person (or of everyone) move into data/trash/<date>_<time>/ (move them
        back to undo); calibrations, blink tests and recordings stay."""
        who = "everyone" if person is None else (person or "No person (data/)")
        if QMessageBox.question(self, "Clear gaze checks", f"Move all of {who}'s gaze checks to data/trash/? Their "
                                "calibrations stay. Move them back by hand to undo.") != QMessageBox.Yes:
            return
        where, n = storage.clear_gaze_checks(person)
        self.refresh()
        self.say(f"Cleared {n} gaze check{'s' if n != 1 else ''} of {who} into "
                 f"{where.relative_to(storage.BASE_DIR.parent).as_posix()}/." if n else f"{who}: no gaze check to clear.")

    def clear_data(self, person: str | None):
        """Every calibration, gaze check, blink test and recording of one person (or of everyone)
        moves into data/trash/<date>_<time>/ (move it back to undo)."""
        who = "everyone" if person is None else (person or "No person (data/)")
        if QMessageBox.question(self, "Clear data", f"Move all of {who}'s calibrations, gaze checks, blink tests "
                                "and recordings to data/trash/? Move them back by hand to undo.") != QMessageBox.Yes:
            return
        where = storage.clear_data(person)
        self.refresh()
        self.data_changed.emit()
        self.say(f"Cleared {who}'s data into {where.relative_to(storage.BASE_DIR.parent).as_posix()}/.")

    def _list_menu(self, lst: QListWidget, pos):
        paths = picked(lst, pos)
        if not paths:
            return
        menu = QMenu(lst)
        if lst is self.removed_list:
            menu.addAction(f"Restore {count(paths, 'calibration')}", lambda: self.restore_files(paths))
        else:
            if lst is self.active_list:
                menu.addAction("Disable (keep, stop training on it)", lambda: self.set_enabled(paths, False))
            elif lst is self.disabled_list:
                menu.addAction("Enable (train on it again)", lambda: self.set_enabled(paths, True))
            add_move_menu(menu, paths, lambda person: self._changed(move_all(paths, person)))
            # Points <-> Line model (the same steps); an old type's file to the special its steps really are
            targets = set.intersection(*(set(storage.copy_targets(p)) for p in paths))
            for t in [m for m in METHOD_BY_ID if m in targets]:
                menu.addAction(f"Copy to {METHOD_BY_ID[t].name} (the same recording trains both)",
                               lambda t=t: self.copy_files(paths, t))
            menu.addAction(f"{I_DELETE}  Remove {count(paths, 'calibration')}", lambda: self.remove_files(paths))
        menu.exec(lst.mapToGlobal(pos))

    def _changed(self, text: str):
        """Files moved, removed, restored, enabled or disabled: the lists, the models' status, the status line."""
        self.refresh()
        self.data_changed.emit()
        self.say(text)

    def set_enabled(self, paths: list[Path], enabled: bool):
        for p in paths:
            set_calibration_enabled(p, enabled)
        self._changed(f"{'Enabled' if enabled else 'Disabled'} {count(paths, 'calibration')}.")

    def copy_files(self, paths: list[Path], method_id: str):
        """Copy to > the same recordings as calibrations of a partner type (Sweep / Points / Line model)."""
        clash = []
        for p in paths:
            try:
                storage.copy_as(p, method_id)
            except FileExistsError:
                clash.append(p.name)
        done = len(paths) - len(clash)
        self._changed(f"Copied {done} calibration{'s' if done != 1 else ''} to {METHOD_BY_ID[method_id].name}."
                      + (f" Already there: {', '.join(clash)}." if clash else ""))

    def remove_files(self, paths: list[Path]):
        """Right-click > Remove (or Delete): into each type's trash folder; Removed lists them to restore."""
        if not paths or QMessageBox.question(self, "Remove calibrations", f"Move {count(paths, 'calibration')} to the "
                                             "trash? Restore from Removed any time.") != QMessageBox.Yes:
            return
        for p in paths:
            remove_calibration(p)
        self._changed(f"Removed {count(paths, 'calibration')}.")

    def restore_files(self, paths: list[Path]):
        """Removed > Restore. A trash file whose name is still in use is the original of an edited
        calibration: restoring it undoes the edits, so that asks first."""
        edited = [p for p in paths if (p.parent.parent / p.name).exists()]
        if edited and QMessageBox.question(self, "Restore", f"{count(edited, 'calibration')}: the untouched original "
                                           "of an edited calibration. Restore it in place of the edited one?"
                                           ) != QMessageBox.Yes:
            return
        for p in paths:
            storage.restore(p)
        self._changed(f"Restored {count(paths, 'calibration')}.")

    def refresh(self):
        """Re-read data/. Runs whenever the page shows, the method changes, or a file was saved."""
        roots = storage.person_roots()
        cols: tuple[list, ...] = ([], [], [], [])             # active, non-active, not usable, removed
        old = [k for k in LEGACY if k not in METHOD_BY_ID]     # the old random specials: not usable, still shown
        for name, root in roots:
            mine: tuple[list, ...] = ([], [], [], [])            # this person's rows, per column
            for m in [*METHOD_BY_ID, *old]:                     # every calibration type, under the person's name
                usable = {p: calibration_usable(p) for p in list_calibrations(m, root)}
                files = ([p for p, ok in usable.items() if ok], list_disabled(m, root),
                         [p for p, ok in usable.items() if not ok], storage.list_removed(m, root))
                for rows, paths in zip(mine, files):
                    if paths:
                        rows += [("type", METHOD_BY_ID[m].name if m in METHOD_BY_ID else model_name(m)), *paths]
            heading = (name or "No person (data/)") + ("   · in use" if root == storage.DATA_DIR else "")
            for col, rows in zip(cols, mine):
                if rows:                                        # people exist: the name above its types
                    col += ([("person", heading)] if len(roots) > 1 else []) + rows
        for lst, rows, empty in zip((self.active_list, self.disabled_list, self.inactive_list, self.removed_list), cols,
                                    ("None active: enable one, or calibrate", "None", "None", "None")):
            fill_tree(lst, rows, empty, storage.calibration_problem if lst is self.inactive_list else None)
        self.gaze_checks.refresh()
        self.blink_scores.refresh()
        self._update_share()
        self.set_check()
        self.refresh_train()
        self.check_table.refresh()
        if self.main.currentIndex() == 7 and not (self.detail_path and self.detail_path.exists()):
            self.main.setCurrentIndex(3)                 # the file on the detail page is gone (Undo, Clear, a move)

    def set_undo(self, what: str):
        """The Undo button under the status line: "Undo <what>", hidden when there is nothing to undo."""
        self.undo_btn.setText(f"{I_UNDO}  Undo {what}")
        self.undo_btn.setToolTip(f"Undo {what} (Ctrl+Z): its files go to the trash; Removed restores a calibration")
        self.undo_btn.setVisible(bool(what))

    # ----- Data > Train: every model, tick some, train them now

    TRAIN_HEAD = ("Model", "Dots from", "Steps from", "Samples", "Error on its dots", "Head matrix", "Status")

    def _train_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 20, 24, 24)
        v.setSpacing(12)
        v.addWidget(label(f"{I_TRAIN}   Train", "sheetTitle"))
        lead = label("Every model the lab makes: a grid's dots alone, or a grid with a special's steps, 24 in all. "
                     "Tick the ones to train for this person and press Train: each learns from the person's "
                     "calibrations now (the newest weighs most), so a gaze check of it starts at once. A model "
                     "without its data says what it needs.", "lead")
        lead.setWordWrap(True)
        v.addWidget(lead)
        self.train_person = ""
        self.train_person_btn, pick = self._person_picker(self.set_train_person)
        v.addLayout(pick)
        row = QHBoxLayout()
        row.setSpacing(8)
        go = button("Train the ticked models", "filled")
        go.clicked.connect(lambda: self.ticked() and self.train_requested.emit(self.train_person, self.ticked()))
        every = button("Tick every ready model", "text")
        every.clicked.connect(lambda: self.tick_models(True))
        none = button("Untick all", "text")
        none.clicked.connect(lambda: self.tick_models(False))
        for b in (go, every, none):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.train_table = QTableWidget(len(MODELS), len(self.TRAIN_HEAD), objectName="table")
        self.train_table.setHorizontalHeaderLabels(self.TRAIN_HEAD)
        self.train_table.verticalHeader().hide()
        self.train_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.train_table.setSelectionMode(QTableWidget.ExtendedSelection)   # Ctrl / Shift-click rows, Space ticks them
        self.train_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.train_table.setWordWrap(False)
        for r, m in enumerate(MODELS):
            for c in range(len(self.TRAIN_HEAD)):
                self.train_table.setItem(r, c, QTableWidgetItem(""))
            self.train_table.item(r, 0).setText(m.name)
            self.train_table.item(r, 0).setData(Qt.UserRole, m.id)
        self.train_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.train_table.horizontalHeader().setStretchLastSection(True)
        QShortcut(QKeySequence(Qt.Key_Space), self.train_table, self._space_ticks, context=Qt.WidgetShortcut)
        v.addWidget(self.train_table, 1)
        note = label("Error on its dots: how far the model's point is from the dots of the newest grid calibration it "
                     "learnt from, as a share of the screen. Low only says the fit is tight; the gaze check, on rings "
                     "it never saw, is the real test. Head matrix: the head-turn correction, the built-in one or the "
                     "special's own head step.", "sheetText")
        note.setWordWrap(True)
        v.addWidget(note)
        self.train_info: dict[tuple, tuple] = {}     # (person, model id) -> (the files it learnt from, the numbers)
        return page

    def set_train_person(self, person: str):
        self.train_person = person
        self.refresh_train()

    def _space_ticks(self):
        """Space ticks (or unticks) every selected row that can be trained."""
        rows = sorted({i.row() for i in self.train_table.selectedIndexes()})
        items = [self.train_table.item(r, 0) for r in rows if self.train_table.item(r, 0).flags() & Qt.ItemIsUserCheckable]
        on = any(i.checkState() != Qt.Checked for i in items)
        for i in items:
            i.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def tick_models(self, on: bool):
        for r in range(self.train_table.rowCount()):
            item = self.train_table.item(r, 0)
            if item.flags() & Qt.ItemIsUserCheckable:
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def ticked(self) -> list[str]:
        """The ids of the ticked models."""
        return [self.train_table.item(r, 0).data(Qt.UserRole) for r in range(self.train_table.rowCount())
                if self.train_table.item(r, 0).checkState() == Qt.Checked]

    def refresh_train(self):
        """Each model's row: what it would learn from for the person on the page, and what it learnt the last
        time it was trained (while those files are unchanged), or what it is missing."""
        self.train_person_btn.setText(f"{self.train_person or 'No person (data/)'}  {I_CHEVRON}")
        root = storage.person_root(self.train_person)
        c = getattr(self.window(), "c", None)
        dim = QColor(c["onSurfaceVariant"]) if c else QColor("gray")     # a model without its data: greyed
        for r, m in enumerate(MODELS):
            grids, specials = storage.model_files(m, root)
            missing = storage.model_ready(m, root)
            item = self.train_table.item(r, 0)
            ticked = item.checkState() == Qt.Checked
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | (Qt.ItemIsUserCheckable if not missing else Qt.NoItemFlags))
            item.setCheckState(Qt.Checked if ticked and not missing else Qt.Unchecked)
            files = tuple((p.name, p.stat().st_mtime_ns) for p in grids + specials)   # as App._job keys a model
            done = self.train_info.get((self.train_person, m.id))
            numbers = done[1] if done and done[0] == files else None
            newest = f"the newest of {len(specials)}" if len(specials) > 1 else "1"   # train() takes the newest that passes
            texts = (f"{len(grids)} {m.grid.name}" if grids else "–",
                     f"{newest} {m.special.name}" if m.special and specials else "–",
                     *(numbers or ("", "", "")),
                     missing or ("Trained" if numbers else "Ready: not trained yet"))
            for col, text in enumerate(texts, start=1):
                self.train_table.item(r, col).setText(text)
            for col in range(self.train_table.columnCount()):
                self.train_table.item(r, col).setData(Qt.ForegroundRole, dim if missing else None)

    def show_trained(self, person: str, results: dict[str, tuple]):
        """The app trained these: model id -> (the files it learnt from, (samples, error, head matrix))."""
        for m, (files, numbers) in results.items():
            self.train_info[(person, m)] = (files, numbers)
        self.refresh_train()

    def showEvent(self, e):
        self.refresh()
        super().showEvent(e)


# ---------------------------------------------------------------- window

class SettingsWindow(QWidget):
    saved = Signal(dict)              # Save changes pressed: every setting, see DEFAULTS

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setWindowTitle("Calibration lab")
        self.resize(1100, 700)
        self.setMinimumSize(960, 560)   # 960: the four Calibrations columns beside the sidebar fit (19 Oct)
        self.store = QSettings("StudyAssistant", "CalibrationLab")
        self.seed = self.store.value("seed", DEFAULTS["seed"])
        self.c = make_scheme(self.seed, False)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.titlebar = self._title_bar()
        root.addWidget(self.titlebar)
        # pages: 0 = testing, 1 = settings; a calibration / check / blink page is added while it runs
        self.testing = TestingPage(self.store.value("method", DEFAULT_METHOD))
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
        self.pages.addWidget(self.testing)
        self.pages.addWidget(self.settings_page)
        root.addWidget(self.pages, 1)
        self.busy = BusyBar(self)                    # runs under the title bar while the app works (App._busy)
        self._before_page = None          # (was visible, was maximized) while a full-screen page runs

        G = 5
        self.grips = [Grip(self, e, cur) for e, cur in [
            (Qt.LeftEdge, Qt.SizeHorCursor), (Qt.RightEdge, Qt.SizeHorCursor),
            (Qt.TopEdge, Qt.SizeVerCursor), (Qt.BottomEdge, Qt.SizeVerCursor),
            (Qt.TopEdge | Qt.LeftEdge, Qt.SizeFDiagCursor), (Qt.BottomEdge | Qt.RightEdge, Qt.SizeFDiagCursor),
            (Qt.TopEdge | Qt.RightEdge, Qt.SizeBDiagCursor), (Qt.BottomEdge | Qt.LeftEdge, Qt.SizeBDiagCursor)]]
        self._grip = G

        self.load(self.read_store())
        # the window reopens where it was left (page, Testing entry and method)
        self.pages.currentChanged.connect(lambda i: i < 2 and self.store.setValue("last_page", i))
        self.testing.side_nav.idClicked.connect(lambda i: self.store.setValue("testing_page", i))
        self.testing.method_changed.connect(lambda m: self.store.setValue("method", m))
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
        h.addWidget(label("Calibration lab", "appname"))
        h.addStretch(1)

        self.nav = QButtonGroup(self)   # exclusive, so clicking the selected tab can't un-select it
        # tab i = page i; the Calibration tab went (20 Oct): Testing > Calibration starts one
        for i, (glyph, name) in enumerate([(I_TESTING, "Testing"), (I_SETTINGS, "Settings")]):
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
        self.nav.idClicked.connect(lambda i: cross_fade(self.pages, i))
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
        reset = SpinButton(I_RESET, "Reset to defaults", "outlined")
        reset.clicked.connect(lambda: self.load(DEFAULTS))   # keys stay: they are managed in their own list
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

        self.camera_btn = button("", "assist")
        self.camera_menu = QMenu(self.camera_btn)
        self.camera_menu.aboutToShow.connect(self._fill_camera_menu)   # list cameras when opened
        self.camera_btn.setMenu(self.camera_menu)
        self.group("Camera", [
            self.row(I_CAMERA, "Webcam", "Used for eye tracking. Changes apply when you save", self.camera_btn),
        ])

        self.person_switch = Switch("Person mode")
        self.person_btn = button("", "assist")
        self.person_menu = QMenu(self.person_btn)
        self.person_menu.aboutToShow.connect(self._fill_person_menu)   # list the people when opened
        self.person_btn.setMenu(self.person_menu)
        self.group("Person mode", [
            self.row(I_PERSON, "Person mode", "On: every file goes under data/<name>/<calibration type>/, one folder "
                     "per person. Off: data/<calibration type>/. Changes apply when you save", self.person_switch),
            self.row(I_PERSON, "Person", "Whose calibrations, checks and blink scores these are. Each person's "
                     "models train on their own files only. Picking or adding a person saves at once", self.person_btn),
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
        return {k: self.store.value(k, d, type=type(d)) for k, d in DEFAULTS.items()}

    def load(self, values: dict):
        self.set_camera(values["camera"])
        self.person_switch.set_checked_now(values["person_mode"])
        self.set_person(values["person"])
        self.mode = values["theme"]
        self.theme_group.button(["system", "light", "dark"].index(self.mode)).setChecked(True)
        self.seed_before_sheet = values["seed"]
        self.set_seed(values["seed"])

    def save(self):
        values = dict(seed=self.seed, theme=self.mode, camera=self.camera,
                      person_mode=self.person_switch.isChecked(), person=self.person)
        for k, v in values.items():
            self.store.setValue(k, v)
        self.saved.emit(values)
        self.save_btn.setText("Saved")
        QTimer.singleShot(1500, lambda: self.save_btn.setText("Save changes"))

    def set_camera(self, name: str):
        self.camera = name
        self.camera_btn.setText(f"{name or 'Not chosen'}  {I_CHEVRON}")

    def set_person(self, name: str):
        self.person = name
        self.person_btn.setText(f"{name or 'Not chosen'}  {I_CHEVRON}")

    def _fill_person_menu(self):
        self.person_menu.clear()
        for name in storage.list_people():
            self.person_menu.addAction(name, lambda n=name: self.pick_person(n))
        self.person_menu.addAction("Add a person…", self.add_person)

    def pick_person(self, name: str):
        """Another person from the menu: person mode on and saved at once, as if Save changes were pressed
        (20 Oct: a person picked but not saved left the files going to the one before, with nothing on screen
        saying so). Picking a person means using them, as adding one does; turn the switch off to stop."""
        self.set_person(name)
        self.person_switch.setChecked(True)
        self.save()

    def add_person(self):
        """A new name: person mode turns on with it, and it is saved at once, so their folder is made and
        the next calibration goes into it."""
        text, ok = QInputDialog.getText(self, "Add a person", "Name (letters, digits, spaces, - and _):")
        name = storage.person_name(text) if ok else ""
        if name:
            self.set_person(name)
            self.person_switch.setChecked(True)
            self.save()

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
        self.pages.setCurrentWidget(self.testing)   # results land in the Testing page's status line
        self.nav.button(0).setChecked(True)
        if not was_visible:
            self.hide()
        elif was_max:
            self.showMaximized()
        else:
            self.showNormal()

    def show_last(self):
        """Bring the window up where it was left: the same page, and the same Testing entry."""
        page = self.pages.widget(max(0, min(self.store.value("last_page", 0, type=int), 1)))
        self.testing.show_page(self.store.value("testing_page", 0, type=int))
        self.show_page(page)

    def show_page(self, page: QWidget):
        """Bring the window up on `page` (self.testing or self.settings_page)."""
        self.pages.setCurrentWidget(page)
        self.nav.button(self.pages.currentIndex()).setChecked(True)
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
        for b in self.findChildren(SpinButton):
            b.set_color(c["primary"] if b.property("kind") == "text" else c["onSurface"])
        self.testing.bar_colors = (c["primary"], mix(c["primary"], c["surface"], .35))
        self.testing.show_results()                  # the comparison's bars in the theme colours
        self.busy.color, self.busy.track = QColor(c["primary"]), QColor(c["primary"])
        self.busy.track.setAlphaF(.2)                # the bar's track: the same colour, faint
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
        self.place_busy()
        G, w, h = self._grip, self.width(), self.height()
        for g, rect in zip(self.grips, [(0, G, G, h - 2 * G), (w - G, G, G, h - 2 * G), (G, 0, w - 2 * G, G),
                                        (G, h - G, w - 2 * G, G), (0, 0, G, G), (w - G, h - G, G, G),
                                        (w - G, 0, G, G), (0, h - G, G, G)]):
            g.setGeometry(*rect)
            g.raise_()
        super().resizeEvent(e)

    def place_busy(self):
        """The bar sits right under the title bar (at the very top on a full-screen page)."""
        self.busy.setGeometry(0, self.titlebar.height() if self.titlebar.isVisible() else 0, self.width(), 3)

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


class MultiCheckDialog(QDialog):
    """Multiple model gaze check: which models take part, and how many of each grid's calibrations they
    train on (the newest first). A row per grid, a chip per model: the grid alone, or with Sweep, Points
    or Line model (a chip only where the person has that special). `available`: model id -> usable
    calibrations of its grid; `last`: the counts chosen before (0 = left out), remembered."""

    def __init__(self, person: str, available: dict[str, int], last: dict[str, int] | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent, Qt.Dialog | Qt.WindowCloseButtonHint)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setWindowTitle("Multiple model gaze check")
        last = last or {}
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(12)
        v.addWidget(label("Which models, and how many calibrations each", "sheetTitle"))
        hint = label(f"Every ticked model predicts from the same camera frames, trained on the newest N of "
                     f"{person or 'No person'}'s calibrations of its grid (and the newest of its special). "
                     "36 rings (6×6), one file per model.", "sheetText")
        hint.setWordWrap(True)
        v.addWidget(hint)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self.chips: dict[str, CheckButton] = {}             # model id -> its chip
        self.picks: dict[str, QPushButton] = {}             # grid id -> its count picker
        self.counts: dict[str, int] = {}                    # grid id -> calibrations to use
        for r, g in enumerate(g for g in GRIDS if any(MODEL_BY_ID[m].grid is g for m in available)):
            most = next(n for m, n in available.items() if MODEL_BY_ID[m].grid is g)
            grid.addWidget(label(g.name, "rowtitle"), r, 0)
            for c, s in enumerate(WITH):
                m = model_for(g.id, s).id
                if m in available:
                    chip = CheckButton("Alone" if s is None else f"+ {METHOD_BY_ID[s].name}", "chip")
                    chip.setChecked(last.get(m, most) > 0)
                    self.chips[m] = chip
                    grid.addWidget(chip, r, 1 + c)
            pick = button("", "assist")
            menu = QMenu(pick)
            for n in range(1, most + 1):
                menu.addAction(f"{n}{' (all)' if n == most else ''}", lambda n=n, g=g.id: self.set_count(g, n))
            pick.setMenu(menu)
            self.picks[g.id] = pick
            mine = [last[m.id] for m in MODELS if m.grid is g and last.get(m.id)]
            self.set_count(g.id, min(max(mine, default=most), most))
            grid.addWidget(pick, r, 1 + len(WITH))
            grid.addWidget(label("calibrations", "rowsub"), r, 2 + len(WITH))
        grid.setColumnStretch(3 + len(WITH), 1)
        v.addLayout(grid)
        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel = button("Cancel", "text")
        cancel.clicked.connect(self.reject)
        ok = button("Start the gaze check", "filled")
        ok.clicked.connect(self.accept)
        actions.addWidget(cancel)
        actions.addWidget(ok)
        v.addLayout(actions)

    def set_count(self, grid_id: str, n: int):
        self.counts[grid_id] = n
        pick = self.picks[grid_id]
        most = len(pick.menu().actions())
        pick.setText(f"{n}{' (all)' if n == most else ''}  {I_CHEVRON}")

    def chosen(self) -> dict[str, int]:
        """model id -> calibrations of its grid to use, for the ticked models only."""
        return {m: self.counts[MODEL_BY_ID[m].grid.id] for m, chip in self.chips.items() if chip.isChecked()}


def pick_camera(names: list[str], current: str = "") -> str | None:
    """Small popup: choose a webcam. Returns its name, or None if closed."""
    store = QSettings("StudyAssistant", "CalibrationLab")
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
    font.setFamilies(TEXT_FONTS)
    app.setFont(font)
    w = SettingsWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
