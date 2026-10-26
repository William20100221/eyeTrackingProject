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
import threading
import json
import sys
from pathlib import Path
from string import Template

from materialyoucolor.dynamiccolor.material_dynamic_colors import MaterialDynamicColors
from materialyoucolor.hct import Hct
from materialyoucolor.scheme.scheme_fidelity import SchemeFidelity
from PySide6.QtCore import (QEasingCurve, QEvent, QPointF, QRectF, QRegularExpression, QSettings, QSize, Qt, QTimer,
                            QUrl, QVariantAnimation, Signal)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QKeySequence, QPainter, QPainterPath,
                           QPalette, QPen, QPixmap, QRegularExpressionValidator, QShortcut)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QAbstractButton, QApplication, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog,
                               QColorDialog, QFileDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from eyetracking.services.camera import camera_names
from eyetracking.services.reading_ai import (DEFAULT_HOSTS, KEY_URLS, KNOWN_MODELS, MAKERS, MODELS, PROVIDER_NAMES,
                                             default_model, list_models, provider_for_key)
from eyetracking.services.screen_text import ocr_languages
from eyetracking.services.secret_store import protect, unprotect
from eyetracking.services.storage import (blink_score_lines, calibration_lines, calibration_usable, file_date,
                                          gaze_check_lines, list_blink_scores, list_calibrations, list_gaze_checks,
                                          trash_file,
                                          list_disabled, list_removed, restore, set_calibration_enabled,
                                          list_readings, reading_messages, reading_title, remove_calibration,
                                          remove_samples)

DEFAULTS = {
    "seed": "#E5483B",          # the logo's red
    "theme": "system",          # system | light | dark
    "font": "Segoe UI",         # a name in FONTS
    "show_ball": True,
    "ball_on_top": True,
    "hide_fullscreen": False,
    "ball_look": "logo",        # logo | water | image: the floating ball's face
    "ball_image": "",           # the user's picture, for "image"
    "ball_ring": "",            # the thin ring around a still ball; "" = the theme colour
    "summary_words": 150,
    "camera": "",               # webcam NAME (Windows can renumber cameras between restarts)
}
BALL_LOOKS = ["logo", "water", "image"]
SWATCHES = ["#3D6CF6", "#00897B", "#2E8B57", "#E8B400", "#E5483B", "#D32F2F", "#D81B60", "#7E57C2"]

PAGE_OF_TAB = {0: 0, 1: 1, 3: 2}          # title-bar tab -> page in the stack (tab 2, Calibration, is full screen)
TAB_OF_PAGE = {v: k for k, v in PAGE_OF_TAB.items()}

TEXT_FONTS = ["Segoe UI Variable Text", "Segoe UI", "Segoe Fluent Icons", "Segoe MDL2 Assets"]
ICON_FONTS = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]   # built into Windows 11 / Windows 10
FONTS = {                   # Settings > Font: the text's families and weight (400 regular, 300 light), all built into Windows
    "Segoe UI": (TEXT_FONTS[:2], 400),
    "Segoe UI Semilight": (["Segoe UI"], 350),
    "Segoe UI Light": (["Segoe UI"], 300),
    "Calibri Light": (["Calibri"], 300),
    "Georgia": (["Georgia"], 400),
}
I_FONT = "\uE8D2"   # Font


def text_font(name: str) -> QFont:
    """The app's font for a Settings > Font choice (an unknown name: the default); the icon fonts last,
    so glyphs inside text still draw."""
    families, weight = FONTS.get(name, FONTS[DEFAULTS["font"]])
    f = QFont()
    f.setFamilies(families + ICON_FONTS)
    f.setWeight(QFont.Weight(weight))
    return f

# Segoe Fluent Icons code points
I_SETTINGS, I_SEARCH = "", ""
I_MIN, I_MAX, I_RESTORE, I_CLOSE = "", "", "", ""
I_CHECK, I_DISMISS, I_RESET, I_CHEVRON = "", "", "", ""
I_THEME, I_PALETTE, I_BALL, I_LAYERS = "", "", "", ""
I_FULLSCREEN, I_SPARKLE, I_LINES = "", "", ""
I_EYE, I_CAMERA = "", ""   # View (eye), Camera
I_KEY = "\uE8D7"   # Permissions (key)
I_ADD = "\uE710"   # Add (+)
I_LANGUAGE = "\uF2B7"   # LocaleLanguage (a globe with an A)
I_TESTING = "\uE9D9"   # Diagnostic
I_BLINK, I_CHART = "\uE7B3", "\uE9D2"   # RedEye, AreaChart
I_BACK, I_DELETE, I_UNDO = "\uE72B", "\uE74D", "\uE7A7"
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
   so each kind's padding gives that width back: focused text never moves or clips (19 Oct: a focused
   "Save changes" lost the end of its text) */
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

/* Settings > AI > Providers: the providers down the left, the chosen one's key, host and models on the right */
#providers { border-top: 1px solid $outlineVariant; }
#providerSide { border-right: 1px solid $outlineVariant; }
QListWidget#providerList { background: transparent; border: none; outline: none; font-size: 14px; }
QListWidget#providerList::item { padding: 7px 8px; border-radius: 8px; }
QListWidget#providerList::item:hover { background: $hover; }
QListWidget#providerList::item:selected { background: $selected; color: $onSurface; }
#providers QLineEdit, #addProvider QLineEdit { min-height: 32px; max-height: 32px; padding: 0 8px; border: 1px solid $outlineVariant;
    border-radius: 6px; background: $surfaceContainerLow; selection-background-color: $primary; selection-color: $onPrimary; }
#providers QLineEdit:focus, #addProvider QLineEdit:focus { border-bottom: 2px solid $primary; }
#providers QLineEdit:disabled { color: $onSurfaceVariant; }
#providerName { font-size: 16px; font-weight: 600; }
#fieldLabel { font-size: 13px; font-weight: 600; }
QPushButton[kind="link"] { color: $primary; font-size: 12px; font-weight: 400; padding: 0 4px; min-height: 20px; max-height: 20px;
    border-radius: 4px; }
QPushButton[kind="link"]:hover { text-decoration: underline; }
QPushButton[kind="link"]:focus { padding: 0 2px; }
QListWidget#modelList { background: $surfaceContainerLow; border: 1px solid $outlineVariant; border-radius: 6px; outline: none;
    font-size: 13px; }
QListWidget#modelList::item { padding: 6px 10px; }
QListWidget#modelList::item:hover { background: $hover; }
QListWidget#modelList::item:selected { background: $selected; color: $onSurface; }

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
QListWidget#sideList[wrap="true"] { padding: 0 12px 0 36px; }   /* wrapped titles: the side padding on the list, */
QListWidget#sideList[wrap="true"]::item { padding: 6px 0; }     /* so a wrapped row is measured at the width it is drawn at */
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


def letter_icon(letter: str, bg: str, fg: str, px: int = 24) -> QIcon:
    """A round mark with one letter: a provider's avatar in Settings > AI (no picture upload: a letter is enough)."""
    pm = QPixmap(px * 2, px * 2)
    pm.setDevicePixelRatio(2)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(bg))
    p.drawEllipse(QRectF(0, 0, px, px))
    f = QFont()
    f.setPixelSize(px // 2)
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(fg))
    p.drawText(QRectF(0, 0, px, px), Qt.AlignCenter, letter[:1].upper())
    p.end()
    return QIcon(pm)


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
    (Segoe's Reset, U+E777) points: turning clockwise, it ran against its own arrowhead (20 Oct)."""

    def __init__(self, glyph: str, text: str, kind: str):
        super().__init__(text)
        self.glyph, self.angle, self.color = glyph, 0.0, "#000000"
        self.setProperty("kind", kind)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self._turn = QVariantAnimation(self, startValue=0.0, endValue=-360.0, duration=550,
                                       easingCurve=QEasingCurve.OutCubic)
        self._turn.valueChanged.connect(self._draw)
        self.clicked.connect(self._turn.start)

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
    def __init__(self, color=None):
        super().__init__()
        self._color = color                  # a function giving the hex to show; None = the theme seed
        self.setFixedSize(20, 20)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 40), 1))
        p.setBrush(QColor(self._color() if self._color else self.window().seed))
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


def fill_list(lst: QListWidget, paths: list[Path], empty: str, name=None):
    """One item per dated file; name(path), if given, is shown after the date (full name in the tooltip)."""
    lst.clear()
    if name and not lst.property("wrap"):         # a title: wrapped to the list's width, never cut at its edge
        lst.setProperty("wrap", True)
        lst.style().unpolish(lst)                 # the [wrap] rules apply from now on
        lst.style().polish(lst)
        lst.setWordWrap(True)
        lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lst.setResizeMode(QListWidget.Adjust)
    for path in paths:
        when = f"{file_date(path):%d %b %Y   %H:%M}"
        item = QListWidgetItem(f"{file_date(path):%d %b %H:%M}  {name(path)}" if name else when)
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
        text_label.setTextFormat(Qt.MarkdownText)       # the AI answers in Markdown: **bold**, bullet lists, headings
        return text_label, text_label
    text_label.setTextFormat(Qt.PlainText)              # yours as typed: a * or _ in an answer is just a character
    bubble = QFrame(objectName="mine")
    QVBoxLayout(bubble).addWidget(text_label)
    bubble.setMaximumWidth(520)
    row = QWidget()
    rh = QHBoxLayout(row)
    rh.setContentsMargins(0, 0, 0, 0)
    rh.addStretch(1)
    rh.addWidget(bubble)
    return row, text_label


class DataPage(QWidget):
    """One kind of recorded data: the files by date on the left, the chosen one in
    words on the right. Right-click a file to remove it (it moves to a trash folder
    beside it)."""

    def __init__(self, title: str, hint: str, list_files, to_lines, empty: str):
        super().__init__()
        self._list, self._lines, self._empty = list_files, to_lines, empty
        h = QHBoxLayout(self)
        h.setContentsMargins(24, 20, 24, 24)
        h.setSpacing(24)
        left = QVBoxLayout()
        left.setSpacing(8)
        left.addWidget(label(title, "sheetTitle"))
        text = label(hint, "sheetText")
        text.setWordWrap(True)
        left.addWidget(text)
        self.files = QListWidget(objectName="sideList")
        self.files.currentItemChanged.connect(lambda it, _: self._show(it))
        self.files.setContextMenuPolicy(Qt.CustomContextMenu)
        self.files.customContextMenuRequested.connect(self._menu)
        left.addWidget(self.files, 1)
        col = QWidget()
        col.setLayout(left)
        col.setFixedWidth(320)
        h.addWidget(col)
        self.lines = QListWidget(objectName="linesList")
        self.lines.setWordWrap(True)
        self.lines.setSelectionMode(QListWidget.NoSelection)
        h.addWidget(self.lines, 1)

    def refresh(self):
        self.files.blockSignals(True)
        fill_list(self.files, self._list(), self._empty)
        self.files.blockSignals(False)
        self.files.setCurrentRow(0)
        self._show(self.files.currentItem())

    def _show(self, item):
        self.lines.clear()
        if item is not None and item.data(Qt.UserRole):
            for line in self._lines(Path(item.data(Qt.UserRole))):
                self.lines.addItem(QListWidgetItem(line))

    def _menu(self, pos):
        item = self.files.itemAt(pos)
        if item is None or not item.data(Qt.UserRole):
            return
        path = Path(item.data(Qt.UserRole))
        menu = QMenu(self.files)
        menu.addAction(f"{I_DELETE}  Remove", lambda: (trash_file(path), self.refresh()))
        menu.exec(self.files.mapToGlobal(pos))


# ---------------------------------------------------------------- testing page

class TestingPage(QWidget):
    """Testing tab. The sidebar picks a page on the right. Tests: what the sample text,
    the gaze check and the blink test do, with the button that starts them. Data: the
    calibrations (active / non-active / not usable), the gaze checks and the blink
    scores, each by date and readable in words."""

    command = Signal(str)          # a start button: "read" | "check" | "blink"
    data_changed = Signal()        # samples or a file removed / disabled / enabled / restored: the app retrains
    undo_requested = Signal()      # the Undo button (or Ctrl+Z): the last calibration / gaze check goes

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
        self.side_nav = QButtonGroup(self)               # the chosen entry stays highlighted: it is the page on the right
        sections = [("Tests", [(I_READ, "Sample text"), (I_EYE, "Gaze check"), (I_BLINK, "Blink test")]),
                    ("Data", [(I_LINES, "Calibrations"), (I_CHART, "Gaze checks"), (I_BLINK, "Blink scores")])]
        i = 0
        for section, items in sections:
            sv.addWidget(label(section, "sideTitle"))
            for glyph, text in items:
                b = button(f"{glyph}   {text}", "nav")
                b.setCheckable(True)
                self.side_nav.addButton(b, i)                # its id = the page index on the right
                sv.addWidget(b)
                i += 1
            sv.addSpacing(8)
        sv.addStretch(1)
        self.status = label("", "rowsub")                # the latest word from the app (the model, a saved file)
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

        # 0-2: the tests (what they do + a start button); 3: calibrations; 4: gaze checks; 5: blink scores;
        # 6: one calibration's detail (from 3)
        self.main = QStackedWidget()
        self.main.addWidget(self._intro(
            I_READ, "Sample text", "Read a short text while the tracker watches where you look.",
            ["Each paragraph turns green once your eyes have rested on it long enough: about a tenth of a second per word.",
             "Press Done, or stare at it, when you finish. The AI then summarises what you skipped and asks you about it, "
             "in the Chat tab.",
             "Open .txt loads a text of your own. Esc leaves without the AI part."],
            "Open the sample text", "read"))
        self.main.addWidget(self._intro(
            I_EYE, "Gaze check", "A full-screen test of the gaze model on 16 rings (4×4): how accurate is it today?",
            ["One ring is lit (white). Look at it and press Enter: that records where the model says you look, and "
             "the next ring lights up. A sample's error is its distance to the lit ring.",
             "A green dot follows where the model thinks you look; it turns grey while your face is lost or you blink.",
             "Do a full round (16) for a fair number. Space pauses. Esc saves and leaves: the file goes to "
             "data/gaze_checks, and Data > Gaze checks shows the average error. Undo (under this sidebar, or Ctrl+Z) "
             "takes it back. It needs a calibration first."],
            "Start the gaze check", "check"))
        self.main.addWidget(self._intro(
            I_BLINK, "Blink test", "Record your blink scores in six eye states, so the blink cut-off can be set from "
            "your own numbers instead of a guess.",
            ["A blink score is a number MediaPipe gives for each eye every frame: 0 = wide open, 1 = shut. The app uses "
             "it to skip frames where the eyes are closing or closed: calibration doesn't record them, and the live gaze "
             "holds its last point instead of jumping when the iris disappears behind the lid.",
             "Today the cut-off is a guess: 0.50 on both eyes. The test asks you to keep your eyes open, look down, "
             "squint, close them, and wink each eye, 6 seconds each after a 3-second countdown, and records the scores.",
             "The result is listed under Data > Blink scores, in words, with the cut-off it suggests. Space starts, "
             "Esc leaves."],
            "Start the blink test", "blink"))
        lists = QWidget()
        mv = QVBoxLayout(lists)
        mv.setContentsMargins(24, 20, 24, 24)
        mv.setSpacing(8)
        mv.addWidget(label("Calibrations", "sheetTitle"))
        hint = label("Saved calibrations by date. Double-click one to read it in words. Ctrl- or Shift-click to pick "
                     "several; right-click (or Delete) to disable them (kept, but they stop training the gaze model), "
                     "enable them again, remove them, or restore removed ones. Removed calibrations wait in the trash. "
                     "Only the active ones train the gaze model.", "sheetText")
        hint.setWordWrap(True)
        mv.addWidget(hint)
        cols = QHBoxLayout()
        cols.setSpacing(24)
        self.active_list = side_list(self.open_detail)
        self.disabled_list = side_list(self.open_detail)
        self.inactive_list = side_list(self.open_detail)
        self.removed_list = side_list(self.open_detail)
        for title, sub, lst in [("Active", "Trains the gaze model", self.active_list),
                                ("Non-active", "Works, but you turned it off", self.disabled_list),
                                ("Not usable", "Another calibration version, or the file can't be read", self.inactive_list),
                                ("Removed", "In the trash: restore them any time", self.removed_list)]:
            lst.setContextMenuPolicy(Qt.CustomContextMenu)
            lst.customContextMenuRequested.connect(lambda pos, lst=lst: self._list_menu(lst, pos))
            QShortcut(QKeySequence.Delete, lst, lambda lst=lst: self.remove_files(picked(lst)),
                      context=Qt.WidgetShortcut)                    # Delete = Remove the picked ones
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
                     "the gaze model retrained. The untouched original stays in data/trash.", "sheetText")
        hint.setWordWrap(True)
        dv.addWidget(hint)
        self.detail = QListWidget(objectName="linesList")
        self.detail.setSelectionMode(QListWidget.ExtendedSelection)
        self.detail.setWordWrap(True)                    # a sample's words are long
        dv.addWidget(self.detail, 1)
        self.remove_btn = button(f"{I_DELETE}  Remove selected", "outlined")
        self.remove_btn.clicked.connect(self.remove_selected)
        dv.addWidget(self.remove_btn, 0, Qt.AlignLeft)
        self.gaze_checks = DataPage(
            "Gaze checks", "Each gaze check's samples (recorded with Enter): the lit ring you looked at, where the "
            "model said, and how far off it was; the average is the error. Right-click a file to remove it.",
            list_gaze_checks, gaze_check_lines, "No gaze check yet")
        self.blink_scores = DataPage(
            "Blink scores", "Each blink test: your scores per eye state, and the cut-off they suggest. Right-click a "
            "file to remove it.", list_blink_scores, blink_score_lines, "No blink test yet")
        self.main.addWidget(self.gaze_checks)
        self.main.addWidget(self.blink_scores)
        self.main.addWidget(detail)
        self.detail_path: Path | None = None
        h.addWidget(self.main, 1)
        self.side_nav.idClicked.connect(lambda i: cross_fade(self.main, i))
        self.show_page(3)                                # Calibrations first: what the model learns from

    def show_page(self, i: int):
        """Open sidebar entry `i` (0-5; the calibration detail, 6, opens as its list)."""
        i = 3 if i == 6 else max(0, min(i, 5))
        self.side_nav.button(i).setChecked(True)
        self.main.setCurrentIndex(i)

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
        self.remove_btn.setVisible(removable and path.parent.name != "trash")   # a removed file: restore it first
        self.main.setCurrentIndex(6)

    def say(self, text: str):
        """A line from the app, in the sidebar's status."""
        self.status.setText(text)

    def set_undo(self, what: str):
        """The Undo button under the status line: "Undo <what>", hidden when there is nothing to undo."""
        self.undo_btn.setText(f"{I_UNDO}  Undo {what}")
        self.undo_btn.setToolTip(f"Undo {what} (Ctrl+Z): its file goes to the trash; Removed restores a calibration")
        self.undo_btn.setVisible(bool(what))

    def remove_selected(self):
        if not (self.detail_path and self.detail_path.exists()):
            self.refresh()                               # gone meanwhile (Undo, a removal): back to the lists
            self.main.setCurrentIndex(3)
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
            menu.addAction(f"{I_DELETE}  Remove {count(paths, 'calibration')}", lambda: self.remove_files(paths))
        menu.exec(lst.mapToGlobal(pos))

    def _changed(self, text: str):
        """Files removed, restored, enabled or disabled: the lists, the model (the app retrains), the status line."""
        self.refresh()
        self.data_changed.emit()
        self.say(text)

    def set_enabled(self, paths: list[Path], enabled: bool):
        """Right-click > Disable / Enable: the files move between data/calibrations and its disabled/."""
        for p in paths:
            set_calibration_enabled(p, enabled)
        self._changed(f"{'Enabled' if enabled else 'Disabled'} {count(paths, 'calibration')}.")

    def remove_files(self, paths: list[Path]):
        """Right-click > Remove (or Delete): into the trash; Removed lists them to restore."""
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
            restore(p)
        self._changed(f"Restored {count(paths, 'calibration')}.")

    def refresh(self):
        """Re-read data/ (new calibrations). Runs whenever the page shows."""
        usable = {p: calibration_usable(p) for p in list_calibrations()}
        fill_list(self.active_list, [p for p, ok in usable.items() if ok], "None active: enable one, or calibrate")
        fill_list(self.disabled_list, list_disabled(), "None")
        fill_list(self.inactive_list, [p for p, ok in usable.items() if not ok], "None")
        fill_list(self.removed_list, list_removed(), "None")
        self.gaze_checks.refresh()
        self.blink_scores.refresh()
        if self.main.currentIndex() == 6 and not (self.detail_path and self.detail_path.exists()):
            self.main.setCurrentIndex(3)                 # the file on the detail page is gone (Undo, a removal)

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

        self.go = button(f"{I_READ}   Go Reading", "go")   # instead of an input bar
        self.go.clicked.connect(self.go_reading)
        self.model_btn = button("", "assist")            # which key and model answer, like Ollama's model picker
        self.model_btn.setToolTip("The AI that writes summaries and marks quizzes")
        self.model_btn.setMenu(QMenu(self.model_btn))
        self.lang_btn = button("", "assist")             # the language of the text you are going to read
        self.lang_btn.setToolTip("The language of the text you will read: the screen is read in it when the app "
                                 "has to read its picture (PDFs, some apps)")
        self.lang_btn.setMenu(QMenu(self.lang_btn))
        bottom = QHBoxLayout()                           # 4: the AI and language pickers, then Go Reading
        bottom.setSpacing(12)
        bottom.addWidget(self.model_btn)
        bottom.addWidget(self.lang_btn)
        bottom.addWidget(self.go, 1)
        v.addLayout(bottom)
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
    shown = Signal(bool)              # the window appeared / went away (the ball's eye follows it)
    ai_changed = Signal()             # a key added / removed / renamed, or another key or model chosen
    language_changed = Signal(str)    # the reading language picked in the chat: its tag ("en-US", "zh-Hans-CN"...)

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setWindowTitle("Study Assistant")
        self.resize(1100, 700)
        self.setMinimumSize(920, 560)
        self.store = QSettings("StudyAssistant", "StudyAssistant")
        self.found_key: tuple[str, str] | None = None   # (provider, key) from the environment: set by the app, not saved
        self.found_ollama: list[list[str]] = []          # the models a running Ollama has installed (set_found)
        self.shown_id = "claude"                         # the provider whose pane Settings > AI shows
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
        self.busy = BusyBar(self)                    # runs under the title bar while the app works (run_busy)
        self._before_page = None          # (was visible, was maximized) while a full-screen page runs

        G = 5
        self.grips = [Grip(self, e, cur) for e, cur in [
            (Qt.LeftEdge, Qt.SizeHorCursor), (Qt.RightEdge, Qt.SizeHorCursor),
            (Qt.TopEdge, Qt.SizeVerCursor), (Qt.BottomEdge, Qt.SizeVerCursor),
            (Qt.TopEdge | Qt.LeftEdge, Qt.SizeFDiagCursor), (Qt.BottomEdge | Qt.RightEdge, Qt.SizeFDiagCursor),
            (Qt.TopEdge | Qt.RightEdge, Qt.SizeBDiagCursor), (Qt.BottomEdge | Qt.LeftEdge, Qt.SizeBDiagCursor)]]
        self._grip = G

        self.load(self.read_store())
        self._migrate_keys()
        self.refresh_providers()
        self.chat.model_btn.menu().aboutToShow.connect(self._fill_ai_menu)
        self.chat.lang_btn.menu().aboutToShow.connect(self._fill_language_menu)
        self._show_language()
        # the window reopens where it was left (page, and the Testing entry); an AI answer still opens the chat
        self.pages.currentChanged.connect(lambda i: i < 3 and self.store.setValue("last_page", i))
        self.testing.side_nav.idClicked.connect(lambda i: self.store.setValue("testing_page", i))
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
                                   else cross_fade(self.pages, PAGE_OF_TAB[i]))
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
        self.font_btn = button("", "assist")
        self.font_menu = QMenu(self.font_btn)
        for name in FONTS:                           # each name in its own font
            self.font_menu.addAction(name, lambda n=name: self.set_font_choice(n)).setFont(text_font(name))
        self.font_btn.setMenu(self.font_menu)
        self.group("Appearance", [
            self.row(I_THEME, "Theme", "Follow Windows, or always use one", chips),
            self.row(I_PALETTE, "Theme colour", "Every colour in the app is made from this one", self.dot, self.hex_label, change),
            self.row(I_FONT, "Font", "The text's typeface. Light and Semilight are thinner", self.font_btn),
        ])

        self.switches = {k: Switch(t) for k, t in [("show_ball", "Show floating ball"),
                                                   ("ball_on_top", "Keep it above other windows"),
                                                   ("hide_fullscreen", "Hide during full-screen apps")]}
        self.ball_look, self.ball_image, self.ball_ring = "logo", "", ""
        looks = QWidget()
        lk = QHBoxLayout(looks)
        lk.setContentsMargins(0, 0, 0, 0)
        lk.setSpacing(8)
        self.look_group = QButtonGroup(self)
        for i, text in enumerate(["Logo", "Water ball", "Image"]):
            b = CheckButton(text, "chip")
            lk.addWidget(b)
            self.look_group.addButton(b, i)
        self.look_group.idClicked.connect(lambda i: self.set_ball_look(BALL_LOOKS[i]))
        self.image_btn = button("Choose image…", "assist")   # shows the chosen file's name
        self.image_btn.clicked.connect(self.choose_ball_image)
        self.ring_dot = ColorDot(lambda: self.ball_ring or self.seed)
        self.ring_hex = label("", "hex")
        self.ring_hex.setMinimumWidth(62)
        ring_btn = button("Change", "text")               # one button: the row must fit beside the open colour sheet
        ring_menu = QMenu(ring_btn)
        ring_menu.addAction("Pick a colour…", self.choose_ring)
        ring_menu.addAction("Theme colour", lambda: self.set_ball_ring(""))
        ring_btn.setMenu(ring_menu)
        self.group("Floating ball", [
            self.row(I_BALL, "Show floating ball", "Opens the assistant from any app", self.switches["show_ball"]),
            self.row(I_LAYERS, "Keep it above other windows", "Turn off to let it hide behind full windows",
                     self.switches["ball_on_top"]),
            self.row(I_FULLSCREEN, "Hide during full-screen apps", "Games and videos stay clear",
                     self.switches["hide_fullscreen"]),
            self.row(I_BALL, "Ball look", "The logo or your own image (still, inside a thin ring), or the moving "
                     "water ball. Applies when you save", looks),
            self.row(I_BALL, "Image", "Your picture for the still ball (png, jpg); it is cropped to a circle",
                     self.image_btn),
            self.row(I_PALETTE, "Ring colour", "The thin ring around the still ball: a colour of yours, or the theme's",
                     self.ring_dot, self.ring_hex, ring_btn),
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
        self.group("AI", [
            self._providers_block(),
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

    def _providers_block(self) -> QWidget:
        """Settings > AI > Providers: the heading row, then the providers down the left (search, Add Provider)
        and the chosen one's switch, key, host and models on the right."""
        block = QWidget()
        bv = QVBoxLayout(block)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(0)
        bv.addWidget(self.row(I_KEY, "Providers",
                              "Claude, ChatGPT, Gemini and Ollama are built in; Add Provider takes any other server that "
                              "speaks the OpenAI or Anthropic API. Each provider keeps its own key (stored encrypted for "
                              "your Windows login, never shown again), API host and models. Click a model to answer "
                              "with it; the switch turns a provider off."))
        self.rows[-1] = (block, self.rows[-1][1] + " keys api claude chatgpt openai gemini ollama local model host custom")
        panel = QFrame(objectName="providers")
        panel.setFixedHeight(440)
        ph = QHBoxLayout(panel)
        ph.setContentsMargins(0, 0, 0, 0)
        ph.setSpacing(0)

        side = QFrame(objectName="providerSide")
        side.setFixedWidth(236)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 12, 12, 12)
        sv.setSpacing(8)
        self.provider_find = QLineEdit(placeholderText="Search providers…")
        self.provider_find.setAccessibleName("Search providers")
        self.provider_find.textChanged.connect(self._filter_providers)
        sv.addWidget(self.provider_find)
        self.provider_list = QListWidget(objectName="providerList")
        self.provider_list.setIconSize(QSize(24, 24))
        self.provider_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # a long name is cut, not scrolled
        self.provider_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.provider_list.customContextMenuRequested.connect(self._provider_menu)
        self.provider_list.currentItemChanged.connect(lambda cur, _: cur and self._show_provider(cur.data(Qt.UserRole)))
        sv.addWidget(self.provider_list, 1)
        add = button(f"{I_ADD}  Add Provider", "outlined")
        add.clicked.connect(self.add_provider)
        sv.addWidget(add)
        ph.addWidget(side)

        detail = QWidget()
        dv = QVBoxLayout(detail)
        dv.setContentsMargins(20, 12, 16, 12)
        dv.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(10)
        self.provider_title = label("", "providerName")
        self.provider_maker = label("", "rowsub")
        head.addWidget(self.provider_title)
        head.addWidget(self.provider_maker, 1)
        self.provider_on = Switch("Provider on")
        self.provider_on.clicked.connect(lambda: self.set_on(self.shown_id, self.provider_on.isChecked()))
        head.addWidget(self.provider_on)
        dv.addLayout(head)

        def heading(text: str, *links) -> tuple[QHBoxLayout, list[QPushButton]]:
            """A field's title with small link buttons beside it, as on a web settings page."""
            h = QHBoxLayout()
            h.setSpacing(8)
            h.addWidget(label(text, "fieldLabel"))
            made = []
            for link_text, slot in links:
                link = button(link_text, "link")
                link.clicked.connect(slot)
                h.addWidget(link)
                made.append(link)
            h.addStretch(1)
            return h, made

        h, (self.key_link, self.forget_link) = heading("API Key", ("Get API key", self._open_key_url),
                                                       ("Forget the saved key", lambda: self.forget_key(self.shown_id)))
        dv.addSpacing(6)
        dv.addLayout(h)
        self.key_field = QLineEdit(placeholderText="Add API key", echoMode=QLineEdit.Password)
        self.key_field.setAccessibleName("API key")
        self.key_eye = self.key_field.addAction(QIcon(), QLineEdit.TrailingPosition)   # show / hide what is typed
        self.key_eye.setToolTip("Show the key")
        self.key_eye.triggered.connect(self._toggle_key)
        self.key_field.editingFinished.connect(lambda: self.set_key(self.shown_id, self.key_field.text()))
        dv.addWidget(self.key_field)

        h, (self.host_link,) = heading("API Host", ("Back to the usual host", lambda: self.set_host(self.shown_id, "")))
        dv.addSpacing(6)
        dv.addLayout(h)
        self.host_field = QLineEdit(placeholderText="https://…")
        self.host_field.setAccessibleName("API host")
        self.host_field.editingFinished.connect(lambda: self.set_host(self.shown_id, self.host_field.text()))
        dv.addWidget(self.host_field)

        mh = QHBoxLayout()
        mh.setSpacing(8)
        mh.addWidget(label("Models", "fieldLabel"), 1)
        self.sync_btn = SpinButton(I_RESET, "Sync models", "outlined")
        self.sync_btn.setToolTip("Ask the provider which models this key can use")
        self.sync_btn.clicked.connect(lambda: self.sync_models(self.shown_id))
        add_model = button(I_ADD, "icon")
        add_model.setToolTip("Add a model by its id")
        add_model.setAccessibleName("Add a model")
        add_model.clicked.connect(lambda: self.add_model(self.shown_id))
        mh.addWidget(self.sync_btn)
        mh.addWidget(add_model)
        dv.addSpacing(6)
        dv.addLayout(mh)
        self.model_list = QListWidget(objectName="modelList")
        self.model_list.setMinimumHeight(100)
        self.model_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.model_list.customContextMenuRequested.connect(self._model_menu_at)
        self.model_list.itemClicked.connect(lambda it: it.data(Qt.UserRole) and self.use(self.shown_id, it.data(Qt.UserRole)))
        dv.addWidget(self.model_list, 1)
        self.keys_note = label("", "rowsub")             # what the last key / sync found; hidden when empty
        self.keys_note.setWordWrap(True)
        self.keys_note.hide()
        dv.addWidget(self.keys_note)
        ph.addWidget(detail, 1)
        bv.addWidget(panel)
        return block

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

    def _toggle_key(self):
        shown = self.key_field.echoMode() == QLineEdit.Normal
        self.key_field.setEchoMode(QLineEdit.Password if shown else QLineEdit.Normal)
        self.key_eye.setToolTip("Show the key" if shown else "Hide the key")

    # ----- AI providers: Claude, ChatGPT, Gemini and Ollama built in, plus custom ones (any server that speaks the
    # OpenAI or Anthropic API). Each has its own key, host and models; a key is only ever stored encrypted (DPAPI)

    def _builtin(self, kind: str) -> dict:
        return {"id": kind, "name": PROVIDER_NAMES[kind], "kind": kind, "secret": "", "last4": "",
                "host": DEFAULT_HOSTS[kind], "models": [list(m) for m in KNOWN_MODELS[kind]], "model": MODELS[kind],
                "on": True, "custom": False}

    def providers(self) -> list[dict]:
        """Every provider, the built-in four first then the custom ones: {id, name, kind (the API it speaks),
        secret (encrypted), last4, host, models [[id, name]], model, on, custom}. A key found in the
        environment / .env fills a built-in one that has none (as "plain": not saved); Ollama's models
        are the ones it has installed."""
        try:
            saved = json.loads(self.store.value("ai_providers", "[]"))
        except ValueError:
            saved = []
        saved = [p for p in saved if isinstance(p, dict) and "id" in p] if isinstance(saved, list) else []
        by_id = {p["id"]: p for p in saved}
        out = [{**self._builtin(k), **by_id.get(k, {})} for k in BRANDS] + [p for p in saved if p.get("custom")]
        for p in out:
            p.pop("plain", None)
            if self.found_key and p["id"] == self.found_key[0] and not p["secret"]:
                p["plain"], p["last4"] = self.found_key[1], self.found_key[1][-4:]
            if p["id"] == "ollama":
                p["models"] = self.found_ollama
                if self.found_ollama and p["model"] not in [m[0] for m in self.found_ollama]:
                    p["model"] = self.found_ollama[0][0]
        return out

    def _save_providers(self, providers: list[dict]):
        self.store.setValue("ai_providers", json.dumps([{k: v for k, v in p.items() if k != "plain"} for p in providers]))

    def _update(self, pid: str, **changes):
        self._save_providers([{**p, **changes} if p["id"] == pid else p for p in self.providers()])

    def provider(self, pid: str) -> dict | None:
        return next((p for p in self.providers() if p["id"] == pid), None)

    def set_found(self, key: tuple[str, str] | None, ollama_models: list):
        """What the app found at start: (provider, key) in the environment / .env, and Ollama's installed models."""
        self.found_key, self.found_ollama = key, [list(m) for m in ollama_models]
        self.refresh_providers()

    def usable(self) -> list[dict]:
        """The providers that can answer: on, with a key (a custom server may need none; Ollama needs its models)."""
        return [p for p in self.providers() if p["on"] and
                (bool(p["models"]) if p["id"] == "ollama" else bool(self.secret_of(p)) or p["custom"])]

    def active(self) -> dict | None:
        """The provider that answers: the chosen one if it can, else the first that can."""
        usable = self.usable()
        want = self.store.value("ai_active", "")
        return next((p for p in usable if p["id"] == want), usable[0] if usable else None)

    def secret_of(self, p: dict) -> str:
        return unprotect(p["secret"]) if p.get("secret") else p.get("plain", "")

    def _migrate_keys(self):
        """Older versions saved one key as "api_key", then several as "ai_keys" (Key 1, Key 2…). Each becomes its
        brand's key; a second key of the same brand becomes a custom provider ("Key 3 (Claude)") on the usual host."""
        old = self.store.value("api_key", "")
        try:
            keys = json.loads(self.store.value("ai_keys", "[]"))
        except ValueError:
            keys = []
        if old and provider_for_key(unprotect(old)):
            key = unprotect(old)
            keys = [{"id": "api_key", "name": "Key 1", "provider": provider_for_key(key), "secret": protect(key),
                     "last4": key[-4:]}] + keys
        if not keys and not old:
            return
        providers = self.providers()
        active = self.store.value("ai_active", "")
        for e in keys:
            kind = e.get("provider")
            if kind not in BRANDS or kind == "ollama" or not e.get("secret"):
                continue
            fields = {"secret": e["secret"], "last4": e.get("last4", ""),
                      "models": e.get("models") or [list(m) for m in KNOWN_MODELS[kind]], "model": e.get("model", MODELS[kind])}
            mine = next(p for p in providers if p["id"] == kind)
            if not mine["secret"]:
                mine.update(fields)
                new_id = kind
            else:
                new_id = new_custom_id(providers)
                providers.append({**self._builtin(kind), **fields, "id": new_id, "custom": True,
                                  "name": f"{e.get('name', 'Key')} ({PROVIDER_NAMES[kind]})"})
            if active == e.get("id"):
                self.store.setValue("ai_active", new_id)
        try:
            chosen = json.loads(self.store.value("ai_extra_models", "{}")).get("ollama")   # Ollama's remembered model
        except (ValueError, AttributeError):
            chosen = None
        if chosen:
            next(p for p in providers if p["id"] == "ollama")["model"] = chosen
        self._save_providers(providers)
        for k in ("api_key", "ai_keys", "ai_extra_models"):
            self.store.remove(k)

    def add_key(self, text: str) -> dict | None:
        """A pasted key (the first-run popup): its prefix says the brand, which gets the key and answers from now on."""
        key = text.strip()
        kind = provider_for_key(key)
        if kind is None:
            self._note("That doesn't look like a Claude (sk-ant-…), Gemini (AQ.… or AIza…) or OpenAI (sk-…) key.")
            return None
        return self.set_key(kind, key)

    def set_key(self, pid: str, text: str) -> dict | None:
        """This provider's key (typed in its field). The provider is asked which models the key can use: that
        also shows the key works. A built-in provider warns about another brand's key, but keeps it (a proxy
        may hand out its own). Then the provider answers from now on."""
        key = text.strip()
        p = self.provider(pid)
        if p is None or not key or key == self.secret_of(p):
            return p                              # nothing typed (the field lost focus), or the key it already has
        if busy():
            return None                           # still asking about the last key: Enter pressed twice
        looks = provider_for_key(key)
        self.place_busy()                         # the provider is asked on a worker thread, the bar running: a
        models, error = run_busy(self.busy, lambda: list_models(p["kind"], key, p["host"]))   # first `import anthropic` took 5 s here
        models = [list(m) for m in models] if not (error and p["custom"]) else p["models"]   # a custom host: not the usual list
        self._update(pid, secret=protect(key), last4=key[-4:], models=models,
                     model=p["model"] if p["model"] in [m[0] for m in models] else default_model(p["kind"], models) if models else "")
        self.key_field.clear()
        self._note(f"{p['name']}: key saved, {len(models)} model{'s' * (len(models) != 1)}."
                   + (f" Couldn't ask for this key's models ({error}), so these are the usual ones." if error else "")
                   + (f" It looks like a {PROVIDER_NAMES[looks]} key, though." if not p["custom"] and looks not in (None, p["kind"]) else ""))
        self.use(pid)
        return self.provider(pid)

    def forget_key(self, pid: str):
        p = self.provider(pid)
        self._update(pid, secret="", last4="")
        self._note(f"{p['name']}: the key is forgotten here; it keeps working on the provider's side.")
        self.refresh_providers()
        self.ai_changed.emit()

    def set_host(self, pid: str, text: str):
        """Where this provider's API is (empty: back to its usual host; a custom provider keeps its own)."""
        p = self.provider(pid)
        host = text.strip().rstrip("/") or (p["host"] if p["custom"] else DEFAULT_HOSTS[p["kind"]])
        if host != p["host"]:
            self._update(pid, host=host)
            self.ai_changed.emit()
        self.refresh_providers()

    def set_on(self, pid: str, on: bool):
        """Off: the provider stays as it is but can't answer and leaves the chat's picker."""
        self._update(pid, on=on)
        if not on and self.store.value("ai_active", "") == pid:
            self.store.setValue("ai_active", "")
        self.refresh_providers()
        self.ai_changed.emit()

    def use(self, pid: str, model: str | None = None):
        """Answer with this provider (and this model, if given) from now on."""
        if model is not None:
            self._update(pid, model=model)
        self.store.setValue("ai_active", pid)
        self.refresh_providers()
        self.ai_changed.emit()

    def sync_models(self, pid: str):
        """Ask the provider (at its host) which models the key can use."""
        p = self.provider(pid)
        if busy():
            return
        self.place_busy()
        models, error = run_busy(self.busy, lambda: list_models(p["kind"], self.secret_of(p), p["host"]))
        models = [list(m) for m in models] if not (error and p["custom"]) else p["models"]
        if pid == "ollama":
            self.found_ollama = models
        else:
            self._update(pid, models=models,
                         model=p["model"] if p["model"] in [m[0] for m in models] else default_model(p["kind"], models) if models else "")
        self._note(f"{p['name']}: {len(models)} model{'s' * (len(models) != 1)}"
                   + (f" (couldn't ask: {error})." if error else "."))
        self.refresh_providers()

    def add_model(self, pid: str):
        """A model the provider didn't list (a custom host with no models API): typed by its id."""
        if pid == "ollama":
            self._note("Ollama's models are the ones it has installed: `ollama pull <model>` in a terminal, then Sync models.")
            return
        mid, ok = QInputDialog.getText(self, "Add a model", "Model id, as the provider names it:")
        mid = mid.strip()
        if not (ok and mid):
            return
        p = self.provider(pid)
        if mid not in [m[0] for m in p["models"]]:
            self._update(pid, models=p["models"] + [[mid, mid]])
        self.use(pid, mid)

    def remove_model(self, pid: str, mid: str):
        p = self.provider(pid)
        models = [m for m in p["models"] if m[0] != mid]
        self._update(pid, models=models, model=p["model"] if p["model"] != mid else (models[0][0] if models else ""))
        self.refresh_providers()
        self.ai_changed.emit()

    def add_provider(self):
        """Add Provider: a custom one from the dialog; it is asked for its models and answers from now on."""
        got = add_provider_dialog(self, self.c)
        if not got:
            return
        providers = self.providers()
        pid = new_custom_id(providers)
        self._save_providers(providers + [{**self._builtin(got["kind"]), "id": pid, "name": got["name"], "host": got["host"],
                                           "models": [], "model": "", "custom": True}])
        self.shown_id = pid
        if got["key"]:
            self.set_key(pid, got["key"])
        else:
            self.sync_models(pid)
            self.use(pid)

    def rename_provider(self, pid: str):
        name, ok = QInputDialog.getText(self, "Rename provider", "Name:", text=self.provider(pid)["name"])
        if ok and name.strip():
            self._update(pid, name=name.strip())
            self.refresh_providers()
            self.ai_changed.emit()

    def remove_provider(self, pid: str):
        p = self.provider(pid)
        if QMessageBox.question(self, "Remove provider", f"Remove {p['name']}? The app forgets its key, host and "
                                "models; the key itself keeps working on the provider's side.") != QMessageBox.Yes:
            return
        self._save_providers([q for q in self.providers() if q["id"] != pid])
        if self.store.value("ai_active", "") == pid:
            self.store.setValue("ai_active", "")
        self.shown_id = "claude"
        self.refresh_providers()
        self.ai_changed.emit()

    def _provider_menu(self, pos):
        item = self.provider_list.itemAt(pos)
        if item is None:
            return
        p = self.provider(item.data(Qt.UserRole))
        menu = QMenu(self.provider_list)
        if p in self.usable():
            menu.addAction("Use this provider", lambda: self.use(p["id"]))
        menu.addAction("Turn off" if p["on"] else "Turn on", lambda: self.set_on(p["id"], not p["on"]))
        if p["custom"]:                                  # the built-in four keep their names and stay
            menu.addAction("Rename…", lambda: self.rename_provider(p["id"]))
            menu.addAction(f"{I_DELETE}  Remove", lambda: self.remove_provider(p["id"]))
        menu.exec(self.provider_list.mapToGlobal(pos))

    def _model_menu_at(self, pos):
        item = self.model_list.itemAt(pos)
        if item is None or not item.data(Qt.UserRole):
            return
        mid = item.data(Qt.UserRole)
        menu = QMenu(self.model_list)
        menu.addAction("Use this model", lambda: self.use(self.shown_id, mid))
        if self.shown_id != "ollama":
            menu.addAction(f"{I_DELETE}  Remove from the list", lambda: self.remove_model(self.shown_id, mid))
        menu.exec(self.model_list.mapToGlobal(pos))

    def _model_menu(self, menu: QMenu, p: dict):
        active = self.active()
        for model_id, name in p["models"] or [[p["model"], p["model"]]]:
            act = menu.addAction(name, lambda _=False, m=model_id: self.use(p["id"], m))
            act.setCheckable(True)
            act.setChecked(active is not None and active["id"] == p["id"] and model_id == p["model"])

    def _fill_ai_menu(self):
        """The chat's picker: every provider that can answer, each with its models; choosing one uses it."""
        menu = self.chat.model_btn.menu()
        menu.clear()
        usable = self.usable()
        for p in usable:
            self._model_menu(menu.addMenu(p["name"]), p)
        if not usable:
            menu.addAction("No AI yet").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Manage providers…", lambda: self.show_page(self.settings_page))

    def _note(self, text: str):
        self.keys_note.setText(text)
        self.keys_note.setVisible(bool(text))

    # ----- the reading language: the chat's picker beside the AI's

    def reading_language(self) -> tuple[str, str]:
        """(tag, name) of the language picked for reading; English until one is picked."""
        return self.store.value("reading_language", "en"), self.store.value("reading_language_name", "English")

    def set_reading_language(self, tag: str, name: str):
        self.store.setValue("reading_language", tag)
        self.store.setValue("reading_language_name", name)
        self._show_language()
        self.language_changed.emit(tag)

    def _show_language(self):
        self.chat.lang_btn.setText(f"{I_LANGUAGE}  {self.reading_language()[1]}  {I_CHEVRON}")

    def _fill_language_menu(self):
        """Every language Windows can read on this PC, the picked one ticked, and where to add more."""
        menu = self.chat.lang_btn.menu()
        menu.clear()
        current = self.reading_language()[0]
        for tag, name in ocr_languages():
            act = menu.addAction(f"{name}   ({tag})", lambda t=tag, n=name: self.set_reading_language(t, n))
            act.setCheckable(True)
            act.setChecked(tag == current)
        if menu.isEmpty():
            menu.addAction("Windows OCR isn't available: the text is read through the apps only").setEnabled(False)
        menu.addSeparator()
        menu.addAction("Add a language… (Windows settings)",
                       lambda: QDesktopServices.openUrl(QUrl("ms-settings:regionlanguage")))

    def _open_key_url(self):
        QDesktopServices.openUrl(QUrl(KEY_URLS[self.provider(self.shown_id)["kind"]]))

    def _filter_providers(self, text: str):
        text = text.strip().lower()
        for i in range(self.provider_list.count()):
            item = self.provider_list.item(i)
            item.setHidden(text not in item.text().lower())

    def refresh_providers(self):
        """The providers list (the one in use marked, off ones said so), the shown provider's pane, and the
        chat's picker button."""
        active = self.active()
        self.provider_list.blockSignals(True)
        self.provider_list.clear()
        for p in self.providers():
            on = active is not None and p["id"] == active["id"]
            icon = letter_icon(p["name"], self.c["primaryContainer" if p["on"] else "surfaceContainerHigh"],
                               self.c["onPrimaryContainer" if p["on"] else "onSurfaceVariant"])
            item = QListWidgetItem(icon, f"{p['name']}{'  ·  in use' if on else ''}{'  (off)' if not p['on'] else ''}")
            item.setData(Qt.UserRole, p["id"])
            self.provider_list.addItem(item)
            if p["id"] == self.shown_id:
                self.provider_list.setCurrentItem(item)
        self.provider_list.blockSignals(False)
        if self.provider_list.currentItem() is None:
            self.provider_list.setCurrentRow(0)           # (a removed provider was shown) -> _show_provider
        else:
            self._show_provider(self.shown_id)
        self._filter_providers(self.provider_find.text())
        self.chat.model_btn.setText(f"{I_SPARKLE}  {active['name']} · {model_label(active)}  {I_CHEVRON}" if active
                                    else f"{I_SPARKLE}  No AI yet  {I_CHEVRON}")

    def _show_provider(self, pid: str):
        """The right pane: this provider's switch, key, host and models."""
        p = self.provider(pid)
        if p is None:
            return
        self.shown_id = pid
        active = self.active()
        ollama = pid == "ollama"
        self.provider_title.setText(p["name"])
        self.provider_maker.setText(("in use  ·  " if active is not None and active["id"] == pid else "")
                                    + (f"speaks the {MAKERS[p['kind']]} API" if p["custom"] else MAKERS[p["kind"]]))
        self.provider_on.set_checked_now(p["on"])
        self.key_field.clear()
        self.key_field.setEnabled(not ollama)
        self.key_field.setPlaceholderText(
            "No key needed: Ollama runs on this PC" if ollama
            else f"From the environment, ending …{p['last4']}. Paste a key to save your own" if p.get("plain")
            else f"Saved, ending …{p['last4']}. Paste a new key to replace it" if p["secret"]
            else "Add API key" + (" (leave empty if the server needs none)" if p["custom"] else ""))
        self.key_link.setVisible(not p["custom"] and p["kind"] in KEY_URLS)
        self.forget_link.setVisible(bool(p["secret"]))
        self.host_field.setText(p["host"])
        self.host_link.setVisible(not p["custom"] and p["host"] != DEFAULT_HOSTS[p["kind"]])
        self.model_list.clear()
        for mid, name in p["models"]:
            item = QListWidgetItem(f"{'●' if mid == p['model'] else '○'}  {name}" + (f"    {mid}" if name != mid else ""))
            item.setData(Qt.UserRole, mid)
            self.model_list.addItem(item)
        if not p["models"]:
            empty = QListWidgetItem("Not running: install Ollama (ollama.com), then start the app again" if ollama
                                    else "No models yet: add a key, then Sync models (or + to type a model's id)")
            empty.setFlags(Qt.ItemIsEnabled)
            self.model_list.addItem(empty)

    def set_ball_look(self, look: str):
        self.ball_look = look
        self.look_group.button(BALL_LOOKS.index(look)).setChecked(True)
        self.image_btn.setEnabled(look == "image")

    def set_ball_image(self, path: str):
        self.ball_image = path
        name = Path(path).name if path else ""
        self.image_btn.setText((name if len(name) <= 24 else name[:21] + "…") if name else "Choose image…")
        self.image_btn.setToolTip(path or "No image yet: the logo is used")

    def choose_ball_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose the ball's image", self.ball_image or str(Path.home()),
                                              "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if path:
            self.set_ball_image(path)
            self.set_ball_look("image")

    def set_ball_ring(self, hex_: str):
        """The ring's colour as #rrggbb, or "" for the theme colour."""
        self.ball_ring = hex_
        self.ring_hex.setText(hex_.upper() if hex_ else "Theme")
        self.ring_dot.update()

    def choose_ring(self):
        colour = QColorDialog.getColor(QColor(self.ball_ring or self.seed), self, "Ring colour")
        if colour.isValid():
            self.set_ball_ring(colour.name())

    def load(self, values: dict):
        for k, sw in self.switches.items():
            sw.set_checked_now(values[k])
        self.set_ball_look(values["ball_look"])
        self.set_ball_image(values["ball_image"])
        self.set_ball_ring(values["ball_ring"])
        self.set_camera(values["camera"])
        self.words.setValue(values["summary_words"])
        self.mode = values["theme"]
        self.theme_group.button(["system", "light", "dark"].index(self.mode)).setChecked(True)
        self.set_font_choice(values["font"])
        self.seed_before_sheet = values["seed"]
        self.set_seed(values["seed"])

    def save(self):
        values = {k: sw.isChecked() for k, sw in self.switches.items()}
        values.update(seed=self.seed, theme=self.mode, font=self.font_name, summary_words=self.words.value(),
                      camera=self.camera, ball_look=self.ball_look, ball_image=self.ball_image, ball_ring=self.ball_ring)
        for k, v in values.items():
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

    def show_last(self):
        """Bring the window up where it was left: the same page, and the same Testing entry."""
        page = self.pages.widget(max(0, min(self.store.value("last_page", 0, type=int), 2)))
        self.testing.show_page(self.store.value("testing_page", 3, type=int))
        self.show_page(page)

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

    def set_font_choice(self, name: str):
        """Settings > Font: every window's text at once, as the theme colour is (kept with Save changes)."""
        self.font_name = name if name in FONTS else DEFAULTS["font"]
        self.font_btn.setText(f"{self.font_name}  {I_CHEVRON}")
        QApplication.setFont(text_font(self.font_name))
        self.apply_theme()                           # the stylesheet again: styled widgets take the new font

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
        if hasattr(self, "chat"):                    # the providers' marks take the new colours
            self.refresh_providers()
        for b in self.findChildren(SpinButton):
            b.set_color(c["primary"] if b.property("kind") == "text" else c["onSurface"])
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

    def place_busy(self):
        """The bar sits right under the title bar (at the very top on a full-screen page)."""
        self.busy.setGeometry(0, self.titlebar.height() if self.titlebar.isVisible() else 0, self.width(), 3)

    def resizeEvent(self, e):
        self.place_busy()
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
        self.shown.emit(True)

    def hideEvent(self, e):
        super().hideEvent(e)
        self.shown.emit(False)

    def _dwm(self, attribute: int, value: int):
        """Set a DWM window attribute (33 = corner preference, 34 = border colour). No-op off Windows 11."""
        if sys.platform == "win32":
            v = ctypes.c_uint(value)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(int(self.winId()), attribute, ctypes.byref(v), 4)


BRANDS = ("claude", "openai", "gemini", "ollama")   # the built-in providers, in the order they are listed


def new_custom_id(providers: list[dict]) -> str:
    """custom1, custom2…: one past the highest number still in use."""
    return f"custom{max([int(p['id'][6:]) for p in providers if p.get('custom')] + [0]) + 1}"


def model_label(p: dict) -> str:
    """The display name of the model this provider uses."""
    return next((name for model_id, name in p["models"] if model_id == p["model"]), p["model"] or "no model")


def add_provider_dialog(parent, c: dict) -> dict | None:
    """Settings > AI > Add Provider: a name, a key and the server's API root under the API it speaks
    (OpenAI or Anthropic). Returns {name, key, host, kind}, or None."""
    dlg = QDialog(parent, Qt.Dialog | Qt.WindowCloseButtonHint)
    dlg.setObjectName("root")
    dlg.setAttribute(Qt.WA_StyledBackground)
    dlg.setWindowTitle("Add custom provider")
    dlg.setStyleSheet(build_qss(c))
    dlg.setFixedWidth(560)
    v = QVBoxLayout(dlg)
    v.setContentsMargins(24, 20, 24, 20)
    v.setSpacing(6)
    v.addWidget(label("Add custom provider", "sheetTitle"))
    hint = label("Any server that speaks the OpenAI or Anthropic API: OpenRouter, DeepSeek, LM Studio, a proxy at "
                 "work… Put its API root under the API it speaks; the request path is shown underneath.", "sheetText")
    hint.setWordWrap(True)
    v.addWidget(hint)
    form = QWidget(objectName="addProvider")
    fv = QVBoxLayout(form)
    fv.setContentsMargins(0, 8, 0, 0)
    fv.setSpacing(6)
    fv.addWidget(label("Provider name *", "fieldLabel"))
    name = QLineEdit(placeholderText="Example: OpenRouter")
    name.setAccessibleName("Provider name")
    fv.addWidget(name)
    fv.addSpacing(6)
    fv.addWidget(label("API key", "fieldLabel"))
    key = QLineEdit(placeholderText="Leave empty if the server needs none", echoMode=QLineEdit.Password)
    key.setAccessibleName("API key")
    fv.addWidget(key)
    fv.addSpacing(10)
    fv.addWidget(label("Endpoint settings", "gh"))
    fields = {}
    for kind, title, path in (("openai", "OpenAI API", "/chat/completions"), ("claude", "Anthropic API", "/v1/messages")):
        fv.addWidget(label(f"{title} base URL", "fieldLabel"))
        edit = QLineEdit(placeholderText="https://example.com/v1")
        edit.setAccessibleName(f"{title} base URL")
        preview = label("Enter the API root URL to see the final request path", "sheetText")
        edit.textChanged.connect(lambda t, pv=preview, path=path: pv.setText(
            f"Requests go to {t.strip().rstrip('/')}{path}" if t.strip() else "Enter the API root URL to see the final request path"))
        fv.addWidget(edit)
        fv.addWidget(preview)
        fv.addSpacing(6)
        fields[kind] = edit
    v.addWidget(form)
    actions = QHBoxLayout()
    actions.setSpacing(8)
    actions.addStretch(1)
    cancel = button("Cancel", "text")
    cancel.clicked.connect(dlg.reject)
    add = button("Add", "filled")
    add.clicked.connect(dlg.accept)
    actions.addWidget(cancel)
    actions.addWidget(add)
    v.addSpacing(8)
    v.addLayout(actions)

    def kind_given() -> str | None:               # OpenAI first when both are filled
        return next((k for k in ("openai", "claude") if fields[k].text().strip()), None)

    def check():
        add.setEnabled(bool(name.text().strip()) and kind_given() is not None)
    for edit in (name, *fields.values()):
        edit.textChanged.connect(check)
    check()
    if dlg.exec() != QDialog.Accepted or not (name.text().strip() and kind_given()):
        return None
    kind = kind_given()
    return {"name": name.text().strip(), "key": key.text().strip(), "host": fields[kind].text().strip().rstrip("/"), "kind": kind}


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
    hint = label("Paste a key from Claude (sk-ant-…), Gemini (AQ.… or AIza…) or OpenAI (sk-…): the app tells them apart "
                 "and remembers it (Settings > AI > Providers: other providers and hosts there). No key? Install Ollama from "
                 "ollama.com and start the app again: it is free and runs on this PC.", "sheetText")
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
