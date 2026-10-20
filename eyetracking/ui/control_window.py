"""The tester version's window (PySide6, Material 3 colours, light or dark as Windows is).

A sidebar of three pages, the status line under it:
    Instructions   the tester's name, the camera, what to do
    Calibration    the 8 recordings as chips (Line model is saved with every Points), each done twice
                   (SETS): the chosen one's steps, what this set still needs, Start; "n/16" in the sidebar
    Gaze check     locked until both sets are done; then twice, every model at once on 36 rings: trained
                   on one set, then on both; "n/2" in the sidebar
The calibration and gaze-check pages fill the screen with no title bar or buttons, then the window comes
back as it was. pick_camera() is the small camera popup used at first start.

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
from PySide6.QtCore import QEasingCurve, QEvent, QRectF, QSettings, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QButtonGroup, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMenu, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from eyetracking.services.camera import camera_names
import eyetracking.services.storage as storage
from eyetracking.services.mailer import TO
from eyetracking.core.methods import METHOD_BY_ID, METHODS, partners

TITLE = "Eye tracking test"
SEED = "#E5483B"            # the logo's red: every colour is made from it
TESTER_METHODS = tuple(m for m in METHODS if m.id != "line-model")   # Line model is saved with every Points
SETS = 2                    # each type calibrated twice; gaze check n: the models trained on the newest n sets

TEXT_FONTS = ["Segoe UI Variable Text", "Segoe UI", "Segoe Fluent Icons", "Segoe MDL2 Assets"]
ICON_FONTS = ["Segoe Fluent Icons", "Segoe MDL2 Assets"]   # built into Windows 11 / Windows 10

# Segoe Fluent Icons code points
I_MIN, I_MAX, I_RESTORE, I_CLOSE = "", "", "", ""
I_CHECK, I_CHEVRON = "", ""
I_INFO, I_EYE, I_CHART = "", "", ""   # Info, View (eye), AreaChart
NAV = [(I_INFO, "Instructions"), (I_EYE, "Calibration"), (I_CHART, "Gaze check")]   # the sidebar = the pages


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

/* buttons */
QPushButton { border: none; border-radius: 12px; min-height: 36px; max-height: 36px; padding: 0 16px; font-size: 13px; font-weight: 600; background: transparent; }
QPushButton[kind="filled"] { background: $primary; color: $onPrimary; }
QPushButton[kind="filled"]:hover { background: $primaryHover; }
QPushButton[kind="filled"]:pressed { background: $primaryPress; }
QPushButton[kind="filled"]:disabled { background: $surfaceContainerHighest; color: $onSurfaceVariant; }

/* chips */
QPushButton[kind="chip"], QPushButton[kind="assist"] { min-height: 30px; max-height: 30px; border-radius: 8px;
    border: 1px solid $outlineVariant; padding: 0 14px; color: $onSurfaceVariant; }
QPushButton[kind="assist"] { color: $onSurface; padding: 0 10px 0 8px; }
QPushButton[kind="assist"]::menu-indicator { image: none; width: 0; }
QPushButton[kind="chip"]:hover, QPushButton[kind="assist"]:hover { background: $hover; }
QPushButton[kind="chip"]:checked { background: $secondaryContainer; color: $onSecondaryContainer; border-color: $secondaryContainer; padding-left: 8px; }
QPushButton[kind="chip"]:checked:hover { background: $secContainerHover; }
QPushButton[kind="chip"]:disabled { color: $outline; border-color: $surfaceContainerHigh; }

/* keyboard focus: a 2px ring on every button. The ring is a border, which eats into the text's room,
   so each kind's padding gives that width back: focused text never moves or clips */
QPushButton:focus { border: 2px solid $secondary; padding: 0 14px; }
QPushButton[kind="nav"]:focus { padding: 0 10px; }
QPushButton[kind="chip"]:focus { border: 2px solid $secondary; padding: 0 13px; }
QPushButton[kind="chip"]:checked:focus { padding: 0 13px 0 7px; }
QPushButton[kind="assist"]:focus { border: 2px solid $secondary; padding: 0 9px 0 7px; }
#caps QPushButton:focus { padding: 0; }

/* the page */
#side { background: $surfaceContainerLow; border-right: 1px solid $outlineVariant; }
QPushButton[kind="nav"] { text-align: left; padding: 0 12px; border-radius: 8px; font-weight: 400; font-size: 14px; }
QPushButton[kind="nav"]:hover { background: $hover; }
QPushButton[kind="nav"]:checked, QPushButton[kind="nav"]:checked:hover { background: $selected; color: $onSurface; }
#rowsub { font-size: 12px; color: $onSurfaceVariant; }
#lead { font-size: 15px; }
#sheetTitle { font-size: 16px; font-weight: 600; }
#sheetText { font-size: 13px; color: $onSurfaceVariant; }
QLineEdit#name { min-height: 34px; max-height: 34px; min-width: 280px; max-width: 280px; padding: 0 6px; border: 1px solid $outlineVariant;
    border-radius: 4px; background: $surfaceContainerLow; selection-background-color: $primary; selection-color: $onPrimary; }
QLineEdit#name:focus { border-bottom: 2px solid $primary; }
QMenu { background: $surfaceContainer; border: 1px solid $outlineVariant; padding: 4px; }
QMenu::item { padding: 8px 32px 8px 12px; border-radius: 4px; }
QMenu::item:selected { background: $hover; }
""")


def build_qss(c: dict) -> str:
    return QSS.substitute(
        c,
        hover=rgba(c["onSurface"], .08),
        selected=rgba(c["onSurface"], .14),     # the chosen sidebar entry: a darker neutral, not a tint
        primaryHover=mix(c["onPrimary"], c["primary"], .08),
        primaryPress=mix(c["onPrimary"], c["primary"], .12),
        secContainerHover=mix(c["onSecondaryContainer"], c["secondaryContainer"], .08),
    )


def is_dark() -> bool:
    return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark


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
    training a model, checking and saving a calibration): M3's indeterminate linear progress
    indicator. It sits over the page, 3 px high, and takes no clicks. It only moves while events are
    processed: App._busy keeps them going."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.color, self.track = QColor(SEED), QColor(229, 72, 59, 50)
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
    cold camera driver, training a dozen models or saving a big calibration could each do that. Returns
    work()'s result; raises its exception. Called again while already running (a click answered
    meanwhile), it just runs work() here: never two pumps."""
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


def method_picker(group: QButtonGroup, methods=METHODS) -> QWidget:
    """Calibration types as chips, a labelled line per row ("In order", "Random order", "Special
    steps"), three chips a line. A chip's id in `group` is its index in METHODS; `methods` picks
    which types."""
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


# ---------------------------------------------------------------- the tester's page

class TestingPage(QWidget):
    """The sidebar picks Instructions, Calibration or Gaze check; the status line sits under it."""

    command = Signal(str)          # a start button: "calibrate" | "check"
    person_changed = Signal(str)   # the name, as typed
    camera_changed = Signal(str)   # a camera picked

    def __init__(self):
        super().__init__()
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        side = QFrame(objectName="side")
        side.setFixedWidth(220)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 12, 12, 12)
        sv.setSpacing(2)
        self.side_nav = QButtonGroup(self)               # the chosen entry stays highlighted: it is the page on the right
        for i, (glyph, text) in enumerate(NAV):
            b = button(f"{glyph}   {text}", "nav")
            b.setCheckable(True)
            self.side_nav.addButton(b, i)                # its id = the page index on the right
            sv.addWidget(b)
        sv.addStretch(1)
        self.status = label("", "rowsub")                # the latest word from the app
        self.status.setWordWrap(True)
        sv.addWidget(self.status)
        h.addWidget(side)

        self.main = QStackedWidget()
        for page in (self._instructions_page(), self._calibrate_page(), self._check_page()):
            self.main.addWidget(page)
        h.addWidget(self.main, 1)
        self.side_nav.idClicked.connect(lambda i: cross_fade(self.main, i))
        self.set_method(TESTER_METHODS[0].id)
        self.show_page(0)

    def show_page(self, i: int):
        self.side_nav.button(i).setChecked(True)
        self.main.setCurrentIndex(i)

    def say(self, text: str):
        """A line from the app, in the sidebar's status."""
        self.status.setText(text)

    @staticmethod
    def _page(glyph: str, title: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 20, 24, 24)
        v.setSpacing(12)
        v.addWidget(label(f"{glyph}   {title}", "sheetTitle"))
        return page, v

    @staticmethod
    def _text(v: QVBoxLayout, text: str, name: str = "lead"):
        w = label(text, name)
        w.setWordWrap(True)
        v.addWidget(w)
        return w

    def _go(self, v: QVBoxLayout, text: str, slot) -> QPushButton:
        go = button(text, "filled")
        go.clicked.connect(slot)
        v.addSpacing(8)
        v.addWidget(go, 0, Qt.AlignLeft)
        v.addStretch(1)
        return go

    # ----- instructions: the name, the camera, what to do

    def _instructions_page(self) -> QWidget:
        page, v = self._page(*NAV[0])
        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.name_edit = QLineEdit(objectName="name")
        self.name_edit.textChanged.connect(self.person_changed.emit)
        self.camera_btn = button("", "assist")
        self.camera_menu = QMenu(self.camera_btn)
        self.camera_menu.aboutToShow.connect(self._fill_camera_menu)   # list the cameras when opened
        self.camera_btn.setMenu(self.camera_menu)
        form.addWidget(label("Your name", "lead"), 0, 0)
        form.addWidget(self.name_edit, 0, 1)
        form.addWidget(label("Camera", "lead"), 1, 0)
        form.addWidget(self.camera_btn, 1, 1, Qt.AlignLeft)
        form.setColumnStretch(2, 1)
        v.addLayout(form)
        v.addSpacing(8)
        for i, step in enumerate(["Type your name and pick the camera that faces you.",
                                  f"Calibration: all {len(TESTER_METHODS)} types, twice ({SETS * len(TESTER_METHODS)} "
                                  "in all). The next one is picked for you.",
                                  "Gaze check: twice, after the calibrations."]):
            self._text(v, f"{i + 1}.  {step}")
        self._text(v, "Sit as you normally would, with your face in good light. When both gaze checks are done, "
                      f"your results are emailed to {TO}.", "sheetText")
        v.addStretch(1)
        return page

    def _fill_camera_menu(self):
        self.camera_menu.clear()
        names = camera_names()
        for n in names:
            self.camera_menu.addAction(n, lambda n=n: self.camera_changed.emit(n))
        if not names:
            self.camera_menu.addAction("No camera found").setEnabled(False)

    def set_camera(self, name: str):
        self.camera_btn.setText(f"{name or 'Not chosen'}  {I_CHEVRON}")

    # ----- calibration: one of the 8, what is left

    def _calibrate_page(self) -> QWidget:
        page, v = self._page(*NAV[1])
        self.method_chips = QButtonGroup(self)
        v.addWidget(method_picker(self.method_chips, TESTER_METHODS))
        self.method_chips.idClicked.connect(lambda i: self.set_method(METHODS[i].id))
        self.method_steps = self._text(v, "")
        self.todo_label = self._text(v, "", "sheetText")
        self.calibrate_go = self._go(v, "Start the calibration", lambda: self.command.emit("calibrate"))
        self.redo_btn = button(f"Redo a calibration  {I_CHEVRON}", "assist")   # its newest one into the trash
        self.redo_menu = QMenu(self.redo_btn)
        self.redo_menu.aboutToShow.connect(self._fill_redo_menu)
        self.redo_btn.setMenu(self.redo_menu)
        v.insertWidget(v.indexOf(self.calibrate_go) + 1, self.redo_btn, 0, Qt.AlignLeft)
        return page

    def _fill_redo_menu(self):
        self.redo_menu.clear()
        for m, n in self.calibrations().items():
            if n:
                self.redo_menu.addAction(f"{m.name} (your last one)", lambda m=m: self.redo(m))

    def redo(self, m):
        """Delete the newest calibration of a type (its Line model copy too) so it can be done again. The
        gaze checks go as well, if any: the models change, so they must be done again."""
        newest = storage.list_calibrations(m.id)[0]
        files = [newest] + [p for o in partners(m)
                            if (p := storage.method_dir(o.id) / f"{o.id}_{newest.stem[-17:]}.json").exists()]
        checks = [p for p in storage.result_files() if p.suffix == ".csv"]
        if QMessageBox.question(self, "Redo a calibration", f"Delete your last {m.name} calibration and do it again?"
                                + (" Your gaze checks are deleted too: you will do them again." if checks else "")
                                ) != QMessageBox.Yes:
            return
        trash = storage.discard(files + checks)
        self.refresh()
        self.say(f"{m.name} deleted (kept in {trash.relative_to(storage.BASE_DIR.parent).as_posix()}/). Do it again.")

    def set_method(self, method_id: str):
        m = METHOD_BY_ID[method_id]
        self.method_id = method_id
        self.method_chips.button(METHODS.index(m)).setChecked(True)
        self.method_steps.setText("\n".join(f"{i + 1}.  {s}" if len(m.steps) > 1 else s for i, s in enumerate(m.steps)))

    def calibrations(self) -> dict:
        """Each of the 8 types -> how many usable calibrations of it the person in use has."""
        return {m: storage.usable_calibrations(m.id) for m in TESTER_METHODS}

    @staticmethod
    def _todo(done: dict) -> list:
        low = min(done.values())
        return [m for m, n in done.items() if n == low] if low < SETS else []

    def todo(self) -> list:
        """The types still to do in the set the person is on ([] once every set is done)."""
        return self._todo(self.calibrations())

    def checks(self) -> int:
        """How many of the gaze checks the person has done."""
        return len(storage.checks_done() & set(range(1, SETS + 1)))

    def refresh(self):
        """At start, on a new name, after each calibration and check: the progress, the next one picked."""
        done = self.calibrations()
        todo, checks = self._todo(done), self.checks()
        calibrated, total = sum(min(n, SETS) for n in done.values()), SETS * len(TESTER_METHODS)
        self.todo_label.setText(f"Set {min(done.values()) + 1} of {SETS}. Still to do: {', '.join(m.name for m in todo)}."
                                if todo else "Both sets done. Next: the gaze check.")
        if todo:
            self.set_method(todo[0].id)               # in order: set 2 starts again at Original 3x3
        for m, n in done.items():                     # exactly SETS of each, one of each per set: only this set's open
            chip = self.method_chips.button(METHODS.index(m))
            chip.plain = f"{m.name}  {min(n, SETS)}/{SETS}"
            chip.setText(f"{I_CHECK}  {chip.plain}" if chip.isChecked() else chip.plain)
            chip.setEnabled(m in todo)
        self.calibrate_go.setEnabled(bool(todo))
        self.redo_btn.setEnabled(any(done.values()))
        for i, n in ((1, f"{calibrated}/{total}"), (2, f"{checks}/{SETS}")):
            self.side_nav.button(i).setText(f"{NAV[i][0]}   {NAV[i][1]}   {n}")
        self.check_go.setEnabled(not todo and checks < SETS)
        self.check_note.setText(
            f"Finish the calibrations first ({calibrated}/{total})." if todo else
            "Both done. Thank you!" if checks == SETS else
            f"Gaze check {checks + 1} of {SETS}: the models trained on "
            f"{'one set' if checks == 0 else 'both sets'} of your calibrations.")

    # ----- the gaze check, twice

    def _check_page(self) -> QWidget:
        page, v = self._page(*NAV[2])
        self._text(v, "Do it twice: first with the models trained on one set of your calibrations, then on both. "
                      f"Look at the white ring and press Enter; it ends after {storage.MULTI_GRID ** 2} rings. "
                      "Space pauses, Esc stops without saving.")
        self.check_note = self._text(v, "")
        self.check_go = self._go(v, "Start the gaze check", lambda: self.command.emit("check"))
        return page


# ---------------------------------------------------------------- window

class MainWindow(QWidget):
    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setObjectName("root")
        self.setAttribute(Qt.WA_StyledBackground)
        self.setWindowTitle(TITLE)
        self.resize(960, 600)
        self.setMinimumSize(760, 520)
        self.store = QSettings("StudyAssistant", "TesterCalibration")   # the name and the camera

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.titlebar = self._title_bar()
        root.addWidget(self.titlebar)
        # the tester's page; a calibration / check page is added while it runs
        self.testing = TestingPage()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.testing)
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
        self.apply_theme()
        QGuiApplication.styleHints().colorSchemeChanged.connect(lambda _: self.apply_theme())

    def _title_bar(self):
        bar = TitleBar(objectName="titlebar")
        bar.setFixedHeight(48)
        h = QHBoxLayout(bar)
        h.setContentsMargins(16, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(AppMark())
        h.addSpacing(12)
        h.addWidget(label(TITLE, "appname"))
        h.addStretch(1)

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

    # ----- full-screen pages (calibration, check)

    def show_fullscreen_page(self, page: QWidget):
        """Show `page` alone on the whole screen: no title bar or buttons.
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
        self.pages.setCurrentWidget(self.testing)   # results land in the page's status line
        if not was_visible:
            self.hide()
        elif was_max:
            self.showMaximized()
        else:
            self.showNormal()

    def apply_theme(self):
        c = make_scheme(SEED, is_dark())
        self.setStyleSheet(build_qss(c))
        for edit in self.findChildren(QLineEdit):
            pal = edit.palette()
            pal.setColor(QPalette.PlaceholderText, QColor(c["onSurfaceVariant"]))
            edit.setPalette(pal)
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


def pick_camera(names: list[str], current: str = "") -> str | None:
    """Small popup: choose a webcam. Returns its name, or None if closed."""
    dlg = QDialog(None, Qt.Dialog | Qt.WindowCloseButtonHint)
    dlg.setObjectName("root")
    dlg.setAttribute(Qt.WA_StyledBackground)
    dlg.setWindowTitle("Choose a camera")
    dlg.setStyleSheet(build_qss(make_scheme(SEED, is_dark())))
    v = QVBoxLayout(dlg)
    v.setContentsMargins(24, 20, 24, 20)
    v.setSpacing(12)
    v.addWidget(label("Which camera faces you?", "sheetTitle"))
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
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
