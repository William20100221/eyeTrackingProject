"""
The whole app, wired together.

    python main.py                       (asks which camera, in the console)
    python main.py --mouse               (no camera: your mouse plays the eyes)
    python main.py --provider gemini     (AI provider, see services/reading_ai.py)

The main window (ui/control_window.py) has three tabs:
    Chat         replies, quiz; "Go Reading" closes the window, back to the ball
    Settings     camera, AI model, theme
    Calibration  25 dots, saves the data, retrains the gaze model
The window and the floating ball swap: click the ball to get the window back.
After reading, the AI summarises what you skipped and quizzes you on it.
Right-click the ball to quit.

What talks to what:
    camera -> tracker -> gaze model -> smoothed screen point
        -> ball halo (face found?), gaze cursor + dwell-click, reading window
    reading window -> (full text, read paragraphs) -> reading AI -> chat tab
"""
from __future__ import annotations

import argparse
import sys
import threading

from PySide6.QtCore import QElapsedTimer, QObject, QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from eyetracking.core.analysis import is_blinking
from eyetracking.core.features import N_FEATURES, gaze_features
from eyetracking.core.gaze_model import GazeModel
from eyetracking.core.reading import mark_unread, unread_word_count
from eyetracking.services import reading_ai
from eyetracking.services.camera import camera_names
from eyetracking.services.reading_ai import MODELS, grade_answer, make_questions, summarize
from eyetracking.services.storage import load_all_calibrations, save_calibration
from eyetracking.services.tracker import EyeTracker
from eyetracking.testing.try_reading_ai import SAMPLE_FULL
from eyetracking.ui.assistant_overlay import DwellController, FloatingBall, GazeCursor
from eyetracking.ui.calibration_window import CalibrationWindow, fake_features_from_mouse
from eyetracking.ui.control_window import MODEL_IDS, TEXT_FONTS, SettingsWindow, pick_camera
from eyetracking.ui.gaze_window import GazeWindow, fake_point_from_mouse

TICK_MS = 33             # ~30 frames a second; the camera read limits it anyway
SMOOTH = 0.3             # share of each new gaze point taken: lower = steadier but laggier
SECONDS_PER_WORD = 0.1   # gaze time a paragraph needs per word to count as read

PARA = "font-size:22px; color:#1F2328; padding:18px; border-radius:12px; background:{};"
UNREAD_BG, READ_BG = "white", "#DDF3E4"


def to_screen(point: tuple[float, float]) -> QPoint:
    """0..1 screen fractions (what the model predicts) -> global pixels, kept on screen."""
    g = QApplication.primaryScreen().geometry()
    x = min(max(point[0], 0.0), 1.0)
    y = min(max(point[1], 0.0), 1.0)
    return QPoint(g.x() + round(x * (g.width() - 1)), g.y() + round(y * (g.height() - 1)))


class ReadingWindow(QWidget):
    """Full-screen text, one block per paragraph. The gaze error (~150 px) is
    far bigger than a line of text (~30 px), so reading is tracked per
    paragraph: it counts as read once the gaze has stayed on it for
    SECONDS_PER_WORD per word, and turns green."""

    finished = Signal(str, str)   # full text, the paragraphs that were read

    def __init__(self, text: str) -> None:
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle("Reading")
        self.setStyleSheet("background:#F4F5F7;")

        open_btn = QPushButton("Open .txt")
        open_btn.clicked.connect(self._open)
        done_btn = QPushButton("Done")
        done_btn.clicked.connect(self.finish)
        self._status = QLabel()
        bar = QHBoxLayout()
        for w in (open_btn, self._status):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(done_btn)

        self._col = QVBoxLayout()
        self._col.setSpacing(48)   # big gaps, so a gaze error lands between paragraphs, not in the next one
        host = QWidget()
        host.setLayout(self._col)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(host)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(120, 30, 120, 30)
        lay.addLayout(bar)
        lay.addWidget(self._scroll)
        self.set_text(text)

    def set_text(self, text: str) -> None:
        while self._col.count():                   # clear the old text
            w = self._col.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        self._seen = [0.0] * len(self._paras)      # seconds of gaze per paragraph
        self._labels = []
        for p in self._paras:
            label = QLabel(p)
            label.setWordWrap(True)
            label.setStyleSheet(PARA.format(UNREAD_BG))
            self._col.addWidget(label)
            self._labels.append(label)
        self._col.addStretch(1)
        self._update_status()

    def feed(self, gaze: QPoint, dt: float) -> None:
        """Called every frame with the gaze point (global pixels) and the seconds since the last one."""
        viewport = self._scroll.viewport()
        if not viewport.rect().contains(viewport.mapFromGlobal(gaze)):
            return                                 # looking at the buttons, or off the text
        for i, label in enumerate(self._labels):
            if label.rect().contains(label.mapFromGlobal(gaze)):
                was_read = self._is_read(i)
                self._seen[i] += dt
                if self._is_read(i) and not was_read:
                    label.setStyleSheet(PARA.format(READ_BG))
                    self._update_status()
                return

    def _is_read(self, i: int) -> bool:
        # ponytail: one fixed reading speed for everyone; measure it per user if it misjudges
        return self._seen[i] >= SECONDS_PER_WORD * len(self._paras[i].split())

    def _update_status(self) -> None:
        done = sum(self._is_read(i) for i in range(len(self._paras)))
        self._status.setText(f"  {done} / {len(self._paras)} paragraphs read")

    def finish(self) -> None:
        read = [p for i, p in enumerate(self._paras) if self._is_read(i)]
        self.finished.emit("\n\n".join(self._paras), "\n\n".join(read))
        self.close()

    def _open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a text", "", "Text files (*.txt)")
        if path:
            with open(path, encoding="utf-8", errors="replace") as f:   # odd characters -> �, not a crash
                self.set_text(f.read())

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()                           # leave without the AI part
        else:
            super().keyPressEvent(event)


class Background(QObject):
    """Runs slow work (AI calls take seconds) on another thread, then hands the
    result back on the GUI thread: only the GUI thread may touch widgets."""

    _done = Signal(object, object)   # what to call next, the result (or the exception)

    def __init__(self) -> None:
        super().__init__()
        self._done.connect(self._deliver)

    def run(self, work, then) -> None:
        def job() -> None:
            try:
                result = work()
            except Exception as e:                 # missing package, no API key, network...
                result = e
            self._done.emit(then, result)          # queued across to the GUI thread
        threading.Thread(target=job, daemon=True).start()

    @Slot(object, object)
    def _deliver(self, then, result) -> None:
        then(result)


class App(QObject):
    def __init__(self, mouse: bool, provider: str) -> None:
        super().__init__()
        self.mouse, self.provider = mouse, provider

        # the main window: Chat + Settings + Calibration tabs (ui/control_window.py)
        self.control = SettingsWindow()
        self.panel = self.control.chat
        self.ball = FloatingBall()
        self.ball.dock_right()
        self.ball.show()
        self.cursor = GazeCursor()
        self.dwell = DwellController(self.cursor)
        self.background = Background()

        self.control.saved.connect(self.apply_settings)
        self.control.calibrate_requested.connect(self.calibrate)
        self.control.shown.connect(lambda shown: self.ball.setVisible(not shown))   # window XOR ball
        self.panel.message_sent.connect(self.answer)
        self.panel.command.connect(lambda c: {"read": self.read, "check": self.check}[c]())
        self.panel.go_reading.connect(self.control.hide)
        self.ball.clicked.connect(lambda: self.control.show_page(self.panel))
        self.ball.menu_actions = [("Settings", self.open_settings)]
        settings = self.control.read_store()

        self.tracker = None
        self.camera = settings["camera"]
        if not mouse:
            try:
                self.tracker = EyeTracker(self._camera_index())
            except Exception as e:                 # no camera, popup closed, MediaPipe failed
                print(f"Camera didn't start ({e}): running in mouse mode instead.")
                self.mouse = True
        self.model = self._load_model()
        self.smoothed: tuple[float, float] | None = None
        self.fullscreen: QWidget | None = None   # calibration / check page, while open
        self.apply_settings({**settings, "camera": self.camera})   # the popup may have just chosen one
        self.reading: ReadingWindow | None = None
        self.full_text = ""
        self.quiz: list[dict] = []                # questions still to ask

        # ponytail: camera + MediaPipe on the GUI thread (~30-50 ms a frame); move to a QThread if the UI stutters
        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)

        self.panel.status.setText(self._model_status())
        self.control.show_page(self.panel)

    # ---- settings + camera -------------------------------------------
    def _camera_index(self) -> int:
        """The saved camera's index; asks with the popup if none is saved or it's gone."""
        names = camera_names()
        if not names:
            raise RuntimeError("no camera found")
        if self.camera not in names:
            chosen = pick_camera(names, self.camera)
            if chosen is None:
                raise RuntimeError("no camera chosen")
            self.camera = chosen
            self.control.store.setValue("camera", chosen)      # remembered for next time
            self.control.set_camera(chosen)
        return names.index(self.camera)

    def apply_settings(self, values: dict) -> None:
        """Startup, and every time Save changes is pressed."""
        self.show_ball = values["show_ball"]
        self.ball.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, values["ball_on_top"])
        self.ball.setVisible(not self.control.isVisible())   # setWindowFlag hides it
        # ponytail: the model choice applies to the claude provider only (--provider gemini/openai keep theirs)
        reading_ai.MODELS["claude"] = MODEL_IDS.get(values["model"], reading_ai.MODELS["claude"])
        self.summary_words = values["summary_words"]
        if self.tracker is not None and values["camera"] != self.camera:   # switch cameras live
            self.tracker.close()
            self.camera = values["camera"]
            try:
                self.tracker = EyeTracker(self._camera_index())
                self.panel.reply(f"Now using the camera: {self.camera}.")
            except Exception as e:
                self.tracker, self.mouse = None, True
                self.panel.reply(f"That camera didn't start ({e}). Mouse mode until you pick another.")

    def open_settings(self) -> None:
        self.control.show_page(self.control.settings_page)

    # ---- gaze ------------------------------------------------------
    def _load_model(self) -> GazeModel | None:
        """Train on every saved calibration, the newest weighted most."""
        self.samples_used = 0
        if self.mouse:
            return None
        samples, weights = load_all_calibrations(N_FEATURES)
        self.samples_used = len(samples)
        return GazeModel().fit(samples, weights) if samples else None

    def _model_status(self) -> str:
        if self.mouse:
            return "Mouse mode: the mouse plays your eyes."
        if self.model is None:
            return "No calibration yet: open the Calibration tab first."
        return f"Gaze model trained on {self.samples_used} samples, the newest calibration counting most."

    def predict(self, features) -> tuple[float, float] | None:
        """Eye features -> (x, y) screen fractions, or None."""
        if features is None:
            return None
        if self.mouse:
            return features                       # the mouse position already is the answer
        if self.model is None:
            return None
        try:
            return self.model.predict(features)
        except ValueError:                        # the saved file holds an older kind of features
            self.model = None
            self.panel.reply("Your saved calibration is from an older version. Type calibrate.")
            return None

    def tick(self) -> None:
        dt = self.clock.restart() / 1000
        if self.mouse:
            face, features = True, fake_point_from_mouse()
        else:
            # read() not read_features(): the ball needs "face found" even while blinking
            r = self.tracker.read()
            face = r is not None and r.face_found
            features = gaze_features(r) if face and not is_blinking(r) else None
        self.ball.set_tracking_state(face)

        point = self.predict(features)
        if point is None:
            return
        # raw gaze jitters; without smoothing the dwell ring never fills
        # ponytail: plain exponential average; swap in a real filter (e.g. One Euro) if lag or jitter bothers
        if self.smoothed is None:
            self.smoothed = point
        else:
            self.smoothed = tuple(s + SMOOTH * (p - s) for s, p in zip(self.smoothed, point))
        gaze = to_screen(self.smoothed)
        self.dwell.feed(gaze)
        self.ball.look_at(gaze)                  # the water-eye ball looks where you look
        if self.reading is not None:
            self.reading.feed(gaze, dt)

    # ---- full-screen windows that read the camera themselves -------
    def _open_fullscreen(self, window: QWidget) -> None:
        """Pause our camera reads (one reader at a time) and hide the
        always-on-top windows so they don't cover the dots."""
        self.timer.stop()
        self.cursor.hide()                        # the ball hides itself: the window is up
        self.fullscreen = window                  # keep a reference, or Python deletes it
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._fullscreen_closed)
        self.control.show_fullscreen_page(window)   # the main window, full screen, no buttons

    def _fullscreen_closed(self) -> None:
        self.fullscreen = None
        self.clock.restart()
        self.timer.start(TICK_MS)

    # ---- commands (tabs + chat sidebar buttons) ------------------------
    def calibrate(self) -> None:
        get = fake_features_from_mouse if self.mouse else self.tracker.read_features
        window = CalibrationWindow(get_features=get)
        window.finished.connect(lambda samples: self._calibrated(window, samples))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, samples: list) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self.panel.reply("Mouse mode: calibration not saved.")
        elif not samples:
            self.panel.reply("No samples: was your face in view? Try calibrate again.")
        else:
            path = save_calibration(samples)
            self.model = self._load_model()       # the new file + all older ones
            self.panel.reply(f"Calibrated with {len(samples)} samples, saved to {path.name}. "
                             f"Gaze model retrained on {self.samples_used} samples.")

    def check(self) -> None:
        if self.mouse:
            get = fake_point_from_mouse
        elif self.model is None:
            self.panel.reply("Calibrate first: open the Calibration tab.")
            return
        else:
            def get():
                return self.predict(self.tracker.read_features())
        self._open_fullscreen(GazeWindow(get_point=get))

    def read(self) -> None:
        if self.model is None and not self.mouse:
            self.panel.reply("Calibrate first: open the Calibration tab.")
            return
        self.control.hide()                       # the ball comes back while you read
        if self.reading is None:
            self.reading = ReadingWindow(SAMPLE_FULL)
            self.reading.finished.connect(self._read_finished)
            self.reading.destroyed.connect(self._reading_closed)
        self.reading.showFullScreen()

    def _reading_closed(self) -> None:
        self.reading = None

    # ---- the reading AI ------------------------------------------------
    def _read_finished(self, full_text: str, read_text: str) -> None:
        self.full_text = full_text
        unread = unread_word_count(mark_unread(full_text, read_text))
        self.control.show_page(self.panel)        # the summary and quiz show in the chat
        self.panel.reply(f"You skipped {unread} of {len(full_text.split())} words. "
                         f"Asking {self.provider}...")
        self.background.run(
            lambda: (summarize(full_text, read_text, "unread", self.provider, self.summary_words),
                     make_questions(full_text, read_text, "unread", self.provider)),
            self._ai_ready)

    def _ai_ready(self, result) -> None:
        if isinstance(result, Exception):
            self._ai_error(result)
            return
        summary, self.quiz = result
        self.panel.reply(summary)
        self._ask_next()

    def _ask_next(self) -> None:
        if self.quiz:
            self.panel.reply(f"Question: {self.quiz[0]['question']}")
        else:
            self.panel.reply("No more questions.")
        self.panel.set_answering(bool(self.quiz))

    def answer(self, text: str) -> None:
        self.panel.set_answering(False)           # back on while the next question is asked
        question = self.quiz.pop(0)
        if text.strip().lower() == "skip":
            self.panel.reply(f"Model answer: {question['model_answer']}")
            self._ask_next()
            return
        self.panel.reply("Checking...")
        self.background.run(
            lambda: grade_answer(self.full_text, question, text, self.provider),
            lambda result: self._graded(question, result))

    def _graded(self, question: dict, result) -> None:
        if isinstance(result, Exception):
            self._ai_error(result)
        else:
            self.panel.reply(f"Score {result['score']}/10. {result['explanation']}\n"
                             f"Model answer: {question['model_answer']}")
        self._ask_next()

    def _ai_error(self, error: Exception) -> None:
        self.panel.reply(f"The AI didn't answer ({type(error).__name__}: {error}). "
                         f"Is the {self.provider} package installed and its API key set? "
                         "See services/reading_ai.py.")

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()                  # camera light off, MediaPipe freed


def main() -> int:
    parser = argparse.ArgumentParser(description="Eye-tracking reading assistant")
    parser.add_argument("--mouse", action="store_true", help="no camera: the mouse plays your eyes")
    parser.add_argument("--provider", choices=list(MODELS), default="claude")
    args, _ = parser.parse_known_args()

    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)   # the ball is the app: quit from its right-click menu
    font = QFont()
    font.setFamilies(TEXT_FONTS)              # the main window's fonts (icon font last, for its glyphs)
    qt_app.setFont(font)
    app = App(mouse=args.mouse, provider=args.provider)
    try:
        return qt_app.exec()
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
