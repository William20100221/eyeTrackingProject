"""
The whole app, wired together.

    python main.py                       (asks which camera, in the console)
    python main.py --mouse               (no camera: your mouse plays the eyes)
    python main.py --provider gemini     (force the AI; normally your key says which, see services/reading_ai.py)

The main window (ui/control_window.py) has three tabs:
    Chat         replies, quiz; "Go Reading" closes the window and locks the ball
    Testing      sample text, gaze check, saved gaze data (active / not usable)
    Calibration  14 line dots, the head step, 16 random dots (core/calibration.py), saves, retrains the gaze model
    Settings     camera, AI key, theme
The camera is on only while something uses the gaze (calibration, the gaze
check, the sample text, Go Reading with the ball's eyes open), then off again
(the ball's halo is grey). It is read on a thread of its own (FrameFeed): nothing waits for a frame.
The floating ball stays on the whole time. Unlocked: click it to open the window.
Locked (after Go Reading): a click opens its eyes and it records what you read in
whatever window you look at; a click closes them and the AI summarises what you
missed, in the chat. Right-click still gives the menu.
After reading, the AI summarises what you skipped and quizzes you on it.
Right-click the ball to quit.

What talks to what:
    camera -> tracker -> gaze model -> smoothed screen point
        -> ball halo (face found?), gaze cursor + dwell-click, reading window
    reading window -> (full text, read paragraphs) -> reading AI -> chat tab
    any window -> screen text (UIA / OCR) + gaze -> ReadingTracker -> (full, read) -> reading AI
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QElapsedTimer, QObject, QPoint, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from eyetracking.core.points_model import TrainedModel, train
from eyetracking.core.quality import check_recording, check_special
from eyetracking.core.models import NO_FRAME, CalibrationRecording
from eyetracking.core.reading import ReadingTracker, count_unit, mark_unread, tokens, unread_word_count, word_count
from eyetracking.core.smoothing import GazeSmoother
from eyetracking.services import reading_ai, screen_text
from eyetracking.services.camera import camera_names
from eyetracking.services.reading_ai import (KEY_ENV, MODELS, PROVIDER_NAMES, grade_answer, key_from_environment,
                                             list_models, make_questions, summarize)
from eyetracking.services.screen_text import lines_on_screen, window_at, window_title
from eyetracking.services.storage import (list_calibrations, load_all_calibrations, load_reading, log_calibration,
                                          migrate_layout, new_reading_path, remove_calibration, save_blink_scores,
                                          save_calibration, save_reading, trash_file)
from eyetracking.services.tracker import EyeTracker, FrameFeed
from eyetracking.testing.try_reading_ai import SAMPLE_FULL
from eyetracking.ui.assistant_overlay import DwellController, FloatingBall, GazeCursor
from eyetracking.core.blink_scores import advice as blink_advice
from eyetracking.ui.blink_window import BlinkWindow, fake_scores_from_mouse
from eyetracking.ui.calibration_window import CalibrationWindow, fake_features_from_mouse
from eyetracking.ui.control_window import (app_icon, TEXT_FONTS, SettingsWindow, busy, full_text_dialog, model_label,
                                          pick_ai, pick_camera, run_busy)
from eyetracking.ui.gaze_window import GazeWindow, fake_point_from_mouse

TICK_MS = 16             # asks for the newest camera frame (read on its own thread) at the screen's pace
SECONDS_PER_WORD = 0.1   # gaze time a paragraph needs per word to count as read
CAPTURE_MS = 1000        # how often the text of the window being read is re-read (scrolling, a new page)

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


STILL_WORKING = "Still working on the last one…"   # a second start while slow work runs


def open_tracker(index: int, closing: threading.Thread | None = None) -> EyeTracker:
    """The camera opened and made ready, on the worker thread (App._busy). `closing`, the camera closed
    last, is waited for first, so the device is free. Then the first frame is read here: the stream
    really starting and MediaPipe's first run would otherwise land on the GUI thread, in the first tick."""
    if closing is not None:
        closing.join()
    tracker = EyeTracker(index)
    tracker.read()
    return tracker


class App(QObject):
    def __init__(self, mouse: bool, provider: str | None = None) -> None:
        super().__init__()
        self.mouse, self.forced_provider = mouse, provider

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
        self.panel.message_sent.connect(self.answer)
        self.panel.chats_deleted.connect(self._chats_deleted)
        self.panel.retry_requested.connect(lambda path: self.retry(Path(path)))
        self.record_path: Path | None = None       # the reading being written (its summary, then its quiz)
        self.control.testing.command.connect(lambda c: {"read": self.read, "check": self.check, "blink": self.blink_test}[c]())
        self.control.testing.data_changed.connect(self._retrain)
        self.control.testing.undo_requested.connect(self.undo)
        self.undo_stack: list[tuple[str, list[Path]]] = []   # (what, the files it made): the newest last
        self.panel.go_reading.connect(self.go_reading)
        self.ball.clicked.connect(self.ball_clicked)
        self.ball.double_clicked.connect(self.toggle_reading)
        self.control.shown.connect(lambda up: setattr(self.ball, "window_open", up))   # issue #12
        self.ball.open_requested.connect(lambda: self._show_window(None))
        self.ball.menu_actions = [("Settings", self.open_settings)]
        # reading from the screen: see go_reading()
        self.reader: ReadingTracker | None = None   # set while the ball's eyes are open
        self.reader_hwnd, self.reader_title = 0, ""
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(CAPTURE_MS)
        self.capture_timer.timeout.connect(self._capture)
        self._capturing = False
        settings = self.control.read_store()
        self.ai_entry: dict | None = None

        self.tracker = None                       # opened by _use_camera(), only while needed (issue #11)
        self.feed: FrameFeed | None = None        # the camera read on its own thread, while it is on
        self._tick_frames = None                  # tick()'s reader of the feed
        self._working = False                     # slow work running (App._busy): a camera start meanwhile is ignored
        self._closing: threading.Thread | None = None   # the last camera, closing on its own thread
        self.camera_users: set[str] = set()       # what needs the gaze right now
        self.camera = settings["camera"]
        migrate_layout()                          # files from older versions, loose in data/, go into their folders
        self.model = self._load_model()
        self.smoothed: tuple[float, float] | None = None
        self.smoother = GazeSmoother()           # the 1-euro filter on the live point (core/smoothing.py)
        self.fullscreen: QWidget | None = None   # calibration / check page, while open
        self.apply_settings({**settings, "camera": self.camera})   # the popup may have just chosen one
        self.reading: ReadingWindow | None = None
        self.full_text = ""
        self.quiz: list[dict] = []                # questions still to ask
        self.record: dict | None = None           # this reading's AI results, saved to data/history as they come

        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)

        # last: the first-run key check answers clicks while it waits, so everything they reach exists by now
        self._choose_ai(provider)
        self.control.ai_changed.connect(self._ai_changed)
        self._set_language(self.control.reading_language()[0])   # the chat's language picker: what OCR reads
        self.control.language_changed.connect(self._set_language)
        self.panel.status.setText(f"{self._model_status()} {self._ai_status()}")
        self.control.show_last()

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
        self.ball.setVisible(self.show_ball and self.fullscreen is None)   # setWindowFlag hides it
        self.summary_words = values["summary_words"]
        self.ball.set_look(values["ball_look"], values["ball_image"], values["ball_ring"] or values["seed"])
        if values["camera"] != self.camera:
            self.camera = values["camera"]
            if self.tracker is not None:          # in use right now: switch at once
                self._stop_feed()
                self.tracker.close()
                self.tracker = None
                if self._open_camera():
                    self.panel.reply(f"Now using the camera: {self.camera}.")

    # ---- the camera: on only while something needs the gaze (issue #11) -----
    # Trade-off: a feature now waits about a second for the camera and MediaPipe
    # when it starts, instead of the camera light being on (and a CPU core busy at
    # 30 frames a second) for the whole time the app runs.
    def _use_camera(self, user: str) -> bool:
        """`user` needs the gaze: open the camera if it is off. False if it can't be opened."""
        if self.mouse:
            return True
        if self._working or busy():               # a click answered while the camera opens, a model trains, a key is checked
            self.panel.status.setText(STILL_WORKING)
            return False
        self.camera_users.add(user)
        if self.tracker is None and not self._open_camera():
            self.camera_users.discard(user)
            return False
        return True

    def _release_camera(self, *users: str) -> None:
        """`users` are done with the gaze; the last one out closes the camera."""
        self.camera_users.difference_update(users)
        if not self.camera_users and self.tracker is not None:
            # camera light off, MediaPipe freed: on its own thread (a driver's release can block for a second);
            # the next open waits for it (open_tracker), so the camera is free again before it's asked for
            old, self.tracker = self.tracker, None
            feed, self.feed, self._tick_frames = self.feed, None, None

            def close() -> None:
                if feed is not None:
                    feed.stop()                   # its read in progress ends first: nothing reads a closed camera
                old.close()
            self._closing = threading.Thread(target=close, daemon=True)
            self._closing.start()
            self._closing.join(0.25)              # usually done by now: the camera is off when this returns
            self.ball.set_tracking_state(None)    # grey halo: camera off
            self.cursor.hide()

    def _frames(self, part: str = "features"):
        """A reader of the camera's frames (FrameFeed.reader), the feed started on first use: each tick
        a window (or tick()) gets the newest frame, or NO_FRAME, instead of waiting for the camera."""
        if self.feed is None:
            self.feed = FrameFeed(self.tracker)
        return self.feed.reader(part)

    def _stop_feed(self) -> None:
        if self.feed is not None:
            self.feed.stop()
            self.feed = self._tick_frames = None

    def _open_camera(self) -> bool:
        """Open the chosen camera and MediaPipe on a worker thread, this one painting meanwhile.
        The chat's status line says so, and the pointer waits, meanwhile."""
        try:
            index = self._camera_index()          # may ask with the popup first
        except RuntimeError as e:                 # no camera, or the popup was closed
            self.panel.reply(f"The camera didn't start ({e}).")
            return False
        # the camera driver can take a minute the first time on some PCs: on a worker thread, the bar running
        try:
            self.tracker = self._busy("Opening the camera…", lambda: open_tracker(index, self._closing))
            return True
        except Exception as e:                    # camera busy, MediaPipe failed
            self.panel.reply(f"The camera didn't start ({e}). Pick another in Settings > Camera.")
            return False

    def _busy(self, text: str, work):
        """work() on a worker thread while this one keeps the window alive (control_window.run_busy): the
        bar runs under the title bar, the chat's status line says `text`, clicks are answered. Windows
        shows "Not Responding" when the GUI thread stops for 5 s, which a cold camera driver or training
        on many calibrations could do on this PC. Returns work()'s result; raises its exception."""
        before = self.panel.status.text()
        self.panel.status.setText(text)
        self.control.place_busy()
        was, self._working = self._working, True     # nested (a click answered meanwhile): the outer job still runs
        try:
            return run_busy(self.control.busy, work)
        finally:
            self._working = was
            if self.panel.status.text() in (text, STILL_WORKING):   # nothing else said meanwhile
                self.panel.status.setText(before)

    def _train_model(self):
        """_load_model on a worker thread, the bar running: training on every calibration takes seconds."""
        return self._busy("Training the gaze model…", self._load_model)

    def _choose_ai(self, forced: str | None) -> None:
        """Which AI answers: the provider chosen in Settings > AI (or the chat's picker). A key in the
        environment / a .env next to main.py fills its provider in, and a running Ollama offers its
        models. Nothing at all on first run: ask once with a popup. --provider overrides."""
        local, _ = list_models("ollama")
        self.control.set_found(key_from_environment(), local)
        if not self.control.usable() and not forced:
            key = pick_ai()                       # first run: paste a key (its brand gets it)
            if key:
                self.control.add_key(key)
        self._use_entry()

    def _use_entry(self) -> None:
        """Answer with the chosen provider and model: its key goes where its package looks, its host and
        model into reading_ai."""
        p = self.ai_entry = self.control.active()
        if p is not None:
            if p["kind"] in KEY_ENV:
                os.environ[KEY_ENV[p["kind"]]] = self.control.secret_of(p)
            reading_ai.MODELS[p["kind"]] = p["model"]
            reading_ai.HOSTS[p["kind"]] = p["host"]
        self.provider = self.forced_provider or (p["kind"] if p else "claude")

    @staticmethod
    def _set_language(tag: str) -> None:
        """The language of the text being read. UI Automation hands over text in any language as it is;
        OCR (PDFs, apps that give no text) must be told which language's letters to look for."""
        screen_text.OCR_LANG = tag

    def _ai_changed(self) -> None:
        self._use_entry()
        self.panel.status.setText(f"{self._model_status()} {self._ai_status()}")

    def _ai_status(self) -> str:
        e = self.ai_entry
        if self.forced_provider and (e is None or e["provider"] != self.forced_provider):
            return f"AI: {PROVIDER_NAMES[self.forced_provider]} (chosen with --provider)."
        if e is None:
            return "No AI yet: add a key in Settings > AI, or install Ollama (ollama.com) for a free local AI."
        return f"AI: {model_label(e)}."

    def open_settings(self) -> None:
        self._show_window(self.control.settings_page)

    # ---- gaze ------------------------------------------------------
    def _load_model(self) -> TrainedModel | None:
        """Train the testers' best model, Points + Random 4x4 (core/points_model.py), on every saved
        calibration that passes its quality check, the newest weighted most."""
        self.samples_used = 0
        if self.mouse:
            return None
        model, self.samples_used = train(load_all_calibrations())
        return model

    def _say(self, text: str) -> None:
        """What happened (a calibration, a gaze check, an undo): in the chat, and in the Testing sidebar's status."""
        self.panel.reply(text)
        self.control.testing.say(text)

    def _did(self, what: str, paths: list[Path]) -> None:
        """A calibration or gaze check just saved these files: Undo can take them back."""
        self.undo_stack.append((what, list(paths)))
        self.control.testing.set_undo(what)

    def undo(self) -> None:
        """Undo the newest calibration or gaze check (as calibration-lab's): its file goes into its trash
        folder, so even an undo can be undone (Data > Calibrations > Removed restores a calibration).
        Never during a full-screen page or while something else runs."""
        if not self.undo_stack or self.fullscreen is not None or self._working:
            return
        what, paths = self.undo_stack.pop()
        gone = [p for p in paths if not p.exists()]          # moved or removed on the Data pages since
        for p in paths:
            if p.exists():
                (remove_calibration if p.suffix == ".json" else trash_file)(p)
        self.control.testing.set_undo(self.undo_stack[-1][0] if self.undo_stack else "")
        self.control.testing.refresh()
        if any(p.suffix == ".json" for p in paths):
            self.model = self._train_model()
            self.panel.status.setText(f"{self._model_status()} {self._ai_status()}")
        left = len(paths) - len(gone)
        said = (f"Undid {what}: {left} file{'s' if left != 1 else ''} moved to the trash"
                + (" (Data > Calibrations > Removed restores it)" if any(p.suffix == ".json" for p in paths) else "")
                + (f"; already gone: {', '.join(p.name for p in gone)}" if gone else "") + ".")
        (self._log_calibration if any(p.suffix == ".json" for p in paths) else self._say)(said)

    def _retrain(self) -> None:
        """Gaze data was removed in the Practice tab: train again on what is left. Asked while other slow
        work runs (a click answered meanwhile), it waits for it: its model must be the last one kept."""
        if self._working or busy():
            QTimer.singleShot(200, self._retrain)
            return
        self.model = self._train_model()
        self.panel.status.setText(f"{self._model_status()} {self._ai_status()}")
        self._say(f"Gaze model retrained on {self.samples_used} samples.")

    def _model_status(self) -> str:
        if self.mouse:
            return "Mouse mode: the mouse plays your eyes."
        if self.model is None:
            return "No calibration yet: open the Calibration tab first."
        return (f"Gaze model trained on {self.samples_used} samples, the newest calibration counting most, "
                f"head turns corrected with {self.model.k_source}.")

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
        except (ValueError, IndexError):          # the saved file holds an older kind of features
            self.model = None
            self.panel.reply("Your saved calibration is from an older version. Type calibrate.")
            return None

    def tick(self) -> None:
        if not self.mouse and self.tracker is None:   # camera off: nothing needs the gaze
            self.clock.restart()
            return
        if self.mouse:
            face, features, t = True, fake_point_from_mouse(), None
        else:
            if self._tick_frames is None:
                self._tick_frames = self._frames("frame")
            got = self._tick_frames()
            if got is NO_FRAME:                       # no camera frame since the last tick: nothing new, not a lost face
                return
            # None during a blink and the frames just after it (eyes reopening);
            # face_found stays True then, so the ball doesn't flash "face lost"
            features, result, t = got
            face = result is not None and result.face_found
        dt = self.clock.restart() / 1000              # since the last frame: the gaze time it stands for
        self.ball.set_tracking_state(face)

        point = self.predict(features)
        if point is not None:
            # raw gaze jitters; without smoothing the dwell ring never fills. The 1-euro filter:
            # steady while the eyes rest, quick when they jump (core/smoothing.py), on the frame's own time
            self.smoothed = self.smoother.feed(point, t)
        elif not face or self.smoothed is None:
            return                                # face lost: nothing to show
        # a blink lands here with point None: HOLD the last smoothed point and
        # keep feeding it, so the cursor and the reading time carry on and the
        # smoothing resumes from where it was (no jump when the eyes reopen) (issue #9)
        gaze = to_screen(self.smoothed)
        if self.reader is None:                  # while Go Reading records: no gaze dot over the page being read
            self.dwell.feed(gaze)
        self.ball.look_at(gaze)                  # the water-eye ball looks where you look
        if self.reader is not None:
            self._feed_reader()
        if self.reading is not None:
            self.reading.feed(gaze, dt)

    # ---- reading from the screen (Go Reading) ----------------------------
    def go_reading(self) -> None:
        """Hide the window and lock the ball: click = eyes open (recording), click = eyes closed (summary)."""
        self.control.hide()
        self.ball.set_locked(True)

    def ball_clicked(self) -> None:
        if not self.ball.locked:
            self._show_window(None)               # where it was left
        elif self.reader is None:
            self.start_reading()
        else:
            self.stop_reading()

    def toggle_reading(self) -> None:
        """Double-click on the ball: record what you read (as Go Reading + a click would), or stop."""
        if self.reader is not None:
            self.stop_reading()
            return
        if not self.ball.locked:
            self.go_reading()
        self.start_reading()

    def _show_window(self, page: QWidget | None) -> None:
        """The window coming back ends a reading (with its summary) and unlocks the ball.
        page None: the page it was left on."""
        if self.reader is not None:
            self.stop_reading()
        self.ball.set_locked(False)
        if page is None:
            self.control.show_last()
        else:
            self.control.show_page(page)

    def start_reading(self) -> None:
        if not self._use_camera("screen"):
            self._show_window(self.panel)         # the reason is in the chat
            return
        self.reader, self.reader_hwnd, self.reader_title = ReadingTracker(), 0, ""
        self.ball.eyes_open = True
        self.cursor.hide()                        # nothing drawn over the page while you read
        self.capture_timer.start()

    def stop_reading(self) -> None:
        """Eyes closed: stop recording; the AI summarises what was missed on the page(s) read."""
        self._release_camera("screen")
        self.capture_timer.stop()
        self.ball.eyes_open = False
        reader, self.reader = self.reader, None
        if reader is None:
            return
        full, read = reader.full_text(), reader.read_text()
        self.ball.set_locked(False)
        self.control.show_page(self.panel)        # the response goes to the chat
        if not word_count(read):
            self.panel.reply(f"I didn't see you read anything on {self.reader_title or 'that window'}.")
            return
        # a copy of the whole text, if the user has one: everything, exactly as written (Skip: the screen's text)
        pasted = full_text_dialog(self.control, self.control.c, self.reader_title)
        full = pasted or full
        unit, marked = count_unit(full), mark_unread(full, read)
        intro = (f"{self.reader_title or 'The text'}: you read {word_count(full) - unread_word_count(marked)} of "
                 f"{word_count(full)} {unit}, {unread_word_count(marked)} unread"
                 + (" (in the copy you pasted)." if pasted else "."))
        self._start_record(full, read, intro)
        self._save_record()                       # saved now: an AI that doesn't answer leaves it to try again
        self.panel.reply(f"{intro} Asking {self._ai_name()} for what you missed...")
        self._ask_ai()

    def _ai_name(self) -> str:
        """The model that answers, as the chat's picker shows it."""
        return model_label(self.ai_entry) if self.ai_entry else self.provider

    def _ask_ai(self) -> None:
        """Ask the AI about the reading in self.record: the summary (and the questions, for the sample text)."""
        r = self.record
        full, read = r["full_text"], r["read_text"]
        self.full_text = full                     # the quiz's grading reads it
        if r.get("quiz"):
            self.background.run(lambda: (summarize(full, read, "unread", self.provider, self.summary_words),
                                         make_questions(full, read, "unread", self.provider)), self._ai_ready)
        else:
            self.background.run(lambda: summarize(full, read, "unread", self.provider, self.summary_words),
                                self._summary_ready)

    def _summary_ready(self, result) -> None:
        if isinstance(result, Exception):
            self._ai_error(result, reading=True)
            return
        self.panel.reply(result)
        self.record["summary"] = result
        self.record.pop("error", None)
        self._save_record()
        if self.record_path is not None:
            self.panel.add_ask_again(str(self.record_path))

    def retry(self, path: Path) -> None:
        """Try again / Ask again on a reading (in the live chat, or a past one in Chats): the AI is asked
        about the same text again, and the reading's file gets the new answer."""
        r = load_reading(path)
        if not r or not r.get("full_text"):
            self.panel.show_live()
            self.panel.reply("This reading was saved before the app kept its text, so it can't be asked again. "
                             "Read it again with Go Reading.")
            return
        self.quiz = []                            # a quiz still waiting belongs to the reading before
        self.panel.set_answering(False)
        self.record, self.record_path = r, path
        self.panel.show_live()
        self.panel.reply(f"Asking {self._ai_name()} again about {r.get('title') or 'that reading'}...")
        self._ask_ai()

    # ---- history: the AI's results, one file per reading --------------------
    def _start_record(self, full_text: str, read_text: str, intro: str, quiz: bool = False) -> None:
        """A reading's file: saved as soon as something was read, with its text, so an AI that didn't answer
        can be asked again later (Chats > the reading > Try again)."""
        words = [m for m in tokens(full_text) if m.group().strip()]
        n = 16 if count_unit(full_text) == "characters" else 6          # the title: the first words / characters
        title = " ".join(full_text[:words[min(n, len(words)) - 1].end()].split()) if words else "A reading"
        self.record = {"title": title, "when": datetime.now().isoformat(timespec="seconds"), "intro": intro,
                       "full_text": full_text, "read_text": read_text, "quiz": quiz, "summary": "", "questions": []}
        self.record_path = new_reading_path()

    def _save_record(self) -> None:
        if self.record_path is None:
            return                                # deleted in the Chats list
        save_reading(self.record_path, self.record)
        self.panel.refresh()                      # the History list shows it at once

    def _chats_deleted(self, paths: list) -> None:
        """Readings deleted in the Chats list: the one still being written (its quiz running) stays deleted."""
        if self.record_path in paths:
            self.record_path = None

    def _capture(self) -> None:
        """Once a second while recording: re-read the window under the gaze, off the GUI thread."""
        if self._capturing or self.smoothed is None:
            return
        hwnd = window_at(*self._gaze_px())
        if hwnd in self._own_windows():           # looking at the ball or the test window: same page as before
            hwnd = self.reader_hwnd
        if not hwnd:                              # the desktop
            return
        if hwnd != self.reader_hwnd:
            self.reader_hwnd, self.reader_title = hwnd, window_title(hwnd)
        self._capturing = True
        self.background.run(lambda: lines_on_screen(hwnd), self._lines_ready)

    def _lines_ready(self, result) -> None:
        self._capturing = False
        if self.reader is not None and result and not isinstance(result, Exception):
            self.reader.show(result)

    def _feed_reader(self) -> None:
        px, py = self._gaze_px()
        self.reader.feed(px, py, time.monotonic())

    def _gaze_px(self) -> tuple[float, float]:
        """The smoothed gaze in physical screen pixels (the frame of the text boxes), from Qt's scaled ones."""
        gaze = to_screen(self.smoothed)
        ratio = QApplication.primaryScreen().devicePixelRatio()
        return gaze.x() * ratio, gaze.y() * ratio

    def _own_windows(self) -> set[int]:
        return {int(w.winId()) for w in (self.ball, self.cursor, self.control)}

    # ---- full-screen windows that read the camera themselves -------
    def _open_fullscreen(self, window: QWidget) -> None:
        """Pause the overlay's tick() (the window reads the frames now) and hide the
        always-on-top windows so they don't cover the dots."""
        self.timer.stop()
        self.cursor.hide()
        self.ball.hide()                          # the dots must not be covered
        self.fullscreen = window                  # keep a reference, or Python deletes it
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._fullscreen_closed)
        self.control.show_fullscreen_page(window)   # the main window, full screen, no buttons

    def _fullscreen_closed(self) -> None:
        self.fullscreen = None
        self._release_camera("calibration", "check", "blink")
        self.ball.setVisible(self.show_ball)
        self.clock.restart()
        self.timer.start(TICK_MS)

    # ---- commands (tabs + chat sidebar buttons) ------------------------
    def calibrate(self) -> None:
        if not self._use_camera("calibration"):
            return
        get = fake_features_from_mouse if self.mouse else self._frames()
        window = CalibrationWindow(get_features=get)
        window.finished.connect(lambda samples: self._calibrated(window, samples))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, rec: CalibrationRecording) -> None:
        """As calibration-lab's: impossible data is rejected and not saved ("Try again"); what the checks
        cleaned out is said; the checks and the save run on a worker. One sitting holds all three steps,
        so both of the lab's checks run: the grid's (check_recording) and steps 1-2's (check_special)."""
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self._say("Mouse mode: calibration not saved.")
            return
        if not rec.dots:
            self._log_calibration("No samples: was your face in view? Try calibrate again.")
            return
        made: list[Path] = []

        def check_and_save() -> str:              # on a worker thread: the checks and big files take seconds here
            grid = check_recording(rec)
            if not grid.ok:                       # the grid trains the model: without it, nothing to save
                return f"Calibration rejected ({grid.summary()}). Try calibrate again."
            steps = check_special(rec, "points")  # the line dots and the head step: the user's own head matrix
            path = save_calibration(rec)          # saved RAW: the checks run again when it trains
            made.append(path)
            said = f"Calibrated with {len(grid.kept.dots)} good dot samples"
            if steps.ok:
                said += (f", {len(steps.kept.sweep)} good line-dot and {len(steps.kept.head)} head-step frames, "
                         f"saved as {path.name}.")
            else:                                 # the grid still trains; head turns: an older sitting's step or the
                said += (f", saved as {path.name}. The line dots / head step can't be used ({steps.summary()}): "
                         "calibrate again to measure your own head turns.")   # built-in matrix (said below)
            cleaned = [r.summary() for r in (grid, steps) if r.ok and r.reasons]
            return said + (f" Cleaned: {'; '.join(cleaned)}." if cleaned else "")

        said = self._busy("Checking and saving the calibration…", check_and_save)
        if made:
            self._did("the calibration", made)
        self.model = self._train_model()          # the new file + all older ones
        if self.model is not None:
            said += f" Gaze model retrained on {self.samples_used} samples, head turns corrected with {self.model.k_source}."
        if made and len(list_calibrations()) == 1:
            # measured on 9 testers (26 Oct): a second sitting in training, 17.3% -> 16.0% of the screen, better for 7 of 9
            said += " A second calibration another day makes the model more accurate: it learns from both sittings."
        self._log_calibration(said)

    def _log_calibration(self, text: str) -> None:
        """What a calibration gave (or why it was rejected, or that it was undone): said, and kept in
        Testing > Data > Calibration log."""
        log_calibration(text)
        self.control.testing.refresh()
        self._say(text)

    def check(self) -> None:
        if not self.mouse and self.model is None:
            self.panel.reply("Calibrate first: open the Calibration tab.")
            return
        if not self._use_camera("check"):
            return
        frames = None if self.mouse else self._frames("stamped")

        def get():
            got = (fake_point_from_mouse(), None) if self.mouse else frames()
            if got is NO_FRAME:                   # no camera frame since the last tick: nothing new
                return NO_FRAME
            features, t = got
            point = self.predict(features)
            if not self.mouse:
                point = self.smoother.feed(point, t)   # the same filter as tick(): the check shows what the overlay would
            if point is not None:
                self.smoothed = point     # keep it: the overlay carries on from here after quitting check (issue #10)
            return point
        window = GazeWindow(get_point=get)
        window.logged.connect(self._gaze_logged)
        self.control.testing.say("Gaze check: look at the lit ring and press Enter; Esc saves and leaves.")
        self._open_fullscreen(window)

    def _gaze_logged(self, path: str) -> None:
        self._did("the gaze check", [Path(path)])
        self.control.testing.refresh()            # Data > Gaze checks lists it
        self._say(f"Gaze samples saved: {Path(path).name}. See Testing > Data > Gaze checks.")

    def blink_test(self) -> None:
        """Testing > Blink test: record blink scores in six eye states, saved under data/blink_scores."""
        if not self._use_camera("blink"):
            return
        if self.mouse:
            get = fake_scores_from_mouse
        else:
            results = self._frames("result")

            def get():
                r = results()
                if r is NO_FRAME:
                    return NO_FRAME
                return (r.blink_score_left, r.blink_score_right) if r is not None and r.face_found else None
        window = BlinkWindow(get_scores=get)
        window.finished.connect(lambda rows: self._blink_done(window, rows))
        self._open_fullscreen(window)

    def _blink_done(self, window: BlinkWindow, rows: list) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        path = save_blink_scores(rows)
        self.control.testing.refresh()
        self._say(f"Blink scores saved to {path.name}. {blink_advice(rows)}")

    def read(self) -> None:
        if self.model is None and not self.mouse:
            self.panel.reply("Calibrate first: open the Calibration tab.")
            return
        if not self._use_camera("reading"):
            return
        if self.reading is None:
            self.reading = ReadingWindow(SAMPLE_FULL)
            self.reading.finished.connect(self._read_finished)
            self.reading.destroyed.connect(self._reading_closed)
        self.reading.showFullScreen()

    def _reading_closed(self) -> None:
        self.reading = None
        self._release_camera("reading")

    # ---- the reading AI ------------------------------------------------
    def _read_finished(self, full_text: str, read_text: str) -> None:
        unread = unread_word_count(mark_unread(full_text, read_text))
        intro = f"You skipped {unread} of {word_count(full_text)} {count_unit(full_text)}."
        self._start_record(full_text, read_text, intro, quiz=True)   # the sample text: its own text, no copy needed
        self._save_record()
        self.control.show_page(self.panel)        # the summary and quiz show in the chat
        self.panel.reply(f"{intro} Asking {self._ai_name()}...")
        self._ask_ai()

    def _ai_ready(self, result) -> None:
        if isinstance(result, Exception):
            self._ai_error(result, reading=True)
            return
        summary, self.quiz = result
        self.record["summary"], self.record["questions"] = summary, list(self.quiz)   # the same dicts: answers land in them
        self.record.pop("error", None)
        self._save_record()
        self.panel.reply(summary)
        if self.record_path is not None:
            self.panel.add_ask_again(str(self.record_path))
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
            question["answer"] = "skip"
            self._save_record()
            self.panel.reply(f"Model answer: {question['model_answer']}")
            self._ask_next()
            return
        self.panel.reply("Checking...")
        self.background.run(
            lambda: grade_answer(self.full_text, question, text, self.provider),
            lambda result: self._graded(question, text, result))

    def _graded(self, question: dict, answer: str, result) -> None:
        if isinstance(result, Exception):
            self._ai_error(result)
        else:
            question.update(answer=answer, score=result["score"], explanation=result["explanation"])
            self._save_record()
            self.panel.reply(f"Score {result['score']}/10. {result['explanation']}\n"
                             f"Model answer: {question['model_answer']}")
        self._ask_next()

    def _ai_error(self, error: Exception, reading: bool = False) -> None:
        """The AI didn't answer: a "Something went wrong" card in the chat. For a reading's summary the error
        is kept in its file and the card has Try again (now, or later from Chats); a quiz grade just moves on."""
        text = f"{type(error).__name__}: {error}"
        if reading:
            self.record["error"] = text
            self._save_record()
        self.panel.add_error(text, str(self.record_path) if reading and self.record_path is not None else None)

    def close(self) -> None:
        self._stop_feed()
        if self.tracker is not None:
            self.tracker.close()                  # camera light off, MediaPipe freed


def main() -> int:
    parser = argparse.ArgumentParser(description="Eye-tracking reading assistant")
    parser.add_argument("--mouse", action="store_true", help="no camera: the mouse plays your eyes")
    parser.add_argument("--provider", choices=list(MODELS), default=None,
                        help="force the AI; normally the app works it out from your key (or uses Ollama)")
    parser.add_argument("--self-test", action="store_true",
                        help="check the model, packages and data folder, then exit (see eyetracking/self_test.py)")
    args, _ = parser.parse_known_args()
    if args.self_test:
        from eyetracking import self_test
        return self_test.run()

    if sys.platform == "win32":               # the taskbar groups a script under python.exe (and its icon)
        import ctypes                         # until the process says it is an app of its own
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StudyAssistant.EyeTracking")
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)   # the ball is the app: quit from its right-click menu
    font = QFont()
    font.setFamilies(TEXT_FONTS)              # the main window's fonts (icon font last, for its glyphs)
    qt_app.setFont(font)
    qt_app.setWindowIcon(app_icon())          # the logo on the taskbar and every window
    app = App(mouse=args.mouse, provider=args.provider)
    try:
        return qt_app.exec()
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
