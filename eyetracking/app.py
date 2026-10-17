"""
The whole app, wired together.

    python main.py                       (asks which camera, in the console)
    python main.py --mouse               (no camera: your mouse plays the eyes)
    python main.py --provider gemini     (force the AI; normally your key says which, see services/reading_ai.py)

The main window (ui/control_window.py) has three tabs:
    Chat         replies, quiz; "Go Reading" closes the window and locks the ball
    Testing      sample text, gaze check, saved gaze data (active / not usable)
    Calibration  line dots + head step + 25 random dots, saves, retrains the gaze model
    Settings     camera, AI key, theme
The camera is on only while something uses the gaze (calibration, the gaze
check, the sample text, Go Reading with the ball's eyes open), then off again
(the ball's halo is grey). The floating ball stays on the whole time. Unlocked: click it to open the window.
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
import html
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
from eyetracking.core.calibration import GazeProfile, training_set
from eyetracking.core.line_model import LineModel
from eyetracking.core.models import CalibrationRecording
from eyetracking.core.quality import check_recording
from eyetracking.core.reading import ReadingTracker, mark_unread, unread_word_count
from eyetracking.services import reading_ai
from eyetracking.services.camera import camera_names
from eyetracking.services.reading_ai import (KEY_ENV, MODELS, grade_answer, key_from_environment, make_questions,
                                             ollama_model, provider_for_key, summarize)
from eyetracking.services.screen_text import lines_on_screen, window_at, window_title
from eyetracking.services.storage import (load_all_calibrations, migrate_layout, new_reading_path, save_blink_scores,
                                          save_calibration, save_reading)
from eyetracking.services.tracker import EyeTracker
from eyetracking.testing.try_reading_ai import SAMPLE_FULL
from eyetracking.ui.assistant_overlay import DwellController, FloatingBall, GazeCursor
from eyetracking.core.blink_scores import advice as blink_advice
from eyetracking.ui.blink_window import BlinkWindow, fake_scores_from_mouse
from eyetracking.ui.calibration_window import CalibrationWindow, fake_features_from_mouse
from eyetracking.ui.control_window import TEXT_FONTS, SettingsWindow, pick_ai, pick_camera
from eyetracking.ui.gaze_window import GazeWindow, fake_point_from_mouse, screen_geometry

TICK_MS = 33             # ~30 frames a second; the camera read limits it anyway
SMOOTH = 0.3             # share of each new gaze point taken: lower = steadier but laggier
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


class ReadingDebug(QLabel):
    """Rough test window, top-left of the screen: the word and line the gaze is
    on, and the running count, while the ball's eyes are open."""

    def __init__(self) -> None:
        super().__init__("", None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet("background:#1F2328; color:white; font-size:16px; padding:8px 12px;")
        self.setFixedWidth(640)
        self.setWordWrap(True)
        self.move(QApplication.primaryScreen().availableGeometry().topLeft() + QPoint(8, 8))
        self._shown = None

    def show_word(self, hit: tuple[str, str] | None, read: int, total: int, title: str) -> None:
        if (hit, read, total) == self._shown:        # 30 calls a second: repaint only on a change
            return
        self._shown = (hit, read, total)
        word, line = hit or ("-", "")
        self.setText(f"<b>{html.escape(word)}</b> &nbsp; {read} / {total} words read &nbsp; "
                     f"<span style='color:#9AA4B2'>{html.escape(title)}</span><br>"
                     f"<span style='color:#C9D1D9'>{html.escape(line)}</span>")


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
        self.control.testing.command.connect(lambda c: {"read": self.read, "check": self.check, "blink": self.blink_test}[c]())
        self.control.testing.data_changed.connect(self._retrain)
        self.panel.go_reading.connect(self.go_reading)
        self.ball.clicked.connect(self.ball_clicked)
        self.ball.double_clicked.connect(self.toggle_reading)
        self.control.shown.connect(lambda up: setattr(self.ball, "window_open", up))   # issue #12
        self.ball.open_requested.connect(lambda: self._show_window(None))
        self.ball.menu_actions = [("Settings", self.open_settings)]
        # reading from the screen: see go_reading()
        self.reader: ReadingTracker | None = None   # set while the ball's eyes are open
        self.reader_hwnd, self.reader_title = 0, ""
        self.debug = ReadingDebug()                  # rough test window: the word under the gaze
        self.capture_timer = QTimer(self)
        self.capture_timer.setInterval(CAPTURE_MS)
        self.capture_timer.timeout.connect(self._capture)
        self._capturing = False
        settings = self.control.read_store()
        self.provider = self._choose_ai(provider, settings["api_key"])

        self.tracker = None                       # opened by _use_camera(), only while needed (issue #11)
        self.camera_users: set[str] = set()       # what needs the gaze right now
        self.camera = settings["camera"]
        migrate_layout()                          # files from older versions, loose in data/, go into their folders
        self.model = self._load_model()
        self.smoothed: tuple[float, float] | None = None
        self.fullscreen: QWidget | None = None   # calibration / check page, while open
        self.apply_settings({**settings, "camera": self.camera})   # the popup may have just chosen one
        self.reading: ReadingWindow | None = None
        self.full_text = ""
        self.quiz: list[dict] = []                # questions still to ask
        self.record: dict | None = None           # this reading's AI results, saved to data/history as they come

        # ponytail: camera + MediaPipe on the GUI thread (~30-50 ms a frame); move to a QThread if the UI stutters
        self.clock = QElapsedTimer()
        self.clock.start()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(TICK_MS)

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
        if values["api_key"] and (provider := self._use_key(values["api_key"])):
            self.provider = self.forced_provider or provider   # a new key switches the AI at once
        self.summary_words = values["summary_words"]
        if values["camera"] != self.camera:
            self.camera = values["camera"]
            if self.tracker is not None:          # in use right now: switch at once
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
        self.camera_users.add(user)
        if self.tracker is None and not self._open_camera():
            self.camera_users.discard(user)
            return False
        return True

    def _release_camera(self, *users: str) -> None:
        """`users` are done with the gaze; the last one out closes the camera."""
        self.camera_users.difference_update(users)
        if not self.camera_users and self.tracker is not None:
            self.tracker.close()                  # camera light off, MediaPipe freed
            self.tracker = None
            self.ball.set_tracking_state(None)    # grey halo: camera off
            self.cursor.hide()

    def _open_camera(self) -> bool:
        """Open the chosen camera and MediaPipe (about a second, on this thread).
        The chat's status line says so, and the pointer waits, meanwhile."""
        try:
            index = self._camera_index()          # may ask with the popup first
        except RuntimeError as e:                 # no camera, or the popup was closed
            self.panel.reply(f"The camera didn't start ({e}).")
            return False
        self.panel.status.setText("Opening the camera…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()              # paint the status before the wait
        # ponytail: opens on the GUI thread (~1 s freeze); move to a QThread if that feels slow
        try:
            self.tracker = EyeTracker(index)
            return True
        except Exception as e:                    # camera busy, MediaPipe failed
            self.panel.reply(f"The camera didn't start ({e}). Pick another in Settings > Camera.")
            return False
        finally:
            QApplication.restoreOverrideCursor()
            self.panel.status.setText(f"{self._model_status()} {self._ai_status()}")

    def _choose_ai(self, forced: str | None, saved_key: str) -> str:
        """Which AI answers. A key wins (saved in Settings, or in the environment / a
        .env next to main.py); its prefix says the provider. No key: Ollama if it runs
        on this PC, else ask once with a popup. --provider overrides the choice."""
        found = key_from_environment()
        key = saved_key or (found[1] if found else "")
        if not key and not forced:
            local = ollama_model()
            if local:
                reading_ai.MODELS["ollama"] = local
                return "ollama"
            key = pick_ai() or ""                 # first run: paste a key (remembered)
            if key:
                self.control.store_key(key)   # encrypted
                self.control.key.setText(key)
        provider = self._use_key(key) if key else None
        return forced or provider or "claude"

    def _use_key(self, key: str) -> str | None:
        """Put the key where its provider's SDK looks; returns that provider (None: not a known key)."""
        provider = provider_for_key(key)
        if provider:
            os.environ[KEY_ENV[provider]] = key
        return provider

    def _ai_status(self) -> str:
        if self.provider == "ollama":
            return f"AI: Ollama on this PC ({reading_ai.MODELS['ollama']})."
        if os.environ.get(KEY_ENV.get(self.provider, "")):
            return f"AI: {self.provider}."
        return "No AI key yet: paste one in Settings, or install Ollama (ollama.com) for a free local AI."

    def open_settings(self) -> None:
        self._show_window(self.control.settings_page)

    # ---- gaze ------------------------------------------------------
    def _load_model(self) -> LineModel | None:
        """Train the line model (core/line_model.py) on every saved
        calibration, the newest weighted most. It works on RAW features."""
        self.samples_used = 0
        if self.mouse:
            return None
        samples, weights, _ = training_set(load_all_calibrations(), scaled=False)
        self.samples_used = len(samples)
        return LineModel(screen_geometry()).fit(samples, weights) if samples else None

    def _retrain(self) -> None:
        """Gaze data was removed in the Practice tab: train again on what is left."""
        self.model = self._load_model()
        self.panel.status.setText(f"{self._model_status()} {self._ai_status()}")
        self.panel.reply(f"Gaze model retrained on {self.samples_used} samples.")

    def _model_status(self) -> str:
        if self.mouse:
            return "Mouse mode: the mouse plays your eyes."
        if self.model is None:
            return "No calibration yet: open the Calibration tab first."
        return f"Gaze model trained on {self.samples_used} samples, the newest calibration counting most."

    def predict(self, features) -> tuple[float, float] | None:
        """RAW eye features -> (x, y) screen fractions, or None."""
        if features is None:
            return None
        if self.mouse:
            return features                       # the mouse position already is the answer
        if self.model is None:
            return None
        try:
            return self.model.predict(features)   # raw features -> 3D line -> screen
        except ValueError:                        # the saved file holds an older kind of features
            self.model = None
            self.panel.reply("Your saved calibration is from an older version. Type calibrate.")
            return None

    def tick(self) -> None:
        dt = self.clock.restart() / 1000
        if not self.mouse and self.tracker is None:   # camera off: nothing needs the gaze
            return
        if self.mouse:
            face, features = True, fake_point_from_mouse()
        else:
            # None during a blink and the frames just after it (eyes reopening);
            # face_found stays True then, so the ball doesn't flash "face lost"
            features = self.tracker.read_features()
            face = self.tracker.face_found
        self.ball.set_tracking_state(face)

        point = self.predict(features)
        if point is not None:
            # raw gaze jitters; without smoothing the dwell ring never fills
            # ponytail: plain exponential average; swap in a real filter (e.g. One Euro) if lag or jitter bothers
            if self.smoothed is None:
                self.smoothed = point
            else:
                self.smoothed = tuple(s + SMOOTH * (p - s) for s, p in zip(self.smoothed, point))
        elif not face or self.smoothed is None:
            return                                # face lost: nothing to show
        # a blink lands here with point None: HOLD the last smoothed point and
        # keep feeding it, so the cursor and the reading time carry on and the
        # smoothing resumes from where it was (no jump when the eyes reopen)
        gaze = to_screen(self.smoothed)
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
        self.debug.show_word(("-", "look at the text you want to read"), 0, 0, "")
        self.debug.show()
        self.capture_timer.start()

    def stop_reading(self) -> None:
        """Eyes closed: stop recording; the AI summarises what was missed on the page(s) read."""
        self._release_camera("screen")
        self.capture_timer.stop()
        self.ball.eyes_open = False
        self.debug.hide()
        reader, self.reader = self.reader, None
        if reader is None:
            return
        full, read = reader.full_text(), reader.read_text()
        self._start_record(full)
        self.ball.set_locked(False)
        self.control.show_page(self.panel)        # the response goes to the chat
        if not read.split():
            self.panel.reply(f"I didn't see you read anything on {self.reader_title or 'that window'}.")
            return
        unread = unread_word_count(mark_unread(full, read))
        self.panel.reply(f"{self.reader_title}: you read {len(read.split())} of {len(full.split())} words, "
                         f"{unread} unread. Asking {self.provider} for what you missed...")
        self.background.run(lambda: summarize(full, read, "unread", self.provider, self.summary_words),
                            self._summary_ready)

    def _summary_ready(self, result) -> None:
        if isinstance(result, Exception):
            self._ai_error(result)
            return
        self.panel.reply(result)
        self.record["summary"] = result
        self._save_record()

    # ---- history: the AI's results, one file per reading --------------------
    def _start_record(self, full_text: str) -> None:
        self.record = {"title": " ".join(full_text.split()[:6]), "when": datetime.now().isoformat(timespec="seconds"),
                       "summary": "", "questions": []}
        self.record_path = new_reading_path()

    def _save_record(self) -> None:
        if not (self.record["summary"] or self.record["questions"]):
            return                                # nothing worth a file yet
        save_reading(self.record_path, self.record)
        self.panel.refresh()                      # the History list shows it at once

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
        read, total = self.reader.progress()
        self.debug.show_word(self.reader.text_at(px, py), read, total, self.reader_title)

    def _gaze_px(self) -> tuple[float, float]:
        """The smoothed gaze in physical screen pixels (the frame of the text boxes), from Qt's scaled ones."""
        gaze = to_screen(self.smoothed)
        ratio = QApplication.primaryScreen().devicePixelRatio()
        return gaze.x() * ratio, gaze.y() * ratio

    def _own_windows(self) -> set[int]:
        return {int(w.winId()) for w in (self.ball, self.cursor, self.debug, self.control)}

    # ---- full-screen windows that read the camera themselves -------
    def _open_fullscreen(self, window: QWidget) -> None:
        """Pause our camera reads (one reader at a time) and hide the
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
        get = fake_features_from_mouse if self.mouse else self.tracker.read_features
        window = CalibrationWindow(get_features=get)
        window.finished.connect(lambda rec: self._calibrated(window, rec))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, rec: CalibrationRecording) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self.panel.reply("Mouse mode: calibration not saved.")
            return
        if not rec.dots:
            self.panel.reply("No samples: was your face in view? Try calibrate again.")
            return
        report = check_recording(rec)             # impossible data -> reject, don't save
        if not report.ok:
            self.panel.reply(f"Calibration rejected: {report.summary()}. Try calibrate again.")
            return
        try:
            GazeProfile.from_recording(report.kept)   # unusable sweep/head step -> don't save it
        except ValueError as e:
            self.panel.reply(f"Calibration didn't work ({e}). Try calibrate again.")
            return
        path = save_calibration(rec)              # saved RAW: the checks rerun on loading
        self.model = self._load_model()           # the new file + all older ones
        note = f" Cleaned: {report.summary()}." if report.reasons else ""
        self.panel.reply(f"Calibrated with {len(report.kept.dots)} good dot samples, saved to {path.name}. "
                         f"Gaze model retrained on {self.samples_used} samples.{note}")

    def check(self) -> None:
        if not self.mouse and self.model is None:
            self.panel.reply("Calibrate first: open the Calibration tab.")
            return

        def get():
            point = self.predict(fake_point_from_mouse() if self.mouse else self.tracker.read_features())
            if point is not None:
                self.smoothed = point     # keep it: the overlay carries on from here after quitting check
            return point
        if not self._use_camera("check"):
            return
        window = GazeWindow(get_point=get)
        window.logged.connect(self._gaze_logged)
        self._open_fullscreen(window)

    def _gaze_logged(self, path: str) -> None:
        self.control.testing.refresh()            # Data > Gaze checks lists it
        self.panel.reply(f"Gaze samples saved: {Path(path).name}. See Testing > Data > Gaze checks.")

    def blink_test(self) -> None:
        """Testing > Blink test: record blink scores in six eye states, saved under data/blink_scores."""
        if not self._use_camera("blink"):
            return
        if self.mouse:
            get = fake_scores_from_mouse
        else:
            def get():
                r = self.tracker.read()
                return (r.blink_score_left, r.blink_score_right) if r is not None and r.face_found else None
        window = BlinkWindow(get_scores=get)
        window.finished.connect(lambda rows: self._blink_done(window, rows))
        self._open_fullscreen(window)

    def _blink_done(self, window: BlinkWindow, rows: list) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        path = save_blink_scores(rows)
        self.control.testing.refresh()
        self.panel.reply(f"Blink scores saved to {path.name}. {blink_advice(rows)}")

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
        self.full_text = full_text
        self._start_record(full_text)
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
        self.record["summary"], self.record["questions"] = summary, list(self.quiz)   # the same dicts: answers land in them
        self._save_record()
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
    parser.add_argument("--provider", choices=list(MODELS), default=None,
                        help="force the AI; normally the app works it out from your key (or uses Ollama)")
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
