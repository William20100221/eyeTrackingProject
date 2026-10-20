"""
The tester version of the calibration lab: a tester types their name, does each calibration type twice
(two sets), then two gaze checks of every model at once: the models trained on one set, then on both.
Once both checks are done, everything is emailed (services/mailer.py); which model is the most accurate
is worked out afterwards, in the lab.

    python main.py                 (a popup asks for the camera on first start)
    python main.py --mouse         (no camera: the mouse plays the eyes; nothing is emailed)

What talks to what:
    camera -> tracker (services/tracker.py) -> 12 features per frame (core/features.py)
      -> CalibrationWindow(type) -> recording -> quality check -> data/<name>/<type>/
      -> GazeWindow: every model the tester's calibrations make (core/methods.py), trained now on the
         newest 1 (then 2) sets, all predicting from the same camera frames on 36 rings (6x6); Enter
         records where each says you look against the lit ring -> one csv per model
      -> after the second check: every file of the tester, zipped and emailed
The camera is on only while one of those full-screen windows runs, read on a thread of its own
(FrameFeed). Slow work (the camera opening, training, checking and saving a calibration) runs on a
worker thread while a thin bar runs under the title bar (App._busy); each email has a thread of its own.
"""
from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.methods import METHOD_BY_ID, MODELS, Method, Model, TrainedModel, partners, train
from eyetracking.core.models import NO_FRAME, CalibrationRecording
from eyetracking.core.quality import check_recording, check_special
from eyetracking.services import mailer, storage
from eyetracking.services.camera import camera_names
from eyetracking.services.storage import load_calibrations, save_calibration, use_person
from eyetracking.services.tracker import EyeTracker, FrameFeed
from eyetracking.ui.calibration_window import CalibrationWindow, fake_features_from_mouse
from eyetracking.ui.control_window import SETS, TEXT_FONTS, MainWindow, app_icon, busy, pick_camera, run_busy
from eyetracking.ui.gaze_window import GazeWindow, fake_point_from_mouse, screen_geometry


STILL_WORKING = "Still working on the last one…"   # a second start while slow work runs


def open_tracker(index: int, closing: threading.Thread | None = None) -> EyeTracker:
    """The camera opened and made ready, on the worker thread (App._busy). `closing`, the camera closed
    last, is waited for first, so the device is free. Then the first frame is read here: the stream
    really starting and MediaPipe's first run would otherwise land on the GUI thread, in the first tick
    of the calibration or the check."""
    if closing is not None:
        closing.join()
    tracker = EyeTracker(index)
    tracker.read()
    return tracker


class App(QObject):
    mailed = Signal(str)                           # what an email did, said from its own thread

    def __init__(self, mouse: bool) -> None:
        super().__init__()
        self.mouse = mouse
        self.control = MainWindow()
        self.testing = self.control.testing
        self.testing.command.connect(lambda c: self._command({"calibrate": self.calibrate, "check": self.check}[c]))
        self.testing.person_changed.connect(self._use_person)
        self.testing.camera_changed.connect(self._set_camera)
        self.mailed.connect(self.testing.say)

        self.tracker: EyeTracker | None = None     # opened by _use_camera(), only while a window needs the gaze
        self._working = False                      # slow work running (App._busy): a second start meanwhile is ignored
        self._closing: threading.Thread | None = None   # the last camera, closing on its own thread
        self.feed: FrameFeed | None = None         # the camera read on its own thread, while a window runs
        self.camera_users: set[str] = set()
        self.fullscreen: QWidget | None = None     # the calibration / check page while it runs
        self.models: dict[tuple, tuple[TrainedModel | None, int]] = {}   # trained models by (model, folder, files)
        self.senders: list[threading.Thread] = []  # emails on their way: close() waits for them
        self.camera = ""
        self._set_camera(self.control.store.value("camera", ""))
        self.person: str | None = None             # the tester's name: set by _use_person
        name = self.control.store.value("person", "")
        self.testing.name_edit.setText(name)       # (its textChanged calls _use_person)
        self._use_person(name)                     # ...but not when it is ""
        self._mail()                               # anything not sent last time
        self.control.show()
        self.testing.name_edit.setFocus()          # the first thing a tester does

    @property
    def method(self) -> Method:
        return METHOD_BY_ID[self.testing.method_id]

    # ---- slow work: off the GUI thread, with the progress bar ------------------
    def _command(self, run) -> None:
        """A start button. Ignored while slow work runs (its window would open on top of it)."""
        if self._working or busy():
            self.testing.say(STILL_WORKING)
            return
        run()

    def _busy(self, text: str, work, status=None):
        """work() on a worker thread while this one keeps the window alive: the progress bar runs, the
        status line says `text` (or `status()`, asked again every 50 ms), the pointer waits, and clicks
        are answered. Windows shows "Not Responding" when the GUI thread stops for 5 s, which a cold
        camera driver, training a dozen models or saving a big calibration could do.
        Returns work()'s result; raises its exception."""
        before, shown = self.testing.status.text(), [text]
        self.testing.status.setText(text)

        def tick() -> None:
            if status is not None:
                shown[0] = status()
                self.testing.status.setText(shown[0])
        self.control.place_busy()
        was, self._working = self._working, True     # nested (a click answered meanwhile): the outer job still runs
        try:
            return run_busy(self.control.busy, work, tick)
        finally:
            self._working = was
            if self.testing.status.text() in (shown[0], STILL_WORKING):   # nothing else said meanwhile
                self.testing.status.setText(before)

    # ---- the name, the camera ------------------------------------------
    def _use_person(self, text: str) -> None:
        """The name typed on the Instructions page: the files go into data/<name>/."""
        name = storage.person_name(text)
        if name == self.person:
            return
        self.person = name
        self.control.store.setValue("person", name)
        use_person(name)
        self.testing.refresh()                    # what this person still has to do

    def _named(self) -> bool:
        """Nothing starts without a name: the results would be no one's."""
        if not self.person:
            self.testing.say("Type your name first.")
            self.testing.show_page(0)
            self.testing.name_edit.setFocus()
        return bool(self.person)

    def _set_camera(self, name: str) -> None:
        self.camera = name
        self.control.store.setValue("camera", name)
        self.testing.set_camera(name)

    def _camera_index(self) -> int:
        """The saved camera's index; asks with the popup if none is saved or it's gone."""
        names = camera_names()
        if not names:
            raise RuntimeError("no camera found")
        if self.camera not in names:
            chosen = pick_camera(names, self.camera)
            if chosen is None:
                raise RuntimeError("no camera chosen")
            self._set_camera(chosen)
        return names.index(self.camera)

    # The camera is on only while a window needs the gaze: a feature waits about a
    # second for it when it starts, and the camera light is off the rest of the time.
    def _use_camera(self, user: str) -> bool:
        if self.mouse:
            return True
        self.camera_users.add(user)
        if self.tracker is None and not self._open_camera():
            self.camera_users.discard(user)
            return False
        return True

    def _release_camera(self, *users: str) -> None:
        self.camera_users.difference_update(users)
        if not self.camera_users and self.tracker is not None:
            # camera light off, MediaPipe freed: on its own thread (a driver's release can block for a second);
            # the next open waits for it (open_tracker), so the camera is free again before it's asked for
            old, self.tracker = self.tracker, None
            feed, self.feed = self.feed, None

            def close() -> None:
                if feed is not None:
                    feed.stop()                   # its read in progress ends first
                old.close()
            self._closing = threading.Thread(target=close, daemon=True)
            self._closing.start()
            self._closing.join(0.25)              # usually done by now: the camera is off when this returns

    def _frames(self, part: str = "features"):
        """A reader of the camera's frames (FrameFeed.reader), the feed started on first use: each tick
        a window gets the newest frame, or NO_FRAME, instead of waiting for the camera."""
        if self.feed is None:
            self.feed = FrameFeed(self.tracker)
        return self.feed.reader(part)

    def _stop_feed(self) -> None:
        if self.feed is not None:
            self.feed.stop()
            self.feed = None

    def _open_camera(self) -> bool:
        """Open the chosen camera and MediaPipe on a worker thread, this one painting meanwhile."""
        try:
            index = self._camera_index()          # may ask with the popup first
        except RuntimeError as e:
            self.testing.say(f"The camera didn't start ({e}).")
            return False
        # the camera driver can take a minute the first time on some PCs: on a worker thread, the bar running
        try:
            self.tracker = self._busy("Opening the camera…", lambda: open_tracker(index, self._closing))
            return True
        except Exception as e:                    # camera busy, MediaPipe failed
            self.testing.say(f"The camera didn't start ({e}). Pick another on the Instructions page.")
            return False

    # ---- the models: trained when the check needs them, kept while their files stay the same -------
    def _model(self, model: Model, root: Path, newest: int) -> tuple[TrainedModel | None, int]:
        """A person's model, trained on their newest `newest` calibrations of its grid and its special's,
        or the one already trained on exactly those files. (model, samples used)."""
        grids, specials = storage.model_files(model, root, newest)
        key = (model.id, str(root), tuple((p.name, p.stat().st_mtime_ns) for p in grids + specials))
        if key not in self.models:
            screen = screen_geometry()            # Qt: here, on the GUI thread; the training itself on a worker
            self.models[key] = self._busy(f"Training the {model.name} model…", lambda: train(
                model, load_calibrations(grids), load_calibrations(specials), screen))
        return self.models[key]

    # ---- emailing: once both checks are done, on a thread of its own ----------------------
    def _mail(self, paths: list[Path] = ()) -> None:
        """These files into the outbox, and everything waiting there emailed, on a thread of its own: an
        email takes seconds, longer on a slow network. Not in mouse mode (nothing real to send)."""
        if self.mouse:
            return
        mailer.queue(paths)

        def send() -> None:
            said = mailer.send_outbox()
            if said:
                self.mailed.emit(said)
        sender = threading.Thread(target=send, daemon=True)
        self.senders = [t for t in self.senders if t.is_alive()] + [sender]
        sender.start()

    # ---- full-screen windows (they read the camera themselves) ----------
    def _open_fullscreen(self, window: QWidget) -> None:
        self.fullscreen = window                  # keep a reference, or Python deletes it
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._fullscreen_closed)
        self.control.show_fullscreen_page(window)   # the main window, full screen, no buttons

    def _fullscreen_closed(self) -> None:
        self.fullscreen = None
        self._release_camera("calibration", "check")

    def calibrate(self) -> None:
        if not self._named() or not self._use_camera("calibration"):
            return
        method = self.method
        get = fake_features_from_mouse if self.mouse else self._frames()
        window = CalibrationWindow(method, get)
        window.finished.connect(lambda rec: self._calibrated(window, method, rec))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, method: Method, rec: CalibrationRecording) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self.testing.say(f"{method.name}: mouse mode, not saved.")
            return
        if not (rec.dots if method.kind == "grid" else rec.head):
            self.testing.say(f"{method.name}: no samples. Was your face in view? Try again.")
            return
        # the file's place is fixed now: a name typed while the checks run must not move it
        target = storage.method_dir(method.id) / f"{method.id}_{datetime.now():{storage.STAMP}}.json"
        made: list[Path] = []

        def check_and_save():                     # on a worker thread: the checks and big files take seconds
            # impossible data -> reject, don't save; a special: its step 1 followed, a head step long enough
            report = check_recording(rec) if method.kind == "grid" else check_special(rec, method.kind)
            if not report.ok:
                return f"{method.name}: not saved ({report.summary()}). Try again."
            path = save_calibration(rec, method.id, target)   # saved RAW
            made.append(path)
            for m in partners(method):            # Points: Line model too (the same steps)
                made.append(storage.copy_as(path, m.id))
            return f"{method.name}: saved."
        said = self._busy("Checking and saving the calibration…", check_and_save)
        self.testing.refresh()
        self.testing.say(said)                    # last: "saved" or why it was not stays in view

    def check(self) -> None:
        """Gaze check n of SETS (the first not done): every model the tester's calibrations make, trained
        now on their newest n sets, all predicting from the same camera frames, on 36 rings (6x6), no
        dot: one csv per model, in data/<name>/<model>/, named "_<n>-calibrations"."""
        if not self._named():
            return
        if self.testing.todo():
            self.testing.say("Finish the calibrations first.")
            return
        left = [n for n in range(1, SETS + 1) if n not in storage.checks_done()]
        if not left:
            self.testing.say("Both gaze checks are done.")
            return
        root, newest = storage.DATA_DIR, left[0]
        models, counts = {}, {}
        for spec in MODELS:
            if storage.model_ready(spec, root):  # its calibrations aren't there
                continue
            model = None if self.mouse else self._model(spec, root, newest)[0]   # mouse mode: the mouse plays each one
            if model is not None or self.mouse:
                models[spec.id], counts[spec.id] = model, len(storage.model_files(spec, root, newest)[0])
        if not models:
            self.testing.say("No calibration yet: calibrate first.")
            return
        if not self._use_camera("check"):
            return
        tracks = {} if self.mouse else {m: model.tracker() for m, model in models.items()}
        frames = None if self.mouse else self._frames("stamped")

        def get_all():
            if self.mouse:
                point = fake_point_from_mouse()
                return {m: point for m in models}
            got = frames()                                # one frame, every model
            return NO_FRAME if got is NO_FRAME else {m: track(*got) for m, track in tracks.items()}
        window = GazeWindow(get_all, {m: storage.method_dir(m, root) for m in models}, counts,
                            grid=storage.MULTI_GRID, show_dot=False, tag="multi")
        window.logged.connect(self._checked)
        self._open_fullscreen(window)

    def _checked(self, paths: list[Path]) -> None:
        self.testing.refresh()
        done = self.testing.checks()
        if done < SETS:
            self.testing.say(f"Gaze check {done}/{SETS} saved. One more to do.")
            return
        self.testing.say("Both gaze checks done. Sending your results…")
        self._mail(storage.result_files())        # everything of this tester, in one email

    def close(self) -> None:
        self._stop_feed()
        if self.tracker is not None:
            self.tracker.close()
        for sender in self.senders:               # an email on its way: let it finish (it gives up after a minute)
            sender.join(2 * mailer.TIMEOUT_S)   # two ports tried


def main() -> int:
    parser = argparse.ArgumentParser(description="Eye tracking test")
    parser.add_argument("--mouse", action="store_true", help="no camera: the mouse plays your eyes")
    args, _ = parser.parse_known_args()
    if sys.platform == "win32":               # the taskbar groups a script under python.exe (and its icon)
        import ctypes                         # until the process says it is an app of its own
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StudyAssistant.EyeTrackingTest")
    qt_app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(TEXT_FONTS)              # the window's fonts (icon font last, for its glyphs)
    qt_app.setFont(font)
    qt_app.setWindowIcon(app_icon())          # the logo on the taskbar and every window
    app = App(mouse=args.mouse)
    try:
        return qt_app.exec()
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
