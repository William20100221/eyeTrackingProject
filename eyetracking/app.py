"""
The calibration lab: every calibration method, the tests and their data, in one window,
to find out which method gives the most accurate gaze model for each person.

    python main.py                 (the camera chosen in Settings; a popup asks on first start)
    python main.py --mouse         (no camera: the mouse plays the eyes; nothing is saved)

The window (ui/control_window.py) has three tabs:
    Testing      Calibration: pick one of the 12 types (core/methods.py) and start it
                 Gaze check: pick a person, a type and how many calibrations, start the check;
                 or every type at once (6x6 rings); or record the camera and run the models later
                 Sample text: read a short text with that model; which paragraphs did it see you read?
                 Blink test, and the data: calibrations, gaze checks, blink scores
    Calibration  starts the chosen type full screen
    Settings     theme, camera, person mode (whose folder the calibrations go into)

Each type keeps its own model, trained from its own folder: data/<person>/<type>/
(data/<type>/ with person mode off; services/storage.py). A model is trained when a gaze
check (or a replay) needs it, from the picked person's newest N calibrations of that type,
and kept until those files change (App.models): nothing trains at start, after a
calibration or on the Data pages. Training is 0.3-1 s per model (core/gaze_model.py).

What talks to what:
    camera -> tracker (services/tracker.py) -> 12 features per frame (core/features.py)
      -> CalibrationWindow(type) -> recording -> quality check -> saved -> models retrained
      -> GazeWindow: the picked person's model of the picked type says where you look;
         one of 16 rings (4x4) is lit, Enter records the point against it -> error per sample
         (every model at once on 36 rings; or in record mode the video + the presses, replayed
         through the tracker and the models afterwards: services/recording.py)
      Sweep / Points / Line model can reuse the person's Original / Random 4x4 as their dots
      Sweep, Points and Line model share every recording (one calibration is saved as all three)
      -> ReadingWindow: the sample text, a paragraph read once the gaze stayed on it long enough
      -> BlinkWindow: blink scores in six eye states
The camera is on only while one of those full-screen windows runs. Slow work (the camera opening,
training, checking and saving a calibration, a replay) runs on a worker thread while a thin bar
runs under the title bar (App._busy), so the window never stops answering.
"""
from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.blink_scores import advice as blink_advice
from eyetracking.core.calibration import GazeProfile
from eyetracking.core.methods import METHOD_BY_ID, Method, TrainedModel, live_clock, partners, train
from eyetracking.core.models import CalibrationRecording
from eyetracking.core.quality import check_recording
from eyetracking.services.camera import camera_names
from eyetracking.services.recording import Recorder, replay
from eyetracking.services.storage import (ensure_layout, load_calibration, save_blink_scores,
                                          save_calibration, save_gaze_check, use_person)
from eyetracking.services import storage
from eyetracking.services.tracker import EyeTracker
from eyetracking.ui.blink_window import BlinkWindow, fake_scores_from_mouse
from eyetracking.ui.calibration_window import CalibrationWindow, fake_features_from_mouse
from eyetracking.ui.control_window import app_icon, TEXT_FONTS, SettingsWindow, busy, pick_camera, run_busy
from eyetracking.ui.gaze_window import GazeWindow, fake_point_from_mouse, screen_geometry
from eyetracking.ui.reading_window import ReadingWindow, summary as reading_summary


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
    def __init__(self, mouse: bool) -> None:
        super().__init__()
        self.mouse = mouse
        self.control = SettingsWindow()
        self.testing = self.control.testing
        self.control.saved.connect(self.apply_settings)
        self.control.calibrate_requested.connect(lambda: self._command(self.calibrate))
        self.testing.command.connect(lambda c: self._command({"calibrate": self.calibrate, "check": self.check,
                                                               "check-all": self.check_all, "sample": self.sample_text,
                                                               "blink": self.blink_test}[c]))
        self.testing.replay_requested.connect(lambda p: self._command(lambda: self._replay(Path(p))))
        self.testing.method_changed.connect(lambda _: self.testing.status.setText(self._model_status()))
        self.testing.data_changed.connect(self._retrain)
        settings = self.control.read_store()

        self.tracker: EyeTracker | None = None     # opened by _use_camera(), only while a window needs the gaze
        self._working = False                      # slow work running (App._busy): a second start meanwhile is ignored
        self._closing: threading.Thread | None = None   # the last camera, closing on its own thread
        self.camera_users: set[str] = set()
        self.camera = settings["camera"]
        self.fullscreen: QWidget | None = None     # the calibration / check / blink page while it runs
        self.models: dict[tuple, tuple[TrainedModel | None, int]] = {}   # trained models by (type, folder, newest N, files)
        self.person = None                         # whose data: set by _use_person, from the settings
        self._use_person(settings)
        self.apply_settings({**settings, "camera": self.camera})
        self.testing.status.setText(self._model_status())
        self.control.show_last()

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
        camera driver, training a dozen models or saving a big calibration could do on this PC.
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
        """Start-up, and every time Save changes is pressed."""
        self._use_person(values)
        if values["camera"] != self.camera:
            self.camera = values["camera"]
            if self.tracker is not None:          # in use right now: switch at once
                self.tracker.close()
                self.tracker = None
                if self._open_camera():
                    self.testing.say(f"Now using the camera: {self.camera}.")

    def _use_person(self, values: dict) -> None:
        """Person mode on: data under data/<name>/; off (or no name): straight under data/.
        A change re-reads every method's files, so each model trains on that person's data only."""
        person = values["person"] if values["person_mode"] else ""
        if person == self.person:
            return
        self.person = person
        root = use_person(person)
        ensure_layout()                           # a new person / folders cut out: made empty
        self.testing.set_method(self.testing.method_id, announce=False)   # the "Saved as" path
        self.testing.refresh()                    # the lists: "in use" moves to this person
        self.testing.set_check(person=person)     # the gaze check tests the person in use, unless picked
        where = root.relative_to(storage.BASE_DIR.parent).as_posix()
        self.testing.say(f"Saving for {person} in {where}/." if person else f"Saving straight under {where}/.")

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
            self._closing = threading.Thread(target=old.close, daemon=True)
            self._closing.start()
            self._closing.join(0.25)              # usually done by now: the camera is off when this returns

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
            self.testing.say(f"The camera didn't start ({e}). Pick another in Settings > Camera.")
            return False

    # ---- the models: trained when a check needs one, kept while its files stay the same -------
    def _model(self, method: Method, root: Path, newest: int | None) -> tuple[TrainedModel | None, int]:
        """A person's model of one type, trained on their newest `newest` calibrations (None: all),
        or the one already trained on exactly those files. (model, samples used)."""
        files = [p for p in storage.list_calibrations(method.id, root) if storage.calibration_usable(p)][:newest]
        key = (method.id, str(root), newest, tuple((p.name, p.stat().st_mtime_ns) for p in files))
        if key not in self.models:
            screen = screen_geometry()            # Qt: here, on the GUI thread; the training itself on a worker
            self.models[key] = self._busy(f"Training the {method.name} model…",
                                          lambda: train(method, storage.load_calibrations(files), screen))
        return self.models[key]

    def _retrain(self, quiet: bool = False) -> None:
        """Calibrations changed (removed, disabled, enabled, cleared; from the Data page): the status
        line follows; a model trains again by itself when a check asks for the changed files."""
        self.testing.status.setText(self._model_status())
        if not quiet:
            self.testing.say(self._model_status())

    def _model_status(self) -> str:
        m = self.method
        if self.mouse:
            return f"{m.name}: mouse mode, the mouse plays your eyes (nothing is saved)."
        n = storage.usable_calibrations(m.id)
        if not n:
            return f"{m.name}: no calibration yet. Start one from Calibration."
        return f"{m.name}: {n} usable calibration{'s' if n != 1 else ''}; the model trains when a gaze check starts."

    # ---- full-screen windows (they read the camera themselves) ----------
    def _open_fullscreen(self, window: QWidget) -> None:
        self.fullscreen = window                  # keep a reference, or Python deletes it
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._fullscreen_closed)
        self.control.show_fullscreen_page(window)   # the main window, full screen, no buttons

    def _fullscreen_closed(self) -> None:
        self.fullscreen = None
        self._release_camera("calibration", "check", "blink", "sample")

    def calibrate(self) -> None:
        if not self._use_camera("calibration"):
            return
        method = self.method
        get = fake_features_from_mouse if self.mouse else self.tracker.read_features
        # Sweep / Points / Line model: the final dots reused from the person's 4x4 calibration (the switch)
        borrowed = storage.dots_source(method) if self.testing.reuse_dots else None
        window = CalibrationWindow(method, get, dots=load_calibration(borrowed).dots if borrowed else None,
                                   dots_from=borrowed.name if borrowed else "")
        window.finished.connect(lambda rec: self._calibrated(window, method, rec, borrowed))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, method: Method, rec: CalibrationRecording,
                    borrowed: Path | None = None) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self.testing.say(f"{method.name}: mouse mode, calibration not saved.")
            return
        if not rec.dots:
            self.testing.say(f"{method.name}: no samples. Was your face in view? Try again.")
            return
        others = partners(method) if self.testing.share_partner else []   # Sweep / Points / Line model: one recording
        # the file's place is fixed now: a person switched in Settings while the checks run must not move it
        target = storage.method_dir(method.id) / f"{method.id}_{datetime.now():{storage.STAMP}}.json"

        def check_and_save():                     # on a worker thread: the checks and big files take seconds here
            report = check_recording(rec)         # impossible data -> reject, don't save
            if not report.ok:
                return f"{method.name}: calibration rejected ({report.summary()}). Try again."
            if method.model != "ridge":
                try:
                    GazeProfile.from_recording(report.kept)   # an unusable sweep / head step -> don't save it
                except ValueError as e:
                    return f"{method.name}: calibration didn't work ({e}). Try again."
            path = save_calibration(rec, method.id, target, dots_from=borrowed.name if borrowed else None)   # saved RAW
            note = f" Cleaned: {report.summary()}." if report.reasons else ""
            note += f" The dots were reused from {borrowed.name}." if borrowed else ""
            for m in others:
                storage.copy_as(path, m.id)
            if others:
                note += f" Also saved as {' and '.join(m.name for m in others)}: all of them train on it."
            return f"{method.name}: calibrated with {len(report.kept.dots)} good dot samples, saved as {path.name}.{note}"
        said = self._busy("Checking and saving the calibration…", check_and_save)
        self.testing.refresh()
        self.testing.say(said)                    # last: "saved as …" or why it was rejected stays in view

    # ---- the gaze check: one model, every model, or a recording ----------------
    def check(self) -> None:
        """The gaze check of the person and calibration type picked on its page: that person's
        model of that type, trained from that person's files only (the newest N)."""
        method = METHOD_BY_ID[self.testing.check_method]
        person = self.testing.check_person
        root = storage.person_root(person)
        who = f"{person or 'No person'}'s {method.name}"
        if self.testing.record_mode:
            self._record_check(storage.CHECK_GRID, [method.id])
            return
        model, count = None, self.testing.check_count             # the newest `count` calibrations train it
        if not self.mouse:
            model, n = self._model(method, root, count)
            if model is None:
                self.testing.say(f"{who}: no usable calibration yet. Calibrate first.")
                return
            self.testing.say(f"{who}: model trained on {n} samples from the newest {count} "
                             f"calibration{'s' if count != 1 else ''}. Look at the lit ring, press Enter.")
        if not self._use_camera("check"):
            return

        track = None if self.mouse else model.tracker()   # the smoothed live point

        def get():
            return fake_point_from_mouse() if self.mouse else track(self.tracker.read_features(), live_clock())
        window = GazeWindow.single(get, method.id, storage.method_dir(method.id, root), count or None)
        window.logged.connect(lambda paths: self._checked(who, paths))
        self._open_fullscreen(window)

    def _models_for(self, root: Path, types: list[str] | None) -> tuple[dict, dict[str, int]]:
        """A person's models: of `types`, each trained on the page's newest N; or, for None, the
        types ticked in the multiple-check popup, each on the N chosen there. (models, calibrations used)."""
        wanted = {m: self.testing.check_count or None for m in types} if types else \
                 {m: n for m, n in self.testing.multi_counts.items() if n}
        models, counts = {}, {}
        for m, newest in wanted.items():
            used = min(newest or 10 ** 6, storage.usable_calibrations(m, root))   # how many files train it (cached)
            if self.mouse:                               # no models in mouse mode: the mouse plays each type that has data
                if used:
                    models[m], counts[m] = None, used
                continue
            model, _ = self._model(METHOD_BY_ID[m], root, newest)
            if model is not None:
                models[m], counts[m] = model, used
        return models, counts

    def check_all(self) -> None:
        """Every type the person has calibrations for, at once, from the same camera frames: one
        file per type, on the 6x6 rings. No dot: it would show one model only."""
        person = self.testing.check_person
        root, who = storage.person_root(person), person or "No person"
        if self.testing.record_mode:
            self._record_check(storage.MULTI_GRID, None)
            return
        models, counts = self._models_for(root, None)
        if not models:
            self.testing.say(f"{who}: no usable calibration of any type yet. Calibrate first.")
            return
        if not self._use_camera("check"):
            return
        self.testing.say(f"{who}: {len(models)} models ready ({', '.join(METHOD_BY_ID[m].name for m in models)}). "
                         "Look at the lit ring, press Enter.")

        tracks = {} if self.mouse else {m: model.tracker() for m, model in models.items()}

        def get_all():
            if self.mouse:
                point = fake_point_from_mouse()
                return {m: point for m in models}
            features, t = self.tracker.read_features(), live_clock()      # one frame, every model
            return {m: track(features, t) for m, track in tracks.items()}
        window = GazeWindow(get_all, {m: storage.method_dir(m, root) for m in models}, counts,
                            grid=storage.MULTI_GRID, show_dot=False, tag="multi")
        window.logged.connect(lambda paths: self._checked(who, paths))
        self._open_fullscreen(window)

    def _record_check(self, grid: int, types: list[str] | None) -> None:
        """Record mode: the camera is recorded with the presses; on Esc the models run over it."""
        person = self.testing.check_person
        who = person or "No person"
        if self.mouse:
            self.testing.say("Record mode needs the camera (not --mouse).")
            return
        if not self._use_camera("check"):
            return
        recorder = Recorder(self.tracker.camera, storage.person_root(person) / storage.VIDEOS, grid)
        window = GazeWindow(lambda: (recorder.frame(), {})[1], {}, grid=grid, show_dot=False, recorder=recorder)
        # the models run once the full-screen page has closed (it takes a while: a wait cursor and a count)
        window.recorded.connect(lambda meta: QTimer.singleShot(0, lambda: self._replay(meta, types)))
        self.testing.say(f"{who}: recording the camera. Look at the lit ring, press Enter; Esc stops and runs the models.")
        self._open_fullscreen(window)

    def _replay(self, meta: Path, types: list[str] | None = None) -> None:
        """Run the person's models of `types` (None: every type) over a recording, as if they had
        been live: one gaze check per type, named after the recording."""
        person = self.testing.check_person
        root, who = storage.person_root(person), person or "No person"
        if self.mouse:
            self.testing.say("Running a recording needs the models (not --mouse).")
            return
        models, counts = self._models_for(root, types)
        if not models:
            self.testing.say(f"{who}: no usable calibration to run over the recording. Calibrate first.")
            return
        done = [0.0]
        try:                                          # a minute's video takes ~30 s: on a worker thread, the bar running
            logs = self._busy("Running the models over the recording…",
                              lambda: replay(meta, {m: model.tracker() for m, model in models.items()},
                                             lambda d: done.__setitem__(0, d)),
                              status=lambda: f"Running {len(models)} models over the recording: {done[0]:.0%}")
        except Exception as e:                        # a damaged or moved recording
            self.testing.say(f"The recording couldn't be run ({e}).")
            return
        stamp = meta.stem[-17:]
        paths = [save_gaze_check(log, m, storage.method_dir(m, root), counts[m], f"video-{stamp}")
                 for m, log in logs.items() if log]
        if paths:
            self._checked(who, paths)
        else:
            self.testing.say(f"{who}: no face was found in the recording, nothing to save.")

    def _checked(self, who: str, paths: list[Path]) -> None:
        self.testing.refresh()
        self.testing.say(f"{who}: gaze check saved as {', '.join(p.name for p in paths)}. See Data > Gaze checks.")

    def sample_text(self) -> None:
        """Tests > Sample text: read a short text with the model picked on the Gaze check page (the
        person, the type, the newest N), and see which paragraphs it says you read."""
        method = METHOD_BY_ID[self.testing.check_method]
        person = self.testing.check_person
        root = storage.person_root(person)
        who = f"{person or 'No person'}'s {method.name}"
        track = None
        if not self.mouse:
            model, n = self._model(method, root, self.testing.check_count)
            if model is None:
                self.testing.say(f"{who}: no usable calibration yet. Calibrate first.")
                return
            track = model.tracker()                   # the smoothed live point, as in the gaze check
        if not self._use_camera("sample"):
            return

        def get():
            return fake_point_from_mouse() if self.mouse else track(self.tracker.read_features(), live_clock())
        window = ReadingWindow(get)
        window.finished.connect(lambda results: self.testing.say(f"{who}, sample text: {reading_summary(results)}"))
        self.testing.say(f"{who}: read the text at your own pace; Esc finishes.")
        self._open_fullscreen(window)

    def blink_test(self) -> None:
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
        QTimer.singleShot(800, window.close)
        path = save_blink_scores(rows)
        self.testing.refresh()
        self.testing.say(f"Blink scores saved as {path.name}. {blink_advice(rows)}")

    def close(self) -> None:
        if self.tracker is not None:
            self.tracker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Eye-tracking calibration lab")
    parser.add_argument("--mouse", action="store_true", help="no camera: the mouse plays your eyes")
    args, _ = parser.parse_known_args()
    if sys.platform == "win32":               # the taskbar groups a script under python.exe (and its icon)
        import ctypes                         # until the process says it is an app of its own
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("StudyAssistant.EyeTracking")
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
