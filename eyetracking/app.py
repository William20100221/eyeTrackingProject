"""
The calibration lab: every calibration method, the tests and their data, in one window,
to find out which method gives the most accurate gaze model for each person.

    python main.py                 (the camera chosen in Settings; a popup asks on first start)
    python main.py --mouse         (no camera: the mouse plays the eyes; nothing is saved)

The window (ui/control_window.py) has three tabs:
    Testing      Calibration: pick one of the 9 types (core/methods.py: three grids in order, three in
                 random order, three specials) and start it
                 Gaze check: pick a person, a model (a grid's dots, alone or with a special) and how
                 many calibrations, start the check; or several models at once (6x6 rings); or record
                 the camera and run the models later
                 Sample text: read a short text with that model; which paragraphs did it see you read?
                 Blink test, and the data: calibrations, Train (any of the 24 models, now), gaze checks,
                 the gaze check table (every model, one tester or everyone), blink scores
    Calibration  starts the chosen type full screen
    Settings     theme, camera, person mode (whose folder the calibrations go into)

Each calibration type has its folder: data/<person>/<type>/ (data/<type>/ with person mode off;
services/storage.py). A model learns from a grid's folder, and a paired model also from a special's
(Sweep + Original 4x4: the person's Original 4x4 calibrations and their newest Sweep). A model is
trained when a gaze check (a replay, Data > Train) needs it, from the picked person's newest N
calibrations of its grid, and kept until those files change (App.models): nothing trains at start,
after a calibration or on the Data pages. Training is 0.3-1 s per model (core/gaze_model.py).

What talks to what:
    camera -> tracker (services/tracker.py) -> 12 features per frame (core/features.py)
      -> CalibrationWindow(type) -> recording -> quality check -> saved (Undo takes it back)
      -> GazeWindow: the picked person's model says where you look; one of 16 rings (4x4) is
         lit, Enter records the point against it -> error per sample (several models at once on
         36 rings; or in record mode the video + the presses, replayed through the tracker and
         the models afterwards: services/recording.py)
      Points and Line model record the same steps: one calibration is saved as both
      -> ReadingWindow: the sample text, a paragraph read once the gaze stayed on it long enough
      -> BlinkWindow: blink scores in six eye states
The camera is on only while one of those full-screen windows runs, read on a thread of its own
(FrameFeed): the windows tick at the screen's pace, never waiting for a frame. Slow work (the camera opening,
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
from eyetracking.core.methods import METHOD_BY_ID, MODEL_BY_ID, Method, Model, TrainedModel, dots_error, partners, train
from eyetracking.core.models import NO_FRAME, CalibrationRecording
from eyetracking.core.quality import check_recording, check_special
from eyetracking.services.camera import camera_names
from eyetracking.services.recording import Recorder, replay
from eyetracking.services.storage import (ensure_layout, load_calibrations, save_blink_scores,
                                          save_calibration, save_gaze_check, use_person)
from eyetracking.services import storage
from eyetracking.services.tracker import EyeTracker, FrameFeed
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
        self.testing.train_requested.connect(lambda person, ids: self._command(lambda: self.train_models(person, ids)))
        self.testing.undo_requested.connect(self.undo)
        self.undo_stack: list[tuple[str, list[Path]]] = []   # (what, the files it made): the newest last
        settings = self.control.read_store()

        self.tracker: EyeTracker | None = None     # opened by _use_camera(), only while a window needs the gaze
        self._working = False                      # slow work running (App._busy): a second start meanwhile is ignored
        self._closing: threading.Thread | None = None   # the last camera, closing on its own thread
        self.feed: FrameFeed | None = None         # the camera read on its own thread, while a window runs
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
                self._stop_feed()
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
        self.testing.set_train_person(person)     # ...and Data > Train trains theirs
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

    def _live(self, track):
        """get() for a window that shows one model's smoothed point: track(features, time) per camera
        frame (NO_FRAME in between), or the mouse in mouse mode."""
        if self.mouse:
            return fake_point_from_mouse
        frames = self._frames("stamped")

        def get():
            got = frames()
            return NO_FRAME if got is NO_FRAME else track(*got)
        return get

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
    def _job(self, model: Model, root: Path, newest: int | None):
        """(cache key, grid files, special files) for a person's model on their newest `newest` grid
        calibrations: the key is the files themselves, so the same files never train twice."""
        grids, specials = storage.model_files(model, root, newest)
        key = (model.id, str(root), tuple((p.name, p.stat().st_mtime_ns) for p in grids + specials))
        return key, grids, specials

    def _model(self, model: Model, root: Path, newest: int | None) -> tuple[TrainedModel | None, int]:
        """A person's model, trained on their newest `newest` calibrations of its grid (None: all) and its
        special's, or the one already trained on exactly those files. (model, samples used)."""
        key, grids, specials = self._job(model, root, newest)
        if key not in self.models:
            screen = screen_geometry()            # Qt: here, on the GUI thread; the training itself on a worker
            self.models[key] = self._busy(f"Training the {model.name} model…", lambda: train(
                model, load_calibrations(grids), load_calibrations(specials), screen))
        return self.models[key]

    def train_models(self, person: str, ids: list[str]) -> None:
        """Data > Train: the ticked models of a person, on all their calibrations, trained now (one job on the
        worker thread, the status counting), so their gaze checks start at once. Each row then says what
        the model learnt from: its samples, its error on its newest grid calibration's dots, its head matrix."""
        root, screen = storage.person_root(person), screen_geometry()
        jobs = [(MODEL_BY_ID[m], *self._job(MODEL_BY_ID[m], root, None)) for m in ids]
        done, results = [0], {}

        def work():
            for model, key, grids, specials in jobs:
                if key not in self.models:
                    self.models[key] = train(model, load_calibrations(grids), load_calibrations(specials), screen)
                trained, n = self.models[key]
                # on the newest grid recording it really learnt from (train() skips one that fails the check)
                error = dots_error(trained, trained.newest) if trained and trained.newest else None
                results[model.id] = (key[2],                             # the files, as the model cache knows them
                                     (str(n), "" if error is None else f"{error:.1%}",
                                      (trained.k_source or "its own 3D line") if trained else "couldn't train"))
                done[0] += 1
        self._busy("Training…", work, status=lambda: f"Training {min(done[0] + 1, len(jobs))} of {len(jobs)}: "
                                                     f"{jobs[min(done[0], len(jobs) - 1)][0].name}…")
        self.testing.show_trained(person, results)
        failed = [MODEL_BY_ID[m].name for m, (_, nums) in results.items() if nums[2] == "couldn't train"]
        self.testing.say(f"{person or 'No person'}: {len(results) - len(failed)} model"
                         f"{'s' if len(results) - len(failed) != 1 else ''} trained."
                         + (f" Couldn't train {', '.join(failed)} (the data didn't pass the quality check)." if failed else ""))

    def _retrain(self, quiet: bool = False) -> None:
        """Calibrations changed (removed, disabled, enabled, cleared; from the Data page): the status
        line follows; a model trains again by itself when a check asks for the changed files."""
        self.testing.status.setText(self._model_status())
        self.testing.refresh_train()
        if not quiet:
            self.testing.say(self._model_status())

    def _model_status(self) -> str:
        m = self.method
        if self.mouse:
            return f"{m.name}: mouse mode, the mouse plays your eyes (nothing is saved)."
        n = storage.usable_calibrations(m.id)
        if not n:
            return f"{m.name}: no calibration yet. Start one from Calibration."
        pairs = (f"; it pairs with each grid calibration of this person: the {m.name} + <grid> models"
                 if m.kind != "grid" else "; its model trains when a gaze check starts")
        return f"{m.name}: {n} usable calibration{'s' if n != 1 else ''}{pairs}."

    # ---- undo: the last calibration or gaze check, its files into the trash ----------
    def _did(self, what: str, paths: list[Path]) -> None:
        """A calibration or gaze check just saved these files: Undo can take them back."""
        self.undo_stack.append((what, list(paths)))
        self.testing.set_undo(what)

    def undo(self) -> None:
        """Undo the newest calibration or gaze check: its files (a calibration's copy, a multiple check's
        one file per model) go into their trash folders, so even an undo can be undone (Data >
        Calibrations > Removed restores a calibration). Never during a full-screen page."""
        if not self.undo_stack or self.fullscreen is not None or self._working:
            return
        what, paths = self.undo_stack.pop()
        gone = [p for p in paths if not p.exists()]          # moved or removed on the Data pages since
        for p in paths:
            if p.exists():
                (storage.remove_calibration if p.suffix == ".json" else storage.trash_file)(p)
        self.testing.set_undo(self.undo_stack[-1][0] if self.undo_stack else "")
        self.testing.refresh()
        self._retrain(quiet=True)
        left = len(paths) - len(gone)
        self.testing.say(f"Undid {what}: {left} file{'s' if left != 1 else ''} moved to the trash"
                         + (" (Data > Calibrations > Removed restores it)" if any(p.suffix == ".json" for p in paths) else "")
                         + (f"; already gone: {', '.join(p.name for p in gone)}" if gone else "") + ".")

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
        get = fake_features_from_mouse if self.mouse else self._frames()
        window = CalibrationWindow(method, get)
        window.finished.connect(lambda rec: self._calibrated(window, method, rec))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, method: Method, rec: CalibrationRecording) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self.testing.say(f"{method.name}: mouse mode, calibration not saved.")
            return
        if not (rec.dots if method.kind == "grid" else rec.head):
            self.testing.say(f"{method.name}: no samples. Was your face in view? Try again.")
            return
        others = partners(method) if self.testing.share_partner else []   # Points / Line model: one recording
        # the file's place is fixed now: a person switched in Settings while the checks run must not move it
        target = storage.method_dir(method.id) / f"{method.id}_{datetime.now():{storage.STAMP}}.json"
        made: list[Path] = []

        def check_and_save():                     # on a worker thread: the checks and big files take seconds here
            # impossible data -> reject, don't save; a special: its step 1 followed, a head step long enough
            report = check_recording(rec) if method.kind == "grid" else check_special(rec, method.kind)
            if not report.ok:
                return f"{method.name}: calibration rejected ({report.summary()}). Try again."
            path = save_calibration(rec, method.id, target)   # saved RAW
            made.append(path)
            note = f" Cleaned: {report.summary()}." if report.reasons else ""
            for m in others:
                made.append(storage.copy_as(path, m.id))
            if others:
                note += f" Also saved as {' and '.join(m.name for m in others)}: both train on it."
            kept = (f"{len(report.kept.dots)} good dot samples" if method.kind == "grid" else
                    f"{len(report.kept.sweep)} good step-1 and {len(report.kept.head)} head-step frames")
            return f"{method.name}: calibrated with {kept}, saved as {path.name}.{note}"
        said = self._busy("Checking and saving the calibration…", check_and_save)
        if made:
            self._did(f"the {method.name} calibration", made)
        self.testing.refresh()
        self.testing.say(said)                    # last: "saved as …" or why it was rejected stays in view

    # ---- the gaze check: one model, every model, or a recording ----------------
    def check(self) -> None:
        """The gaze check of the person and model picked on its page: that person's model, trained
        from that person's files only (the newest N of its grid, and its special's)."""
        model_spec = MODEL_BY_ID[self.testing.check_model]
        person = self.testing.check_person
        root = storage.person_root(person)
        who = f"{person or 'No person'}'s {model_spec.name}"
        if self.testing.record_mode:
            self._record_check(storage.CHECK_GRID, [model_spec.id])
            return
        model, count = None, self.testing.check_count             # the newest `count` grid calibrations train it
        if not self.mouse:
            missing = storage.model_ready(model_spec, root)
            model, n = (None, 0) if missing else self._model(model_spec, root, count)
            if model is None:
                self.testing.say(f"{who}: {missing or 'its calibrations did not pass the quality check'}. Calibrate first.")
                return
            self.testing.say(f"{who}: model trained on {n} samples from the newest {count} {model_spec.grid.name} "
                             f"calibration{'s' if count != 1 else ''}"
                             + (f" and the {model_spec.special.name}" if model_spec.special else "")
                             + ". Look at the lit ring, press Enter.")
        if not self._use_camera("check"):
            return
        get = self._live(None if self.mouse else model.tracker())   # the smoothed live point
        window = GazeWindow.single(get, model_spec.id, storage.method_dir(model_spec.id, root), count or None)
        window.logged.connect(lambda paths: self._checked(who, paths))
        self._open_fullscreen(window)

    def _models_for(self, root: Path, ids: list[str] | None) -> tuple[dict, dict[str, int]]:
        """A person's models: of `ids`, each trained on the page's newest N; or, for None, the
        models ticked in the multiple-check popup, each on the N chosen there. (models, grid calibrations used)."""
        wanted = {m: self.testing.check_count or None for m in ids} if ids else \
                 {m: n for m, n in self.testing.multi_counts.items() if n}
        models, counts = {}, {}
        for m, newest in wanted.items():
            spec = MODEL_BY_ID[m]
            if storage.model_ready(spec, root):          # its data is missing (moved or removed meanwhile)
                continue
            used = len(storage.model_files(spec, root, newest)[0])   # how many grid files train it (cached)
            if self.mouse:                               # no models in mouse mode: the mouse plays each model that has data
                models[m], counts[m] = None, used
                continue
            model, _ = self._model(spec, root, newest)
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
        self.testing.say(f"{who}: {len(models)} models ready ({', '.join(MODEL_BY_ID[m].name for m in models)}). "
                         "Look at the lit ring, press Enter.")

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
        window.logged.connect(lambda paths: self._checked(who, paths))
        self._open_fullscreen(window)

    def _record_check(self, grid: int, ids: list[str] | None) -> None:
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
        window.recorded.connect(lambda meta: QTimer.singleShot(0, lambda: self._replay(meta, ids)))
        self.testing.say(f"{who}: recording the camera. Look at the lit ring, press Enter; Esc stops and runs the models.")
        self._open_fullscreen(window)

    def _replay(self, meta: Path, ids: list[str] | None = None) -> None:
        """Run the person's models `ids` (None: the ones ticked in the popup) over a recording, as if they had
        been live: one gaze check per model, named after the recording."""
        person = self.testing.check_person
        root, who = storage.person_root(person), person or "No person"
        if self.mouse:
            self.testing.say("Running a recording needs the models (not --mouse).")
            return
        models, counts = self._models_for(root, ids)
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
        self._did("the gaze check" if len(paths) == 1 else f"the gaze check ({len(paths)} files)", paths)
        self.testing.refresh()
        self.testing.say(f"{who}: gaze check saved as {', '.join(p.name for p in paths)}. See Data > Gaze checks.")

    def sample_text(self) -> None:
        """Tests > Sample text: read a short text with the model picked on the Gaze check page (the
        person, the model, the newest N), and see which paragraphs it says you read."""
        spec = MODEL_BY_ID[self.testing.check_model]
        person = self.testing.check_person
        root = storage.person_root(person)
        who = f"{person or 'No person'}'s {spec.name}"
        track = None
        if not self.mouse:
            missing = storage.model_ready(spec, root)
            model, n = (None, 0) if missing else self._model(spec, root, self.testing.check_count)
            if model is None:
                self.testing.say(f"{who}: {missing or 'its calibrations did not pass the quality check'}. Calibrate first.")
                return
            track = model.tracker()                   # the smoothed live point, as in the gaze check
        if not self._use_camera("sample"):
            return
        window = ReadingWindow(self._live(track))
        window.finished.connect(lambda results: self.testing.say(f"{who}, sample text: {reading_summary(results)}"))
        self.testing.say(f"{who}: read the text at your own pace; Esc finishes.")
        self._open_fullscreen(window)

    def blink_test(self) -> None:
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
        QTimer.singleShot(800, window.close)
        path = save_blink_scores(rows)
        self.testing.refresh()
        self.testing.say(f"Blink scores saved as {path.name}. {blink_advice(rows)}")

    def close(self) -> None:
        self._stop_feed()
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
