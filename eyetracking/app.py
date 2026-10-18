"""
The calibration lab: every calibration method, the tests, and their data, in one window.

    python main.py                 (the camera chosen in Settings; a popup asks on first start)
    python main.py --mouse         (no camera: the mouse plays the eyes; nothing is saved)

The window (ui/control_window.py) has three tabs:
    Testing      Calibration (pick a method, start it), the tests (gaze check, blink test),
                 and the data (calibrations, gaze checks, blink scores, the log)
    Calibration  starts the chosen method full screen
    Settings     camera, theme
The methods are in core/methods.py; each one keeps its own model, trained from
its own folder: data/<person>/<method>/, or data/<method>/ with person mode off
(services/storage.py).

What talks to what:
    camera -> tracker (services/tracker.py) -> features (core/features.py)
      -> CalibrationWindow (one per method) -> recording -> quality check -> saved -> model trained
      -> GazeWindow: the chosen method's model predicts where you look; Enter records samples
      -> BlinkWindow: blink scores in six eye states
The camera is on only while one of those windows runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.blink_scores import advice as blink_advice
from eyetracking.core.calibration import GazeProfile, point_grid
from eyetracking.core.methods import METHOD_BY_ID, Method, TrainedModel, train
from eyetracking.core.models import CalibrationRecording
from eyetracking.core.quality import check_recording
from eyetracking.services.camera import camera_names
from eyetracking.services.storage import (ensure_layout, load_all_calibrations, save_blink_scores, save_calibration,
                                          use_person)
from eyetracking.services import storage
from eyetracking.services.tracker import EyeTracker
from eyetracking.ui.blink_window import BlinkWindow, fake_scores_from_mouse
from eyetracking.ui.calibration_window import CalibrationWindow, fake_features_from_mouse
from eyetracking.ui.control_window import TEXT_FONTS, SettingsWindow, pick_camera
from eyetracking.ui.gaze_window import GazeWindow, fake_point_from_mouse, screen_geometry


class App(QObject):
    def __init__(self, mouse: bool) -> None:
        super().__init__()
        self.mouse = mouse
        self.control = SettingsWindow()
        self.testing = self.control.testing
        self.control.saved.connect(self.apply_settings)
        self.control.calibrate_requested.connect(self.calibrate)
        self.testing.command.connect(lambda c: {"calibrate": self.calibrate, "check": self.check, "blink": self.blink_test}[c]())
        self.testing.method_changed.connect(lambda _: self.testing.status.setText(self._model_status()))
        self.testing.data_changed.connect(self._retrain)
        settings = self.control.read_store()

        self.tracker: EyeTracker | None = None     # opened by _use_camera(), only while a window needs the gaze
        self.camera_users: set[str] = set()
        self.camera = settings["camera"]
        self.fullscreen: QWidget | None = None     # the calibration / check / blink page while it runs
        self.models: dict[str, TrainedModel] = {}  # one model per method, trained from its own folder
        self.samples_used: dict[str, int] = {}
        self.person = None                         # whose data: set by _use_person, from the settings
        self._use_person(settings)
        self.apply_settings({**settings, "camera": self.camera})
        self._retrain(quiet=True)
        self.control.show_last()

    @property
    def method(self) -> Method:
        return METHOD_BY_ID[self.testing.method_id]

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
        for m in METHOD_BY_ID.values():
            self._train(m)
        self.testing.set_method(self.testing.method_id, announce=False)   # lists + "Saved as" path
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
            self.tracker.close()                  # camera light off, MediaPipe freed
            self.tracker = None

    def _open_camera(self) -> bool:
        """Open the chosen camera and MediaPipe (about a second, on this thread)."""
        try:
            index = self._camera_index()          # may ask with the popup first
        except RuntimeError as e:
            self.testing.say(f"The camera didn't start ({e}).")
            return False
        self.testing.status.setText("Opening the camera…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        # ponytail: opens on the GUI thread (~1 s freeze); move to a QThread if that feels slow
        try:
            self.tracker = EyeTracker(index)
            return True
        except Exception as e:                    # camera busy, MediaPipe failed
            self.testing.say(f"The camera didn't start ({e}). Pick another in Settings > Camera.")
            return False
        finally:
            QApplication.restoreOverrideCursor()

    # ---- the models: one per method ---------------------------------------
    def _train(self, method: Method) -> None:
        if self.mouse:
            return
        model, n = train(method, load_all_calibrations(method.id), screen_geometry())
        self.samples_used[method.id] = n
        if model is not None:
            self.models[method.id] = model
        else:
            self.models.pop(method.id, None)

    def _retrain(self, quiet: bool = False) -> None:
        """Calibrations changed (removed, disabled, enabled; any type, from the Data page): train every model again."""
        for m in METHOD_BY_ID.values():
            self._train(m)
        self.testing.status.setText(self._model_status())
        if not quiet:
            self.testing.say(self._model_status())

    def _model_status(self) -> str:
        m = self.method
        if self.mouse:
            return f"{m.name}: mouse mode, the mouse plays your eyes (nothing is saved)."
        if m.id not in self.models:
            return f"{m.name}: no calibration yet. Start one from Calibration."
        return f"{m.name}: model trained on {self.samples_used[m.id]} samples, the newest calibration counting most."

    # ---- full-screen windows (they read the camera themselves) ----------
    def _open_fullscreen(self, window: QWidget) -> None:
        self.fullscreen = window                  # keep a reference, or Python deletes it
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._fullscreen_closed)
        self.control.show_fullscreen_page(window)   # the main window, full screen, no buttons

    def _fullscreen_closed(self) -> None:
        self.fullscreen = None
        self._release_camera("calibration", "check", "blink")

    def calibrate(self) -> None:
        if not self._use_camera("calibration"):
            return
        method = self.method
        get = fake_features_from_mouse if self.mouse else self.tracker.read_features
        window = CalibrationWindow(method, get)
        window.finished.connect(lambda rec: self._calibrated(window, method, rec))
        self._open_fullscreen(window)

    def _calibrated(self, window: CalibrationWindow, method: Method, rec: CalibrationRecording) -> None:
        QTimer.singleShot(800, window.close)      # show "done" briefly
        if self.mouse:
            self.testing.say(f"{method.name}: mouse mode, calibration not saved.")
            return
        if not rec.dots:
            self.testing.say(f"{method.name}: no samples. Was your face in view? Try again.")
            return
        report = check_recording(rec)             # impossible data -> reject, don't save
        if not report.ok:
            self.testing.say(f"{method.name}: calibration rejected ({report.summary()}). Try again.")
            return
        if method.model != "ridge":
            try:
                GazeProfile.from_recording(report.kept)   # an unusable sweep / head step -> don't save it
            except ValueError as e:
                self.testing.say(f"{method.name}: calibration didn't work ({e}). Try again.")
                return
        path = save_calibration(rec, method.id)   # saved RAW: the checks rerun on loading
        self._train(method)
        note = f" Cleaned: {report.summary()}." if report.reasons else ""
        self.testing.say(f"{method.name}: calibrated with {len(report.kept.dots)} good dot samples, saved as "
                         f"{path.name}. Model trained on {self.samples_used[method.id]} samples.{note}")
        self.testing.refresh()
        self.testing.status.setText(self._model_status())

    def check(self) -> None:
        method = self.method
        model = self.models.get(method.id)
        if not self.mouse and model is None:
            self.testing.say(f"{method.name}: calibrate first.")
            return
        if not self._use_camera("check"):
            return

        def get():
            return fake_point_from_mouse() if self.mouse else model.predict(self.tracker.read_features())
        window = GazeWindow(get_point=get, targets=point_grid(method.grid), method_id=method.id)
        window.logged.connect(lambda path: (self.testing.refresh(), self.testing.say(
            f"{method.name}: gaze samples saved as {Path(path).name}. See Data > Gaze checks.")))
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
    qt_app = QApplication(sys.argv)
    font = QFont()
    font.setFamilies(TEXT_FONTS)              # the window's fonts (icon font last, for its glyphs)
    qt_app.setFont(font)
    app = App(mouse=args.mouse)
    try:
        return qt_app.exec()
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
