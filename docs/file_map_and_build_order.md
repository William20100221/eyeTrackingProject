# File Map & Build Order

## Quick version

**What exists and works:** `core/models.py` (data shapes), `core/analysis.py` (blink logic). That's it.

**What's empty:** `services/camera.py`, `services/landmarker.py`, `ui/main_window.py`, `ui/worker.py`, `ui/gaze_overlay/`, `main.py` (scratch — delete it).

**What doesn't exist yet but must:** `core/features.py`, `core/gaze_model.py`, `core/calibration.py`, `ui/calibration_window.py`.

**Build order:** camera → landmarker → features → calibration → model → overlay. Each step should *visibly work* before you start the next.

**Next step:** get a live webcam window with face landmarks drawn on it, using a throwaway OpenCV script — not Qt yet.

---

# Part A — What each file does *right now*

## `main.py` — 11 lines, broken, delete it

```python
app  = PySide6.QtWidgets.QApplication([])
app2 = QApplication([])          # RuntimeError: only one QApplication allowed
app3 = QtWidgets.QApplication([])
```

Scratch code testing three PySide6 import styles. Qt permits exactly one `QApplication` per process, so line 9 raises. Nothing imports it. This will become your real entry point later.

## `eyetracking/core/models.py` — 17 lines, works

Defines the **shape of your data**:

```python
@dataclass(frozen=True)
class HeadPose:
    yaw, pitch, roll: float

@dataclass(frozen=True)
class FrameResult:
    timestamp_ms: int
    face_found: bool
    head_pose: HeadPose | None
    blink_score_left: float
    blink_score_right: float
```

`frozen=True` = immutable once created. Every other module agrees on these shapes, so they can be developed independently.

**Limitation:** `FrameResult` carries no landmark or eye data. Gaze estimation cannot be built on this as-is — extending it is unavoidable.

## `eyetracking/core/analysis.py` — 25 lines, works

Three pure functions over `FrameResult`:

| Function | Does | Note |
|---|---|---|
| `is_blinking(result)` | both blink scores > 0.5 | `and` = wink doesn't count |
| `count_blinks(results)` | counts not-blinking → blinking transitions | no debouncing |
| `is_looking_away(result, yaw_limit=30)` | `abs(yaw) > 30`, or no face | head direction only, not gaze |

No imports outside `.models` — this is the layering rule working correctly. Testable with hand-written data, no webcam.

## Everything else — empty

`services/camera.py`, `services/landmarker.py`, `ui/main_window.py`, `ui/worker.py`, `ui/gaze_overlay/__init__.py`, and all the `__init__.py` package markers.

`models/face_landmarker.task` (3.7 MB) is present — the MediaPipe model file is downloaded and ready.

---

# Part B — The files you need

Signatures below are **specifications, not code to copy**. They say what each piece is responsible for; you decide how.

## `core/` — pure logic, no MediaPipe / no Qt / no OpenCV

### `models.py` (extend the existing file)

```python
@dataclass(frozen=True)
class EyeFeatures:
    """The numbers your model learns from. ~4-8 floats."""
    # your chosen features go here

@dataclass(frozen=True)
class GazePoint:
    x: float          # screen coords, 0..1 or pixels — pick one and document it
    y: float

@dataclass(frozen=True)
class CalibrationSample:
    features: EyeFeatures
    target: GazePoint     # where the dot WAS — the "right answer"
```

`FrameResult` also needs a way to carry eye data. Two options, both defensible:

- **Add `features: EyeFeatures | None`** — clean, small, but ties the landmarker to your feature choices.
- **Add raw `landmarks`** — flexible (you can change features later without re-recording), but leaks a MediaPipe-shaped list into `core/`.

Whichever you pick, say why in your report.

### `features.py` — NEW, the critical file

```python
def extract_features(landmarks) -> EyeFeatures:
    """Landmark list -> a few numbers describing where the eyes point."""

def _iris_centre(landmarks, iris_indices) -> tuple[float, float]:
    """Average of the iris ring points."""

def _normalise(iris, corner_a, corner_b) -> tuple[float, float]:
    """Iris position relative to eye corners, scaled by eye width.
    This is what makes the features head-position independent."""
```

Everything downstream depends on this being right. See `step_feature_extraction_worksheet.md`.

### `calibration.py` — NEW

```python
class CalibrationSession:
    def __init__(self, points: list[GazePoint]): ...
    def current_target(self) -> GazePoint | None
    def add_sample(self, features: EyeFeatures) -> None
    def next_target(self) -> None
    def is_complete(self) -> bool
    def samples(self) -> list[CalibrationSample]
```

Holds calibration *state* — which dot is showing, what's been collected. Deliberately knows nothing about drawing; the UI asks it what to show. That split means you can unit-test the calibration flow without opening a window.

### `gaze_model.py` — NEW, the ML

```python
class GazeModel:
    def fit(self, samples: list[CalibrationSample]) -> None:
        """Train. Internally: build X (features) and y (targets), call sklearn."""

    def predict(self, features: EyeFeatures) -> GazePoint:
        """Live prediction."""

    def is_trained(self) -> bool
    def save(self, path) -> None
    def load(self, path) -> None
```

Note: **two** regressors — one predicting screen x, one predicting screen y — or one multi-output model. sklearn's `LinearRegression` handles multi-output directly.

### `smoothing.py` — NEW, optional but cheap

```python
class Smoother:
    def update(self, point: GazePoint) -> GazePoint:
        """Average recent predictions to kill jitter."""
```

Raw per-frame predictions jump around. A rolling average or exponential smoother makes the dot look far better for ~15 lines. Trade-off: more smoothing = steadier but laggier. Worth a sentence in Evaluation.

## `services/` — talks to the outside world

### `capture.py`

```python
class Camera:
    def __init__(self, index: int = 0): ...
    def read_rgb(self) -> ndarray | None   # OpenCV gives BGR; MediaPipe wants RGB
    def release(self) -> None
    def __enter__ / __exit__               # so `with Camera() as cam:` works
```

The BGR→RGB conversion is a classic silent bug: nothing crashes, detection just gets worse. Do it here, once.

### `landmarker.py` (you're rewriting this)

```python
class Landmarker:
    def __init__(self, model_path): ...
    def process(self, rgb_frame, timestamp_ms: int) -> FrameResult
    def close(self) -> None
```

Changes from the old version:
- **Keep the landmarks.** The old one discarded all 478 points — that was the fatal flaw.
- **Check `face_landmarks`, not `face_blendshapes`,** for whether a face was found.
- **Verify the yaw/pitch mapping empirically** rather than trusting the formula (the old code appears to have had them swapped).
- Add `close()` — MediaPipe holds native resources.
- Ensure iris landmarks are enabled (you want 478 points, not 468).

⚠️ `detect_for_video()` requires timestamps that **strictly increase**. Repeat or go backwards and it throws.

### `storage.py` — NEW, optional

```python
def save_calibration(samples, path) -> None
def load_calibration(path) -> list[CalibrationSample]
```

Lets you re-train without re-calibrating. Big time-saver while developing, and gives you a fixed dataset to compare model types against — which is exactly the evidence your Testing section wants.

## `ui/` — the screen

### `worker.py`

```python
class FrameWorker(QObject):
    frame_ready = Signal(object)     # FrameResult
    gaze_ready  = Signal(object)     # GazePoint

    def run(self) -> None            # the capture loop
    def stop(self) -> None
```

**Why this file must exist:** Qt has one UI thread. Reading the webcam and running MediaPipe takes ~20-40ms per frame. Do that on the UI thread and the whole app freezes. The worker runs the loop on a `QThread` and reports results via signals, which Qt delivers safely across threads.

Cost: threading is the most bug-prone part of the project. Never touch a widget from inside `run()` — emit a signal instead.

### `main_window.py`

```python
class MainWindow(QMainWindow):
    # video preview, "Calibrate" button, "Start tracking" button,
    # status text, starts/stops the worker
```

### `calibration_window.py` — NEW

```python
class CalibrationWindow(QWidget):
    # fullscreen, black, draws one dot at a time
    # per dot: show -> wait ~1s for the eye to settle -> collect ~30 frames -> next
    finished = Signal(list)      # the collected samples
```

Fullscreen matters: the dot's on-screen position *is* the training label, so you need the whole screen and known coordinates.

The settle-then-collect timing is a real design decision — collect too early and you record the eye mid-movement, which poisons your training data.

### `gaze_overlay/`

```python
class GazeOverlay(QWidget):
    # frameless, always-on-top, transparent window
    # draws a dot at the predicted gaze point
    def set_point(self, point: GazePoint) -> None
```

## Root

### `main.py` — replace the scratch entirely

```python
def main() -> None:
    app = QApplication(sys.argv)    # exactly one
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
```

### `tests/` — NEW, small but high value

`test_analysis.py`, `test_features.py`, `test_calibration.py`. Your `core/` layer is pure functions over dataclasses — perfect for testing, no webcam needed. This is the payoff for the layered architecture, and it's directly worth marks.

---

# Part C — Build order

Rule: **each step must visibly work before you start the next.** Never build three things then debug them together.

| # | Goal | Files | Done when |
|---|---|---|---|
| 1 | See yourself with landmarks drawn | `capture.py`, `landmarker.py`, throwaway script | Dots track your face live |
| 2 | Eye features that make sense | `features.py`, `models.py` | Numbers move sensibly as you look around |
| 3 | Collect calibration data | `calibration.py`, `calibration_window.py` | Dots appear; samples saved to disk |
| 4 | Train and predict | `gaze_model.py` | Predicted (x,y) printed live |
| 5 | See the gaze dot | `gaze_overlay/`, `worker.py`, `main_window.py` | Dot follows your eyes on screen |
| 6 | Measure and polish | `smoothing.py`, `tests/` | Accuracy numbers for your report |

## The big early decision: OpenCV window or Qt?

**Recommendation: OpenCV (`cv2.imshow`) for steps 1-2, Qt from step 3 on.**

- **For OpenCV first:** a ~20-line script gets you a live window immediately. No threads, no signals, no event loop. If landmarks don't appear you know it's MediaPipe or your camera — nothing else can be at fault. You'll be debugging feature maths in step 2, and you want *zero* unrelated complexity while doing it.
- **Against:** it's throwaway code you'll delete, and it means writing the display twice.
- **Why Qt still wins later:** fullscreen calibration, transparent always-on-top overlays, and buttons are painful-to-impossible in OpenCV and natural in Qt.
- **Why not Qt from the start:** if landmarks don't show up, the cause could be the camera, MediaPipe, the thread, the signal, or the paint code. Five suspects instead of two. With a month's deadline, that's the expensive path.

Delete the throwaway script once step 2 passes — but **screenshot it first**, it's Development Log evidence.

---

# Part D — Missing dependencies

`pyproject.toml` currently lists `mediapipe`, `opencv-contrib-python`, `pyside6`. You'll also need:

- **`scikit-learn`** — the regression model (step 4)
- **`numpy`** — array maths (arrives with mediapipe/opencv anyway, but declare it if you import it directly)
- **`pytest`** — for `tests/` (dev dependency)

Add them before step 4 so you don't stall mid-step.
