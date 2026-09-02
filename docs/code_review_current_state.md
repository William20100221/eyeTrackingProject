# What's in the project so far — and why

## Quick version

**The architecture (3 layers) — good choice**
- `core/` = pure logic, no camera/UI. `services/` = talks to outside world (MediaPipe, camera). `ui/` = screen.
- Good: each layer testable on its own; you can test blink logic with fake data, no webcam needed.
- Bad: more folders/files for a small project — extra navigation overhead.

**`models.py` — dataclasses `FrameResult` / `HeadPose`**
- Why: gives every frame a fixed, named shape instead of loose tuples/dicts.
- Good: `frozen=True` means nothing can accidentally change a result later; typed fields catch mistakes.
- Bad: **it has no landmark data** — this is the blocker for your gaze work.

**`landmarker.py` — MediaPipe wrapper**
- Why: hides all MediaPipe detail behind one `process()` method.
- Good: if you swap MediaPipe out later, only this file changes.
- Bad: it *throws away* the 478 landmarks and returns only 4 numbers. Gaze needs those landmarks.
- ⚠️ **Likely bug:** `yaw` and `pitch` look swapped in `_matrix_to_pose`. Test before trusting.

**`analysis.py` — blink counting + looking away**
- Good: pure functions (data in → answer out), trivially unit-testable. Great report evidence.
- Bad: `BLINK_THRESHOLD = 0.5` is a guess, same for everyone; no debouncing on `count_blinks`.

**Not written yet:** `capture.py`, `worker.py`, `main_window.py`, `gaze_overlay/` — all empty.

**Should delete:** `step1.py` (dead stub), `main.py` (scratch — it would crash).

---

## Furthermore — detail on each

### The layered architecture

Files are split into `core/`, `services/`, `ui/`. This is a standard pattern (sometimes called layered or hexagonal architecture).

The rule being followed: **`core/` never imports from `services/` or `ui/`.** Look at `analysis.py` — it only imports `.models`. No MediaPipe, no camera, no Qt.

- **Why this is good:** you can run and test your blink logic by handing it a list of hand-written `FrameResult` objects. No webcam, no waiting, no "is my face lit properly". This makes automated testing possible, which is directly worth marks in your Testing section.
- **Why it costs you:** for a project this size, it's arguably over-engineered. You now have 4 folders and several `__init__.py` files to hold ~60 lines of real code. A single-file script would have been faster to write.
- **Verdict for your report:** defensible — the cost is small and it's the reason your logic is testable. Say exactly that, including the cost.

### `models.py` — why dataclasses?

```python
@dataclass(frozen=True)
class FrameResult:
    timestamp_ms: int
    face_found: bool
    head_pose: HeadPose | None
    blink_score_left: float
    blink_score_right: float
```

The alternative would be passing a plain dict `{"blink_left": 0.3, ...}` or a tuple `(123, True, pose, 0.3, 0.4)`.

- **Good — named fields:** `result.blink_score_left` says what it is. `result[3]` does not. Typos become errors instead of silent bugs.
- **Good — `frozen=True`:** the object can't be modified after creation. This prevents a whole class of bug where some function quietly changes a frame's data and you can't find where.
- **Good — `head_pose: HeadPose | None`:** the type itself documents that pose may be missing (no face), forcing you to handle it.
- **Bad — rigidity:** every new piece of data means editing the class. You'll hit this immediately when you add gaze features.
- **The real problem:** the landmark positions aren't stored. MediaPipe computes 478 3D points per face, this class keeps none of them. **You cannot do gaze estimation from this data.** Changing that is your next step.

### `landmarker.py` — the wrapper pattern

The class hides MediaPipe behind one method: `process(rgb_frame, timestamp_ms) -> FrameResult`.

- **Good — one place to change:** MediaPipe's API is verbose and version-unstable. All of that mess lives in one file. The rest of your code just sees `FrameResult`.
- **Good — `RunningMode.VIDEO`:** this mode tracks the face across frames instead of re-detecting from scratch, so it's faster and less jittery than `IMAGE` mode.
- **Bad — data loss:** it converts rich output down to 4 numbers and discards the rest. Fine for a blink detector, fatal for a gaze tracker.
- **Bad — face-found check is indirect:** it tests `if not raw.face_blendshapes`. That only works because `output_face_blendshapes=True` is set. Turn that option off and the code silently reports "no face" forever. Checking `raw.face_landmarks` would be more honest.
- **Bad — no cleanup:** `FaceLandmarker` holds native (C++) resources and has a `.close()`. Nothing calls it. Wrapping in a context manager (`__enter__`/`__exit__`) would be tidier.
- **Bad — brittle path:** `parents[2]` counts folders upward to find `models/`. Move this file one folder and it breaks with a confusing error.
- **Gotcha — timestamps must always increase.** `detect_for_video` throws if you pass a timestamp that's the same as or lower than the previous one. Worth knowing before you write the camera loop.

### The likely yaw/pitch bug

```python
pitch = math.degrees(math.asin(-m[2][0]))
yaw   = math.degrees(math.atan2(m[2][1], m[2][2]))
roll  = math.degrees(math.atan2(m[1][0], m[0][0]))
```

This is the standard formula for pulling Euler angles out of a rotation matrix. In that standard formula:
- `asin(-m[2][0])` gives the rotation about the **vertical (Y) axis** — that's **yaw** (shaking your head "no")
- `atan2(m[2][1], m[2][2])` gives the rotation about the **side-to-side (X) axis** — that's **pitch** (nodding "yes")

So the two labels look swapped. `roll` is correct.

**Why it matters:** `is_looking_away()` in `analysis.py` checks `abs(head_pose.yaw) > 30`. If `yaw` actually contains pitch, then looking up/down triggers "looking away" and turning your head left/right doesn't.

**How to test it (do this, don't take my word for it):** print all three values live. Turn your head left and right — whichever number swings is yaw. Nod up and down — whichever swings is pitch. Sign conventions vary between libraries, so an empirical check is the right move regardless.

**Report value:** this is an excellent Development Log entry — "I inherited/wrote code with swapped axes, found it by empirical testing, fixed it." Showing you found and fixed a bug through testing scores better than code that was never questioned.

### `analysis.py` — the blink logic

```python
def is_blinking(result):
    return (result.blink_score_left > 0.5 and result.blink_score_right > 0.5)
```

- **Good — pure functions.** Input data → output answer, no side effects, no hidden state. The easiest kind of code to test and to explain.
- **`and` not `or`:** requires *both* eyes closed, so a wink isn't counted as a blink. Correct for blink counting — worth stating as a deliberate choice in your report.
- **Bad — magic threshold.** `0.5` is arbitrary and identical for everyone. Eyelid shape, glasses, and camera angle all shift blendshape scores. A per-user calibrated threshold would be better; at minimum, justify the 0.5 with a measurement.

```python
def count_blinks(results):
    # counts rising edges: not-blinking -> blinking
```

- **Good — edge detection.** Counting transitions (not frames) is the right approach; a 5-frame blink counts as 1, not 5.
- **Bad — no debouncing/hysteresis.** If a score hovers right at 0.500 it can flicker above/below across frames and inflate the count. The standard fix is two thresholds (e.g. must exceed 0.6 to start a blink, must drop below 0.4 to end it).

```python
def is_looking_away(result, yaw_limit=30.0):
    if not result.face_found or result.head_pose is None:
        return True
```

- **Good — safe default.** Face missing → assume not looking. Fails in the harmless direction.
- **Bad — conflates two things.** "No face detected" and "head turned away" are different states with different causes (bad lighting vs. actual head turn), but both return `True`, so you can't tell them apart or debug them separately.
- **Bad — head pose ≠ gaze.** You can face straight ahead and look sideways with your eyes. This function measures head direction only. That limitation is exactly what your gaze model is meant to solve — good framing for your Introduction.

### Files to delete

**`step1.py`** — contains only `from mediapipe.tasks.python import *`. Nothing imports it. Wildcard imports are bad practice anyway (they dump unknown names into your namespace). Delete it.

**`main.py`** — creates three `QApplication` objects:
```python
app  = PySide6.QtWidgets.QApplication([])
app2 = QApplication([])   # RuntimeError here
```
Qt allows exactly **one** `QApplication` per process; the second call raises `RuntimeError`. This file is clearly scratch code trying out three different import styles — it was never meant to run. Replace it with your real entry point.

### The gap for gaze tracking

Your pipeline currently is:

```
camera (missing) → Landmarker → FrameResult{pose, blinks} → analysis
```

What gaze estimation needs:

```
camera → Landmarker → FrameResult{ + eye/iris landmarks } → features → model → (x, y)
```

The missing link is that `Landmarker.process()` sees the landmarks and drops them. Everything downstream is starved of the one thing gaze prediction depends on.

**A note on AI-generated code for your report:** the code you have is well-structured but *over-fitted to blink detection* — it made an early assumption about what data mattered and threw the rest away. That's a genuinely interesting Evaluation point: clean architecture doesn't protect you from designing the wrong data model, and the boundary between layers is exactly where that mistake gets locked in.
