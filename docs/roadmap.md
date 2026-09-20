# Roadmap — where you are and what's left

*Last updated 2026-09-07. Deadline is roughly early October — about 4 weeks.*

## Quick version

| Step | What | Files | Status |
|---|---|---|---|
| 1 | Live webcam preview | `services/camera.py`, `testing/capture.py` | ✅ **done** |
| 2 | Face landmarks drawn on your face | `services/landmarker.py` | ⬅️ **you are here** |
| 3 | Eye features that respond to gaze | `core/features.py`, `core/models.py` | ⬜ |
| 4 | Collect calibration data | `core/calibration.py`, `ui/calibration_window.py` | ⬜ |
| 5 | Train the model + predict | `core/gaze_model.py` | ⬜ |
| 6 | Gaze dot on screen | `ui/` — overlay, worker, main window | ⬜ |
| 7 | Measure accuracy + write it up | `tests/`, `core/smoothing.py`, report | ⬜ |

**Rough time budget:** steps 2-3 are ~1 week (the fiddly part), 4-5 ~1 week, 6 ~1 week, 7 the last week. If you fall behind, **step 6 is the one to cut down** — an OpenCV window showing the gaze dot proves the same thing as a polished Qt overlay.

**Next step:** get MediaPipe running and draw the 478 landmarks onto your camera preview.

---

## Step 2 — Landmarks (do this now)

**Goal:** the same preview window as step 1, but with dots drawn on your face that follow it as you move.

### What changes

Your loop grows by one participant:

```
Camera.read_rgb()  ->  Landmarker.process()  ->  draw dots  ->  imshow
```

`camera.py` doesn't change at all. That's the payoff for the split you built.

### Questions to work through

**Setup**
- [ ] The three imports are in `step_camera_guide.md` (Furthermore section). What is each one for?
- [ ] `FaceLandmarkerOptions` needs a path to `models/face_landmarker.task`. How do you build that path so it works no matter where the script is run from? (`pathlib.Path(__file__)` is the usual route — what do `.resolve()` and `.parents[n]` do?)
- [ ] Which `RunningMode` do you want, and why?

**The timestamp**
- [ ] `detect_for_video()` needs a **strictly increasing** integer, in milliseconds. Two options: a frame counter (`frame_no * 33`), or real time (`time.perf_counter()`).
- [ ] Which is more honest about what actually happened? Which is simpler? Pick one and note why.

**Reading the result**
- [ ] `process()` gives you a result object. How do you tell whether a face was found? (Not via blendshapes — that was the old code's mistake.)
- [ ] Print `len(result.face_landmarks[0])`. **You want 478.** If you get 468, the iris landmarks are missing and you'll need to find the option that enables them.
- [ ] Print one landmark. What are its fields, and what range are the numbers in?

**Drawing them**
- [ ] Landmarks are normalized 0-1. `cv.circle` needs integer pixels. What's the conversion?
- [ ] Where do you get the frame's width and height? (Hint: `frame.shape` — what are its three values, and in what order?)
- [ ] `cv.circle` wants BGR colour. Which image are you drawing onto — the RGB one or the BGR one? Does it matter for where the dots land?

### Done when

Dots cover your face and track it as you move. Two extra checks worth doing while you're there:

- [ ] **Count = 478?** Confirms iris landmarks are on.
- [ ] **Yaw/pitch check:** if you also extract head pose, print all three angles. Turn your head left-right — which number moves? Nod — which moves? The old code appeared to have these swapped. Finding that yourself is a strong Development Log entry.

**Screenshot it.**

---

## Step 3 — Eye features

**Goal:** a handful of numbers that change when your *eyes* move and stay still when only your *head* moves.

Full worksheet already written: `step_feature_extraction_worksheet.md`.

**Done when:** you print your features live, look far left then far right, and the numbers move consistently and in the same direction each time.

⚠️ **This is the step that decides whether the project works.** A model can only find patterns that already exist in your numbers. Don't rush past it — if features are bad, no amount of model tuning saves you.

---

## Step 4 — Calibration

**Goal:** show dots on screen, record features while the user looks at each one, save the pairs.

**This is where Qt starts.** Fullscreen black window, one dot at a time, known screen coordinates.

Key decisions to make and write down:
- **How many dots?** 5 (corners + centre) is the minimum, 9 (3×3 grid) is standard, 16 gives more data but takes longer and people's attention drifts.
- **Timing per dot.** Show it → wait ~1s for the eye to settle → collect ~30 frames. Collect too early and you record the eye *mid-movement*, which poisons the training data.
- **Rejecting bad samples.** Blinks and lost faces during collection — drop them or keep them?

**Also build `services/storage.py` here** — save the samples to a file. Otherwise every model experiment means re-calibrating, and you'll waste hours.

**Done when:** you run calibration and a data file appears with ~9 targets × ~30 samples in it.

---

## Step 5 — The model

**Goal:** learn features → screen (x, y).

This is smaller than you expect — probably 30 lines. `LinearRegression` handles multiple outputs, so one model predicts both x and y.

Concepts you'll need to explain in the report:
- **Features (X) vs targets (y)** — inputs vs the right answers
- **Fitting** — the computer choosing the line/curve that best matches your examples
- **Overfitting** — memorising your 9 dots instead of learning the pattern. With few calibration points and many features, this is a genuine risk
- **Train/test split** — hold back some calibration points, predict them, measure the error. This is how you get a *number* for your Evaluation section

Worth trying **two** models (e.g. `LinearRegression` vs polynomial features, or `Ridge`) and comparing measured error. Two models compared with numbers is far stronger evidence than one model that "seems to work."

**Done when:** predicted (x, y) prints live and roughly follows where you're looking.

---

## Step 6 — Show the gaze dot

**Goal:** a dot on screen that follows your eyes.

Needs `ui/worker.py` (QThread — keeps the UI from freezing), `ui/gaze_overlay/` (transparent always-on-top window), `ui/main_window.py` (buttons), and a real `main.py`.

**Threading is the most bug-prone part of the project.** The rule: never touch a widget from inside the worker's `run()` — emit a signal instead.

**If you're short on time, cut this down.** A full-screen OpenCV window with a drawn circle demonstrates the same result. Say in your report that you scoped the UI down to protect the core functionality — that's a defensible engineering decision, not a failure.

---

## Step 7 — Measure and write up

- **Accuracy:** show a dot at a known place, look at it, record the prediction, measure the error in pixels. Repeat across the screen. Average error is your headline number.
- **Where it fails:** move your head after calibrating, dim the lights, try glasses on/off, try another person. Record what breaks — limitations honestly stated score better than pretending there are none.
- **`core/smoothing.py`:** ~15 lines, makes the dot look far better. Note the trade: steadier vs laggier.
- **`tests/`:** unit tests for `analysis.py`, `features.py`, `calibration.py`. Your `core/` layer is pure functions over dataclasses — this is the payoff for the architecture.

---

## Things to do at some point (small, easy to forget)

- [ ] Delete or replace `main.py` — it's still the broken 3×`QApplication` scratch
- [ ] Check what `eyetracking/testing/play.py` is — leftover, or something you want?
- [ ] Add `scikit-learn` to `pyproject.toml` before step 5
- [ ] Remove the unused `import numpy as np` from `camera.py` and `capture.py`
- [ ] Add `__enter__`/`__exit__` to `Camera` so release is guaranteed
- [ ] Record a short test video and add a `FakeCamera` that replays it — same interface, no webcam, identical input every run. Makes measured comparisons possible, which Testing needs

---

## Furthermore — why this order

Each step ends with **something you can see**. That's deliberate.

The alternative — build all the files, then run it — means when it doesn't work you have six suspects and no way to isolate them. With this order, if step 3's numbers look wrong, step 2 already proved the landmarks were fine, so the bug is in the code you wrote most recently.

**The cost:** you write some throwaway code (`testing/capture.py` will eventually be replaced by the Qt version), and the project doesn't look impressive until quite late. That's a real trade-off, and you should say so in your Evaluation — but with a fixed deadline, always-having-something-that-works beats a big-bang integration that might fail in week 4.

**The other reason:** if you run out of time, you stop at whatever step you reached and still have a working demo of *that*. A half-built version of everything demos nothing.
