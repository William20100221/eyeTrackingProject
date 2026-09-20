# 3-Week Schedule — 20 Sep to 11 Oct 2026

*~1–2 hours/day. 21 days, with 3 buffer days built in.*

## Quick version

| Week | Goal | Milestone |
|---|---|---|
| **1** (Sep 20–26) | See the eyes | Features respond to your gaze |
| **2** (Sep 27–Oct 3) | Make it predict | Predicted (x, y) follows your eyes |
| **3** (Oct 4–11) | Prove it + write it | Accuracy numbers + finished report |

**Two decisions that make this fit:**

1. **Drop Qt entirely.** Use OpenCV fullscreen windows for calibration and the gaze dot. Saves ~5 days.
2. **Write the report daily**, 10 minutes at the end of each session. Not in week 3.

**Hard checkpoints — if you miss these, cut scope immediately:**
- **Day 7 (Sep 26):** features move sensibly when you look around
- **Day 13 (Oct 2):** live predicted (x, y) appears on screen

---

## The Qt decision

You have PySide6 installed and the original plan used it for calibration and a transparent overlay. **Skip it.**

- **For skipping:** Qt adds threading (`QThread`, signals), a second event loop, widget painting, and a fullscreen window class — realistically 4–6 of your remaining days, spent on presentation rather than on gaze tracking. OpenCV gives you fullscreen in two lines (`cv.namedWindow` + `cv.setWindowProperty` with `WINDOW_FULLSCREEN`), and you already know OpenCV.
- **Against skipping:** a transparent always-on-top overlay over *other* applications is genuinely not possible in OpenCV — you'd get a fullscreen window with a dot instead. The app looks like a demo rather than a product.
- **Verdict:** the assessment marks your gaze tracking, your testing and your reasoning — not window polish. A working tracker with measured accuracy beats a beautiful UI wrapped around an untested model.

**Write this in your report as a deliberate decision**, not an omission:

> Given the remaining time, I scoped the interface down to OpenCV windows rather than a Qt overlay. This protected the core gaze-estimation work and the testing phase. The cost is that the gaze marker is shown in a dedicated fullscreen window rather than overlaid on other applications — a presentation limitation, not a functional one.

That reads as engineering judgement. "I ran out of time" does not.

---

# Week 1 — See the eyes (Sep 20–26)

### Day 1 — Sun 20 Sep · Finish `landmarker.py`
- Add `running_mode=vision.RunningMode.VIDEO` to your options
- Uncomment `detect()`. Replace `mp.Image.create_from_file(...)` — that's for file paths. You have a numpy array, so use the `mp.Image(image_format=..., data=...)` form
- Call `detect_for_video(image, timestamp_ms)`. Where does the strictly-increasing timestamp come from?
- Clean up the duplicate imports (lines 14/18, 15/17) and the ignored `MODEL_PATH` parameter
- **Done when:** `print(len(result.face_landmarks[0]))` shows **478**. If it's 468, iris landmarks are off — find the option that enables them.

### Day 2 — Mon 21 Sep · Draw the landmarks
- Run `drawing.py` first. If the `drawing_utils` / `drawing_styles` imports fail, don't fight it — write your own loop with `cv.circle` over all 478 points. Simpler, and you'll understand it.
- Landmarks are normalized 0–1; `cv.circle` needs integer pixels. `frame.shape` gives you what you need — what are its three values, in what order?
- **Done when:** dots cover your face and track it. **Screenshot it.**

### Day 3 — Tue 22 Sep · Find the eye landmarks
- Print anything in `mediapipe.solutions.face_mesh` with `IRIS` or `EYE` in the name
- Those are sets of *connection pairs* — how do you get unique point indices out of them?
- Draw **only** iris + eye-corner points, in a different colour
- **Done when:** the coloured dots sit on your irises and eye corners. Write the index numbers in your dev log.

### Day 4 — Wed 23 Sep · `core/features.py`
- Iris centre = average of the iris ring points
- Normalise: iris position relative to the eye corners, divided by eye width
- Add `EyeFeatures` to `models.py`
- **Done when:** features print live, once per frame.

### Day 5 — Thu 24 Sep · Test the features (gaze)
- Look far left, far right, up, down
- **Do the numbers move consistently and in the same direction each time?**
- If they don't, fix it now. Everything downstream depends on this.

### Day 6 — Fri 25 Sep · Test the features (head)
- Keep staring at one point, slide your head left/right, then lean closer
- **Do the features stay roughly still?** They should. If they swing with your head, your normalisation isn't cancelling head movement — revisit Day 4.

### Day 7 — Sat 26 Sep · ⚑ CHECKPOINT + buffer
- Catch up on anything unfinished
- Write up week 1 in your dev log
- **If features don't work yet, stop and fix them.** Do not move on. A model trained on bad features cannot work.

---

# Week 2 — Make it predict (Sep 27–Oct 3)

### Day 8 — Sun 27 Sep · Data shapes + `calibration.py`
- `models.py`: add `GazePoint`, `CalibrationSample`
- `calibration.py`: `CalibrationSession` — tracks which dot is showing, collects samples. **No drawing code in it.**

### Day 9 — Mon 28 Sep · Fullscreen dot
- OpenCV fullscreen window, black, one white dot at a known position
- Decide your dot grid — **9 (3×3) is the sweet spot.** 5 is too few for accuracy; 16 takes long enough that attention drifts.

### Day 10 — Tue 29 Sep · Full calibration flow
- Loop over all 9 dots: show → wait ~1s for the eye to settle → collect ~30 frames → next
- **The settle delay matters.** Collect immediately and you record the eye mid-movement, which poisons the training data.
- What should happen if the face is lost or they blink mid-collection?

### Day 11 — Wed 30 Sep · `services/storage.py`
- Save/load samples as JSON
- Run a real calibration and get a data file
- **Do this before the model.** Otherwise every experiment means re-calibrating, and you'll lose hours.

### Day 12 — Thu 1 Oct · `core/gaze_model.py`
- Add `scikit-learn` to `pyproject.toml` first
- `fit(samples)`: build X (features) and y (targets), call `LinearRegression`. It handles both x and y outputs in one model.
- `predict(features) -> GazePoint`
- ~30 lines. Smaller than you expect.

### Day 13 — Fri 2 Oct · ⚑ CHECKPOINT — live prediction
- Load the saved calibration, train, print predicted (x, y) live
- **Does it roughly follow where you're looking?** It'll be rough. That's fine — rough is the milestone.

### Day 14 — Sat 3 Oct · Buffer + dev log

---

# Week 3 — Prove it + write it (Oct 4–11)

### Day 15 — Sun 4 Oct · Gaze dot on screen
- Fullscreen window, circle drawn at the predicted point
- This is your demo. Record a short screen capture.

### Day 16 — Mon 5 Oct · Smoothing + measurement script
- `core/smoothing.py` — rolling average over the last N predictions. ~15 lines, huge visual improvement.
- Note the trade: more smoothing = steadier but laggier
- Write a script that shows a dot at a known point, records the prediction, and computes the error in pixels

### Day 17 — Tue 6 Oct · Measure accuracy
- Run the measurement across ~9 screen positions. Average error in pixels = **your headline number.**
- Train a second model (polynomial features, or `Ridge`) on the *same* saved data and compare
- **Two models compared with numbers is far stronger evidence than one model that "seems to work."**

### Day 18 — Wed 7 Oct · Find where it breaks
- Move your head after calibrating. Dim the lights. Glasses on/off. Try another person.
- Record what happens. **Honest limitations score better than pretending there are none.**

### Day 19 — Thu 8 Oct · Tests + cleanup
- 3–4 unit tests for `core/` (`analysis.py`, `features.py`, `calibration.py`) — pure functions, no webcam needed. This is the payoff for your layered architecture.
- Delete `main.py` (the broken 3×`QApplication` scratch), `play.py`, `path_test.py` if unused
- Remove unused `import numpy as np` from `camera.py` and `capture.py`

### Day 20 — Fri 9 Oct · Report, part 1
Introduction · Research · Requirements · Design
*(structure is in `documentation_structure.md`)*

### Day 21 — Sat 10 Oct · Report, part 2
Development Log · Testing · Evaluation · References

### Sun 11 Oct · Final buffer
Re-read, fix gaps, submit.

---

## Furthermore — how to not fall behind again

**Write the dev log daily, 10 minutes.** Date, what you did, what broke, what you decided and why. This is the single highest-value habit here: the Development Log and Evaluation sections are worth real marks, and they're built from details you will not remember in three weeks. It's also the part that collapses first under time pressure — which is exactly why it goes at the end of each session, not at the end of the project.

**If you lose days again, cut in this order:**
1. Smoothing (Day 16) — cosmetic
2. The second model comparison (Day 17) — nice evidence, not essential
3. Unit tests (Day 19) — down to one or two
4. The gaze-dot display (Day 15) — printing coordinates proves the same thing

**Never cut:** feature testing (Days 5–6), accuracy measurement (Day 17), or the report (Days 20–21). A tracker with no measured accuracy has nothing to evaluate, and an unwritten report scores zero regardless of how good the code is.

**The two checkpoints are real.** If features don't work on Day 7, or nothing predicts by Day 13, tell me and we re-plan then — not in week 3.
