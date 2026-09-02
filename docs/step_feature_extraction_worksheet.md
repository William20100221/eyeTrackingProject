# Step: Eye Feature Extraction — worksheet

Work through these in order. Write your answers down — they become your Development Log.

## Quick version — what you're solving

- MediaPipe gives you ~478 face landmarks per frame. Your code currently throws them all away.
- You need to turn *some* of those points into a small set of numbers ("features") that describe **where the eyes are pointing**.
- Those numbers become the input to your regression model.
- The hard part is **not** getting the landmarks. It's choosing numbers that mean the same thing regardless of where your head is.

---

## Part 1 — Explore what you actually have

Don't write features yet. First look at the data.

- [ ] In `Landmarker.process()`, the MediaPipe result is the variable `raw`. What attributes does it have besides `face_blendshapes`? (`print(dir(raw))`)
- [ ] Find the one holding landmark positions. How many landmarks are in it?
- [ ] Print one landmark. What fields does it have? What's the range of the values — are they pixels, or something else?
- [ ] Does that range change if you resize the camera window? Why might MediaPipe have chosen that range?

**Write down:** what a single landmark looks like, and what its numbers mean.

## Part 2 — Find the eye landmarks

You need to know *which* of the ~478 indices are the iris and the eye corners.

- [ ] MediaPipe ships named groups of indices. Look inside `mediapipe.solutions.face_mesh` — print anything with `IRIS` or `EYE` in the name.
- [ ] Those constants are sets of *connections* (pairs of point indices). How would you get the unique point indices out of a set of pairs?
- [ ] Search for the "MediaPipe canonical face model" landmark index diagram to cross-check visually.
- [ ] Verify: draw your chosen points onto the camera image with OpenCV. Do they land on the iris and eye corners?

**Write down:** the index numbers you settled on, and how you verified them.

> ⚠️ Iris landmarks only exist if refinement is enabled. Check whether your `FaceLandmarkerOptions` is producing them — if you get exactly 468 landmarks instead of 478, that's your answer.

## Part 3 — The key problem (think before coding)

This is the part that decides whether your project works.

- [ ] Stare at a fixed point on screen. Now slide your head 10cm to the right, still staring at the same point. **Does the raw x-coordinate of your iris change?** Does your gaze target change?
- [ ] So: is a raw iris coordinate a good feature? What exactly is wrong with it?
- [ ] What could you measure the iris position **relative to**, so that sliding your head cancels out? (Hint: what else moves with your head at exactly the same rate?)
- [ ] Now lean 20cm closer to the camera. The whole face gets bigger in the image. What happens to your relative measurement? How do you cancel *that* out?
- [ ] Should your features include head pose (yaw/pitch) as well as eye position? Argue both sides.

**Write down:** your feature list, and *why each one is in it*. This is prime Design-section material.

## Part 4 — Build it

- [ ] Where should this code live — `core/` or `services/`? Justify using the layering rule (`core/` must not import MediaPipe).
- [ ] `FrameResult` has no landmark field. What's the minimum change to get eye data downstream?
- [ ] Write the function. Suggested shape: landmarks in → a small list/tuple of floats out.
- [ ] Test: print your features live. Look far left, far right, up, down. **Do the numbers move in a consistent, sensible way?** If they don't, your model can never learn from them.

---

## Furthermore — why this step matters more than the model

Beginners assume the ML model is the clever part. It usually isn't.

A regression model can only find patterns that **already exist in the numbers you feed it**. If your features don't cleanly separate "eyes moved" from "head moved", no model — not linear regression, not a neural network — can un-mix them. It will just learn noise and predict badly.

This is the single most common reason webcam gaze projects fail, and it's a strong Evaluation point for your report: *"most of my effort went into feature design rather than model choice, because the model can only be as good as its inputs."*

**Two design tensions worth recording, both genuinely double-sided:**

**More features vs. fewer.** Adding head pose, both eyes, iris size etc. gives the model more to work with. But every extra feature needs more calibration points to pin down — with only ~9 calibration dots, too many features means the model memorises your calibration and fails everywhere else (overfitting). Fewer features generalise better but may miss real effects.

**Normalising away head movement vs. keeping it.** If you fully cancel head position, the model handles head movement gracefully but loses information — because in real life, where your head points genuinely *does* correlate with where you look. If you keep raw head data, the model can exploit that correlation during calibration but breaks the moment you sit differently. There is no clean right answer; state which you chose and what it costs you.

**On testing your features:** the "look far left / far right" check in Part 4 is not a formality. A feature that doesn't change when your gaze changes is useless; a feature that changes when your gaze *doesn't* is worse than useless. Catching that now costs minutes. Catching it after you've built calibration and training costs days.
