# Step: `capture.py` — guide

## Quick version

- **`capture.py` uses OpenCV only. No MediaPipe.** MediaPipe belongs in `landmarker.py`.
- Job: open the webcam, hand out frames in the format MediaPipe wants, close cleanly.
- Imports you need: `cv2`, and `numpy` only if you type-hint frames.
- Four things to get right: **BGR→RGB**, **the mirror decision**, **`.read()` can fail**, **always release**.
- On Windows, `cv2.VideoCapture(0)` can take 3-10 seconds to open. That's normal, not a bug.

---

## Why no MediaPipe here

Your architecture rule is that each file has one reason to change:

```
camera.py     -> knows about webcams          (OpenCV)
landmarker.py -> knows about face detection   (MediaPipe)
features.py   -> knows about eye maths        (pure Python)
```

If `capture.py` also ran MediaPipe, you couldn't test your camera without MediaPipe working, and you couldn't feed the landmarker a saved video file for testing. Keeping them apart means you can debug one at a time — which matters a lot when something doesn't work.

**Cost of this split:** more files, and a tiny bit of plumbing to connect them. Worth it.

---

## Imports for `capture.py`

```python
import cv2                    # everything camera-related
import numpy as np            # ONLY if you want to type-hint frames as np.ndarray
```

That's genuinely it. No mediapipe, no PySide6.

`cv2` comes from the `opencv-contrib-python` package already in your `pyproject.toml`. The package name and the import name being different is normal and catches people out.

---

## What to build

A `Camera` class with roughly this shape — you write the bodies:

```python
class Camera:
    def __init__(self, index: int = 0) -> None: ...
    def read_rgb(self) -> np.ndarray | None: ...
    def release(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args): ...
```

### Questions to work through

**Opening**
- [ ] `cv2.VideoCapture(0)` — what does the `0` mean? What would `1` be? What if you passed a filename string instead?
- [ ] It doesn't raise an exception when the camera fails to open. So how do you find out it failed? (Look for a method on the capture object that answers this.)
- [ ] Where should you check that — in `__init__`, or the first time you read? Which gives a clearer error message?

**Reading a frame**
- [ ] `cap.read()` returns **two** things. What are they? Why two — what's the first one for?
- [ ] When can the first value be `False` even though the camera opened fine? (Think: unplugged mid-run, another app grabbed the camera, end of a video file.)
- [ ] Your method returns `np.ndarray | None`. What does `None` mean to the caller, and what should they do about it?

**Colour order — the silent bug**
- [ ] OpenCV frames come back in **BGR** order. MediaPipe wants **RGB**. What happens if you skip the conversion — a crash, or just worse detection?
- [ ] Find the OpenCV function that converts between colour spaces. What constant do you pass it?
- [ ] Why convert *here* rather than in `landmarker.py`? (Hint: how many places would need to remember, if you did it later?)

**Shutting down**
- [ ] What happens if your program exits without releasing the camera? Try it — then try to open the camera again.
- [ ] `__enter__` / `__exit__` let you write `with Camera() as cam:`. What does `__enter__` need to return? When exactly does `__exit__` run — including when your code crashes?

**Frame size (optional, do it later)**
- [ ] `cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)` requests a resolution. Why "requests"? What should you do to find out what you actually got?
- [ ] Bigger frames = more detail for the iris, but slower. Which matters more for your project?

---

## The mirror decision — think about this one properly

A raw webcam image is **not** mirrored: it shows you as other people see you. Most video apps flip it horizontally so it behaves like a mirror.

Work it out:

- [ ] You're facing the camera and you look at the **right** edge of your screen. In the *unflipped* camera image, does your iris appear to move left or right? (Draw it if you need to — remember the camera is facing you, so your right is the image's left.)
- [ ] Now with a horizontal flip applied. Which way does it move?
- [ ] Which of those two will be easier for *you* to sanity-check when you're staring at printed feature numbers?

**Does it affect the ML?** Barely — a flip is just a sign change, and the regression will learn either version equally well.

**What it does affect:**
1. **Your ability to debug.** If left/right feel backwards, you'll waste hours doubting correct code.
2. **Consistency — this one can break everything.** If you calibrate flipped and predict unflipped, your model's x-axis is inverted and the gaze dot runs the wrong way. The bug looks like "my model is bad" rather than "my image is mirrored", so it's genuinely hard to find.

**The rule:** decide once, do it in `capture.py`, and never flip anywhere else. Write your choice in your Design section.

- [ ] Find the OpenCV function that flips an image. What does its second argument control?

---

## Windows gotchas

- **Slow first open.** `cv2.VideoCapture(0)` can take 3-10 seconds on Windows — it's the default MSMF backend probing devices. Passing `cv2.CAP_DSHOW` as a second argument usually opens much faster. Trade-off: DirectShow is older and can offer different resolutions. Try the default first; only switch if the wait annoys you.
- **Camera in use.** If Teams/Zoom/Camera app has the webcam, `isOpened()` may be `True` but reads fail or the frame is black. Close other apps before blaming your code.
- **Privacy setting.** Windows Settings → Privacy & security → Camera must allow desktop apps. If every read fails immediately, check here before debugging further.

---

## "No loop in the class" — what that means

### The rule

**`Camera` hands out ONE frame per call. It never loops.**
Whoever *uses* `Camera` writes the loop.

Think of a vending machine: you press the button, you get one drink. It doesn't spray drinks at you continuously, and it doesn't decide what you do with them.

### ❌ Loop inside the class

```python
class Camera:
    def run(self):
        while True:
            ok, frame = self._cap.read()
            cv2.imshow("preview", frame)
            if cv2.waitKey(1) == ord("q"):
                break

# using it:
cam = Camera()
cam.run()          # never returns. Your program is stuck here.
```

What's wrong:

- **It never returns.** Nothing after `cam.run()` can ever execute.
- **It decides how to display.** Now your camera file imports window code — the layering rule is broken.
- **Only one behaviour.** It always shows a preview. What if you want to feed frames to MediaPipe instead? Or draw calibration dots? You'd need `run_with_landmarks()`, `run_calibration()`, `run_qt()`... one method per use.
- **Qt won't accept it.** Qt runs its *own* event loop. Two infinite loops can't both be in charge.
- **Untestable.** You can't unit-test a function that never returns.

### ✅ Loop outside the class

```python
class Camera:
    def read_rgb(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # no while, no imshow, no waitKey
```

Now the same class serves every part of your app, because each caller brings its own loop:

```python
# Test script - display it
while True:
    frame = cam.read_rgb()
    if frame is None: break
    cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) == ord("q"): break
```

```python
# Landmarker test - detect a face
while True:
    frame = cam.read_rgb()
    if frame is None: break
    result = landmarker.process(frame, timestamp)
    print(result.face_found)
```

```python
# Qt worker - report to the UI
def run(self):
    while self._running:
        frame = self._camera.read_rgb()
        if frame is None: break
        self.frame_ready.emit(frame)     # no window code here either
```

**Same `Camera` class in all three.** You never edit it. That's the whole point.

### The principle

> A class that **supplies** data shouldn't decide **when** it's asked for it, or **what happens** to it afterwards.

`Camera` is a supplier. The loop belongs to the consumer, because only the consumer knows what it's doing with the frames and when to stop.

**Cost, honestly:** every caller has to write its own `while` loop, so you repeat that structure ~3 times across the project. A single `run()` would have been less typing. The trade is duplication now versus flexibility later — and since you already know you need three different consumers, flexibility wins here. Worth stating exactly that way in your Design section.

---

## How to test it

Before `landmarker.py` exists, write a **throwaway script outside your package** (not in `eyetracking/`):

- [ ] Loop: read a frame, show it with `cv2.imshow`, break on a keypress.
- [ ] ⚠️ `cv2.imshow` expects **BGR**. Your `read_rgb()` returns RGB — so the preview will look blue-tinted unless you convert back. That's not a bug; it's proof your conversion is working.
- [ ] Check: is the image mirrored the way you decided?
- [ ] Unplug or block the camera mid-run — does your code handle the failed read without crashing?

---

# Furthermore — how MediaPipe works (preview for the next step)

You'll need this for `landmarker.py`, not `capture.py`. Short version so it's not a black box.

## The idea

MediaPipe is a library of **already-trained** ML models packaged as "Tasks". You never train anything. You:

1. Download a `.task` file — the trained model's weights. **You already have this**: `models/face_landmarker.task` (3.7 MB).
2. Create a task object, pointing it at that file.
3. Feed it images. Get structured results back.

It runs on CPU, in real time, and the "AI" part is entirely inside the `.task` file.

## What Face Landmarker gives you per frame

| Output | What it is | Enabled by |
|---|---|---|
| `face_landmarks` | ~478 3D points on the face, **including the iris** | always on |
| `face_blendshapes` | ~52 expression scores 0-1 (e.g. `eyeBlinkLeft`) | `output_face_blendshapes=True` |
| `facial_transformation_matrixes` | 4×4 matrix → head rotation/position | `output_facial_transformation_matrixes=True` |

**For gaze, `face_landmarks` is the one that matters** — that's the data your old `landmarker.py` was throwing away.

Landmark coordinates are **normalized 0-1**, not pixels: `x=0.5` means halfway across the frame. Multiply by frame width/height to get pixels. This makes them resolution-independent — useful, and a nice detail for your report.

## Running modes

| Mode | Use | Notes |
|---|---|---|
| `IMAGE` | single photos | no tracking between frames |
| `VIDEO` | frame sequences | tracks across frames — smoother, faster. **Use this.** |
| `LIVE_STREAM` | async callback | fastest, but callbacks add complexity |

`VIDEO` mode needs a **strictly increasing** timestamp per frame. Pass the same value twice and it throws.

- [ ] Where will you get that timestamp from? (`time` module? a frame counter? What are the pros and cons of each?)

## Imports for `landmarker.py` (next step, not now)

```python
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
```

- `mp.Image` — MediaPipe's own image wrapper; you'll wrap your numpy frame in it
- `mp_python.BaseOptions` — points at the `.task` file
- `vision.FaceLandmarkerOptions`, `vision.FaceLandmarker`, `vision.RunningMode` — the task itself

Note this is the modern **Tasks API**. Most tutorials online use the old `mp.solutions.face_mesh` API, which is deprecated and looks completely different. If example code you find doesn't match the imports above, it's the old API — don't mix them.

**Exception:** `mp.solutions.face_mesh` still contains useful *constants* (`FACEMESH_LEFT_IRIS`, etc.) listing which landmark indices belong to which facial feature. Borrowing those constants is fine — just don't use the old detection API.
