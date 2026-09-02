# OpenCV (`cv2`) — what you actually need to know

## Quick version

**Does `capture.py` open the camera? Yes.** You can't convert an image you don't have. The full job is:

```
open camera  ->  grab frame  ->  convert BGR to RGB  ->  hand it out  ->  close camera
```

The colour conversion is one line. The rest is managing the camera.

**The 8 functions you need:**

| Function | Plain English |
|---|---|
| `cv2.VideoCapture(0)` | Turn the webcam on |
| `cap.isOpened()` | Did that work? True/False |
| `cap.read()` | Grab one frame. Returns `(worked?, image)` |
| `cap.release()` | Turn the webcam off |
| `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | Swap colour order |
| `cv2.flip(img, 1)` | Mirror left-right |
| `cv2.imshow("name", img)` | Pop up a window showing the image |
| `cv2.waitKey(1)` | Wait 1ms for a keypress — **required** or the window never draws |

**Websites:** start with the OpenCV "Getting Started with Videos" tutorial (link below) — it's about 30 lines of code and covers most of the above.

---

## First: what *is* an image, in code?

This underpins everything else.

An image is a **numpy array of numbers** with shape `(height, width, 3)`.

- A 480×640 webcam frame is an array of shape `(480, 640, 3)`
- Each pixel has **3 numbers**, each 0-255, describing how much of each colour it has
- `frame[0][0]` gives you the top-left pixel — something like `[52, 130, 201]`

**BGR vs RGB is just the order of those 3 numbers.**

| Order | `[52, 130, 201]` means |
|---|---|
| **RGB** (normal) | Red=52, Green=130, Blue=201 → a blue-ish colour |
| **BGR** (OpenCV) | Blue=52, Green=130, Red=201 → an orange-ish colour |

Same numbers, opposite meaning. OpenCV uses BGR for historical reasons (1990s camera hardware). Almost everything else — MediaPipe included — uses RGB.

**So `cvtColor` isn't doing anything clever.** It's swapping the 1st and 3rd number of every pixel. That's the whole operation.

### ⚠️ The common misunderstanding

It's tempting to think *"camera images are blue-ish, and MediaPipe wants pink, so I convert them."*

**That's not it.** The camera captures a perfectly normal-looking face. The numbers are correct. Only the **order they're labelled in** differs.

Think of a date: `05/11`. Is that 5 November, or 11 May? The digits are identical — the confusion comes from two people using different conventions, not from the digits being wrong. BGR vs RGB is exactly that.

**Walk through one skin pixel:**

| Step | Values | |
|---|---|---|
| Real skin colour | R=200, G=150, B=120 | warm pink |
| OpenCV stores it (BGR order) | `[120, 150, 200]` | same colour, just written B-first |
| MediaPipe reads it as RGB | R=120, G=150, B=200 | **light blue** |

The blue-ness is **created by the misreading**. It was never in the image.

(Because skin is red-dominant, swapping R and B always makes it blue-dominant. That's why the mistake shows up so clearly on faces — and why your `imshow` preview goes blue when you show RGB data.)

**So the correct statement is:**

> MediaPipe's face model was trained on real faces with normal skin tones. OpenCV stores frames with the channels in a different order (BGR). If I pass them straight to MediaPipe, it reads channel 0 as red when it's actually blue, so it sees a blue-grey face instead of a pink one — nothing like its training data. Converting to RGB re-orders the channels so MediaPipe interprets them the way it expects.

**Why this bug is nasty:** nothing crashes. The arrays are the same shape either way, and no code can check a convention that isn't stored anywhere. MediaPipe just detects slightly worse, with no error message.

---

## The functions, properly

### `cv2.VideoCapture(index)` — open the camera

```python
cap = cv2.VideoCapture(0)
```

`0` = your default webcam. `1` = a second camera if you have one. You can also pass a **filename string** to read a video file instead — same object, same methods. That's useful later: you can test your whole pipeline on a recorded video instead of sitting in front of the camera every time.

**Important:** this does **not** raise an error if it fails. You get an object back either way. It's `cap.isOpened()` that tells you the truth.

### `cap.isOpened()` — did it work?

```python
if not cap.isOpened():
    # camera busy, missing, or blocked by Windows privacy settings
```

Returns `True`/`False`. Always check — otherwise you get confusing failures later instead of a clear one now.

### `cap.read()` — grab one frame

```python
ok, frame = cap.read()
```

Returns **two** values:
- `ok` — a bool: did we actually get a frame?
- `frame` — the numpy array (or `None` if it failed)

`ok` can be `False` even when the camera opened fine: unplugged mid-run, another app stole it, or (for a video file) you hit the end. **Always check `ok` before using `frame`** — otherwise you'll pass `None` to MediaPipe and get a crash a few functions away from the real cause.

The `ok, frame = ...` line is Python tuple unpacking — `read()` returns a 2-tuple and you're naming both parts.

### `cap.release()` — turn it off

Frees the camera so other apps (and your next run) can use it. **Forget this and your camera can stay locked until you restart Python.** This is exactly what `__enter__`/`__exit__` are for — see below.

### `cv2.cvtColor(img, code)` — convert colour

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

The `code` is a constant naming the conversion. The ones you'll meet:

| Constant | Does |
|---|---|
| `cv2.COLOR_BGR2RGB` | BGR → RGB (for MediaPipe) |
| `cv2.COLOR_RGB2BGR` | back again (for `imshow`) |
| `cv2.COLOR_BGR2GRAY` | to greyscale — you don't need it here |

Returns a **new** array; the original is unchanged.

### `cv2.flip(img, flipCode)` — mirror

```python
mirrored = cv2.flip(frame, 1)
```

| `flipCode` | Effect |
|---|---|
| `1` | horizontal (left↔right) — the mirror effect |
| `0` | vertical (upside down) |
| `-1` | both |

`1` is the one you want, if you want it at all. See the mirror discussion in `step_camera_guide.md`.

### `cv2.imshow(name, img)` + `cv2.waitKey(ms)` — show a window

```python
cv2.imshow("preview", frame)
key = cv2.waitKey(1)
```

These two **always go together.** `imshow` only queues the image; `waitKey` is what actually draws the window and processes events. Call `imshow` without `waitKey` and you get a grey frozen box.

`waitKey(1)` = wait up to 1 millisecond for a keypress, then carry on. It returns the key code, or `-1` if nothing was pressed.

The standard "quit on q" line you'll see everywhere:

```python
if cv2.waitKey(1) & 0xFF == ord("q"):
    break
```

- `ord("q")` → the number 113 (the character code for `q`)
- `& 0xFF` → keeps only the low 8 bits, working around a platform quirk where `waitKey` returns extra bits

⚠️ **`imshow` expects BGR.** If you show an RGB image, it'll look blue/orange-tinted. When testing `read_rgb()`, that tint is *proof your conversion worked* — not a bug.

### `cv2.destroyAllWindows()` — close the windows

Call it after your loop ends.

### Bonus: `cv2.circle(img, centre, radius, colour, thickness)`

```python
cv2.circle(frame, (320, 240), 3, (0, 255, 0), -1)
```

You'll need this in the next step to draw landmarks. `centre` is `(x, y)` in **pixels** as ints. `colour` is a **BGR** tuple, so `(0, 255, 0)` is green. `thickness=-1` means filled.

### Bonus: `cap.set()` / `cap.get()` — camera settings

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
actual = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # may NOT be 1280
```

`set()` is a **request**, not a command — the camera picks the nearest size it supports. Always `get()` afterwards if the value matters.

Trade-off for your project: a bigger frame means more pixels across the iris (better precision) but slower processing. Start at default; only change it if precision turns out to be the problem.

---

## What `__enter__` / `__exit__` are for

These let your class work with `with`:

```python
with Camera() as cam:
    frame = cam.read_rgb()
# release() runs automatically here — even if the code above crashed
```

- `__enter__` runs at the `with`, and whatever it **returns** is what `as cam` becomes (normally `return self`)
- `__exit__` runs when the block ends — **including when an exception is thrown**

That last part is the point. Without it, a crash mid-loop skips your `release()` and leaves the camera locked. This pattern is called a **context manager**, and it's the same mechanism as `with open(...) as f:` for files.

---

## The capture loop — line by line

### The one idea that makes it click

**A webcam does not give you "a video". It gives you one photo at a time.**

There is no video object, no "recording" that runs in the background. When you ask the camera for a frame, you get **one still image**. That's it.

To get moving video, you ask again. And again. About 30 times a second. Show each one as it arrives and a human eye sees smooth motion — the same trick as a flipbook.

So this:

```python
while True:
    ok, frame = cap.read()
```

is not "start recording". It's **"take a photo, take a photo, take a photo, forever."**

Once that clicks, "how do I finish capturing?" answers itself: **you stop asking for photos.** You break out of the loop.

### The tutorial code, translated

```python
cap = cv.VideoCapture(0)          # 1. pick up the camera
if not cap.isOpened():            # 2. did that work?
    print("Cannot open camera")
    exit()

while True:                       # 3. keep going until told to stop
    ok, frame = cap.read()        # 4. take ONE photo

    if not ok:                    # 5. EXIT A: camera stopped working
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   # 6. do something to it
    cv.imshow('frame', gray)                       # 7. put it on screen

    if cv.waitKey(1) == ord('q'):                  # 8. EXIT B: user pressed q
        break

cap.release()                     # 9. put the camera down
cv.destroyAllWindows()            # 10. close the window
```

| Line | Plain English |
|---|---|
| 1 | Ask Windows for the webcam |
| 2 | Check we got it — busy, missing, or blocked all fail here |
| 3 | `while True` = "loop forever, until someone breaks out" |
| 4 | Take one photo. `ok` says whether it worked |
| 5 | If the camera died mid-run, stop looping |
| 6 | Your processing — greyscale in the tutorial, **BGR→RGB in yours** |
| 7 | Draw it in a window |
| 8 | Wait 1ms for a keypress. If it was `q`, stop looping |
| 9 | Hand the camera back to the operating system |
| 10 | Close the preview window |

### "How can I finish capturing?"

The loop has exactly **two exits**, both `break`:

- **You press `q`** (line 8) — the normal way you quit
- **A read fails** (line 5) — camera unplugged, another app stole it, or a video file ended

`break` jumps out of the `while` and continues at line 9.

⚠️ **`release()` is OUTSIDE the loop.** Look at the indentation — this trips people up:

```python
while True:
    ok, frame = cap.read()    # indented -> runs every frame (~30x/sec)
    ...
cap.release()                 # NOT indented -> runs ONCE, after the loop ends
```

Everything indented under `while` runs 30 times a second. `cap.release()` sits at the left margin, so it runs once, at the very end. When the tutorial says *"at the end, don't forget to release"*, it means **at the end of the whole program**, not the end of each frame.

### What `release()` actually does

While your program holds the camera, **no other program can use it** — the operating system gives it to one app at a time. `release()` hands it back.

Forget it, and:
- Zoom/Teams/the Camera app can't open the webcam
- **Your own next run can't open it either** — which looks like "my code broke" when actually the previous run never let go

Try it: run a script that opens the camera and crashes without releasing, then run it again. That's why `__enter__`/`__exit__` are worth the trouble — they guarantee `release()` runs even when your code crashes.

### How this maps to your `capture.py`

The tutorial puts everything in one file. You're splitting it:

| Tutorial line | Goes in your `Camera` class |
|---|---|
| `cv.VideoCapture(0)` + `isOpened()` | `__init__` |
| `cap.read()` + `cvtColor` | `read_rgb()` |
| `cap.release()` | `release()` |
| the `while` loop, `imshow`, `waitKey` | **stays outside** — in your test script, later in the Qt worker |

That last row is the important one. **Your `Camera` class contains no loop.** It just hands out one frame per call. Whoever's using it decides how often to ask — a test script with `cv.imshow`, or later a Qt worker thread. That's what makes the class reusable in both.

---

## Websites

**OpenCV official Python tutorials** — start here:
- Root index: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **Getting Started with Videos**: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html — `VideoCapture`, `read`, `imshow`, `waitKey`, `release`. This is basically your `capture.py` in tutorial form. **Read this one first.**
- Drawing Functions: https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html — `circle`, `line`, `putText`. For the next step.
- Changing Colorspaces: https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html — `cvtColor`.

*(If a deep link breaks, go to the root index and navigate — OpenCV occasionally reshuffles URLs.)*

**MediaPipe Face Landmarker (Python)** — for the next step, not now:
- https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python

**Searching for help:** put "opencv python" in front of your query. And **check the date** — OpenCV answers age well, but MediaPipe answers from before ~2023 use the old deprecated API and won't match your code.

---

## Furthermore — why a class at all?

You could just write functions. But the camera has **state**: it's either open or closed, and the `cap` object must be kept alive between frames.

```python
# Without a class - who holds cap? Every caller has to.
cap = open_camera()
frame = read_frame(cap)
close_camera(cap)
```

A class keeps `cap` inside itself (`self._cap`), so the rest of your program just does `cam.read_rgb()` and never sees the OpenCV object at all.

**The real payoff:** later you can write a `FakeCamera` class with the same three methods that replays a saved video or returns test images. Your landmarker, features, and model code won't know the difference — you swap one line. That means you can test your gaze pipeline **without a webcam, without your face, and with identical input every time**, which is what makes measured comparisons possible.

That's a strong Design-section point: *"I wrapped the camera in a class so it could be substituted with a recorded source for repeatable testing."*

**The cost:** ~30 extra lines and one more concept, for a thing that could have been three functions. On a one-month deadline that's a fair thing to question — but the testability payoff is real, and it's the kind of decision an assessment wants to see you justify rather than stumble into.
