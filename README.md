## Eye-Tracking Calibration Lab

The calibration and testing side of the webcam eye-tracking project, on its own: every
calibration method in one window, each with its own data folder and its own model, so
they can be run and compared under the same conditions. No reading tracker, AI or
floating ball on this branch (see `master` and the `new-calibrate-*` branches for those).

## Install Requirements

Mediapipe, OpenCV, scikit-learn, PySide6 (Qt) and materialyoucolor (`uv sync`).

## Run

```
python main.py            # the camera chosen in Settings (a popup asks on first start)
python main.py --mouse    # no camera: the mouse plays your eyes, nothing is saved
```

## Methods

| Method | What you do | Model |
|---|---|---|
| Original 3×3 / 4×4 / 5×5 | look at each dot, row by row | a straight-line fit (ridge) on the 6 eye features, each frame first turned back to the calibration's posture with the built-in head-rotation matrix |
| Sweep | follow a gliding dot across and down, the head step, then 16 dots (4×4) | the same fit, the head-rotation matrix measured by your own head step |
| Points | 14 still dots along a line, the head step, 16 dots (4×4) | the same as Sweep |
| Line model | the Points procedure | a 3D line from each eye through the screen, blended with the fitted line |
| Random 3×3 / 4×4 / 5×5 / Sweep / Points / Line model | each type above, its final dots in a new random order each time | the same as its in-order type |

Every type records the same 12 numbers per frame, so their files can be compared. Sweep, Points and
Line model end on a 4×4 grid on this branch, the same 16 dots as Original 4×4. The live point goes
through a 1€ filter (steady while the eyes rest, quick when they jump). Why the models are what they
are, with the measurements: `docs/model_improvements.md`.

## Testing tab

- **Calibration:** pick a type (two rows: *In order*, *Random order*), read what it does, start it.
  Space starts, Space pauses / carries on, Esc leaves (it asks first once samples are collected:
  they aren't saved).
- **Gaze check:** pick a **person**, a **type** and how many of their calibrations to use (the newest
  *N*; *None* when there are none). 16 rings (4×4), one lit at a time: look at it, press Enter. Each
  sample's error is its distance to the lit ring. A table compares every type's latest and mean error
  for that person; **Refresh** re-reads the data folder.
  - **Multiple model gaze check:** every type the person has calibrations for at once, predicting from
    the same camera frames, on 36 rings (6×6), no dot; one file per type.
  - **Record mode** (switch at the top): the webcam is recorded with the moment of each Enter instead of
    predicting live; afterwards the models are run over the recording as if live, and **Recordings**
    runs any model (even a later one) over an old recording.
- **Calibration, Sweep / Points / Line model:** a switch reuses the person's newest Original 4×4
  (Random 4×4 for the random twins) as the final 16 dots, so only the first steps are recorded.
- **Blink test:** your blink scores in six eye states, with the cut-off they suggest.
- **Data:** calibrations (by person > type > date; active / non-active / not usable; readable in words,
  samples removable), gaze checks (by person > type > date), blink scores, and a log of this run.

## Settings

Theme and colour, the webcam, and **person mode**: on, every file goes under `data/<person>/`; off,
straight under `data/`. The person in use is the one whose calibrations train the models.

## Where the data goes

By person, then by calibration type. Settings > Person mode picks the person; with it off,
the same folders sit straight in `data/`.

```
data/<person>/<method>/<method>_<date>_<time>.json             a calibration: trains that method's model
data/<person>/<method>/gaze-check_<method>_<date>_<time>.csv   a gaze check of that method
data/<person>/<method>/disabled/                                kept, turned off
data/<person>/<method>/trash/                                   removed files and the originals of edited ones
data/<person>/blink-test/blink-scores_<date>_<time>.csv
data/<person>/gaze-videos/gaze-video_<date>_<time>.mp4 + .json   record-mode gaze checks (the video of your face)
```

`<method>` is one of `original-3x3`, `original-4x4`, `original-5x5`, `sweep`, `points`, `line-model`,
`random-3x3`, `random-4x4`, `random-5x5`, `random-sweep`, `random-points`, `random-line-model`.
The other branches' folders in `data/` (`calibrations/`, `gaze_checks/`, `blink_scores/`, `history/`) are left alone.

## Privacy

- Everything stays on your computer; nothing is sent anywhere.
- No camera images are saved, except in the gaze check's **record mode**, which you turn on yourself:
  the video goes in `data/<person>/gaze-videos/` and you can delete it any time.
