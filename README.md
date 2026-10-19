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

## Calibrations and models

Nine calibration types, a row each on the Calibration page:

| Row | Types | What you do |
|---|---|---|
| In order | Original 3×3 / 4×4 / 5×5 | look at each dot, row by row |
| Random order | Random 3×3 / 4×4 / 5×5 | the same dots, in a new random order each time |
| Special steps | Sweep | follow a gliding dot across, then down; then the head step (turn, nod, tilt, lean, eyes on the centre dot) |
| | Points | 14 still dots along the same two lines; then the head step |
| | Line model | the Points procedure (a Points calibration is saved as a Line model one too, and back) |

A special records no dots: it pairs with any grid calibration of the same person, so the 9 types make
**24 models**:

| Model | Learns from |
|---|---|
| a grid alone (6) | a straight-line fit (ridge) on the 6 eye features of the grid's dots, each frame turned back to the calibration's posture with the built-in head-rotation matrix |
| Sweep + a grid (6) | the same fit on the grid's dots **and** the gliding dot's frames (each matched to where the dot was a moment before: the eye trails it), with the head-rotation matrix of the Sweep's own head step |
| Points + a grid (6) | the same on the grid's dots **and** the 14 line dots, its own head matrix |
| Line model + a grid (6) | a 3D line from each eye through the screen, fitted on the grid's dots and the 14 line dots |

Before 20 Oct 2026 Sweep, Points and Line model each ended on 4×4 dots, had random twins, and one recording
was saved as all three, so Sweep and Points were the same model on the same data. Their gaze checks are
still listed as the old types ("Sweep (old)" …); the old random-* folders show under Not usable. Every
type records the same 12 numbers per frame. The live point goes through a 1€ filter. Why the models are
what they are: `docs/model_improvements.md`; the pairing: `docs/calibration_lab_dev_log.md` addendum 17.

## Testing tab

- **Calibration:** pick a type (three rows), read what it does, start it. Space starts, Space pauses /
  carries on, Esc leaves (it asks first once samples are collected: they aren't saved).
- **Gaze check:** pick a **person**, a model (**Dots from** a grid, **With** nothing or a special) and how
  many of that grid's calibrations to use (the newest *N*). 16 rings (4×4), one lit at a time: look at
  it, press Enter. A table compares every model's latest and mean error for that person.
  - **Multiple model gaze check:** the models you tick in its popup (a row per grid: alone, + Sweep,
    + Points, + Line model) at once, from the same camera frames, on 36 rings (6×6); one file per model.
  - **Record mode:** the webcam is recorded with the moment of each Enter; the models run over it
    afterwards as if live, and **Recordings** runs any model over an old recording.
- **Sample text, Blink test.**
- **Undo** (under the status line, or Ctrl+Z): the last calibration or gaze check goes to the trash
  (a calibration's copy, and every file of a multiple check, with it); Removed restores a calibration.
- **Data:**
  - **Calibrations** by person > type > date: active / non-active / not usable (point at one to see
    why) / removed; readable in words, samples removable. Ctrl- or Shift-click picks several (scrolling
    keeps them, a click anywhere else lets go). **Clear data…** moves everything of one person, or of
    everyone, into `data/trash/<date>_<time>/`.
  - **Train:** the 24 models of a person, tick some (or every ready one) and train them now: each row
    says what it learnt from, or what it still needs.
  - **Gaze checks** by person > model > date.
  - **Gaze check table:** every model's errors like a spreadsheet, for one tester or everyone side by
    side (coloured per tester from their best to their worst); click a heading to sort; **Download** as
    .csv (this table, every tester one row per model, or everyone side by side).
  - **Blink scores.**

## Settings

Theme and colour, the webcam, and **person mode**: on, every file goes under `data/<person>/`; off,
straight under `data/`. Picking or adding a person saves at once. The person in use is the one whose
calibrations train the models.

## Where the data goes

By person, then by calibration type. Settings > Person mode picks the person; with it off,
the same folders sit straight in `data/`.

```
data/<person>/<type>/<type>_<date>_<time>.json                 a calibration
data/<person>/<model>/gaze-check_<model>_<date>_<time>.csv     a gaze check of a model (a grid's own model: the grid's folder)
data/<person>/<type>/disabled/                                  kept, turned off
data/<person>/<type>/trash/                                     removed files, undone ones, the originals of edited ones
data/<person>/blink-test/blink-scores_<date>_<time>.csv
data/<person>/gaze-videos/gaze-video_<date>_<time>.mp4 + .json   record-mode gaze checks (the video of your face)
data/trash/<date>_<time>/<person>/...                           everything cleared with Clear data…
```

`<type>` is one of `original-3x3`, `original-4x4`, `original-5x5`, `random-3x3`, `random-4x4`, `random-5x5`,
`sweep`, `points`, `line-model`; `<model>` is a type's id for a grid alone, or `<special>-<grid>`
(`sweep-original-4x4`, `line-model-random-5x5` …). Folders of the old types (`random-sweep` …) are read, never written.
The other branches' folders in `data/` (`calibrations/`, `gaze_checks/`, `blink_scores/`, `history/`) are left alone.

## Privacy

- Everything stays on your computer; nothing is sent anywhere.
- No camera images are saved, except in the gaze check's **record mode**, which you turn on yourself:
  the video goes in `data/<person>/gaze-videos/` and you can delete it any time.
