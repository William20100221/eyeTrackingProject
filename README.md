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
| Original 3×3 / 4×4 / 5×5 | look at each dot, row by row | ridge curve on the raw features |
| Sweep | follow a gliding dot across and down, the head step, then 25 dots | ridge curve on eye numbers scaled by your own range |
| Points | 14 still dots along a line, the head step, 25 dots in random order | the same scaled curve |
| Line model | the Points procedure | a 3D line from each eye through the screen, blended with the curve |

Every method records the same 12 numbers per frame, so their files can be compared.

## Testing tab

- **Calibration:** pick a method, read what it does, start it (Space starts, Space pauses, Esc leaves).
- **Gaze check:** a dot where the chosen method's model thinks you look, rings on its calibration dots; Enter records a sample.
- **Blink test:** your blink scores in six eye states, with the cut-off they suggest.
- **Data:** the chosen method's calibrations (active / non-active / not usable, each readable in words, samples removable),
  every gaze check, the blink scores, and a log of what happened.

## Where the data goes

By person, then by calibration type. Settings > Person mode picks the person; with it off,
the same folders sit straight in `data/`.

```
data/<person>/<method>/<method>_<date>_<time>.json             a calibration: trains that method's model
data/<person>/<method>/gaze-check_<method>_<date>_<time>.csv   a gaze check of that method
data/<person>/<method>/disabled/                                kept, turned off
data/<person>/<method>/trash/                                   removed files and the originals of edited ones
data/<person>/blink-test/blink-scores_<date>_<time>.csv
```

`<method>` is one of `original-3x3`, `original-4x4`, `original-5x5`, `sweep`, `points`, `line-model`.
The other branches' folders in `data/` (`calibrations/`, `gaze_checks/`, `blink_scores/`, `history/`) are left alone.

## Privacy

- Everything stays on your computer; no camera images are saved or sent anywhere.
