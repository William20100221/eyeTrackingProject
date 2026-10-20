## Eye Tracking Test (tester version of the calibration lab)

The calibration lab cut down for testers: instructions, two sets of calibrations and two gaze checks.
When both checks are done, all of the tester's files are emailed to zh25764@my.westlake.school.nz.
Which model is the most accurate is worked out afterwards, in the lab (`calibration-lab` branch):
unzip each email into its `data/` folder.

## Install Requirements

Mediapipe, OpenCV, scikit-learn, PySide6 (Qt) and materialyoucolor (`uv sync`).

Emailing needs `mail.ini` beside `main.py`. It is not in git (the repo is public and it holds a
password), so copy it onto each tester's computer along with the app. Use an account made only for
sending, for example a Gmail account with an app password:

```
[smtp]
host = smtp.gmail.com
user = <the sending address>
password = <its app password>
```

On each tester's computer, check it before they start: `python -m eyetracking.services.mailer` sends
a test email and says what failed. It tries port 465, then 587 (a network may block one).

## Run

```
python main.py            # a popup asks for the camera on first start
python main.py --mouse    # no camera: the mouse plays your eyes, nothing is emailed
```

## What the tester does

1. Instructions: type their name, pick the camera.
2. Calibration: the 8 recordings (Original and Random 3×3 / 4×4 / 5×5, Sweep, Points; each Points is
   also saved as Line model), exactly twice each: set 1 is one of each, then set 2. Only the types still
   open in the current set can be started; each chip shows `n/2`. "Redo a calibration" deletes a type's
   last one (into `data/<name>/trash/`, with the gaze checks, which then have to be done again).
3. Gaze check, locked until the 16 are done, then twice: every model (24) at once from the same camera
   frames, on 36 rings, trained on the newest set (files `_1-calibrations`), then on both sets
   (`_2-calibrations`). It saves and ends by itself after the last ring; Esc before that saves nothing.

The sidebar shows the progress (`Calibration 5/16`, `Gaze check 1/2`). Files go into
`data/<name>/<type>/`, the lab's layout. After the second check everything is emailed in one zip. If
that fails (no internet, no `mail.ini`), the zip is put on the Desktop with a note to email it by
hand, and the email is tried again at the next start (`data/outbox.txt`).

## Self-checks (no webcam, no network)

```
python -m eyetracking.testing.test_camera_use     # the tester app: name, 2 sets, 2 checks, one email
python -m eyetracking.testing.test_mailer         # the outbox, the zip, port 587, the Desktop copy
python -m eyetracking.testing.test_gaze_log
python -m eyetracking.testing.test_frame_feed
python -m eyetracking.testing.test_blinks
python -m eyetracking.testing.test_head_step
python -m eyetracking.testing.test_methods
```
