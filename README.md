## Eye Tracking Test (tester version of the calibration lab)

The calibration lab cut down for testers: instructions, the calibrations and one gaze check. Each
result is emailed to zh25764@my.westlake.school.nz as soon as it is saved. Which model is the most
accurate is worked out afterwards, in the lab (`calibration-lab` branch): unzip each email into its
`data/` folder.

## Install Requirements

Mediapipe, OpenCV, scikit-learn, PySide6 (Qt) and materialyoucolor (`uv sync`).

Emailing needs `mail.ini` beside `main.py` (not in git: it holds a password). Use an account made
only for sending, for example a Gmail account with an app password:

```
[smtp]
host = smtp.gmail.com
port = 465
user = <the sending address>
password = <its app password>
```

## Run

```
python main.py            # a popup asks for the camera on first start
python main.py --mouse    # no camera: the mouse plays your eyes, nothing is emailed
```

## What the tester does

1. Instructions: type their name, pick the camera.
2. Calibration: the 8 recordings, one after another (Original and Random 3×3 / 4×4 / 5×5, Sweep,
   Points; each Points is also saved as Line model). The next one to do is picked after each.
3. Gaze check: every model their calibrations make (up to 24), all at once from the same camera
   frames, on 36 rings. It ends by itself after the last ring.

Files go into `data/<name>/<type>/`, the lab's layout. A file not sent yet (no internet) waits in
`data/outbox.txt` and goes with the next email, or at the next start.

## Self-checks (no webcam, no network)

```
python -m eyetracking.testing.test_camera_use     # the tester app: name, calibrations, check, emails
python -m eyetracking.testing.test_mailer         # the outbox and the zip
python -m eyetracking.testing.test_gaze_log
python -m eyetracking.testing.test_frame_feed
python -m eyetracking.testing.test_blinks
python -m eyetracking.testing.test_head_step
python -m eyetracking.testing.test_methods
```
