"""
Entry point:  python main.py

Starts the calibration lab (eyetracking/app.py): one window with every
calibration method, the gaze check, the blink test and their data.
Add --mouse to try it without a camera.
"""
import sys

from eyetracking.app import main

if __name__ == "__main__":
    sys.exit(main())
