"""
Entry point:  python main.py

Starts the tester version of the calibration lab (eyetracking/app.py): the
instructions, every calibration and the gaze check; each result is emailed.
Add --mouse to try it without a camera.
"""
import sys

from eyetracking.app import main

if __name__ == "__main__":
    sys.exit(main())
