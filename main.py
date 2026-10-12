"""
Entry point:  python main.py

Starts the whole app (eyetracking/app.py): floating ball + chat panel,
with calibrate / check / read commands. Add --mouse to try it without a camera.
"""
import sys

from eyetracking.app import main

if __name__ == "__main__":
    sys.exit(main())
