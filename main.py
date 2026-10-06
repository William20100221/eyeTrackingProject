"""
Entry point:  python main.py

For now this starts the calibration window (the finished part of the app).
When ui/main_window.py exists, start that here instead.
"""
import sys

from eyetracking.ui.calibration_window import main

if __name__ == "__main__":
    sys.exit(main())
