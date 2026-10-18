"""Locate project files without depending on the current working directory.

Two ways the app runs:
  * from source (`python main.py`): files sit in the project folder, data in `data/`.
  * installed (frozen by PyInstaller, see installer/): the read-only files are
    unpacked into the bundle folder (`sys._MEIPASS`), and the program folder can't
    be written to, so data goes to %LOCALAPPDATA%\\StudyAssistant\\data instead.
"""

import os
import sys
from pathlib import Path

# This module's own location. Fixed at import time, independent of the cwd.
_MODULE_PATH = Path(__file__).resolve()

# True inside the installed app: PyInstaller sets sys.frozen.
FROZEN = bool(getattr(sys, "frozen", False))

# Where bundled read-only files live: PyInstaller's bundle folder, or the project root.
BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", _MODULE_PATH.parent))

# A relative fragment: the tail attached to each folder while searching.
# Never call .resolve() on this -- that would measure it against the cwd.
MODEL_RELPATH = Path("models") / "face_landmarker.task"

# Saved calibrations, gaze checks, blink scores, history. From source: the project's
# data/ (this file sits in the project root). Installed: the user's local app data,
# because an installed program may not write next to its own files.
if FROZEN:
    DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "StudyAssistant" / "data"
else:
    DATA_DIR = _MODULE_PATH.parent / "data"


def find_model(relpath: Path = MODEL_RELPATH) -> Path:
    """Return an absolute path to the model file.

    Installed: inside the bundle folder. From source: walks upward from this
    module's location, checking each ancestor folder, so it works wherever the
    application was launched from.

    Raises:
        FileNotFoundError: if no folder contains `relpath`.
    """
    for folder in (BUNDLE_DIR, *_MODULE_PATH.parents):
        candidate = folder / relpath
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find {relpath} in {BUNDLE_DIR} or any parent folder of {_MODULE_PATH}"
    )


if __name__ == "__main__":
    print("found:", find_model())
    print("data:", DATA_DIR, "(installed)" if FROZEN else "(from source)")
