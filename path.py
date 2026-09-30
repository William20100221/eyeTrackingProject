"""Locate project files without depending on the current working directory."""

from pathlib import Path

# This module's own location. Fixed at import time, independent of the cwd.
_MODULE_PATH = Path(__file__).resolve()

# A relative fragment: the tail attached to each folder while searching.
# Never call .resolve() on this -- that would measure it against the cwd.
MODEL_RELPATH = Path("models") / "face_landmarker.task"

# Saved calibrations. This file sits in the project root, so its folder IS the root.
DATA_DIR = _MODULE_PATH.parent / "data"


def find_model(relpath: Path = MODEL_RELPATH) -> Path:
    """Return an absolute path to the model file.

    Walks upward from this module's location, checking each ancestor folder.
    Works regardless of where the application was launched from.

    Raises:
        FileNotFoundError: if no ancestor folder contains `relpath`.
    """
    for folder in _MODULE_PATH.parents:
        candidate = folder / relpath
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find {relpath} in any parent folder of {_MODULE_PATH}"
    )


if __name__ == "__main__":
    print("found:", find_model())