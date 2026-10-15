"""
Save and load calibrations as JSON, so one calibration can be reused for
many model experiments without sitting through it again.

Files go in <project root>/data, named by date and time, e.g.
    data/calibration_2026-10-14_140312.json

File format (FORMAT below). Every feature list is RAW gaze_features():
    {
      "format":   "gaze-calibration-v3",
      "saved_at": "2026-10-14T14:03:12",
      "sweep": [ {"features": [...12], "target_x": .., "target_y": ..}, ... ],
      "head":  [ ...same, target = the centre dot... ],
      "dots":  [ ...same, target = the grid dot... ]
    }
A file without exactly this "format" value (older versions, other files)
is not loaded: the code has no way to read anything else.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core.models import CalibrationRecording, CalibrationSample
from path import DATA_DIR

FORMAT = "gaze-calibration-v3"   # v3: 12 features (iris height vs the eye corners added)
PARTS = ("sweep", "head", "dots")


def save_calibration(rec: CalibrationRecording, path: Path | None = None) -> Path:
    """Write one calibration to `path` (default: a new timestamped file in data/).
    Returns the path it wrote to."""
    now = datetime.now()
    if path is None:
        path = DATA_DIR / f"calibration_{now:%Y-%m-%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {"format": FORMAT, "saved_at": now.isoformat(timespec="seconds")}
    data |= {part: [asdict(s) for s in getattr(rec, part)] for part in PARTS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_calibration(path: Path) -> CalibrationRecording:
    """Read one file. Raises ValueError if it isn't in the current FORMAT."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"not in the current format ({FORMAT})")
    # JSON has no tuples: features come back as a list, so turn it back
    parts = [tuple(CalibrationSample(tuple(s["features"]), s["target_x"], s["target_y"])
                   for s in data[part]) for part in PARTS]
    return CalibrationRecording(*parts)


def load_all_calibrations() -> list[CalibrationRecording]:
    """Every readable calibration in data/, NEWEST FIRST (date-time names sort
    in time order). Unreadable or other-format files are skipped, with a message."""
    recordings = []
    for path in sorted(DATA_DIR.glob("calibration_*.json"), reverse=True):
        try:
            recordings.append(load_calibration(path))
        except (OSError, ValueError, KeyError, TypeError) as e:   # old format, damaged, hand-edited
            print(f"Skipping {path.name}: {e}")
    return recordings
