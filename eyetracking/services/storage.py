"""
Save and load calibration samples as JSON, so one calibration can be reused
for many model experiments without sitting through the dots again.

Files go in <project root>/data, named by date and time, e.g.
    data/calibration_2026-09-29_140312.json
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core.models import CalibrationSample
from path import DATA_DIR


def save_calibration(samples: list[CalibrationSample], path: Path | None = None) -> Path:
    """Write samples to `path` (default: a new timestamped file in data/).
    Returns the path it wrote to."""
    now = datetime.now()
    if path is None:
        path = DATA_DIR / f"calibration_{now:%Y-%m-%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "saved_at": now.isoformat(timespec="seconds"),
        "samples": [asdict(s) for s in samples],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_calibration(path: Path) -> list[CalibrationSample]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # JSON has no tuples: features come back as a list, so turn it back
    return [CalibrationSample(tuple(s["features"]), s["target_x"], s["target_y"])
            for s in data["samples"]]


def latest_calibration() -> Path | None:
    """Newest saved file, or None if nothing has been saved yet.
    Works because the date-time names sort in time order."""
    files = sorted(DATA_DIR.glob("calibration_*.json"))
    return files[-1] if files else None
