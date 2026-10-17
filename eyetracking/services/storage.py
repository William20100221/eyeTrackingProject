"""
Save and load calibrations as JSON, so one calibration can be reused for
many model experiments without sitting through it again.

Files go in <project root>/data, named by date and time, e.g.
    data/calibration_2026-10-14_140312.json

File format (FORMAT below). Every feature list is RAW gaze_features():
    {
      "format":   "gaze-calibration-v2",
      "saved_at": "2026-10-14T14:03:12",
      "sweep": [ {"features": [...10], "target_x": .., "target_y": ..}, ... ],
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

from eyetracking.core.calibration import describe_target
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from path import DATA_DIR

STAMP = "%Y-%m-%d_%H%M%S"            # the date-time part of every file name
HISTORY_DIR = DATA_DIR / "history"   # chat + quiz history, one file per app run
STEP_NAMES = {'sweep': 'Step 1, dots on a line', 'head': 'Step 2, head moving, eyes on the centre dot', 'dots': 'Step 3, the dot grid'}


def file_date(path: Path) -> datetime:
    """calibration_2026-09-29_140312.json -> 29 Sep 2026 14:03:12"""
    return datetime.strptime(path.stem.split("_", 1)[1], STAMP)

FORMAT = "gaze-calibration-v2"
PARTS = ("sweep", "head", "dots")


def save_calibration(rec: CalibrationRecording, path: Path | None = None) -> Path:
    """Write one calibration to `path` (default: a new timestamped file in data/).
    Returns the path it wrote to."""
    now = datetime.now()
    if path is None:
        path = DATA_DIR / f"calibration_{now:{STAMP}}.json"
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


def list_calibrations() -> list[Path]:
    """Every saved calibration file, newest first."""
    return sorted(DATA_DIR.glob("calibration_*.json"), reverse=True)


def calibration_usable(path: Path) -> bool:
    """True if this version of the app can train on the file: right format and
    feature count. Other versions' files and damaged ones are listed as not usable."""
    try:
        load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def calibration_text(path: Path) -> str:
    """One calibration in words, a line per sample, step by step:
    "#12 came from dot (2, 1) of the 5x5 grid: 0.412, 0.533, ..." """
    try:
        rec = load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"{path.name}\n\nCan't show this file: it was saved by a different calibration version ({e})."
    lines = [f"Calibration of {file_date(path):%d %b %Y, %H:%M}"]
    n = 0
    for part in PARTS:
        samples = getattr(rec, part)
        lines += ["", f"{STEP_NAMES[part]}: {len(samples)} samples"]
        for s in samples:
            n += 1
            numbers = ", ".join(f"{v:.3f}" for v in s.features)
            lines.append(f"#{n} came from {describe_target(s.target_x, s.target_y)}: {numbers}")
    return "\n".join(lines)


# ---- chat + quiz history ------------------------------------------------
def new_chat_path() -> Path:
    return HISTORY_DIR / f"chat_{datetime.now():{STAMP}}.json"


def save_chat(path: Path, messages: list[dict]) -> None:
    """messages: [{"who": "you" | "assistant", "text": ..., "at": iso time}, ...]"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)


def list_chats() -> list[Path]:
    """Every saved conversation, newest first."""
    return sorted(HISTORY_DIR.glob("chat_*.json"), reverse=True)


def chat_text(path: Path) -> str:
    """A saved conversation as plain text, a paragraph per message."""
    try:
        with open(path, encoding="utf-8") as f:
            messages = json.load(f)["messages"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"Can't read {path.name} ({e})."
    return "\n\n".join(f"{'You' if m['who'] == 'you' else 'Assistant'} ({m['at'][11:16]}):\n{m['text']}"
                       for m in messages)
