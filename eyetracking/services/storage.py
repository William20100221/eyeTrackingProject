"""
Save and load calibration samples as JSON, so one calibration can be reused
for many model experiments without sitting through the dots again.

Files go in <project root>/data, named by date and time, e.g.
    data/calibration_2026-09-29_140312.json
    data/history/chat_2026-09-29_141502.json     (one per app run: chat + quizzes)
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core.calibration import describe_target
from eyetracking.core.features import N_FEATURES
from eyetracking.core.models import CalibrationSample
from path import DATA_DIR

STAMP = "%Y-%m-%d_%H%M%S"            # the date-time part of every file name
HISTORY_DIR = DATA_DIR / "history"


def file_date(path: Path) -> datetime:
    """calibration_2026-09-29_140312.json -> 29 Sep 2026 14:03:12"""
    return datetime.strptime(path.stem.split("_", 1)[1], STAMP)


def save_calibration(samples: list[CalibrationSample], path: Path | None = None) -> Path:
    """Write samples to `path` (default: a new timestamped file in data/).
    Returns the path it wrote to."""
    now = datetime.now()
    if path is None:
        path = DATA_DIR / f"calibration_{now:{STAMP}}.json"
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


def load_all_calibrations(n_features: int, older_weight: float = 0.5
                          ) -> tuple[list[CalibrationSample], list[float]]:
    """Every saved calibration, newest first, plus a weight per sample:
    newest file 1.0, every older file `older_weight` (0.5).
    Older sessions still help, but today's seat, camera and light count most.
    Files with a different feature count (an older version) are skipped."""
    # ponytail: every older file weighs the same; decay by age/date if old sessions start to hurt
    samples, weights = [], []
    for path in sorted(DATA_DIR.glob("calibration_*.json"), reverse=True):
        try:
            found = load_calibration(path)
        except (OSError, ValueError, KeyError, TypeError) as e:   # damaged or hand-edited file
            print(f"Skipping {path.name}: can't read it ({e})")
            continue
        if not found or len(found[0].features) != n_features:
            print(f"Skipping {path.name}: older calibration format, not used for training")
            continue
        weights += [older_weight if samples else 1.0] * len(found)   # first one kept = newest
        samples += found
    return samples, weights


def list_calibrations() -> list[Path]:
    """Every saved calibration file, newest first."""
    return sorted(DATA_DIR.glob("calibration_*.json"), reverse=True)


def calibration_usable(path: Path) -> bool:
    """True if this version of the app can train on the file: right format and
    feature count. Other versions' files and damaged ones are listed as not usable."""
    try:
        found = load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return bool(found) and len(found[0].features) == N_FEATURES


def calibration_text(path: Path) -> str:
    """One calibration in words, a line per sample:
    "#12 came from dot (2, 1) of the 5x5 grid: 0.412, 0.533, ..." """
    try:
        samples = load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"{path.name}\n\nCan't show this file: it was saved by a different calibration version ({e})."
    width = len(samples[0].features) if samples else 0
    lines = [f"Calibration of {file_date(path):%d %b %Y, %H:%M}: {len(samples)} samples, {width} numbers each.", ""]
    for i, s in enumerate(samples, 1):
        numbers = ", ".join(f"{v:.3f}" for v in s.features)
        lines.append(f"#{i} came from {describe_target(s.target_x, s.target_y)}: {numbers}")
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


def latest_calibration() -> Path | None:
    """Newest saved file, or None if nothing has been saved yet.
    Works because the date-time names sort in time order."""
    files = sorted(DATA_DIR.glob("calibration_*.json"))
    return files[-1] if files else None
