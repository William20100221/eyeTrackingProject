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
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core.calibration import describe_target
from eyetracking.core.features import describe_features, head_degrees
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from path import DATA_DIR

STAMP = "%Y-%m-%d_%H%M%S"            # the date-time part of every file name
HISTORY_DIR = DATA_DIR / "history"   # the AI's summary + quiz, one file per reading
STEP_NAMES = {'sweep': 'Step 1, following the moving dot', 'head': 'Step 2, head moving, eyes on the centre dot', 'dots': 'Step 3, the dot grid'}


def file_date(path: Path) -> datetime:
    """calibration_2026-09-29_140312.json -> 29 Sep 2026 14:03:12"""
    return datetime.strptime(path.stem.split("_", 1)[1], STAMP)

FORMAT = "gaze-calibration-v2"
PARTS = ("sweep", "head", "dots")


def save_calibration(rec: CalibrationRecording, path: Path | None = None) -> Path:
    """Write one calibration to `path` (default: a new timestamped file in data/).
    Returns the path it wrote to. Nothing is written without dot samples."""
    if not rec.dots:
        raise ValueError("no dot samples: nothing worth saving")
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
    """Every calibration file in data/ (the ones that can train the model), newest first."""
    return sorted(DATA_DIR.glob("calibration_*.json"), reverse=True)


# ---- disabled: works, but the user turned it off --------------------------
# Kept in data/disabled/. Training only reads data/calibration_*.json, so a file
# there is left out without any check in the training code; enabling moves it back.
DISABLED_DIR = DATA_DIR / "disabled"


def list_disabled() -> list[Path]:
    """Calibrations the user disabled, newest first."""
    return sorted(DISABLED_DIR.glob("calibration_*.json"), reverse=True)


def set_calibration_enabled(path: Path, enabled: bool) -> Path:
    """Move a calibration between data/ (trains the model) and data/disabled/ (kept, doesn't).
    Returns its new path."""
    folder = DATA_DIR if enabled else DISABLED_DIR
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    if target != path:
        shutil.move(str(path), str(target))
    return target


def calibration_usable(path: Path) -> bool:
    """True if this version of the app can train on the file: right format and
    feature count. Other versions' files and damaged ones are listed as not usable."""
    try:
        load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def _parts(rec: CalibrationRecording) -> list[tuple[str, tuple]]:
    return [(STEP_NAMES[part], getattr(rec, part)) for part in PARTS]


def _spread(rows: list) -> float:
    """The widest standard deviation over the columns of rows (0 for a single row)."""
    return max((statistics.pstdev(col) for col in zip(*rows)), default=0.0) if len(rows) > 1 else 0.0


def _range_words(samples) -> str:
    """How much of the eyes' and the head's range a step used."""
    if not samples:
        return ""
    f = [s.features for s in samples]
    span = [max(v[i] for v in f) - min(v[i] for v in f) for i in range(4)]
    yaw, pitch = zip(*(head_degrees(v) for v in f))
    return (f"Eyes moved {span[0]:.0%} across and {span[2]:.0%} down (right eye), {span[1]:.0%} / {span[3]:.0%} (left); "
            f"head turned over {max(yaw) - min(yaw):.0f}° and nodded over {max(pitch) - min(pitch):.0f}°.")


def calibration_lines(path: Path) -> list[tuple[str, int | None, str | None]]:
    """The viewer's lines as (text, sample index, dot key). A sample line carries its
    index (as remove_samples counts them) and its dot; a dot's summary line carries
    the dot only (= all of its samples); headings carry neither."""
    try:
        parts = _parts(load_calibration(path))
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [(f"Can't show this file: it was saved by a different calibration version ({e}).", None, None)]
    lines = [(f"Calibration of {file_date(path):%d %b %Y, %H:%M}", None, None)]
    n = 0
    for name, samples in parts:
        lines += [("", None, None), (f"{name}: {len(samples)} samples. {_range_words(samples)}", None, None)]
        by_dot: dict = {}
        for s in samples:
            by_dot.setdefault((s.target_x, s.target_y), []).append((n, s))
            n += 1
        grouped = len(by_dot) <= 50       # a moving dot (the sweep) has a target per frame: no per-dot lines
        for (tx, ty), group in by_dot.items():
            key = f"{name} {tx} {ty}"
            if grouped:
                eyes = _spread([s.features[:4] for _, s in group])
                head = _spread([head_degrees(s.features) for _, s in group])
                lines.append((f"{describe_target(tx, ty)}: {len(group)} sample{'s' if len(group) != 1 else ''}, "
                              f"eyes {'steady' if eyes < 0.02 else 'wandering'} (spread {eyes:.0%}), "
                              f"head {'still' if head < 2 else 'moving'} (±{head:.0f}°)", None, key))
            where = "" if grouped else f" ({describe_target(tx, ty)})"
            lines += [(f"      #{i + 1}{where}: {describe_features(s.features)}", i, key) for i, s in group]
    return lines


def calibration_text(path: Path) -> str:
    """One calibration in words (the viewer's lines joined)."""
    return "\n".join(text for text, _, _ in calibration_lines(path))


# ---- removing: the untouched original always stays in data/trash/ ------------
TRASH_DIR = DATA_DIR / "trash"


def keep_original(path: Path) -> None:
    """Before a file's first edit, copy it to data/trash/ (undo = copy it back)."""
    backup = TRASH_DIR / path.name
    if not backup.exists():
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def remove_samples(path: Path, indexes: set[int]) -> int:
    """Drop the samples numbered `indexes` (as calibration_lines counts them) and
    rewrite the file in its own format. Returns how many were dropped."""
    keep_original(path)
    rec = load_calibration(path)
    n, kept = 0, {}
    for part in PARTS:
        kept[part] = []
        for s in getattr(rec, part):
            if n not in indexes:
                kept[part].append(s)
            n += 1
    if kept["dots"]:
        save_calibration(CalibrationRecording(**{p: tuple(v) for p, v in kept.items()}), path)
    else:
        remove_calibration(path)        # no dots left: the file goes too
    return n - sum(len(v) for v in kept.values())


def remove_calibration(path: Path) -> Path:
    """Move a calibration out of data/ into data/trash/, so it no longer trains the model."""
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    target = TRASH_DIR / path.name
    if target.exists():
        path.unlink()                   # the untouched original is already there
    else:
        shutil.move(str(path), str(target))
    return target


# ---- history: the AI's results, one file per reading ------------------------
def new_reading_path() -> Path:
    return HISTORY_DIR / f"reading_{datetime.now():{STAMP}}.json"


def save_reading(path: Path, record: dict) -> None:
    """record: {"title": the text's first words, "when": iso time, "summary": str,
    "questions": [{"question", "model_answer", "answer"?, "score"?, "explanation"?}, ...]}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def list_readings() -> list[Path]:
    """Every saved reading (summary + quiz), newest first."""
    return sorted(HISTORY_DIR.glob("reading_*.json"), reverse=True)


def reading_title(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["title"]
    except (OSError, ValueError, KeyError, TypeError):
        return path.name


def reading_messages(path: Path) -> list[tuple[str, bool]]:
    """A saved reading as chat messages (text, mine): the summary and the questions are
    the assistant's, your answers are yours."""
    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError) as e:
        return [(f"Can't read {path.name} ({e}).", False)]
    out = [(r["summary"], False)] if r.get("summary") else []
    for i, q in enumerate(r.get("questions", []), 1):
        out.append((f"Question {i}: {q['question']}", False))
        if "answer" in q:
            out.append((q["answer"], True))
        if "score" in q:
            out.append((f"Score {q['score']}/10. {q['explanation']}\nModel answer: {q['model_answer']}", False))
        elif "answer" in q:
            out.append((f"Model answer: {q['model_answer']}", False))
    return out


