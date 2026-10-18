"""
What the lab writes to disk, and reads back in words.

    data/                                                   (person mode: data/people/<name>/, same tree inside)
      calibrations/<method>/<method>_<date>_<time>.json    what a method's model learns from
      calibrations/<method>/disabled/...                   kept, but turned off by the user
      calibrations/<method>/trash/...                      removed files, and the originals of edited ones
      calibrations/older-versions/...                      files from before the lab (not used)
      gaze_checks/<method>/gaze-check_<method>_<date>_<time>.csv
      blink_scores/blink-scores_<date>_<time>.csv

<method> is a method id from core/methods.py (original-3x3, original-4x4,
original-5x5, sweep, points, line-model), so a file name says which method made
it and when. Every calibration file is in one format (FORMAT below), with the
method and its grid size inside:

    {"format": "gaze-calibration-v3", "method": "sweep", "grid": 5, "saved_at": ...,
     "sweep": [{"features": [...12], "target_x": .., "target_y": ..}, ...],   step 1 (empty for Original)
     "head":  [...the same, at the centre dot...],                              step 2 (empty for Original)
     "dots":  [...the same, at the grid dots...]}
"""
from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core import blink_scores
from eyetracking.core.calibration import describe_target, grid_positions
from eyetracking.core.features import describe_features, head_degrees
from eyetracking.core.methods import METHOD_BY_ID
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from path import DATA_DIR

STAMP = "%Y-%m-%d_%H%M%S"                   # the date-time part of every file name
BASE_DIR = DATA_DIR                         # data/ itself; use_person() moves the folders below
CALIB_DIR = DATA_DIR / "calibrations"
GAZE_CHECK_DIR = DATA_DIR / "gaze_checks"
BLINK_DIR = DATA_DIR / "blink_scores"
OLDER = "older-versions"                    # where files from before the lab go
FORMAT = "gaze-calibration-v3"              # 12 features per frame
PARTS = ("sweep", "head", "dots")
STEP_NAMES = {"sweep": "Step 1, the eye range", "head": "Step 2, the head moving, eyes on the centre dot",
              "dots": "The dot grid"}


def file_date(path: Path) -> datetime:
    """sweep_2026-09-29_140312.json -> 29 Sep 2026 14:03:12 (any of the dated file names)."""
    return datetime.strptime(path.stem[-17:], STAMP)


def method_of(path: Path) -> str:
    """The method a file belongs to: its folder (a disabled / trash file: the folder above)."""
    folder = path.parent
    if folder.name in ("disabled", "trash"):
        folder = folder.parent
    return folder.name


# ---- person mode: each person's files in their own folder -----------------------
def person_name(text: str) -> str:
    """What a typed name becomes as a folder: letters, digits, spaces, - and _ only
    (no "..", slashes or characters Windows refuses). "" = no usable name."""
    return re.sub(r"[^\w \-]", "", text).strip(" .")


def use_person(name: str) -> Path:
    """'' = files straight under data/; a name = under data/people/<name>/ (the same
    calibrations / gaze_checks / blink_scores tree inside). Returns the folder used now.
    Every function below reads the folders at call time, so this switches them all."""
    global DATA_DIR, CALIB_DIR, GAZE_CHECK_DIR, BLINK_DIR
    name = person_name(name)
    DATA_DIR = BASE_DIR / "people" / name if name else BASE_DIR
    CALIB_DIR, GAZE_CHECK_DIR, BLINK_DIR = DATA_DIR / "calibrations", DATA_DIR / "gaze_checks", DATA_DIR / "blink_scores"
    return DATA_DIR


def list_people() -> list[str]:
    """The people with a folder under data/people/, A to Z."""
    folder = BASE_DIR / "people"
    return sorted((p.name for p in folder.iterdir() if p.is_dir()), key=str.lower) if folder.is_dir() else []


def ensure_layout() -> list[Path]:
    """Create every data folder that is missing (the whole data/ tree, or one method's
    folder cut out to keep another person's files apart) and return the ones made.
    Saving also creates its own folder, so a cut never breaks a save; this only makes
    the empty skeleton visible again. Safe at every start."""
    folders = [BLINK_DIR] + [d / m for m in METHOD_BY_ID for d in (CALIB_DIR, GAZE_CHECK_DIR)]
    made = [f for f in folders if not f.is_dir()]
    for f in made:
        f.mkdir(parents=True, exist_ok=True)      # a test may point two kinds at one folder
    return made


def migrate_layout() -> int:
    """Files from before the lab (calibration_*.json loose in data/ or data/calibrations/,
    gaze_check_*.csv loose in data/ or data/gaze_checks/) are moved into an
    older-versions folder, so they stay but don't mix with the lab's files. Safe at every start."""
    moves = [(DATA_DIR, "calibration_*.json", CALIB_DIR / OLDER), (CALIB_DIR, "calibration_*.json", CALIB_DIR / OLDER),
             (DATA_DIR, "gaze_check_*.csv", GAZE_CHECK_DIR / OLDER), (GAZE_CHECK_DIR, "gaze_check_*.csv", GAZE_CHECK_DIR / OLDER),
             (DATA_DIR, "blink_scores_*.csv", BLINK_DIR)]
    n = 0
    for folder, pattern, target in moves:
        for path in list(folder.glob(pattern)):
            target.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target / path.name))
            n += 1
    return n


# ---- calibrations ---------------------------------------------------------------
def method_dir(method_id: str) -> Path:
    return CALIB_DIR / method_id


def save_calibration(rec: CalibrationRecording, method_id: str, path: Path | None = None) -> Path:
    """Write one calibration to its method's folder (or `path`). Returns the path.
    Nothing is written without dot samples."""
    if not rec.dots:
        raise ValueError("no dot samples: nothing worth saving")
    now = datetime.now()
    if path is None:
        path = method_dir(method_id) / f"{method_id}_{now:{STAMP}}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = METHOD_BY_ID[method_id].grid if method_id in METHOD_BY_ID else 5
    data = {"format": FORMAT, "method": method_id, "grid": grid, "saved_at": now.isoformat(timespec="seconds")}
    data |= {part: [asdict(s) for s in getattr(rec, part)] for part in PARTS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_calibration(path: Path) -> CalibrationRecording:
    """Read one file. Raises ValueError if it isn't in the current FORMAT."""
    return _load(path)[0]


def _load(path: Path) -> tuple[CalibrationRecording, int]:
    """(the recording, its grid size)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"not in the current format ({FORMAT})")
    # JSON has no tuples: features come back as a list, so turn it back
    parts = [tuple(CalibrationSample(tuple(s["features"]), s["target_x"], s["target_y"])
                   for s in data[part]) for part in PARTS]
    return CalibrationRecording(*parts), int(data.get("grid", 5))


def list_calibrations(method_id: str) -> list[Path]:
    """A method's active calibrations (the ones its model learns from), newest first."""
    return sorted(method_dir(method_id).glob(f"{method_id}_*.json"), reverse=True)


def list_disabled(method_id: str) -> list[Path]:
    """A method's calibrations the user turned off, newest first."""
    return sorted((method_dir(method_id) / "disabled").glob(f"{method_id}_*.json"), reverse=True)


def load_all_calibrations(method_id: str) -> list[CalibrationRecording]:
    """Every readable active calibration of a method, NEWEST FIRST. Unreadable files are skipped, with a message."""
    recordings = []
    for path in list_calibrations(method_id):
        try:
            recordings.append(load_calibration(path))
        except (OSError, ValueError, KeyError, TypeError) as e:   # damaged, hand-edited
            print(f"Skipping {path.name}: {e}")
    return recordings


def calibration_usable(path: Path) -> bool:
    """True if the lab can train on the file: the current format."""
    try:
        load_calibration(path)
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def set_calibration_enabled(path: Path, enabled: bool) -> Path:
    """Move a calibration between its method's folder (trains the model) and that
    folder's disabled/ (kept, doesn't). Training only reads the method's folder, so a
    disabled file is left out without any check in the training code. Returns the new path."""
    folder = method_dir(method_of(path))
    if not enabled:
        folder = folder / "disabled"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    if target != path:
        shutil.move(str(path), str(target))
    return target


# ---- a calibration in words ----------------------------------------------------
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
        rec, grid = _load(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [(f"Can't show this file: it was saved by a different version ({e}).", None, None)]
    method = METHOD_BY_ID.get(method_of(path))
    lines = [(f"{method.name if method else method_of(path)}, {file_date(path):%d %b %Y, %H:%M}", None, None)]
    n = 0
    for part in PARTS:
        samples = getattr(rec, part)
        if not samples:
            n += 0
            continue
        name = STEP_NAMES[part] if part != "dots" else f"The {grid}x{grid} dot grid"
        lines += [("", None, None), (f"{name}: {len(samples)} samples. {_range_words(samples)}", None, None)]
        by_dot: dict = {}
        for s in samples:
            by_dot.setdefault((s.target_x, s.target_y), []).append((n, s))
            n += 1
        grouped = len(by_dot) <= 50       # a moving dot (the sweep) has a target per frame: no per-dot lines
        for (tx, ty), group in by_dot.items():
            key = f"{part} {tx} {ty}"
            if grouped:
                eyes = _spread([s.features[:4] for _, s in group])
                head = _spread([head_degrees(s.features) for _, s in group])
                lines.append((f"{describe_target(tx, ty, grid)}: {len(group)} sample{'s' if len(group) != 1 else ''}, "
                              f"eyes {'steady' if eyes < 0.02 else 'wandering'} (spread {eyes:.0%}), "
                              f"head {'still' if head < 2 else 'moving'} (±{head:.0f}°)", None, key))
            where = "" if grouped else f" ({describe_target(tx, ty, grid)})"
            lines += [(f"      #{i + 1}{where}: {describe_features(s.features)}", i, key) for i, s in group]
    return lines


def calibration_text(path: Path) -> str:
    """One calibration in words (the viewer's lines joined)."""
    return "\n".join(text for text, _, _ in calibration_lines(path))


# ---- removing: the untouched original always stays in the method's trash/ -------
def _trash_dir(path: Path) -> Path:
    return method_dir(method_of(path)) / "trash"


def keep_original(path: Path) -> None:
    """Before a file's first edit, copy it to its method's trash/ (undo = copy it back)."""
    backup = _trash_dir(path) / path.name
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
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
        save_calibration(CalibrationRecording(**{p: tuple(v) for p, v in kept.items()}), method_of(path), path)
    else:
        remove_calibration(path)        # no dots left: the file goes too
    return n - sum(len(v) for v in kept.values())


def remove_calibration(path: Path) -> Path:
    """Move a calibration into its method's trash/, so it no longer trains the model."""
    target = _trash_dir(path) / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        path.unlink()                   # the untouched original is already there
    else:
        shutil.move(str(path), str(target))
    return target


# ---- gaze checks: the samples recorded with Enter, per method ------------------
def save_gaze_check(samples: list[tuple[int, float, float]], method_id: str, folder: Path | None = None) -> Path:
    """(index, x, y) rows -> data/gaze_checks/<method>/gaze-check_<method>_<time>.csv. Returns the path."""
    folder = folder or GAZE_CHECK_DIR / method_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"gaze-check_{method_id}_{datetime.now():{STAMP}}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y"])
        w.writerows((i, f"{x:.4f}", f"{y:.4f}") for i, x, y in samples)
    return path


def list_gaze_checks() -> list[Path]:
    """Every method's gaze checks (and the older ones), newest first."""
    return sorted(GAZE_CHECK_DIR.glob("*/*.csv"), key=file_date, reverse=True)


def _nearest_dot(x: float, y: float, grid: int) -> tuple[int, int, float]:
    """(column, row) of the closest calibration dot of an n x n grid, counted from the
    top left, and how far it is (as a fraction of the screen)."""
    positions = grid_positions(grid)
    col = min(range(len(positions)), key=lambda i: abs(positions[i] - x))
    row = min(range(len(positions)), key=lambda i: abs(positions[i] - y))
    return col + 1, row + 1, math.hypot(positions[col] - x, positions[row] - y)


def gaze_check_lines(path: Path) -> list[str]:
    """A gaze check in words: one line per sample, with the nearest calibration dot and the distance."""
    try:
        with open(path, newline="") as f:
            rows = [(int(i), float(x), float(y)) for i, x, y in list(csv.reader(f))[1:]]
    except (OSError, ValueError) as e:
        return [f"Can't read {path.name} ({e})."]
    method = METHOD_BY_ID.get(method_of(path))
    grid = method.grid if method else 5
    near = [_nearest_dot(x, y, grid) for _, x, y in rows]
    average = sum(d for _, _, d in near) / len(near) if near else 0.0
    spacing = grid_positions(grid)[1] - grid_positions(grid)[0]
    lines = [f"Gaze check with {method.name if method else method_of(path)}, {file_date(path):%d %b %Y, %H:%M}: "
             f"{len(rows)} samples. If you were looking at the rings, the average distance to the nearest "
             f"calibration dot is the error: {average:.0%} of the screen (the dots are {spacing:.1%} apart).", ""]
    for (i, x, y), (col, row, d) in zip(rows, near):
        lines.append(f"#{i}: {x:.0%} across, {y:.0%} down; nearest calibration dot ({col}, {row}), {d:.0%} of the screen away")
    return lines


# ---- blink scores: from the blink test ------------------------------------------
def save_blink_scores(rows: list[blink_scores.Row], folder: Path | None = None) -> Path:
    """(state, ms, left, right) rows -> data/blink_scores/blink-scores_<time>.csv."""
    folder = folder or BLINK_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"blink-scores_{datetime.now():{STAMP}}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["state", "t_ms", "left", "right"])
        w.writerows((state, ms, f"{left:.4f}", f"{right:.4f}") for state, ms, left, right in rows)
    return path


def list_blink_scores() -> list[Path]:
    return sorted(BLINK_DIR.glob("blink*scores_*.csv"), key=file_date, reverse=True)


def load_blink_scores(path: Path) -> list[blink_scores.Row]:
    with open(path, newline="") as f:
        return [(state, int(ms), float(left), float(right)) for state, ms, left, right in list(csv.reader(f))[1:]]


def blink_score_lines(path: Path) -> list[str]:
    """A blink test in words: each state's scores, then the cut-off it suggests."""
    try:
        rows = load_blink_scores(path)
    except (OSError, ValueError) as e:
        return [f"Can't read {path.name} ({e})."]
    return [f"Blink test of {file_date(path):%d %b %Y, %H:%M}: {len(rows)} frames.", ""] + blink_scores.lines(rows)


def trash_file(path: Path) -> Path:
    """Move a recorded csv into a trash folder beside it (move it back to undo)."""
    trash = path.parent / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    target = trash / path.name
    target.unlink(missing_ok=True)
    shutil.move(str(path), str(target))
    return target
