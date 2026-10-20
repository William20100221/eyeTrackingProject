"""
What the tester version writes to disk, the same layout as the lab's, so its files drop into the lab's data/:

    data/<name>/<type>/<type>_<date>_<time>.json      a calibration (one of the 9 types in core/methods.py)
    data/<name>/<model>/gaze-check_<model>_multi_<n>-calibrations_<date>_<time>.csv
                                                      a gaze check of a model, trained on n grid calibrations
    data/outbox.txt                                   files saved but not emailed yet (services/mailer.py)

<type> is a calibration type: original-3x3, original-4x4, original-5x5, random-3x3, random-4x4,
random-5x5, sweep, points, line-model. A grid's own model has the grid's id, so its checks sit with its
calibrations; a paired model's folder (sweep-original-4x4/ ...) is made by its first check.

Every calibration file is in one format (FORMAT below), with its type and grid size inside:

    {"format": "gaze-calibration-v3", "method": "sweep", "grid": 0, "saved_at": ...,
     "sweep": [{"features": [...12], "target_x": .., "target_y": ..}, ...],   step 1 (empty for a grid)
     "head":  [...the same, at the centre dot...],                              step 2 (empty for a grid)
     "dots":  [...the same, at the grid dots...]}                               (empty for a special)

A gaze check is a csv of index, x, y (where the model said), target_x, target_y (the lit ring).
"""
from __future__ import annotations

import csv
import functools
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core.methods import LEGACY, METHOD_BY_ID, MODEL_BY_ID, Model
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.core.quality import MIN_HEAD_FRAMES, step_one_problem
from path import DATA_DIR

STAMP = "%Y-%m-%d_%H%M%S"                   # the date-time part of every file name
BASE_DIR = DATA_DIR                         # data/ itself; use_person() moves DATA_DIR into data/<name>/
# never a name: the type and model folders, and the lab's other folders in data/
RESERVED = {k.lower() for k in (*METHOD_BY_ID, *MODEL_BY_ID, *LEGACY)} | {
    "blink-test", "gaze-videos", "calibrations", "gaze_checks", "blink_scores", "history", "people", "disabled",
    "trash", "older-versions"}
FORMAT = "gaze-calibration-v3"              # 12 features per frame
PARTS = ("sweep", "head", "dots")


def _method_folder(path: Path) -> Path:
    """The method folder a file sits in (a disabled / trash file: the folder above)."""
    return path.parent.parent if path.parent.name in ("disabled", "trash") else path.parent


def method_of(path: Path) -> str:
    """The method a file belongs to: its folder's name."""
    return _method_folder(path).name


# ---- each tester's files in their own folder -------------------------------------
def person_name(text: str) -> str:
    """What a typed name becomes as a folder: letters, digits, spaces, - and _ only
    (no "..", slashes or characters Windows refuses). "" = no usable name, also for
    a name taken by a data folder (a type, a model, the lab's other folders)."""
    name = re.sub(r"[^\w \-]", "", text).strip(" .")
    return "" if name.lower() in RESERVED else name


def use_person(name: str) -> Path:
    """'' = files straight under data/; a name = under data/<name>/ (the same type
    folders inside). Returns the folder used now. Every function below reads DATA_DIR
    at call time, so this switches them all."""
    global DATA_DIR
    name = person_name(name)
    DATA_DIR = BASE_DIR / name if name else BASE_DIR
    return DATA_DIR


# ---- calibrations ---------------------------------------------------------------
def method_dir(method_id: str, root: Path | None = None) -> Path:
    """A method's folder: its calibrations and its gaze checks (of the person in use, or under `root`)."""
    return (root or DATA_DIR) / method_id


def save_calibration(rec: CalibrationRecording, method_id: str, path: Path | None = None) -> Path:
    """Write one calibration to its method's folder (or `path`). Returns the path.
    Nothing is written without dot samples (a grid) or head-step frames (a special)."""
    if not rec.dots and not rec.head:
        raise ValueError("no dot samples and no head step: nothing worth saving")
    now = datetime.now()
    if path is None:
        path = method_dir(method_id) / f"{method_id}_{now:{STAMP}}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = METHOD_BY_ID[method_id].grid if method_id in METHOD_BY_ID else 0
    if rec.dots and not grid:                       # an older special / old type's file (it ended on dots): their size
        grid = round(len({(s.target_x, s.target_y) for s in rec.dots}) ** 0.5)
    data = {"format": FORMAT, "method": method_id, "grid": grid, "saved_at": now.isoformat(timespec="seconds")}
    data |= {part: [asdict(s) for s in getattr(rec, part)] for part in PARTS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))   # compact: no indent (0.78 -> 0.46 MB, 84 -> 23 ms for a Sweep)
    return path


def load_calibration(path: Path) -> CalibrationRecording:
    """Read one file. Raises ValueError if it isn't in the current FORMAT."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"not in the current format ({FORMAT})")
    # JSON has no tuples: features come back as a list, so turn it back
    return CalibrationRecording(*[tuple(CalibrationSample(tuple(s["features"]), s["target_x"], s["target_y"])
                                        for s in data[part]) for part in PARTS])


def copy_as(path: Path, method_id: str) -> Path:
    """The same recording saved as another type's calibration (Points / Line model share theirs),
    in the same person's folder, with the same date and time and "same_as" naming the
    original. Never overwrites: FileExistsError. Returns the new path."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    target = _method_folder(path).parent / method_id / f"{method_id}_{path.stem[-17:]}.json"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    data |= {"method": method_id, "same_as": path.name}
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    return target


def list_calibrations(method_id: str, root: Path | None = None) -> list[Path]:
    """A method's calibrations (the ones its model learns from), newest first."""
    return sorted(method_dir(method_id, root).glob(f"{method_id}_*.json"), reverse=True)


def load_calibrations(paths) -> list[CalibrationRecording]:
    """These files, read in order. One that can't be read (damaged, hand-edited, or moved while a
    model trains) is skipped, with a message."""
    recordings = []
    for path in paths:
        try:
            recordings.append(load_calibration(path))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"Skipping {path.name}: {e}")
    return recordings


def calibration_usable(path: Path) -> bool:
    """True if a model can train on the file (calibration_problem says why not)."""
    return calibration_problem(path) is None


def calibration_problem(path: Path) -> str | None:
    """Why a calibration can't train a model, in words, or None. Cheap checks only (the full quality
    check runs when a model trains, on the worker thread): the current format; a grid has dots; a
    special has a head step long enough and a step 1 of its own kind (a gliding dot for Sweep, still
    dots for Points / Line model). Remembered per file version (size, modification time)."""
    try:
        st = path.stat()
    except OSError:
        return "the file is gone"
    return _problem(str(path), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=4096)
def _problem(path: str, size: int, mtime_ns: int) -> str | None:
    kind = method_of(Path(path))
    try:
        rec = load_calibration(Path(path))
    except (OSError, ValueError, KeyError, TypeError) as e:
        return f"it can't be read ({e})"
    if kind not in METHOD_BY_ID:
        return f"an old type ({LEGACY.get(kind, kind)})"
    method = METHOD_BY_ID[kind]
    if method.kind == "grid":
        return None if rec.dots else "no dot samples"
    if len(rec.head) < MIN_HEAD_FRAMES:
        return f"a head step of only {len(rec.head)} frames (need {MIN_HEAD_FRAMES})"
    return step_one_problem(rec.sweep, method.kind)


def usable_calibrations(method_id: str, root: Path | None = None) -> int:
    """How many of a type's calibrations can train a model."""
    return sum(calibration_usable(p) for p in list_calibrations(method_id, root))


def model_files(model: Model, root: Path | None = None) -> tuple[list[Path], list[Path]]:
    """What a model of a person trains on: their usable calibrations of its grid, newest first, and every
    usable calibration of its special (train() takes the newest that passes the full check)."""
    usable = lambda m: [p for p in list_calibrations(m, root) if calibration_usable(p)]
    return usable(model.grid.id), usable(model.special.id) if model.special else []


def model_ready(model: Model, root: Path | None = None) -> str | None:
    """None if a person has what the model trains on; else what is missing, in words."""
    grids, specials = model_files(model, root)
    missing = [m.name for m, files in ((model.grid, grids), (model.special, specials)) if m and not files]
    return f"needs a usable {' and a '.join(missing)} calibration" if missing else None


# ---- gaze checks: every model at once, on the same rings ---------------------------
CHECK_GRID = 4     # one model's check (gaze_window's runner): the 4x4 rings (16)
MULTI_GRID = 6     # the app's check: every model at once, on 6x6 rings (36)
Check = tuple[int, float, float, float, float]   # index, where the model said (x, y), the lit ring (x, y)


def save_gaze_check(samples: list[Check], method_id: str, folder: Path | None = None,
                    calibrations: int | None = None, tag: str = "") -> Path:
    """(index, x, y, ring x, ring y) rows -> <method>/gaze-check_<method>[_<tag>][_<n>-calibrations]_<time>.csv,
    n = how many calibrations trained the model; tag = "multi" (every model at once). Returns the path."""
    folder = folder or method_dir(method_id)
    folder.mkdir(parents=True, exist_ok=True)
    used = f"_{calibrations}-calibrations" if calibrations else ""
    path = folder / f"gaze-check_{method_id}{'_' + tag if tag else ''}{used}_{datetime.now():{STAMP}}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "target_x", "target_y"])
        w.writerows((i, f"{x:.4f}", f"{y:.4f}", f"{tx:.4f}", f"{ty:.4f}") for i, x, y, tx, ty in samples)
    return path
