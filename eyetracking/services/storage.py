"""
What the lab writes to disk, and reads back in words. By person, then by calibration type:

    data/<person>/                                    person mode on (Settings > Person mode)
    data/                                             person mode off: the same folders straight in data/
      <type>/<type>_<date>_<time>.json                a calibration: what the type's model learns from
      <type>/gaze-check_<type>[_<n>-calibrations]_<date>_<time>.csv
                                                      a gaze check of that type's model, trained on n calibrations
      <type>/disabled/                                calibrations kept, but turned off by the user
      <type>/trash/                                   removed files, and the originals of edited ones
      blink-test/blink-scores_<date>_<time>.csv       blink tests (they belong to no type)
      gaze-videos/gaze-video_<date>_<time>.mp4 + .json  record-mode gaze checks (services/recording.py)
    data/trash/<date>_<time>/<person or data>/...      everything cleared with Data > Clear data (move back to undo)

<type> is a method id from core/methods.py: original-3x3, original-4x4, original-5x5,
sweep, points, line-model, and the random twins random-3x3 ... random-line-model. So a
file name says which type made it and when, and its folders say whose it is. Missing
folders are made at start (ensure_layout): a person's folder can be cut out and moved
away by hand. The other branches' folders in data/ (calibrations/, gaze_checks/,
blink_scores/, history/) are never read or moved here, and can't be used as a person's name.

Every calibration file is in one format (FORMAT below), with its type and grid size inside:

    {"format": "gaze-calibration-v3", "method": "sweep", "grid": 4, "saved_at": ...,
     "sweep": [{"features": [...12], "target_x": .., "target_y": ..}, ...],   step 1 (empty for Original)
     "head":  [...the same, at the centre dot...],                              step 2 (empty for Original)
     "dots":  [...the same, at the grid dots...]}

A gaze check is a csv of index, x, y (where the model said), target_x, target_y (the lit ring).
"""
from __future__ import annotations

import csv
import functools
import json
import math
import os
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
BASE_DIR = DATA_DIR                         # data/ itself; use_person() moves DATA_DIR into data/<person>/
BLINK = "blink-test"                        # the blink tests' folder, beside the method folders
VIDEOS = "gaze-videos"                      # record-mode recordings of the gaze check
KINDS = (*METHOD_BY_ID, BLINK, VIDEOS)      # the folders in a person's folder (or in data/ with person mode off)
# never a person: the lab's own folders, and the other branches' folders in the shared data/
RESERVED = {k.lower() for k in KINDS} | {"calibrations", "gaze_checks", "blink_scores", "history", "people",
                                         "disabled", "trash", "older-versions"}
FORMAT = "gaze-calibration-v3"              # 12 features per frame
PARTS = ("sweep", "head", "dots")
STEP_NAMES = {"sweep": "Step 1, the eye range", "head": "Step 2, the head moving, eyes on the centre dot",
              "dots": "The dot grid"}


def file_date(path: Path) -> datetime:
    """sweep_2026-09-29_140312.json -> 29 Sep 2026 14:03:12 (any of the dated file names)."""
    return datetime.strptime(path.stem[-17:], STAMP)


def _method_folder(path: Path) -> Path:
    """The method folder a file sits in (a disabled / trash file: the folder above)."""
    return path.parent.parent if path.parent.name in ("disabled", "trash") else path.parent


def method_of(path: Path) -> str:
    """The method a file belongs to: its folder's name."""
    return _method_folder(path).name


# ---- person mode: each person's files in their own folder -----------------------
def person_name(text: str) -> str:
    """What a typed name becomes as a folder: letters, digits, spaces, - and _ only
    (no "..", slashes or characters Windows refuses). "" = no usable name, also for
    a name taken by a data folder (a method, blink-test, the other branches' folders)."""
    name = re.sub(r"[^\w \-]", "", text).strip(" .")
    return "" if name.lower() in RESERVED else name


def use_person(name: str) -> Path:
    """'' = files straight under data/; a name = under data/<name>/ (the same method
    folders inside). Returns the folder used now. Every function below reads DATA_DIR
    at call time, so this switches them all."""
    global DATA_DIR, BLINK_DIR
    name = person_name(name)
    DATA_DIR = BASE_DIR / name if name else BASE_DIR
    BLINK_DIR = DATA_DIR / BLINK
    return DATA_DIR


BLINK_DIR = DATA_DIR / BLINK


def person_root(name: str) -> Path:
    """A person's folder ('' = data/ itself)."""
    name = person_name(name)
    return BASE_DIR / name if name else BASE_DIR


def person_of(path: Path) -> str:
    """Whose file it is: the person folder above its method folder ('' = straight under data/)."""
    root = _method_folder(path).parent
    return "" if root == BASE_DIR else root.name


def person_roots() -> list[tuple[str, Path]]:
    """Every place data can be: ("", data/) first, then each person A to Z with their folder."""
    return [("", BASE_DIR)] + [(name, BASE_DIR / name) for name in list_people()]


def list_people() -> list[str]:
    """The people with a folder in data/ (a folder holding method or blink-test folders), A to Z."""
    if not BASE_DIR.is_dir():
        return []
    return sorted((p.name for p in BASE_DIR.iterdir()
                   if p.is_dir() and p.name.lower() not in RESERVED and any((p / k).is_dir() for k in KINDS)),
                  key=str.lower)


def ensure_layout() -> list[Path]:
    """Create every folder of the current person (or of data/) that is missing: all of
    them for a new person, or the one cut out to keep someone's files apart. Returns the
    ones made. Saving also creates its own folder, so a cut never breaks a save; this
    only makes the empty skeleton visible again. Safe at every start."""
    made = [DATA_DIR / k for k in KINDS if not (DATA_DIR / k).is_dir()]
    for f in made:
        f.mkdir(parents=True)
    return made


# ---- clearing: everything of one person (or of everyone) into one dated trash folder ----------
def clear_data(person: str | None) -> Path:
    """Move all of a person's recorded files ('' = the ones straight under data/, None = everyone's)
    into data/trash/<date>_<time>/<person or data>/, folders kept, so a clear can be undone by moving
    them back. The person in use gets their empty folders again. Returns the trash folder."""
    stamp = BASE_DIR / "trash" / f"{datetime.now():{STAMP}}"
    for name, root in person_roots():
        if person is not None and name != person_name(person):
            continue
        target = stamp / (name or "data")
        target.mkdir(parents=True, exist_ok=True)
        for kind in KINDS:
            if (root / kind).is_dir():
                shutil.move(str(root / kind), str(target / kind))
        if name and root.is_dir() and not any(root.iterdir()):
            root.rmdir()
    ensure_layout()
    return stamp


# ---- calibrations ---------------------------------------------------------------
def method_dir(method_id: str, root: Path | None = None) -> Path:
    """A method's folder: its calibrations and its gaze checks (of the person in use, or under `root`)."""
    return (root or DATA_DIR) / method_id


def save_calibration(rec: CalibrationRecording, method_id: str, path: Path | None = None,
                     dots_from: str | None = None) -> Path:
    """Write one calibration to its method's folder (or `path`). Returns the path.
    Nothing is written without dot samples. `dots_from`: the file the dots were reused
    from (a Sweep / Points / Line model calibration that skipped its own dots)."""
    if not rec.dots:
        raise ValueError("no dot samples: nothing worth saving")
    now = datetime.now()
    if path is None:
        path = method_dir(method_id) / f"{method_id}_{now:{STAMP}}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    grid = METHOD_BY_ID[method_id].grid if method_id in METHOD_BY_ID else 5
    data = {"format": FORMAT, "method": method_id, "grid": grid, "saved_at": now.isoformat(timespec="seconds")}
    if dots_from:
        data["dots_from"] = dots_from
    data |= {part: [asdict(s) for s in getattr(rec, part)] for part in PARTS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))   # compact: no indent (0.78 -> 0.46 MB, 84 -> 23 ms for a Sweep)
    return path


def load_calibration(path: Path) -> CalibrationRecording:
    """Read one file. Raises ValueError if it isn't in the current FORMAT."""
    return _load(path)[0]


def _load(path: Path) -> tuple[CalibrationRecording, int, str | None, str | None]:
    """(the recording, its grid size, the file its dots were reused from or None, the file it is a copy of or None)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"not in the current format ({FORMAT})")
    # JSON has no tuples: features come back as a list, so turn it back
    parts = [tuple(CalibrationSample(tuple(s["features"]), s["target_x"], s["target_y"])
                   for s in data[part]) for part in PARTS]
    return CalibrationRecording(*parts), int(data.get("grid", 5)), data.get("dots_from"), data.get("same_as")


def copy_as(path: Path, method_id: str) -> Path:
    """The same recording saved as another type's calibration (Sweep / Points / Line model share theirs),
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
    """A method's active calibrations (the ones its model learns from), newest first."""
    return sorted(method_dir(method_id, root).glob(f"{method_id}_*.json"), reverse=True)


def list_disabled(method_id: str, root: Path | None = None) -> list[Path]:
    """A method's calibrations the user turned off, newest first."""
    return sorted((method_dir(method_id, root) / "disabled").glob(f"{method_id}_*.json"), reverse=True)


def load_all_calibrations(method_id: str, root: Path | None = None) -> list[CalibrationRecording]:
    """Every readable active calibration of a method (of the person in use, or under `root`), NEWEST
    FIRST. Unreadable files are skipped, with a message."""
    return load_calibrations(list_calibrations(method_id, root))


def load_calibrations(paths) -> list[CalibrationRecording]:
    """These files, read in order. One that can't be read (damaged, hand-edited, or moved or removed on
    the Data page while a model trains) is skipped, with a message."""
    recordings = []
    for path in paths:
        try:
            recordings.append(load_calibration(path))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"Skipping {path.name}: {e}")
    return recordings


def dots_source(method, root: Path | None = None) -> Path | None:
    """The newest usable 4x4 calibration a Sweep / Points / Line model calibration can take its
    dots from: Original 4x4 for an in-order type, Random 4x4 for a random twin. None if there
    is none (or the type does its own dots anyway)."""
    if method.kind == "grid":
        return None
    source = "random-4x4" if method.random_order else "original-4x4"
    return next((p for p in list_calibrations(source, root) if calibration_usable(p)), None)


def calibration_usable(path: Path) -> bool:
    """True if the lab can train on the file: the current format. Remembered per file version (its
    size and modification time), so the lists, the chips and the reuse switch don't parse every file
    again on each click: 42 files were ~0.5 s a click on 19 Oct (several seconds on a cold start)."""
    try:
        st = path.stat()
    except OSError:
        return False
    return _usable(str(path), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=4096)
def _usable(path: str, size: int, mtime_ns: int) -> bool:
    try:
        load_calibration(Path(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def set_calibration_enabled(path: Path, enabled: bool) -> Path:
    """Move a calibration between its method's folder (trains the model) and that
    folder's disabled/ (kept, doesn't). Training only reads the method's folder, so a
    disabled file is left out without any check in the training code. Returns the new path."""
    folder = _method_folder(path)
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
        rec, grid, dots_from, same_as = _load(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [(f"Can't show this file: it was saved by a different version ({e}).", None, None)]
    method = METHOD_BY_ID.get(method_of(path))
    lines = [(f"{method.name if method else method_of(path)}, {file_date(path):%d %b %Y, %H:%M}"
              + (f" (the dots reused from {dots_from})" if dots_from else "")
              + (f" (the same recording as {same_as})" if same_as else ""), None, None)]
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
    return _method_folder(path) / "trash"


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
    rec, _, dots_from, _ = _load(path)
    n, kept = 0, {}
    for part in PARTS:
        kept[part] = []
        for s in getattr(rec, part):
            if n not in indexes:
                kept[part].append(s)
            n += 1
    if kept["dots"]:
        save_calibration(CalibrationRecording(**{p: tuple(v) for p, v in kept.items()}), method_of(path), path, dots_from)
    else:
        remove_calibration(path)        # no dots left: the file goes too
    return n - sum(len(v) for v in kept.values())


def list_removed(method_id: str, root: Path | None = None) -> list[Path]:
    """A method's calibrations in its trash/ (removed, or the untouched original of an edited one), newest first."""
    return sorted((method_dir(method_id, root) / "trash").glob(f"{method_id}_*.json"), reverse=True)


def restore(path: Path) -> Path:
    """A file in a trash/ folder back into its folder. A file of the same name there is the edited
    version of this original: it is replaced. Returns the new path."""
    target = _method_folder(path) / path.name
    os.replace(path, target)
    return target


def move_to_person(path: Path, person: str) -> Path:
    """A file into the same folder of another person ('' = straight under data/): the same type,
    and disabled/ or trash/ if it was there. Never overwrites: FileExistsError. Returns the new path."""
    folder = _method_folder(path)
    target = person_root(person) / folder.name / path.relative_to(folder)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def remove_calibration(path: Path) -> Path:
    """Move a calibration into its method's trash/, so it no longer trains the model."""
    target = _trash_dir(path) / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        path.unlink()                   # the untouched original is already there
    else:
        shutil.move(str(path), str(target))
    return target


# ---- gaze checks: one person's model of one calibration type, on the same rings -------
CHECK_GRID = 4     # every type is checked on the 4x4 rings (16), one lit at a time: the same test for all
MULTI_GRID = 6     # the multiple-model check: every model at once, on 6x6 rings (36)
Check = tuple[int, float, float, float, float]   # index, where the model said (x, y), the lit ring (x, y)


def save_gaze_check(samples: list[Check], method_id: str, folder: Path | None = None,
                    calibrations: int | None = None, tag: str = "") -> Path:
    """(index, x, y, ring x, ring y) rows -> <method>/gaze-check_<method>[_<tag>][_<n>-calibrations]_<time>.csv,
    n = how many calibrations trained the model; tag = "multi" (every model at once) or
    "video-<recording time>" (run over a recording). Returns the path."""
    folder = folder or method_dir(method_id)
    folder.mkdir(parents=True, exist_ok=True)
    used = f"_{calibrations}-calibrations" if calibrations else ""
    path = folder / f"gaze-check_{method_id}{'_' + tag if tag else ''}{used}_{datetime.now():{STAMP}}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "x", "y", "target_x", "target_y"])
        w.writerows((i, f"{x:.4f}", f"{y:.4f}", f"{tx:.4f}", f"{ty:.4f}") for i, x, y, tx, ty in samples)
    return path


def list_gaze_checks(method_id: str, root: Path | None = None) -> list[Path]:
    """One calibration type's gaze checks (of the person in use, or under `root`), newest first."""
    return sorted(method_dir(method_id, root).glob("gaze-check_*.csv"), key=file_date, reverse=True)


def read_gaze_check(path: Path) -> list[Check]:
    """The rows of a gaze check. A file from before the lit rings (index, x, y only) gets the
    nearest 5x5 ring as its target: an estimate, it can't tell where the eyes really were."""
    positions = grid_positions(CHECK_GRID)
    near = lambda v: min(positions, key=lambda p: abs(p - v))
    with open(path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    return [(int(r[0]), float(r[1]), float(r[2]), *((float(r[3]), float(r[4])) if len(r) >= 5
                                                     else (near(float(r[1])), near(float(r[2])))))
            for r in rows]


def check_calibrations(path: Path) -> int | None:
    """How many calibrations trained the model a gaze check tested (from its name; None if not recorded)."""
    found = re.search(r"_(\d+)-calibrations_", path.name)
    return int(found.group(1)) if found else None


def usable_calibrations(method_id: str, root: Path | None = None) -> int:
    """How many of a type's active calibrations can train a model: the most a gaze check can use."""
    return sum(calibration_usable(p) for p in list_calibrations(method_id, root))


def check_grid(rows: list[Check]) -> int:
    """Which rings a gaze check used (4x4, 6x6 ...), from its targets: every one on that grid."""
    for n in (CHECK_GRID, MULTI_GRID, 5, 3):
        if rows and all(tx in grid_positions(n) and ty in grid_positions(n) for _, _, _, tx, ty in rows):
            return n
    return CHECK_GRID


def check_source(path: Path) -> str:
    """How a gaze check was made, in words: "", "with every model at once" or "from the recording of <time>"."""
    if "_multi_" in path.name:
        return " with every model at once"
    found = re.search(r"_video-(\d{4}-\d{2}-\d{2}_\d{6})_", path.name)
    return f" from the recording of {datetime.strptime(found.group(1), STAMP):%d %b %Y, %H:%M}" if found else ""


def check_error(rows: list[Check]) -> float | None:
    """The mean distance from where the model said to the lit ring, as a fraction of the screen."""
    return sum(math.hypot(x - tx, y - ty) for _, x, y, tx, ty in rows) / len(rows) if rows else None


def gaze_check_lines(path: Path) -> list[str]:
    """A gaze check in words: whose model, which type, the average error, then each sample."""
    try:
        rows = read_gaze_check(path)
    except (OSError, ValueError, IndexError) as e:
        return [f"Can't read {path.name} ({e})."]
    method = METHOD_BY_ID.get(method_of(path))
    who = person_of(path) or "No person"
    error = check_error(rows)
    used = check_calibrations(path)
    trained = f" (trained on the newest {used} calibration{'s' if used != 1 else ''})" if used else ""
    grid = check_grid(rows)
    lines = [f"Gaze check of {who}'s {method.name if method else method_of(path)} model{trained}{check_source(path)}, "
             f"{file_date(path):%d %b %Y, %H:%M}: {len(rows)} samples on {grid}x{grid} rings, average error "
             f"{error or 0:.1%} of the screen (from where the model said to the lit ring).", ""]
    for i, x, y, tx, ty in rows:
        lines.append(f"#{i}: looking at {describe_target(tx, ty, grid)}; the model said {x:.0%} across, "
                     f"{y:.0%} down, {math.hypot(x - tx, y - ty):.1%} of the screen away")
    return lines


def method_results(root: Path | None = None) -> list[tuple[str, int, int, float | None, float | None]]:
    """For one person (or data/), per calibration type: (method id, calibrations, gaze checks,
    the latest check's error, the mean error over every check) - the comparison of the methods."""
    out = []
    for m in METHOD_BY_ID:
        errors = []
        for path in list_gaze_checks(m, root):
            try:
                errors.append(check_error(read_gaze_check(path)))
            except (OSError, ValueError, IndexError):
                pass
        errors = [e for e in errors if e is not None]
        out.append((m, len(list_calibrations(m, root)), len(errors), errors[0] if errors else None,
                    sum(errors) / len(errors) if errors else None))
    return out


# ---- recordings: record-mode gaze checks (the video + the presses) --------------
def list_recordings(root: Path | None = None) -> list[Path]:
    """A person's recordings (their json), newest first."""
    return sorted(((root or DATA_DIR) / VIDEOS).glob("gaze-video_*.json"), key=file_date, reverse=True)


def recording_label(path: Path) -> str:
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return f"{file_date(path):%d %b %Y, %H:%M}: {len(meta['presses'])} samples on {meta['grid']}x{meta['grid']} rings"
    except (OSError, ValueError, KeyError):
        return f"{file_date(path):%d %b %Y, %H:%M}: can't read"


# ---- blink scores: from the blink test ------------------------------------------
def save_blink_scores(rows: list[blink_scores.Row], folder: Path | None = None) -> Path:
    """(state, ms, left, right) rows -> blink-test/blink-scores_<time>.csv."""
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
