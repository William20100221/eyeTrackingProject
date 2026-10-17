"""
Save and load calibration samples as JSON, so one calibration can be reused
for many model experiments without sitting through the dots again.

Files go in <project root>/data, named by date and time, e.g.
    data/calibration_2026-09-29_140312.json

File format (FORMAT below): {"format": "gaze-calibration-v1", "saved_at": ...,
"samples": [{"features": [...10], "target_x": .., "target_y": ..}, ...]}.
Files from before the tag have no "format" but do have "samples": they count as v1.
A file in any other format (the 3-step branches write v2 / v3) is not loaded.
    data/history/reading_2026-09-29_141502.json  (one per reading: the AI's summary + quiz)
"""
from __future__ import annotations

import json
import shutil
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from eyetracking.core.calibration import describe_target
from eyetracking.core.features import N_FEATURES, describe_features, head_degrees
from eyetracking.core.models import CalibrationSample
from path import DATA_DIR

STAMP = "%Y-%m-%d_%H%M%S"            # the date-time part of every file name
HISTORY_DIR = DATA_DIR / "history"   # the AI's summary + quiz, one file per reading
FORMAT = "gaze-calibration-v1"


def file_date(path: Path) -> datetime:
    """calibration_2026-09-29_140312.json -> 29 Sep 2026 14:03:12"""
    return datetime.strptime(path.stem.split("_", 1)[1], STAMP)


def save_calibration(samples: list[CalibrationSample], path: Path | None = None) -> Path:
    """Write samples to `path` (default: a new timestamped file in data/).
    Returns the path it wrote to. Nothing is written for an empty calibration."""
    if not samples:
        raise ValueError("no samples: nothing worth saving")
    now = datetime.now()
    if path is None:
        path = DATA_DIR / f"calibration_{now:{STAMP}}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "format": FORMAT,
        "saved_at": now.isoformat(timespec="seconds"),
        "samples": [asdict(s) for s in samples],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_calibration(path: Path) -> list[CalibrationSample]:
    """Read one file. Raises ValueError if it isn't in this version's FORMAT."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    tag = data.get("format", FORMAT if "samples" in data else None) if isinstance(data, dict) else None
    if tag != FORMAT:
        raise ValueError(f"not in the current format ({FORMAT})")
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


def _parts(samples: list[CalibrationSample]) -> list[tuple[str, tuple]]:
    return [("The dot grid", tuple(samples))]


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
    samples = load_calibration(path)
    kept = [s for i, s in enumerate(samples) if i not in indexes]
    if kept:
        save_calibration(kept, path)
    else:
        remove_calibration(path)        # nothing left: the file goes too
    return len(samples) - len(kept)


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


def latest_calibration() -> Path | None:
    """Newest saved file, or None if nothing has been saved yet.
    Works because the date-time names sort in time order."""
    files = sorted(DATA_DIR.glob("calibration_*.json"))
    return files[-1] if files else None
