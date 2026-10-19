"""
Audit a data folder for the 24 models (core/methods.py MODELS), read-only: per person, which
calibration files to DELETE, RESTORE, REPLACE (recalibrate) or ADD so that every model can be trained.

    python -m eyetracking.testing.audit_data <data folder>
    python -m eyetracking.testing.audit_data --self-check      (a made-up person in a temp folder)

Nothing is written, moved or deleted: the files are only read, and the report is printed.

Who: every person storage.list_people() finds, and the files straight under data/ ("no person").
What: each type folder (the 9 types) and each old type folder (random-sweep, random-points,
random-line-model), every active, disabled/ and trash/ file; also the files Data > Clear data put in
data/trash/<date>_<time>/<person>/ (listed; offered back only when nothing nearer serves). Per file:
  * format: readable, FORMAT, its method and grid the folder's, its name the folder's prefix, a grid's
    dot count and order (row by row for Original, shuffled for Random)
  * storage.calibration_problem: the cheap check (the lab's Not usable column)
  * the full quality check the training runs: quality.check_recording for a grid, check_special for a
    special with the folder's kind (an old type: its steps, as its new type's kind), and what it dropped
  * same_as (saved as a copy of which file) and dots_from (dots reused from which file)
  * step 1's kind (a gliding dot or still dots) against the folder's: a Points recording saved as Sweep
    is wrong data for Sweep since 20 Oct
  * identical content: the same step 1 + head step (one recording) or the same dots as another file
  * a special's head step: frames kept, and how far yaw / pitch / roll moved against what the head step
    asks (calibration.HEAD_MOVES); Sweep's pursuit lag; an old special's own or reused 4x4 dots
WEAK = passes the check, but the head step is thin: under WEAK_HEAD_FRAMES frames, or an angle whose
5-95% span is under WEAK_SPAN of the span the step asks (±target). Then per person and type: is there an
active file that passes the full check and is not weak? The lists DELETE, RESTORE, REPLACE / RECALIBRATE,
ADD and OPTIONAL MOVES name the files; and how many of the 24 models can train now, and after RESTORE.
Conservative: a file is put up for deletion only when its recording is kept in another file, or it is a
copy saved under the wrong kind, or it fails the check while the person has a usable file of that type.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import eyetracking.services.storage as storage
from eyetracking.core.calibration import HEAD_MOVES, point_grid
from eyetracking.core.head_correction import DEFAULT_K, fit_k
from eyetracking.core.methods import METHOD_BY_ID, METHODS, MODELS, partners, pursuit_samples
from eyetracking.core.quality import STILL_DOTS_MAX, check_recording, check_special

warnings.filterwarnings("ignore")             # spearmanr on a constant column, etc.

OLD = {"random-sweep": "sweep", "random-points": "points", "random-line-model": "line-model"}   # old type -> new type
# ponytail: thresholds read off the real head steps of 20 Oct (spans 13-66 deg); tune if a weak one trains well
WEAK_HEAD_FRAMES = 100        # kept head-step frames: 60 trains (MIN_HEAD_FRAMES), under ~100 the matrix is noisy
WEAK_SPAN = 0.8               # an angle's 5-95% span under 80% of the step's ±target (yaw 24, pitch / roll 16 deg)


@dataclass(eq=False)
class File:
    path: Path
    person: str
    folder: str                       # a type id, or an old type id
    where: str                        # active | disabled | trash | cleared <stamp>[/disabled|/trash]
    meta: dict = field(default_factory=dict)
    rec: object = None
    problems: list[str] = field(default_factory=list)    # format
    cheap: str | None = None                             # storage.calibration_problem
    ok: bool = False                                     # the full check, as the folder's kind
    reasons: list[str] = field(default_factory=list)
    weak: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    step1: str = "none"                                  # gliding | still | none
    keys: tuple = ("", "", "")                           # hashes of step 1, head step, dots

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def rel(self) -> str:
        """<type>/[disabled/ | trash/]<name>; a cleared file from trash/<stamp>/ on."""
        parts = self.path.parts
        if self.cleared:
            return "/".join(parts[parts.index("trash", len(parts) - 6):])
        return "/".join(parts[-2 if self.where == "active" else -3:])

    @property
    def cleared(self) -> bool:
        return self.where.startswith("cleared")

    @property
    def kind(self) -> str:
        """What the folder holds now: grid | sweep | points."""
        return METHOD_BY_ID[OLD.get(self.folder, self.folder)].kind

    @property
    def real_kind(self) -> str:
        """What step 1 holds: sweep (a gliding dot) | points (still dots) | grid (none)."""
        return {"gliding": "sweep", "still": "points"}.get(self.step1, "grid")

    @property
    def usable(self) -> bool:
        return self.ok and not self.problems

    @property
    def good(self) -> bool:
        return self.usable and not self.weak

    @property
    def recording(self) -> tuple:
        return self.keys[:2]


def digest(part) -> str:
    return hashlib.sha1(repr([(s.features, s.target_x, s.target_y) for s in part]).encode()).hexdigest()[:10] if part else ""


def dot_order(dots) -> str:
    """'in order' (row by row, the order they first appear in) or 'shuffled'."""
    seen = list(dict.fromkeys((s.target_x, s.target_y) for s in dots))
    return "in order" if seen == sorted(seen, key=lambda t: (t[1], t[0])) else "shuffled"


def head_words(head) -> tuple[str, list[str]]:
    """(what the head step measured, why it is weak)."""
    f = np.array([s.features for s in head], float)
    spans, weak = [], []
    for text, i, target, _ in HEAD_MOVES[:3]:              # yaw, pitch, roll (the distance is not in K)
        lo, hi = np.percentile(f[:, i] - np.median(f[:, i]), [5, 95])
        name = ("yaw", "pitch", "roll")[i - 4]
        spans.append(f"{name} {lo:+.0f}/{hi:+.0f} (sd {f[:, i].std():.1f})")
        if hi - lo < WEAK_SPAN * 2 * target:
            weak.append(f"{name} moved over only {hi - lo:.0f} deg (asks ±{target:.0f})")
    if len(head) < WEAK_HEAD_FRAMES:
        weak.append(f"only {len(head)} head-step frames kept")
    K = fit_k(head)
    corr = float(np.corrcoef(K.ravel(), DEFAULT_K.ravel())[0, 1]) if K is not None else float("nan")
    return f"head step {len(head)} frames, deg: " + ", ".join(spans) + f"; own K vs default K r={corr:+.2f}", weak


def audit_file(path: Path, person: str, folder: str, where: str) -> File:
    f = File(path, person, folder, where)
    try:
        f.meta = json.loads(path.read_text(encoding="utf-8"))
        f.rec, grid, _, _ = storage._load(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        f.problems.append(f"can't be read ({e})")
        return f
    m, rec = f.meta, f.rec
    if m.get("method") != folder:
        f.problems.append(f"method '{m.get('method')}' in a {folder}/ folder")
    if not path.name.startswith(folder + "_"):
        f.problems.append(f"name not prefixed {folder}_")
    f.keys = (digest(rec.sweep), digest(rec.head), digest(rec.dots))
    if rec.sweep:
        f.step1 = "gliding" if len({(s.target_x, s.target_y) for s in rec.sweep}) > STILL_DOTS_MAX else "still"
    if not f.cleared:
        f.cheap = storage.calibration_problem(path)
        if where != "active" and (path.parent.parent / path.name).exists():
            f.info.append("the untouched original of the edited active file of the same name")
    f.info.insert(0, f"{len(rec.sweep)} step-1 ({f.step1}) / {len(rec.head)} head / {len(rec.dots)} dot frames"
                  + (f", same_as {m['same_as']}" if m.get("same_as") else "")
                  + (f", dots_from {m['dots_from']}" if m.get("dots_from") else ""))
    if f.kind == "grid":
        method = METHOD_BY_ID[folder]
        report = check_recording(rec)
        f.ok, f.reasons = report.ok, report.reasons
        n, order = len({(s.target_x, s.target_y) for s in rec.dots}), dot_order(rec.dots)
        f.info.append(f"{n} dots ({order})")
        if n != method.dots or grid != method.grid:
            f.problems.append(f"{n} dots / grid {grid}, not {method.dots} / {method.grid}")
        if method.random_order != (order == "shuffled"):
            f.problems.append(f"dots {order} in a {method.name} folder")
        return f
    report = check_special(rec, f.kind)
    f.ok, f.reasons = report.ok, report.reasons
    if report.ok:
        words, f.weak = head_words(report.kept.head)
        f.info.append(words)
        if f.kind == "sweep":
            lags = [len(p) - len(pursuit_samples(p)) for p in
                    ([s for s in report.kept.sweep if s.target_y == 0.5], [s for s in report.kept.sweep if s.target_y != 0.5])]
            f.info.append(f"pursuit lag {lags[0]} frames across, {lags[1]} down")
        else:
            f.info.append(f"{len({(s.target_x, s.target_y) for s in report.kept.sweep})} of 13 line-dot places kept")
    if rec.dots:                               # an old special / old type ended on 4x4 dots: its own, or reused?
        f.info.append(f"its {len({(s.target_x, s.target_y) for s in rec.dots})} old dots: {dot_order(rec.dots)}, "
                      + (f"reused from {m['dots_from']}" if m.get("dots_from") else "its own")
                      + f", grid check {'passes' if check_recording(rec).ok else 'fails'} (unused since 20 Oct)")
    return f


def scan(data: Path) -> dict[str, list[File]]:
    """Every calibration file, by person ('' = straight under data/)."""
    folders = [m.id for m in METHODS] + list(OLD)
    people = {"": data, **{p: data / p for p in storage.list_people()}}
    out: dict[str, list[File]] = {p: [] for p in people}
    for person, root in people.items():
        for folder in folders:
            for sub, where in (("", "active"), ("disabled", "disabled"), ("trash", "trash")):
                out[person] += [audit_file(p, person, folder, where) for p in sorted((root / folder / sub).glob("*.json"))]
    for stamp in sorted(p for p in (data / "trash").glob("*") if p.is_dir()):     # Data > Clear data
        for who in sorted(p for p in stamp.iterdir() if p.is_dir()):
            person = "" if who.name == "data" else who.name
            for folder in folders:
                for sub in ("", "disabled", "trash"):
                    where = f"cleared {stamp.name}" + (f"/{sub}" if sub else "")
                    out.setdefault(person, []).extend(
                        audit_file(p, person, folder, where) for p in sorted((who / folder / sub).glob("*.json")))
    return out


def label(person: str) -> str:
    return person or "(no person: straight under data/)"


def why(f: File) -> str:
    return "; ".join(f.problems + f.reasons[-1:])


def plan(files: list[File], everyone: list[File]) -> tuple[dict[str, list[str]], list[File]]:
    """The lists for one person, and the files RESTORE brings back."""
    live = [f for f in files if not f.cleared]
    act = {k: [] for k in ("OK", "DELETE", "RESTORE", "REPLACE", "ADD", "OPTIONAL")}
    restored: list[File] = []
    deleted: set[int] = set()
    same = lambda f, g: f is not g and f.keys[1] and f.recording == g.recording
    newest_first = lambda fs: sorted(sorted(fs, key=lambda f: f.name[-22:], reverse=True), key=lambda f: (
        f.cleared, f.where != "active", not f.good, f.meta.get("same_as") is not None))    # the best, then the newest

    def delete(f: File, text: str) -> None:
        if id(f) not in deleted:
            deleted.add(id(f))
            act["DELETE"].append(f"{f.rel} ({f.where}): {text}")

    # 1. specials and old types: step 1 of the folder's kind? a copy of a recording kept elsewhere?
    for f in live:
        if f.kind == "grid" or f.rec is None:
            continue
        if f.step1 != "none" and f.real_kind != f.kind:
            homes = newest_first([g for g in everyone if same(f, g) and g.kind == f.real_kind])
            if homes:
                delete(f, f"a {f.real_kind.title()} recording saved as {f.folder}; the real one is {homes[0].rel}")
            else:
                act["OPTIONAL"].append(f"{f.rel}: the only copy of a {f.real_kind} recording, in the wrong folder: "
                                       f"move it to {f.real_kind}/ (prefix {f.real_kind}_, \"method\": \"{f.real_kind}\")")
        elif f.folder in OLD:
            twins = [g for g in everyone if same(f, g) and g.folder in METHOD_BY_ID and g.real_kind == g.kind]
            if twins:
                delete(f, f"old type; the same recording is {twins[0].rel}")
            elif f.meta.get("same_as") and any(same(f, g) for g in live):
                delete(f, f"old type, a copy of {f.meta['same_as']}, which stays")

    # 2. per type: an active file that passes and is not weak?
    for m in METHODS:
        mine = [f for f in files if f.folder == m.id]
        active = [f for f in mine if f.where == "active"]
        wrong = lambda f: f.kind != "grid" and f.step1 != "none" and f.real_kind != f.kind
        failing = [f for f in active if not f.usable and not wrong(f)]
        old = newest_first([f for f in live if OLD.get(f.folder) in {m.id, *(p.id for p in partners(m))} and f.usable
                            and id(f) not in deleted])
        move = (f"{m.name}: {old[0].rel}'s steps pass as {m.name}{' (weak)' if old[0].weak else ''}: move it to "
                f"{m.id}/ (prefix {m.id}_, \"method\": \"{m.id}\"; its old dots are ignored)") if old else ""
        if any(f.good for f in active):
            weak = [f.name for f in active if f.usable and f.weak]
            act["OK"].append(f"{m.name}: " + ", ".join(f.name for f in active if f.good)
                             + (f" (also, weak: {', '.join(weak)})" if weak else ""))
            for f in failing:
                delete(f, f"fails the check ({why(f)}); training already skips it")
            continue
        passing = [f for f in active if f.usable]
        spare = newest_first([f for f in mine if f.where != "active" and f.usable])
        mate = [f for p in partners(m) for f in live if f.folder == p.id and f.where == "active" and f.usable]
        if passing:
            act["REPLACE"].append(f"{m.name}: {passing[0].name} trains but is weak ({'; '.join(passing[0].weak)}): "
                                  "recalibrate, keep it until the new one is in" + (" - or the optional move" if old else ""))
        elif spare:
            g = spare[0]
            restored.append(g)
            act["RESTORE"].append(f"{m.name}: {g.rel}" + (" (Enable)" if g.where == "disabled" else
                                  f" (move it back to {m.id}/)" if g.cleared else " (Data > Removed > Restore)"))
            if g.weak:
                act["REPLACE"].append(f"{m.name}: the restored {g.name} is weak ({'; '.join(g.weak)}): recalibrate "
                                      "when you can" + (" - or the optional move, which is not weak" if old and old[0].good else ""))
        elif mate:
            act["ADD"].append(f"{m.name}: right-click > Copy to {m.name} on {mate[0].rel} (the same procedure, no "
                              f"recalibration), or calibrate {m.name}")
        elif failing:
            act["REPLACE"].append(f"{m.name}: recalibrate; {failing[0].name} fails: {why(failing[0])}")
        else:
            act["ADD"].append(f"{m.name}: calibrate it (none on disk)" + (" - or the optional move" if old else ""))
        if old and (passing or not spare or spare[0].weak):        # nothing good restored: the move is worth naming
            act["OPTIONAL"].append(move)

    # 3. old types' real recordings not yet named: a second calibration of their new type, or kept as they are
    for f in live:
        if f.folder in OLD and f.usable and id(f) not in deleted and f.meta.get("same_as") is None \
                and not any(o.startswith(f"{METHOD_BY_ID[OLD[f.folder]].name}: {f.rel}") for o in act["OPTIONAL"]):
            act["OPTIONAL"].append(f"{f.rel}: a real {METHOD_BY_ID[OLD[f.folder]].name} recording of an old type"
                                   + (" (in its trash)" if f.where == "trash" else "") + (", weak" if f.weak else "")
                                   + f": keep it; moved to {OLD[f.folder]}/ (prefix and \"method\" changed) it would "
                                   "be a second calibration of that type")
    # 4. failing files away from training (trash, disabled, old types), when the person has that type usable
    have = {f.folder for f in live if f.where == "active" and f.usable} | {f.folder for f in restored}
    for f in live:
        target = OLD.get(f.folder, f.folder)
        if not f.usable and (f.where != "active" or f.folder in OLD) and f.rec is not None and target in have \
                and not (f.kind != "grid" and f.step1 != "none" and f.real_kind != f.kind):
            delete(f, f"fails the check ({why(f)}); the person has a usable {METHOD_BY_ID[target].name}")
    # 5. an old special's own 4x4 dots, where the person has no such grid
    for f in live:
        if f.kind != "grid" and f.rec is not None and f.rec.dots and not f.meta.get("dots_from"):
            grid = "random-4x4" if dot_order(f.rec.dots) == "shuffled" else "original-4x4"
            n = len({(s.target_x, s.target_y) for s in f.rec.dots})
            if grid not in have and check_recording(f.rec).ok:
                act["OPTIONAL"].append(f"{METHOD_BY_ID[grid].name}: {f.rel} ended on its own {n} of 16 dots "
                                       f"({dot_order(f.rec.dots)}, they pass the grid check): as a file in {grid}/ "
                                       f"({grid}_ prefix, \"method\": \"{grid}\", \"grid\": 4) they would train it")
    return act, restored


def trainable(files: list[File], restored=()) -> int:
    """How many of the 24 models have an active (or restored) file of each part that passes the full check."""
    ready = {f.folder for f in files if (f.where == "active" or f in restored) and f.usable and f.folder in METHOD_BY_ID}
    return sum(m.grid.id in ready and (m.special is None or m.special.id in ready) for m in MODELS)


def self_check() -> None:
    """A made-up person in a temp folder (never data/): an Original 3x3, a Points recording also saved as
    Sweep (the 19 Oct copy, storage.copy_as), the real Sweep in sweep/trash/. The plan must delete the copy,
    restore the Sweep, offer Copy to for Line model and ask for the 5 other grids."""
    import tempfile
    from eyetracking.core.calibration import sweep_points
    from eyetracking.core.line_model import ScreenGeometry, features_for
    from eyetracking.core.models import CalibrationRecording, CalibrationSample
    rng, screen = np.random.default_rng(1), ScreenGeometry(27.7, 18.9)
    still = lambda: [rng.normal(0, 2), rng.normal(0, 2), rng.normal(0, 1), rng.normal(0, 1), rng.normal(-5, 1), rng.normal(-65, 1)]
    turn = lambda: [rng.uniform(-18, 18), rng.uniform(-12, 12), rng.uniform(-12, 12), 0, -5, -65]
    frames = lambda t, n, head=still: [CalibrationSample(features_for(screen, t, head(), (2, -1), 0.9, rng), *t) for _ in range(n)]
    head = tuple(frames((0.5, 0.5), 200, turn))
    line = tuple(s for t in sweep_points() for s in frames(t, 20))
    glide = tuple(s for x in np.linspace(0.05, 0.95, 60) for s in frames((round(float(x), 4), 0.5), 1)) + \
        tuple(s for y in np.linspace(0.05, 0.95, 60) for s in frames((0.5, round(float(y), 4)), 1))
    tmp = Path(tempfile.mkdtemp()) / "Tester"
    storage.save_calibration(CalibrationRecording((), (), tuple(s for t in point_grid(3) for s in frames(t, 25))),
                             "original-3x3", tmp / "original-3x3" / "original-3x3_2026-10-19_120000.json")
    points = storage.save_calibration(CalibrationRecording(line, head, ()), "points", tmp / "points" / "points_2026-10-19_120200.json")
    storage.copy_as(points, "sweep")                                                  # the copy of 19 Oct
    storage.save_calibration(CalibrationRecording(glide, head[::-1], ()), "sweep", tmp / "sweep" / "trash" / "sweep_2026-10-19_120100.json")
    storage.BASE_DIR = tmp.parent
    files = scan(tmp.parent)["Tester"]
    act, restored = plan(files, files)
    assert [line.split(" ")[0] for line in act["DELETE"]] == ["sweep/sweep_2026-10-19_120200.json"], act["DELETE"]
    assert act["RESTORE"] == ["Sweep: sweep/trash/sweep_2026-10-19_120100.json (Data > Removed > Restore)"], act
    assert len(act["ADD"]) == 6 and "Copy to Line model on points/points_2026-10-19_120200.json" in act["ADD"][-1], act["ADD"]
    assert not act["REPLACE"] and (trainable(files), trainable(files, restored)) == (2, 3)
    print("self-check ok")


def main() -> int:
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        return 0
    data = Path(sys.argv[1] if len(sys.argv) > 1 else storage.BASE_DIR).resolve()
    storage.BASE_DIR = data                     # list_people reads it; nothing is written
    people = scan(data)
    everyone = [f for fs in people.values() for f in fs]
    summary = []
    for person, files in people.items():
        print(f"\n{'=' * 100}\n{label(person)}\n{'=' * 100}")
        for f in files:
            state = ("OK" if f.good else "WEAK" if f.usable else "FAILS") + ("+FORMAT" if f.problems else "")
            print(f"  [{state:5}] {f.where:9} {f.rel}")
            lines = f.info + [f"format: {'; '.join(f.problems)}"] * bool(f.problems) \
                + [f"Not usable in the lab: {f.cheap}"] * bool(f.cheap) + [f"check: {'; '.join(f.reasons)}"] * bool(f.reasons) \
                + [f"weak: {'; '.join(f.weak)}"] * bool(f.weak)
            twins = [g.rel + ("" if g.person == person else f" ({label(g.person)})")
                     for g in everyone if g is not f and g.keys[1] and g.recording == f.recording]
            dots = [g.rel for g in everyone if g is not f and g.keys[2] and g.keys[2] == f.keys[2]]
            lines += [f"same recording (step 1 + head) as: {', '.join(twins)}"] * bool(twins) \
                + [f"same dots as: {', '.join(dots)}"] * bool(dots)
            print("\n".join(f"           {line}" for line in lines))
        if not any(f.where == "active" for f in files):
            print("\n  No active calibrations: not a tester. Nothing to add unless someone calibrates here.")
            summary.append(f"{label(person)}: no active calibrations, nothing to do.")
            continue
        act, restored = plan(files, everyone)
        ready = sum(storage.model_ready(m, data / person if person else data) is None for m in MODELS)
        now, after = trainable(files), trainable(files, restored)
        print(f"\n  Models: {ready}/24 the lab offers now (cheap check), {now}/24 pass the full check now, "
              f"{after}/24 after RESTORE")
        for key, title in (("OK", "OK (an active file passes the full check, not weak)"), ("DELETE", "DELETE"),
                           ("RESTORE", "RESTORE"), ("REPLACE", "REPLACE / RECALIBRATE"), ("ADD", "ADD"),
                           ("OPTIONAL", "OPTIONAL MOVES")):
            print(f"  {title}:\n" + "\n".join(f"    - {line}" for line in act[key] or ["-"]))
        summary.append(f"{label(person)}: {now}/24 models now, {after}/24 after RESTORE; "
                       + ", ".join(f"{k} {len(act[k])}" for k in ("DELETE", "RESTORE", "REPLACE", "ADD", "OPTIONAL")))
    print(f"\n{'=' * 100}\nSummary\n" + "\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
