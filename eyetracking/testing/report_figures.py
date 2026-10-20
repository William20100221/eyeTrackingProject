"""
The report's charts, made from the data folder alone: every number is read or computed from the
calibration files and gaze-check CSVs, nothing is typed in.

    python -m eyetracking.testing.report_figures data/ [--person NAME] [--out docs/img/report] [--screen 1440x960]

Read: every calibration (data/[<person>/]<type>/<type>_<date>_<time>.json, the current format,
storage.load_calibration; not disabled/ or trash/) and every gaze check (data/[<person>/]<model>/
gaze-check_<model>[_multi][_<n>-calibrations]_<date>_<time>.csv: index, x, y, target_x, target_y as
fractions of the screen, storage.read_gaze_check). Errors in px use --screen (the laptop the checks were
made on: 1440 x 960); "% of the screen" is the app's own measure, the distance in screen fractions. A
prediction is off screen when x or y is outside 0..1. Written into --out, each PNG (200 dpi) with a CSV of
the numbers it plots, the files it came from printed, and every number printed as a table:

  multi_check        per type with a *_multi_* check: median and mean error (px, % of the screen), every
                     type predicting from the same camera frames; off-screen predictions counted
  one_vs_two         per type: the error of checks of a model trained on 1 calibration vs 2 or more
  across_vs_down     per type: mean |dx| and |dy| in px (the vertical weakness)
  ring_map           the best and the worst type by mean error: the error at each ring, on a 1440 x 960 outline
                     (each circle's radius is that ring's mean error, to scale)
  versions           the model versions on each grid calibration file, leave-one-dot-out (px):
                       LinearRegression on 18 raw coordinates (6 eye landmarks' x, y, z: the 4 Oct baseline)
                       ridge on 10 features (the v2 file's: the 4 eye ratios + the 6 head numbers)
                       polynomial (degree 2) + ridge on all 12 (the model until 18 Oct)
                       the head-corrected line: ridge on the 6 eye features turned back to the calibration's
                       posture (core/methods.py train(), today's grid model)
                     No file holds landmark coordinates (the 30 Sep file is gone; v2 and v3 store features),
                     so the first is shown as "not in the files", never as a number.
  feature_correlation  each of the 12 features against the dot's x and y: Spearman, per file, the mean
  feature_subsets    leave-one-dot-out error (px) with eye features only, head features only, both
  heat_25            the newest 5x5 calibration (Original or Random): each dot's leave-one-dot-out error (px),
                     today's model, as a 5 x 5 heat map
  lodo               the same calibration's dots on the screen outline, each a circle whose radius is its error:
                     the dot left out (gaze_model.leave_one_dot_out) beside a random 80/20 split of the frames
                     (seed SPLIT_SEED), which tests frames whose own dot trained the model: why a random split lies
  k_matrix           DEFAULT_K (yaw, pitch, roll x the 6 eye features, change per degree x 1000) beside the K
                     fit_k gives from each person's head steps (every special's, the old types' too; a copy
                     saved under another type counted once), the mean per person
  one_euro           the newest record-mode video (data/<person>/gaze-videos/), every frame re-run through the
                     tracker and the person's model with the recorded frames_ms: the raw point, the old 30 %
                     moving average and the 1-euro filter over 4 s around one ring change (from the presses),
                     each with its settling time
  same_vs_next       the multiple checks' median per type (same sitting) beside bench_train's mean error per type
                     over every pair of one person's calibrations (next sitting, bench_train.new_sitting_errors),
                     with the old 12-feature curve on the same pairs as a reference line
  blink_timeline     one blink inside a blink test's open-eye states (the newest recording with one): both
                     eyes' scores over 3 s, run through the tracker's own blink rules (EyeTracker.read_features
                     fed the recorded scores; BLINK_ONSET_RISE, BLINK_RECOVERY_MS... imported): the open level,
                     the onset line, the regions open / onset / closed / recovery shaded, and the frames where
                     the gaze point is held
  accuracy_degrees   every gaze check's error as a visual angle (the screen's size in cm: --screen-cm or Windows;
                     each person's eye distance: the median head z of their calibrations), per model and each
                     person's best, beside published trackers' accuracies as reference lines (PUBLISHED, with
                     their sources: literature, not this data)
  blink_states       every blink test's states as a box plot per eye, with the candidate cut-offs (today's
                     BLINK_THRESHOLD, the highest open-eye score, the lowest closed one, their midpoint if
                     there is a gap). Blink tests: each person's blink-test/, and data/blink_scores/ (the other
                     branches' folder, the same columns) when no --person is given

Calibration dots go through the quality check first (core/quality.check_recording), as training does.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                       # files only: headless, and in tests
import matplotlib.pyplot as plt              # noqa: E402
import numpy as np                           # noqa: E402
from scipy.stats import spearmanr            # noqa: E402

from eyetracking.core.calibration import grid_positions                                        # noqa: E402
from eyetracking.core.features import EYE, FEATURE_NAMES, HEAD, IRIS_DOWN                     # noqa: E402
from eyetracking.core.gaze_model import GazeModel, leave_one_dot_out                           # noqa: E402
from eyetracking.core.head_correction import DEFAULT_K, corrected, fit_k, reference_angles     # noqa: E402
from eyetracking.core.quality import MIN_HEAD_FRAMES                                           # noqa: E402
from eyetracking.core.smoothing import GazeSmoother                                            # noqa: E402
from eyetracking.core.methods import GRIDS, LEGACY, MODELS, model_name                         # noqa: E402
from eyetracking.core.models import CalibrationSample                                          # noqa: E402
from eyetracking.core.quality import check_recording                                           # noqa: E402
from eyetracking.services import storage                                                       # noqa: E402

warnings.filterwarnings("ignore")             # sklearn / scipy chatter on tiny folds
DPI = 200
OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "report"
CHECK_IDS = [m.id for m in MODELS] + list(LEGACY)
EYE_F = list(range(EYE.start, EYE.stop)) + list(range(IRIS_DOWN.start, IRIS_DOWN.stop))
HEAD_F = list(range(HEAD.start, HEAD.stop))
VERSIONS = ("LinearRegression, 18 raw coordinates", "ridge, 10 features", "poly 2 + ridge, 12 features",
            "head-corrected line (today)")
BLUE, RED, GREY = "#1E88E5", "#E5483B", "#9E9E9E"


# ---------------------------------------------------------------- reading the folder
def collect(data: Path, person: str | None):
    """Of the folder: calibrations [(person, type, path, recording)], checks [(person, model id, path, rows)],
    head steps [(person, path, head samples)] (every special's, the old types' too; a copy saved under another
    type counted once), record-mode videos [(person, json path)] and blink tests [(person, csv path)]: each
    person's blink-test/, and (for everyone) the other branches' data/blink_scores/, oldest first."""
    storage.BASE_DIR = data.resolve()
    storage.use_person("")
    cals, checks, heads, videos, blinks, seen = [], [], [], [], [], set()
    for name, root in storage.person_roots():
        if person is not None and name != person:
            continue
        for kind in ["sweep", "points", "line-model", *(k for k in LEGACY if k.startswith("random-"))]:
            for path in storage.list_calibrations(kind, root):
                try:
                    head = storage.load_calibration(path).head
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                key = hash(tuple(s.features for s in head))
                if len(head) >= MIN_HEAD_FRAMES and key not in seen:
                    seen.add(key)
                    heads.append((name, path, head))
        videos += [(name, p) for p in storage.list_recordings(root)]
        blinks += [(name, p) for p in sorted((root / storage.BLINK).glob("blink*scores_*.csv"), key=storage.file_date)]
        for g in GRIDS:
            for path in storage.list_calibrations(g.id, root):
                try:
                    cals.append((name, g.id, path, storage.load_calibration(path)))
                except (OSError, ValueError, KeyError, TypeError) as e:
                    print(f"  skipped {path} ({e})")
        for m in CHECK_IDS:
            for path in storage.list_gaze_checks(m, root):
                try:
                    rows = storage.read_gaze_check(path)
                except (OSError, ValueError, IndexError) as e:
                    print(f"  skipped {path} ({e})")
                    continue
                if rows:
                    checks.append((name, m, path, rows))
    if person is None:                 # the other branches' blink tests, the same columns (state, t_ms, left, right)
        blinks += [("data/blink_scores", p) for p in sorted((storage.BASE_DIR / "blink_scores").glob("blink*scores_*.csv"),
                                                               key=storage.file_date)]
    return cals, checks, heads, videos, blinks


# ---------------------------------------------------------------- writing
def write(out: Path, name: str, fig, header, rows, sources) -> None:
    """The PNG, its CSV, the files it came from and the numbers, printed."""
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(["" if v is None else round(v, 4) if isinstance(v, float) else v for v in r] for r in rows)
    print(f"\n== {name}.png / {name}.csv: from {len(sources)} file{'s' if len(sources) != 1 else ''}")
    for s in sources:
        print(f"   {s}")
    widths = [max(len(str(h)), *(len(_cell(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(header)]
    print("   " + "  ".join(str(h).ljust(n) for h, n in zip(header, widths)))
    for r in rows:
        print("   " + "  ".join(_cell(v).ljust(n) for v, n in zip(r, widths)))
    if not rows:
        print("   (no data for this figure in the folder)")


def _cell(v) -> str:
    return "" if v is None else f"{v:.4g}" if isinstance(v, float) else str(v)


def empty(ax, what: str) -> None:
    ax.text(0.5, 0.5, f"No data: {what}", ha="center", va="center", transform=ax.transAxes, color=GREY)
    ax.set_axis_off()


def px_errors(rows, screen):
    """Per sample: the error in px, in screen fractions, dx px, dy px, off screen?"""
    W, H = screen
    return [(math.hypot((x - tx) * W, (y - ty) * H), math.hypot(x - tx, y - ty), abs(x - tx) * W, abs(y - ty) * H,
             not (0 <= x <= 1 and 0 <= y <= 1)) for _, x, y, tx, ty in rows]


def bars(ax, labels, series: dict, unit: str):
    """Grouped bars: series name -> values (None: no bar)."""
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))
    for k, (name, vals) in enumerate(series.items()):
        v = [np.nan if a is None else a for a in vals]
        ax.bar(x + (k - (len(series) - 1) / 2) * width, v, width, label=name)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(unit)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# ---------------------------------------------------------------- the figures
def multi_table(checks, screen):
    """Per type with a *_multi_* check: (name, checks, samples, median px, mean px, median %, mean %, off screen),
    the lowest mean first; and the files."""
    # ponytail: every multiple check of a type pooled over people and sessions; a panel per session if they differ
    per = defaultdict(list)
    files = defaultdict(set)
    for _, m, path, rows in checks:
        if "_multi_" in path.name:
            per[m] += px_errors(rows, screen)
            files[m].add(path)
    order = sorted(per, key=lambda m: np.mean([e[0] for e in per[m]]))
    table = [(model_name(m), len(files[m]), len(per[m]), float(np.median([e[0] for e in per[m]])),
              float(np.mean([e[0] for e in per[m]])), float(np.median([e[1] for e in per[m]])) * 100,
              float(np.mean([e[1] for e in per[m]])) * 100, sum(e[4] for e in per[m])) for m in order]
    return table, sorted({str(p) for v in files.values() for p in v})


def multi_check(checks, screen, out):
    table, files = multi_table(checks, screen)
    fig, ax = plt.subplots(figsize=(11, 5))
    if table:
        bars(ax, [t[0] for t in table], {"median": [t[3] for t in table], "mean": [t[4] for t in table]}, "error (px)")
        # px and % are not one scale (the screen isn't square, % is the app's distance in fractions): % as labels
        for i, t in enumerate(table):
            ax.annotate(f"{t[6]:.1f}%" + (f"\n{t[7]} off screen" if t[7] else ""), (i + 0.2, t[4]), ha="center",
                        va="bottom", fontsize=7, color=RED if t[7] else "black")
        ax.set_title(f"Several models at once on the same frames (*_multi_* checks): median and mean error; "
                     f"mean as % of the screen above; {screen[0]} x {screen[1]} px", fontsize=9)
    else:
        empty(ax, "no *_multi_* gaze check")
    write(out, "multi_check", fig, ["type", "checks", "samples", "median_px", "mean_px", "median_pct", "mean_pct",
                                    "off_screen"], table, files)


def one_vs_two(checks, screen, out):
    # ponytail: the calibration count comes from the check's name; checks from before 18 Oct carry none and are left out
    per = defaultdict(lambda: ([], []))
    files = set()
    for _, m, path, rows in checks:
        n = storage.check_calibrations(path)
        if n:
            per[m][0 if n == 1 else 1].extend(e[0] for e in px_errors(rows, screen))
            files.add(str(path))
    order = sorted(per, key=CHECK_IDS.index)
    mean = lambda v: float(np.mean(v)) if v else None
    table = [(model_name(m), len(per[m][0]), mean(per[m][0]), len(per[m][1]), mean(per[m][1])) for m in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    if table:
        bars(ax, [t[0] for t in table], {"1 calibration": [t[2] for t in table],
                                         "2 or more": [t[4] for t in table]}, "mean error (px)")
        ax.set_title("A model trained on one calibration vs two or more (*_1-calibrations_* / *_2-calibrations_* "
                     "checks); a missing bar: no such check", fontsize=9)
    else:
        empty(ax, "no check names how many calibrations trained it")
    write(out, "one_vs_two", fig, ["type", "samples_1", "mean_px_1", "samples_2plus", "mean_px_2plus"], table,
          sorted(files))


def across_vs_down(checks, screen, out):
    per = defaultdict(list)
    for _, m, path, rows in checks:
        per[m] += px_errors(rows, screen)
    order = sorted(per, key=CHECK_IDS.index)
    table = [(model_name(m), len(per[m]), float(np.mean([e[2] for e in per[m]])), float(np.mean([e[3] for e in per[m]])))
             for m in order]
    fig, ax = plt.subplots(figsize=(11, 5))
    if table:
        bars(ax, [t[0] for t in table], {"across |dx|": [t[2] for t in table], "down |dy|": [t[3] for t in table]},
             "mean error (px)")
        ax.set_title(f"Across vs down, every gaze check ({screen[0]} x {screen[1]} px)", fontsize=9)
    else:
        empty(ax, "no gaze check")
    write(out, "across_vs_down", fig, ["type", "samples", "mean_abs_dx_px", "mean_abs_dy_px"], table,
          sorted(str(p) for _, _, p, _ in checks))


def ring_map(checks, screen, out):
    per = defaultdict(list)
    for _, m, path, rows in checks:
        per[m] += [(tx, ty, x, y) for _, x, y, tx, ty in rows]
    W, H = screen
    err = {m: np.mean([math.hypot((x - tx) * W, (y - ty) * H) for tx, ty, x, y in v]) for m, v in per.items()}
    picks = []
    if err:
        best, worst = min(err, key=err.get), max(err, key=err.get)
        picks = [("best", best)] + ([("worst", worst)] if worst != best else [])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    table = []
    for ax, (which, m) in zip(axes, picks):
        rings = defaultdict(list)
        for tx, ty, x, y in per[m]:
            rings[(tx, ty)].append((x, y))
        ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, color="black", lw=1))
        for (tx, ty), pts in sorted(rings.items()):
            e = float(np.mean([math.hypot((x - tx) * W, (y - ty) * H) for x, y in pts]))
            mx, my = np.mean(pts, axis=0)
            ax.add_patch(plt.Circle((tx * W, ty * H), e, fill=False, lw=0.8, alpha=0.7,   # outlines: to scale they overlap
                                    color=RED if which == "worst" else BLUE))
            ax.plot(tx * W, ty * H, "k+", ms=6)
            ax.annotate("", (mx * W, my * H), (tx * W, ty * H), arrowprops=dict(arrowstyle="->", lw=0.7, color="#555"))
            ax.text(tx * W, ty * H - 12, f"{e:.0f}", ha="center", va="bottom", fontsize=7)
            table.append((which, model_name(m), tx, ty, len(pts), e, float(mx), float(my)))
        ax.set_xlim(-200, W + 200)
        ax.set_ylim(H + 150, -150)
        ax.set_aspect("equal")
        ax.set_title(f"{which}: {model_name(m)}, mean {err[m]:.0f} px", fontsize=9)
        ax.set_xticks([0, W / 2, W])
        ax.set_yticks([0, H / 2, H])
    for ax in axes[len(picks):]:
        empty(ax, "no gaze check")
    fig.suptitle(f"Error at each ring ({W} x {H} px screen): circle radius = the ring's mean error, to scale; "
                 "arrow = to the mean prediction", fontsize=9)
    write(out, "ring_map", fig, ["which", "type", "ring_x", "ring_y", "samples", "mean_error_px", "mean_x", "mean_y"],
          table, sorted({str(p) for _, m, p, _ in checks if any(m == k for _, k in picks)}))


def _cleaned(rec):
    report = check_recording(rec)
    return list(report.kept.dots) if report.ok else None


def _lodo(samples, screen, make_model, transform=lambda s: s) -> dict:
    data = [CalibrationSample(tuple(transform(s.features)), s.target_x, s.target_y) for s in samples]
    return leave_one_dot_out(data, *screen, make_model=make_model)


def _head_corrected(samples):
    ref = reference_angles(samples)
    return lambda f: corrected(f, DEFAULT_K, ref)


def versions(cals, screen, out):
    table, sources = [], []
    for person, kind, path, rec in cals:
        dots = _cleaned(rec)
        if not dots:
            print(f"  versions: {path.name} fails the quality check, left out")
            continue
        sources.append(str(path))
        errs = [None,                        # 18 raw coordinates: no file holds landmarks
                _lodo(dots, screen, GazeModel, lambda f: f[:10]),
                _lodo(dots, screen, lambda: GazeModel(degree=2)),
                _lodo(dots, screen, GazeModel, _head_corrected(dots))]
        table.append((person or "data", path.name, *(None if e is None else float(np.mean(list(e.values()))) for e in errs)))
    fig, ax = plt.subplots(figsize=(9, 5))
    if table:
        cols = list(zip(*[t[2:] for t in table]))
        means = [None if all(v is None for v in c) else float(np.mean([v for v in c if v is not None])) for c in cols]
        x = np.arange(len(VERSIONS))
        ax.bar(x, [np.nan if m is None else m for m in means], color=[GREY, BLUE, BLUE, RED])
        for i, c in enumerate(cols):
            got = [v for v in c if v is not None]
            ax.scatter([i] * len(got), got, s=10, color="black", alpha=0.5, zorder=3)
            ax.text(i, (means[i] or 0) + 5, "not in the files:\nno file holds\nlandmark coordinates" if means[i] is None
                    else f"{means[i]:.0f} px", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, VERSIONS, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("held-out error, leave-one-dot-out (px)")
        ax.set_title(f"Model versions on the same calibration files ({len(table)} files; dots = one file each)",
                     fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        table.append(("mean", f"{len(table)} files", *means))
    else:
        empty(ax, "no calibration passes the quality check")
    write(out, "versions", fig, ["person", "file", *VERSIONS], table, sources)


def feature_correlation(cals, out):
    per_file, sources = [], []
    for person, kind, path, rec in cals:
        dots = _cleaned(rec)
        if not dots or len({(s.target_x, s.target_y) for s in dots}) < 3:
            continue
        F = np.array([s.features for s in dots])
        r = []
        for t in ("target_x", "target_y"):
            tv = [getattr(s, t) for s in dots]
            r.append([spearmanr(F[:, i], tv)[0] for i in range(F.shape[1])])
        per_file.append(np.nan_to_num(np.array(r)))
        sources.append(str(path))
    table = []
    fig, ax = plt.subplots(figsize=(11, 5))
    if per_file:
        mean = np.mean(per_file, axis=0)
        table = [(name, float(mean[0, i]), float(mean[1, i])) for i, name in enumerate(FEATURE_NAMES)]
        bars(ax, [t[0] for t in table], {"vs dot x": [t[1] for t in table], "vs dot y": [t[2] for t in table]},
             "Spearman correlation (mean over files)")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-1, 1)
        ax.set_title(f"Each feature against where the dot was ({len(per_file)} calibration files)", fontsize=9)
    else:
        empty(ax, "no calibration with 3 or more dots")
    write(out, "feature_correlation", fig, ["feature", "spearman_x", "spearman_y"], table, sources)


def feature_subsets(cals, screen, out):
    subsets = {"eye features only": EYE_F, "head features only": HEAD_F, "both": EYE_F + HEAD_F}
    table, sources = [], []
    for person, kind, path, rec in cals:
        dots = _cleaned(rec)
        if not dots:
            continue
        sources.append(str(path))
        table.append((person or "data", path.name, *(float(np.mean(list(_lodo(dots, screen, GazeModel,
                                                                                  lambda f, c=cols: [f[i] for i in c]).values())))
                                                     for cols in subsets.values())))
    fig, ax = plt.subplots(figsize=(8, 5))
    if table:
        means = [float(np.mean(c)) for c in list(zip(*[t[2:] for t in table]))]
        ax.bar(list(subsets), means, color=[BLUE, GREY, RED])
        for i, c in enumerate(zip(*[t[2:] for t in table])):
            ax.scatter([i] * len(c), c, s=10, color="black", alpha=0.5, zorder=3)
            ax.text(i, means[i] + 5, f"{means[i]:.0f} px", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("held-out error, leave-one-dot-out (px)")
        ax.set_title(f"Which features carry the gaze: ridge on each subset ({len(table)} files)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        table.append(("mean", f"{len(table)} files", *means))
    else:
        empty(ax, "no calibration passes the quality check")
    write(out, "feature_subsets", fig, ["person", "file", *subsets], table, sources)


def newest_5x5(cals):
    """(person, type, path, cleaned dots) of the newest 5x5 calibration (Original or Random) that passes the
    quality check, or None."""
    fives = sorted((c for c in cals if c[1] in ("original-5x5", "random-5x5")), key=lambda c: storage.file_date(c[2]))
    for person, kind, path, rec in reversed(fives):
        dots = _cleaned(rec)
        if dots:
            return person, kind, path, dots
    return None


def _title(pick) -> str:
    person, kind, path, _ = pick
    return f"{model_name(kind)}, {person or 'no person'}, {storage.file_date(path):%d %b %Y %H:%M}"


def heat_25(cals, screen, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    table, sources = [], []
    pick = newest_5x5(cals)
    if pick:
        dots = pick[3]
        errs = _lodo(dots, screen, GazeModel, _head_corrected(dots))
        pos = grid_positions(5)
        grid = np.full((5, 5), np.nan)
        for (tx, ty), e in errs.items():
            if round(tx, 4) in pos and round(ty, 4) in pos:
                grid[pos.index(round(ty, 4)), pos.index(round(tx, 4))] = e
                table.append((pos.index(round(tx, 4)) + 1, pos.index(round(ty, 4)) + 1, tx, ty, float(e)))
        im = ax.imshow(grid, cmap="RdYlGn_r")
        for (r, c), e in np.ndenumerate(grid):
            ax.text(c, r, "" if np.isnan(e) else f"{e:.0f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(5), [f"{p:.0%}" for p in pos])
        ax.set_yticks(range(5), [f"{p:.0%}" for p in pos])
        fig.colorbar(im, ax=ax, label="held-out error (px)")
        ax.set_title(f"{_title(pick)}: each dot left out in turn (today's model)", fontsize=9)
        sources.append(str(pick[2]))
    else:
        empty(ax, "no 5x5 calibration passes the quality check")
    write(out, "heat_25", fig, ["column", "row", "dot_x", "dot_y", "error_px"], table, sources)


def _outline(ax, screen, title):
    W, H = screen
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, color="black", lw=1))
    ax.set_xlim(-200, W + 200)
    ax.set_ylim(H + 150, -150)
    ax.set_aspect("equal")
    ax.set_xticks([0, W / 2, W])
    ax.set_yticks([0, H / 2, H])
    ax.set_title(title, fontsize=9)


SPLIT_SEED = 0                                      # the random 80/20 split, fixed so a rerun draws the same split


def lodo(cals, screen, out):
    """One calibration's dots (the newest 5x5), each with its held-out error as a circle of that radius: left,
    the dot left out (the model never saw it); right, a random 80/20 split of the frames, where every dot's
    other frames are in the training set, so the "held-out" frames are near-copies of trained ones."""
    W, H = screen
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    table, sources = [], []
    pick = newest_5x5(cals)
    if pick:
        dots = pick[3]
        sources.append(str(pick[2]))
        out_err = _lodo(dots, screen, GazeModel, _head_corrected(dots))
        fix = _head_corrected(dots)
        data = [CalibrationSample(tuple(fix(s.features)), s.target_x, s.target_y) for s in dots]
        order = np.random.default_rng(SPLIT_SEED).permutation(len(data))
        cut = int(0.8 * len(data))
        model = GazeModel().fit([data[i] for i in order[:cut]])
        split = defaultdict(list)
        for i in order[cut:]:
            s = data[i]
            x, y = model.predict(s.features)
            split[(s.target_x, s.target_y)].append(math.hypot((x - s.target_x) * W, (y - s.target_y) * H))
        for ax, (name, errs, colour) in zip(axes, (("dot left out", out_err, RED),
                                                   ("random 80/20 split of the frames",
                                                    {d: float(np.mean(v)) for d, v in split.items()}, BLUE))):
            for (tx, ty), e in errs.items():
                ax.add_patch(plt.Circle((tx * W, ty * H), e, color=colour, alpha=0.2))
                ax.plot(tx * W, ty * H, "k+", ms=6)
                ax.text(tx * W, ty * H - 10, f"{e:.0f}", ha="center", va="bottom", fontsize=7)
            _outline(ax, screen, f"{name}: mean {np.mean(list(errs.values())):.0f} px")
        for d in sorted(out_err):
            table.append((d[0], d[1], float(out_err[d]), float(np.mean(split[d])) if d in split else None,
                          len(split.get(d, []))))
        held, split_mean = np.mean(list(out_err.values())), np.mean([np.mean(v) for v in split.values()])
        fig.suptitle(f"{_title(pick)}: each circle's radius is the dot's error, to scale ({W} x {H} px).\nThe random split "
                     f"tests frames whose own dot also trained the model: {split_mean:.0f} px against {held:.0f} px "
                     f"with the dot left out ({(held - split_mean) / held:+.0%} lower)", fontsize=9)
    else:
        for ax in axes:
            empty(ax, "no 5x5 calibration passes the quality check")
    write(out, "lodo", fig, ["dot_x", "dot_y", "dot_left_out_px", "random_split_px", "random_split_test_frames"],
          table, sources)


K_ROWS = ("yaw", "pitch", "roll")
K_COLS = [FEATURE_NAMES[i].split(" (")[0] for i in EYE_F]


def k_matrix(heads, out):
    """DEFAULT_K beside the K each person's head steps give (fit_k on each, the mean per person), x 1000."""
    per = defaultdict(list)
    for person, path, head in heads:
        K = fit_k(head)
        if K is not None:
            per[person or "no person"].append(K)
    panels = [("built-in (DEFAULT_K)", DEFAULT_K, 0)] + [(f"{p} ({len(ks)} head step{'s' if len(ks) != 1 else ''})",
                                                          np.mean(ks, axis=0), len(ks)) for p, ks in sorted(per.items())]
    vmax = max(float(np.abs(K).max()) for _, K, _ in panels) * 1000
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    table = []
    for ax, (name, K, n) in zip(axes[0], panels):
        im = ax.imshow(K * 1000, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for (r, c), v in np.ndenumerate(K * 1000):
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=7)
            table.append((name, n, K_ROWS[r], K_COLS[c], float(v)))
        ax.set_xticks(range(len(K_COLS)), K_COLS, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(3), K_ROWS, fontsize=8)
        ax.set_title(name, fontsize=9)
    fig.colorbar(im, ax=axes[0].tolist(), label="change of the eye feature per degree, x 1000", shrink=0.8)
    fig.suptitle("The head-rotation matrix K: built in, and fitted from each person's head steps (core/head_correction.py)",
                 fontsize=9)
    write(out, "k_matrix", fig, ["matrix", "head_steps", "angle", "eye_feature", "per_degree_x1000"], table,
          [str(p) for _, p, _ in heads])


OLD_SMOOTH = 0.3                                    # the moving average before the 1-euro filter: 30 % of each new point
WINDOW_MS = 2000                                    # each side of the ring change: 4 s in all
SETTLE_BAND = 0.1                                   # settled: within 10 % of the jump of its own final value, for good


def smoothing_traces(ms, raw):
    """The raw predictions, the old moving average (each frame takes 30 % of the new point) and the 1-euro
    filter, frame by frame, on the frames' own times. None (no face, a blink) holds the filters."""
    ema, euro, out = None, GazeSmoother(), {"raw": [], "moving average 30 %": [], "1-euro": []}
    for t, p in zip(ms, raw):
        e = None
        if p is not None:
            ema = p if ema is None else tuple(s + OLD_SMOOTH * (v - s) for s, v in zip(ema, p))
            e = euro.feed(p, t / 1000)
        out["raw"].append(p)
        out["moving average 30 %"].append(ema if p is not None else None)
        out["1-euro"].append(e)
    return out


def settling_ms(t, v, change_ms) -> float | None:
    """How long after the ring change the trace stays within SETTLE_BAND of the jump from its level before
    (the last 0.5 s) to its level after (the window's last 0.5 s); None if it never moved."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    before = v[(t >= change_ms - 500) & (t < change_ms)]
    after = v[t >= t.max() - 500]
    if not len(before) or not len(after):
        return None
    lo, hi = float(np.median(before)), float(np.median(after))
    if abs(hi - lo) < 1e-9:
        return None
    outside = (t >= change_ms) & (np.abs(v - hi) > SETTLE_BAND * abs(hi - lo))
    last = t[outside].max() if outside.any() else change_ms
    return float(last - change_ms)


def _video_model(person, cals):
    """The person's model for a video: today's Original 5x5 if they have one, else their grid with the most dots."""
    from eyetracking.core.methods import model_for, train
    mine = [c for c in cals if c[0] == person]
    if not mine:
        return None, None
    kind = "original-5x5" if any(c[1] == "original-5x5" for c in mine) else max(mine, key=lambda c: len(c[3].dots))[1]
    recs = [c[3] for c in sorted((c for c in mine if c[1] == kind), key=lambda c: storage.file_date(c[2]), reverse=True)]
    model, _ = train(model_for(kind), recs, [])
    return model, kind


def video_predictions(meta: Path, model) -> tuple[list[int], list, dict]:
    """Every frame of a record-mode video through the tracker and the model, unsmoothed: (frame times ms,
    points or None, the json)."""
    from eyetracking.services.recording import VideoFrames, read_recording
    from eyetracking.services.tracker import EyeTracker
    data = read_recording(meta)
    source = VideoFrames(meta.with_name(data["video"]), data["frames_ms"])
    tracker = EyeTracker(source=source)
    ms, pts = [], []
    try:
        while True:
            features = tracker.read_features()
            if source.done:
                break
            ms.append(source.timestamp_ms)
            pts.append(model.predict(features))
    finally:
        tracker.close()
    return ms, pts, data


def one_euro(videos, cals, screen, out):
    """The newest record-mode video re-run with the person's model: raw, the old 30 % moving average and the
    1-euro filter over 4 s around one ring change (the presses), each with its settling time."""
    W, H = screen
    fig, ax = plt.subplots(figsize=(10, 4.6))
    table, sources = [], []
    for person, meta in sorted(videos, key=lambda v: storage.file_date(v[1]), reverse=True):
        model, kind = _video_model(person, cals)
        if model is None:
            print(f"  one_euro: no usable calibration of {person or 'no person'} for {meta.name}")
            continue
        ms, pts, data = video_predictions(meta, model)
        presses = sorted(data["presses"], key=lambda p: p["t_ms"])
        # a ring change = a press (its ring goes out, the next lights): with 2 s of frames on each side
        changes = [(k, p) for k, p in enumerate(presses[:-1]) if ms and p["t_ms"] - WINDOW_MS >= ms[0]
                   and p["t_ms"] + WINDOW_MS <= ms[-1]]
        if not changes:
            print(f"  one_euro: {meta.name} has no ring change with 2 s of frames each side")
            continue
        jump = lambda kp: max(abs(presses[kp[0] + 1]["target_x"] - kp[1]["target_x"]) * W,
                              abs(presses[kp[0] + 1]["target_y"] - kp[1]["target_y"]) * H)
        k, p = max(changes, key=jump)                         # the biggest jump: the clearest step
        nxt = presses[k + 1]
        axis = 0 if abs(nxt["target_x"] - p["target_x"]) * W >= abs(nxt["target_y"] - p["target_y"]) * H else 1
        scale = (W, H)[axis]
        traces = smoothing_traces(ms, pts)
        keep = [i for i, t in enumerate(ms) if p["t_ms"] - WINDOW_MS <= t <= p["t_ms"] + WINDOW_MS]
        t = [(ms[i] - p["t_ms"]) / 1000 for i in keep]
        for name, colour in (("raw", GREY), ("moving average 30 %", BLUE), ("1-euro", RED)):
            v = [np.nan if traces[name][i] is None else traces[name][i][axis] * scale for i in keep]
            settle = settling_ms([ms[i] for i in keep], v, p["t_ms"])
            ax.plot(t, v, color=colour, lw=1 if name == "raw" else 1.8,
                    label=f"{name}: settles in {settle:.0f} ms" if settle is not None else f"{name}: never settles")
            table += [(name, ms[i] - p["t_ms"], None if np.isnan(x) else float(x), settle) for i, x in zip(keep, v)]
        ring = [(p["target_x"], p["target_y"])[axis] * scale if tt < 0 else (nxt["target_x"], nxt["target_y"])[axis] * scale
                for tt in t]
        ax.plot(t, ring, "k--", lw=1, label="the lit ring")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("seconds from the ring change")
        ax.set_ylabel(f"gaze {'x' if axis == 0 else 'y'} (px)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title(f"{person or 'no person'}'s {model_name(kind)} model re-run on {meta.name} (frame times from the "
                     f"recording); settled = within {SETTLE_BAND:.0%} of the jump for good", fontsize=9)
        sources.append(str(meta))
        break
    if not sources:
        empty(ax, "no record-mode gaze video (data/<person>/gaze-videos/) with a ring change and a usable model")
    write(out, "one_euro", fig, ["trace", "ms_from_change", "value_px", "settling_ms"], table, sources)


def _old_curve_fit(mtype, rec):
    """The model until 18 Oct, on one file: polynomial (degree 2) + ridge on all 12 raw features."""
    dots = _cleaned(rec)
    if not dots:
        return None
    model = GazeModel(degree=2).fit(dots)
    return lambda f: model.predict(f) if f is not None and len(f) == len(dots[0].features) else None


def same_vs_next(checks, data: Path, person, screen, out):
    """Left: the multiple checks' median per type (the model checked in the sitting it was calibrated in).
    Right: bench_train's error per type over every pair of one person's calibrations (trained on one, tested
    on another: the next sitting), with the old 12-feature curve on the same pairs as a reference line."""
    from eyetracking.testing.bench_train import grid_files, new_sitting_errors
    multi, multi_files = multi_table(checks, screen)
    files = grid_files(data, person)
    now, used = new_sitting_errors(files)
    old, _ = new_sitting_errors(files, fit=_old_curve_fit)
    old_mean = float(np.mean([e for v in old.values() for e in v])) * 100 if old else None
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    table = [("same sitting (multi check median)", t[0], t[2], t[5]) for t in multi]
    if multi:
        axes[0].bar(range(len(multi)), [t[5] for t in multi], color=BLUE)
        axes[0].set_xticks(range(len(multi)), [t[0] for t in multi], rotation=35, ha="right", fontsize=8)
        axes[0].set_ylabel("median error (% of the screen)")
        axes[0].set_title("Same sitting: the multiple checks' median", fontsize=9)
        axes[0].grid(axis="y", alpha=0.3)
    else:
        empty(axes[0], "no *_multi_* gaze check")
    order = [g.id for g in GRIDS if g.id in now]
    if order:
        means = [float(np.mean(now[m])) * 100 for m in order]
        axes[1].bar(range(len(order)), means, color=RED)
        for i, v in enumerate(means):
            axes[1].text(i, v * 1.05, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        axes[1].set_xticks(range(len(order)), [model_name(m) for m in order], rotation=35, ha="right", fontsize=8)
        if old_mean is not None:
            axes[1].axhline(old_mean, color="black", ls="--", lw=1,
                            label=f"the old 12-feature curve (poly 2 + ridge), same pairs: {old_mean:.1f}%")
            axes[1].legend(fontsize=8)
        axes[1].set_ylabel("mean error (% of the screen, log scale)")
        axes[1].set_yscale("log")                    # the old curve is ~100x worse on a new sitting: a linear axis hides the bars
        axes[1].set_title("Next sitting: trained on one calibration, tested on another of the same person (bench_train)",
                          fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)
        table += [("next sitting (bench_train mean)", model_name(m), len(now[m]), float(np.mean(now[m])) * 100)
                  for m in order]
        if old_mean is not None:
            table.append(("next sitting, the old 12-feature curve", "all types",
                          sum(len(v) for v in old.values()), old_mean))
    else:
        empty(axes[1], "no person with two grid calibrations")
    write(out, "same_vs_next", fig, ["what", "type", "samples_or_pairs", "error_pct"], table,
          multi_files + [str(p) for p in used])


# ---------------------------------------------------------------- blinks
REGIONS = (("open", "#FFFFFF"), ("onset", "#FFE0B2"), ("closed", "#EF9A9A"), ("recovery", "#C5CAE9"))


def tracker_regions(rows):
    """The recorded blink scores through the tracker's own blink rules (EyeTracker.read_features, fed frames
    as test_blinks does: no camera), frame by frame: (t ms, left, right, region, the open level it compared
    with, gaze point held?). Region from the tracker's state after the frame: closed (a full blink),
    recovery (within BLINK_RECOVERY_MS of one), onset (the score rose more than BLINK_ONSET_RISE above the
    open level, for up to BLINK_ONSET_MAX_MS), else open. Nothing of its logic is copied here."""
    from eyetracking.core.models import FrameResult, HeadPose
    from eyetracking.services.tracker import BLINK_RECOVERY_MS, EyeTracker
    lm = np.random.default_rng(0).uniform(0.2, 0.8, (478, 3))      # landmarks only so features can be made
    tracker = EyeTracker.__new__(EyeTracker)                       # ponytail: the tracker's private state, as test_blinks
    tracker.face_found, tracker._open_level, tracker._onset_ms, tracker._blink_ms = False, None, None, None
    out = []
    for _, t, left, right in rows:
        level = tracker._open_level
        frame = FrameResult(int(t), True, HeadPose(0, 0, 0), float(left), float(right), lm)
        tracker.read = lambda f=frame: f
        held = tracker.read_features() is None
        if tracker._blink_ms == t:
            region = "closed"
        elif tracker._blink_ms is not None and t - tracker._blink_ms < BLINK_RECOVERY_MS:
            region = "recovery"
        elif tracker._onset_ms is not None:
            region = "onset"
        else:
            region = "open"
        out.append((int(t), float(left), float(right), region, level if level is not None else max(left, right), held))
    return out


def blink_timeline(blinks, out):
    """One blink in a blink test's open-eye states (the newest recording with one): the scores over 3 s around
    it, the tracker's regions shaded, its open level and onset line, and when the gaze point is held."""
    from eyetracking.core.analysis import BLINK_THRESHOLD
    from eyetracking.core.blink_scores import NAMES, OPEN_STATES
    from eyetracking.services.tracker import BLINK_ONSET_MAX_MS, BLINK_ONSET_RISE, BLINK_RECOVERY_MS
    fig, ax = plt.subplots(figsize=(10, 4.6))
    table, sources = [], []
    for who, path in reversed(blinks):
        rows = storage.load_blink_scores(path)
        for state in OPEN_STATES:                       # natural blinks while the eyes are meant to be open
            seq = [r for r in rows if r[0] == state]
            frames = tracker_regions(seq)
            closed = [f for f in frames if f[3] == "closed"]
            if not closed:
                continue
            mid = closed[0][0]
            win = [f for f in frames if mid - 1500 <= f[0] <= mid + 1500]
            t = [(f[0] - mid) / 1000 for f in win]
            for (f, x0), x1 in zip(zip(win, t), t[1:] + [t[-1] + 0.033]):
                ax.axvspan(x0, x1, color=dict(REGIONS)[f[3]], lw=0)
            ax.plot(t, [f[1] for f in win], color=BLUE, lw=1.4, label="left eye score")
            ax.plot(t, [f[2] for f in win], color=RED, lw=1.4, label="right eye score")
            ax.plot(t, [f[4] for f in win], color="black", lw=1, label="open level (the tracker's)")
            ax.plot(t, [f[4] + BLINK_ONSET_RISE for f in win], color="black", lw=1, ls="--",
                    label=f"onset: open level + {BLINK_ONSET_RISE} (BLINK_ONSET_RISE)")
            ax.axhline(BLINK_THRESHOLD, color=GREY, lw=1, ls=":", label=f"closed: both eyes > {BLINK_THRESHOLD}")
            held = [x for f, x in zip(win, t) if f[5]]
            ax.scatter(held, [-0.06] * len(held), marker="s", s=8, color="black", label="gaze point held (frame skipped)")
            for name, colour in REGIONS[1:]:
                ax.bar(0, 0, color=colour, label=f"{name}" + {"onset": f" (up to {BLINK_ONSET_MAX_MS} ms)",
                                                              "recovery": f" ({BLINK_RECOVERY_MS} ms)"}.get(name, ""))
            ax.set_ylim(-0.1, 1.05)
            ax.set_xlabel("seconds from the first closed frame")
            ax.set_ylabel("blink score (0 open, 1 shut)")
            ax.legend(fontsize=7, loc="upper left", ncol=2)
            ax.set_title(f"A blink during \"{NAMES[state]}\" ({who}: {path.name}), through the tracker's blink rules "
                         "(services/tracker.py)", fontsize=9)
            table = [(f[0] - mid, f[1], f[2], f[4], f[3], "held" if f[5] else "live") for f in win]
            sources.append(str(path))
            break
        if sources:
            break
    if not sources:
        empty(ax, "no blink-test recording with a blink while the eyes were meant to be open")
    write(out, "blink_timeline", fig, ["ms_from_blink", "left", "right", "open_level", "region", "gaze_point"],
          table, sources)


def blink_states(blinks, out):
    """Every blink test's states as box plots of each eye's score, with the candidate cut-offs: today's 0.5,
    the highest open-eye score (95th percentile of open / looking down / half closed), the lowest closed
    score (5th percentile) and the midpoint between them when there is a gap."""
    from eyetracking.core.analysis import BLINK_THRESHOLD
    from eyetracking.core.blink_scores import NAMES, OPEN_STATES, STATES, summary
    rows = [r for _, p in blinks for r in storage.load_blink_scores(p)]
    fig, ax = plt.subplots(figsize=(11, 5))
    table = []
    states = [s for s, _ in STATES if any(r[0] == s for r in rows)]
    if states:
        data, pos, colours = [], [], []
        for i, s in enumerate(states):
            for k, (eye, colour) in enumerate((("left", BLUE), ("right", RED))):
                v = [r[2 + k] for r in rows if r[0] == s]
                data.append(v)
                pos.append(i * 3 + k)
                colours.append(colour)
                q = np.percentile(v, [0, 5, 50, 95, 100])
                table.append((NAMES[s], eye, len(v), *(float(x) for x in q)))
        box = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=True,
                         flierprops=dict(markersize=2, alpha=0.4))
        for patch, colour in zip(box["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.5)
        ax.set_xticks([i * 3 + 0.5 for i in range(len(states))], [NAMES[s] for s in states])
        s = summary(rows)
        opens = [s[k][eye][2] for k in OPEN_STATES if k in s for eye in ("left", "right")]
        cuts = [(BLINK_THRESHOLD, "today's cut-off (BLINK_THRESHOLD)", "black", "-")]
        if opens:
            cuts.append((max(opens), "highest open-eye score (95th pct)", BLUE, "--"))
        if "closed" in s:
            cuts.append((min(s["closed"]["left"][0], s["closed"]["right"][0]), "lowest closed score (5th pct)", RED, "--"))
        if len(cuts) == 3 and cuts[2][0] > cuts[1][0]:
            cuts.append(((cuts[1][0] + cuts[2][0]) / 2, "midpoint of the gap", GREY, ":"))
        for value, name, colour, style in cuts:
            ax.axhline(value, color=colour, ls=style, lw=1, label=f"{name}: {value:.2f}")
        ax.plot([], [], color=BLUE, lw=6, alpha=0.5, label="left eye")
        ax.plot([], [], color=RED, lw=6, alpha=0.5, label="right eye")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_ylabel("blink score (0 open, 1 shut)")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Blink scores per eye state ({len(blinks)} blink test{'s' if len(blinks) != 1 else ''}, "
                     f"{len(rows)} frames): where a cut-off would have to go", fontsize=9)
        table += [("cut-off", name, None, value, None, None, None, None) for value, name, _, _ in cuts]
    else:
        empty(ax, "no blink-test recording")
    write(out, "blink_states", fig, ["state", "eye", "frames", "min", "p5", "median", "p95", "max"], table,
          [str(p) for _, p in blinks])

# published accuracies, for reference lines only: none of them is this data (mean error, degrees of visual angle)
PUBLISHED = (("Tobii / EyeLink (infrared, lab)", 0.5, "https://researchwiki.solo.universiteitleiden.nl/xwiki/wiki/"
              "researchwiki.solo.universiteitleiden.nl/view/Physiology/Eye%20Tracking/Eye%20Tracker%20Comparison/"),
             ("GazeCloud API, its own claim", 1.0, "https://gazerecorder.com/faq/"),
             ("Labvanced webcam (Kaduk et al. 2023)", 1.4, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11289017/"),
             ("GazeRecorder, measured (Frontiers 2024, head still)", 2.65,
              "https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1369566/full"),
             ("WebGazer (Papoutsaki et al. 2016)", 4.17, "https://par.nsf.gov/biblio/10024076"))


def screen_cm(arg: str | None) -> tuple[float, float]:
    """The screen's size in cm: --screen-cm WxH, or what Windows reports for the main screen (through Qt)."""
    if arg:
        return tuple(float(v) for v in arg.lower().split("x"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    mm = app.primaryScreen().physicalSize()
    return mm.width() / 10, mm.height() / 10


def eye_distances(cals) -> dict[str, float]:
    """Per person: how far the eyes were from the camera, cm: the median of MediaPipe's head z over every frame
    of their grid calibrations (approximate: MediaPipe's head position is not a metric measurement)."""
    z = defaultdict(list)
    for person, _, _, rec in cals:
        z[person] += [-s.features[HEAD.start + 5] for s in rec.dots]
    return {p: float(np.median(v)) for p, v in z.items() if v}


def visual_angle(x, y, tx, ty, size, distance) -> float:
    """Degrees at the eye between the lit ring and the prediction (screen fractions), the eye on the screen's
    centre line `distance` cm away."""
    W, H = size
    eye = np.array([0.0, 0.0, -distance])
    a = np.array([(tx - .5) * W, (ty - .5) * H, 0.0]) - eye
    b = np.array([(x - .5) * W, (y - .5) * H, 0.0]) - eye
    return math.degrees(math.acos(float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def accuracy_degrees(cals, checks, size, out):
    """Every gaze check's error as a visual angle: per model (every person pooled) and each person's best model,
    beside published webcam and infrared trackers' accuracies (reference lines, not this data)."""
    dist = eye_distances(cals)
    per, perp, sources = defaultdict(list), defaultdict(list), []
    for person, m, path, rows in checks:
        if person not in dist:
            print(f"  accuracy_degrees: no calibration of {person or 'no person'} to give a distance; {path.name} left out")
            continue
        d = [visual_angle(x, y, tx, ty, size, dist[person]) for _, x, y, tx, ty in rows]
        per[m] += d
        perp[(person, m)] += d
        sources.append(str(path))
    table = [("model", model_name(m), "everyone", None, len(v), float(np.mean(v)), float(np.median(v)))
             for m, v in sorted(per.items(), key=lambda kv: np.mean(kv[1]))]
    best = []
    for person in sorted({p for p, _ in perp}):
        mine = {m: v for (p, m), v in perp.items() if p == person}
        b = min(mine, key=lambda m: np.mean(mine[m]))
        best.append(("best of a person", model_name(b), person or "no person", dist[person], len(mine[b]),
                     float(np.mean(mine[b])), float(np.median(mine[b]))))
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.28 * len(table) + 1.5)), gridspec_kw={"width_ratios": [3, 2]})
    if table:
        for ax, rows, title in ((axes[0], table, "Every model, every person's gaze checks pooled"),
                                (axes[1], best, "Each person's best model (the lowest of all: a little optimistic)")):
            labels = [r[1] if r[2] == "everyone" else f"{r[2]}: {r[1]}" for r in rows]
            y = np.arange(len(rows))[::-1]
            ax.barh(y, [r[5] for r in rows], height=0.8 if ax is axes[0] else 0.45, color=BLUE, alpha=0.8, label="mean")
            ax.scatter([r[6] for r in rows], y, color="black", s=12, zorder=3, label="median")
            for yy, r in zip(y, rows):
                ax.text(r[5] + 0.05, yy, f"{r[5]:.2f}° (n={r[4]})", va="center", fontsize=7)
            for (name, v, _), colour in zip(PUBLISHED, ("#2E7D32", "#7E57C2", "#00897B", RED, "#8D6E63")):
                ax.axvline(v, color=colour, ls="--", lw=1, label=f"{name}: {v}°")
            ax.set_yticks(y, labels, fontsize=7)
            ax.set_xlabel("error, degrees of visual angle")
            ax.set_title(title, fontsize=9)
            ax.grid(axis="x", alpha=0.3)
        axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.08))   # under the panel: off the bars
        fig.suptitle(f"Gaze-check accuracy in degrees: screen {size[0]:.1f} x {size[1]:.1f} cm; each person's eye distance "
                     f"from their calibrations ({', '.join(f'{p or chr(45)} {d:.0f} cm' for p, d in sorted(dist.items()))}). "
                     "Dashed: published trackers, other protocols, not this data", fontsize=9)
    else:
        for ax in axes:
            empty(ax, "no gaze check of a person with a calibration")
    write(out, "accuracy_degrees", fig, ["what", "model", "person", "eye_distance_cm", "samples", "mean_deg",
                                         "median_deg"], table + best + [("published", n, u, None, None, v, None)
                                                                        for n, v, u in PUBLISHED], sources)

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The report's figures, from the data folder")
    parser.add_argument("data", type=Path)
    parser.add_argument("--person", help="one person's files only ('' = the files straight under data/)")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--screen", default="1440x960", help="the screen the checks were made on, in px")
    parser.add_argument("--screen-cm", help="its size in cm, WxH (default: what Windows reports for the main screen)")
    args = parser.parse_args(argv)
    screen = tuple(int(v) for v in args.screen.lower().split("x"))
    cals, checks, heads, videos, blinks = collect(args.data, args.person)
    print(f"{len(cals)} calibrations, {len(checks)} gaze checks, {len(heads)} head steps, {len(videos)} gaze videos "
          f"under {args.data.resolve()}" + (f" ({args.person or 'no person'})" if args.person is not None else ""))
    multi_check(checks, screen, args.out)
    one_vs_two(checks, screen, args.out)
    across_vs_down(checks, screen, args.out)
    ring_map(checks, screen, args.out)
    versions(cals, screen, args.out)
    feature_correlation(cals, args.out)
    feature_subsets(cals, screen, args.out)
    heat_25(cals, screen, args.out)
    lodo(cals, screen, args.out)
    k_matrix(heads, args.out)
    one_euro(videos, cals, screen, args.out)
    same_vs_next(checks, args.data, args.person, screen, args.out)
    blink_timeline(blinks, args.out)
    blink_states(blinks, args.out)
    accuracy_degrees(cals, checks, screen_cm(args.screen_cm), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
