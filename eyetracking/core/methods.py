"""
The calibrations the lab records, and the models they train.

    METHODS                  the 9 calibration types, three rows of three, in the app's order
    METHOD_BY_ID["sweep"]    one by its id (the id is also its data folder and file prefix)
    MODELS                   the 24 models: each grid alone, and each grid paired with a special
    MODEL_BY_ID["sweep-original-4x4"]
    LEGACY                   the six types before 20 Oct 2026 whose gaze checks are still on disk
    train(model, grids, specials, screen) -> (TrainedModel | None, samples used)

Calibration types (what you record), a row each:
  In order      Original 3x3 / 4x4 / 5x5   the dots, row by row
  Random order  Random 3x3 / 4x4 / 5x5     the same dots, shuffled anew for each calibration
  Special       Sweep        step 1: a dot glides edge to edge, across then down (the eyes pursue it)
                Points       step 1: 14 still dots along the same two lines (the eyes fixate each)
                Line model   the Points procedure, for the geometric model (a Points calibration is
                             saved as a Line model one too, and the other way round: partners())
                and each special's step 2, the head step (head turning, eyes on the centre dot)
A special records no grid of its own: it is paired with a grid calibration.

Models (what learns, and what a gaze check tests): a grid alone, or a special + a grid:
  <grid>               ridge (a straight line) on the grid's dots, every frame turned back to the
                       calibration's posture with the built-in head-rotation matrix (head_correction.py)
  Sweep + <grid>       the same fit on the grid's dots AND the sweep's pursuit frames (each matched to
                       where the dot was a moment before: the eye trails a moving dot), with the
                       head-rotation matrix measured by the sweep's own head step
  Points + <grid>      the same on the grid's dots AND the 14 still line dots, its own head matrix
  Line model + <grid>  a 3D line from each eye through the screen (line_model.py), fitted on the
                       grid's dots and the 14 line dots
So Sweep and Points differ by their first step (pursuit frames vs fixations) and by their own head
steps. Until 20 Oct 2026 every special was a type of its own that ended on 4x4 dots, step 1 trained
nothing, and one recording was saved as all three: Sweep and Points were the same model on the same
data, and their gaze checks came out identical (docs/calibration_lab_dev_log.md addendum 17).

Every type records the same 12 features per frame (core/features.py), so the files of different
types can be compared, and every file carries its type and grid size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from eyetracking.core.gaze_model import GazeModel
from eyetracking.core.head_correction import DEFAULT_K, corrected, fit_k, reference_angles
from eyetracking.core.line_model import LineModel, ScreenGeometry
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.core.quality import check_recording, check_special
from eyetracking.core.smoothing import GazeSmoother


@dataclass(frozen=True)
class Method:
    """A calibration type: what the user records."""
    id: str                       # its data folder (data/[<person>/]<id>/), and the file prefix
    name: str
    kind: str                     # grid | sweep | points
    grid: int = 0                 # dots per side of its grid (0: a special, which records none)
    random_order: bool = False    # the grid shuffled for each calibration (the "Random order" row)
    about: str = ""               # one or two sentences for the page
    steps: tuple[str, ...] = ()   # what the user does, in order

    @property
    def row(self) -> str:
        return "Special steps" if self.kind != "grid" else "Random order" if self.random_order else "In order"

    @property
    def dots(self) -> int:
        return self.grid * self.grid


GRID = "Look at each dot until it moves on, head still."
HEAD = "Keep looking at the centre dot and move your head as far as the bar asks: turn, nod, tilt, lean."
LINE = "Look at each dot along a line across, then down the middle: 14 dots."
PAIR = " No dots of its own: pair it with any grid calibration (Original or Random) to make the {0} + <grid> models."

ORIGINALS = (
    Method("original-3x3", "Original 3×3", "grid", 3,
           about="The plain method on a 3×3 grid: 9 dots row by row, then a straight-line fit on the eye features. "
                 "The quickest, but 9 dots can only fit roughly.",
           steps=(GRID + " 9 dots, about 25 seconds.",)),
    Method("original-4x4", "Original 4×4", "grid", 4,
           about="The plain method on a 4×4 grid: 16 dots row by row. A middle ground between speed and fit.",
           steps=(GRID + " 16 dots, about 40 seconds.",)),
    Method("original-5x5", "Original 5×5", "grid", 5,
           about="The original method: 25 dots row by row, then a straight-line fit (ridge) on the 6 eye features, "
                 "each frame turned back to the calibration's posture with the built-in head-rotation matrix. "
                 "The baseline every other model is compared with.",
           steps=(GRID + " 25 dots, about a minute.",)),
)


def randomised(m: Method) -> Method:
    """The "Random order" twin of an in-order grid: the same dots, shuffled anew for each calibration."""
    return replace(m, id="random-" + m.id.removeprefix("original-"), name="Random " + m.name.removeprefix("Original "),
                   random_order=True, steps=(m.steps[-1].rstrip(".") + ", in random order.",),
                   about=f"{m.name} with its {m.dots} dots in random order, new for each calibration, so the eyes "
                         f"can't move to the next dot before it appears. Compare it with {m.name} to see what the "
                         "order changes.")


SPECIALS = (
    Method("sweep", "Sweep", "sweep",
           about="Two steps: a dot glides edge to edge, across then down, and your eyes follow it (a pursuit: "
                 "hundreds of frames, each matched to where the dot was a moment before, as the eye trails it), "
                 "then the head step, which measures how YOUR head turn shifts the eye numbers." + PAIR.format("Sweep"),
           steps=("Follow the gliding dot with your eyes, head still: across, then down.", HEAD)),
    Method("points", "Points", "points",
           about="Two steps: 14 still dots along the same two lines (the eye rests on each, so every frame is a "
                 "fixation, none mid-movement), then the head step." + PAIR.format("Points"),
           steps=(LINE, HEAD)),
    Method("line-model", "Line model", "points",
           about="The Points procedure, for the geometric model: a 3D line from each eye through the screen (the "
                 "eyes converge at about 6 degrees), fitted on the grid's dots and the 14 line dots, blended with "
                 "the plain 3D line by a leave-one-dot-out check." + PAIR.format("Line model"),
           steps=(LINE, HEAD)),
)
GRIDS = ORIGINALS + tuple(randomised(m) for m in ORIGINALS)
METHODS = GRIDS + SPECIALS
METHOD_BY_ID = {m.id: m for m in METHODS}
DEFAULT_METHOD = "original-5x5"
# Points and Line model share their whole procedure, so one recording serves both (addendum 13): a
# calibration of one is saved as the other too. Sweep's first step differs, so it shares nothing.
SHARED = ({"points", "line-model"},)


def partners(m: Method) -> list[Method]:
    """The other types that record exactly the same steps as this one (in the app's order), or []."""
    group = next((g for g in SHARED if m.id in g), set())
    return [p for p in METHODS if p.id in group and p.id != m.id]


@dataclass(frozen=True)
class Model:
    """What learns and what a gaze check tests: a grid's dots, alone or with a special's steps."""
    id: str                       # its gaze checks' folder: the grid's own id, or <special>-<grid>
    name: str
    grid: Method
    special: Method | None = None

    @property
    def kind(self) -> str:
        """ridge (the grid alone) | profile (+ Sweep / Points: own head matrix, step 1) | line (+ Line model)."""
        return "ridge" if self.special is None else "line" if self.special.id == "line-model" else "profile"


MODELS = tuple(Model(g.id, g.name, g) for g in GRIDS) + \
    tuple(Model(f"{s.id}-{g.id}", f"{s.name} + {g.name}", g, s) for s in SPECIALS for g in GRIDS)
MODEL_BY_ID = {m.id: m for m in MODELS}
DEFAULT_MODEL = "original-5x5"
# the types before 20 Oct 2026: a special + its own 4x4 dots in one file. Their gaze checks stay readable.
LEGACY = {"sweep": "Sweep (old)", "points": "Points (old)", "line-model": "Line model (old)",
          "random-sweep": "Random Sweep (old)", "random-points": "Random Points (old)",
          "random-line-model": "Random Line model (old)"}


def model_for(grid_id: str, special_id: str | None = None) -> Model:
    return MODEL_BY_ID[f"{special_id}-{grid_id}" if special_id else grid_id]


def model_name(model_id: str) -> str:
    """Any gaze check's model in words: one of the 24, or an old type."""
    return MODEL_BY_ID[model_id].name if model_id in MODEL_BY_ID else LEGACY.get(model_id, model_id)


class TrainedModel:
    """Answers predict(features) for one model: the fitted regression on the 6 eye features
    turned back to the calibration's posture (K, ref: core/head_correction.py), or the line
    model on the raw features (K is None)."""

    def __init__(self, model: Model, inner, K: np.ndarray | None = None, ref: np.ndarray | None = None,
                 k_source: str = "") -> None:
        self.model, self.inner, self.K, self.ref, self.k_source = model, inner, K, ref, k_source

    def predict(self, features: Sequence[float] | None) -> tuple[float, float] | None:
        if features is None:
            return None
        try:
            if self.K is not None:
                features = corrected(features, self.K, self.ref)
            return self.inner.predict(features)
        except (ValueError, IndexError):        # features of another size (an older file's)
            return None

    def tracker(self) -> Callable[[Sequence[float] | None, float], tuple[float, float] | None]:
        """A live predictor: predict() with the 1-euro filter on top (core/smoothing.py), so the
        point is steady while the eyes rest and quick when they jump. Call it with every frame's
        features and the frame's time in seconds; None (a blink) comes back as None and the
        filter keeps its place. The replay of a recording uses the same, with the recorded times."""
        smoother = GazeSmoother()
        return lambda features, t: smoother.feed(self.predict(features), t)


def dots_error(trained: TrainedModel, rec: CalibrationRecording) -> float | None:
    """The mean distance from the model's point to the dot over a recording's dots, as a share of the screen."""
    errors = [float(np.hypot(p[0] - s.target_x, p[1] - s.target_y)) for s in rec.dots
              if (p := trained.predict(s.features)) is not None]
    return sum(errors) / len(errors) if errors else None


def live_clock() -> float:
    """Seconds, for tracker(): the same clock for every frame of a session."""
    return time.perf_counter()


def kept_recordings(recordings: Sequence[CalibrationRecording]) -> list[CalibrationRecording]:
    """The recordings that pass the quality check, cleaned, newest first."""
    kept = []
    for rec in recordings:
        report = check_recording(rec)
        if report.ok:
            kept.append(report.kept)
        else:
            print(f"Skipping a calibration: {report.summary()}")
    return kept


MAX_LAG = 8              # frames tried for the pursuit lag: up to ~0.27 s at 30 frames a second
EXTRA_WEIGHT = 0.5       # a special's step-1 frames: another sitting than the dots, weighted like an older calibration


def pursuit_samples(sweep: Sequence[CalibrationSample]) -> list[CalibrationSample]:
    """The gliding dot's frames, each paired with where the dot was a few frames EARLIER: the eye trails a
    moving dot (the pursuit lag) and the camera's picture is older than the dot's place when it is read.
    The shift is measured per pass (across, then down) as the one whose eye numbers line up best with the
    dot (|correlation|), as pursuit calibration does (Pfeuffer et al., UIST 2013). Frames are in time
    order; a dropped frame (a blink) makes the shift a frame off there.
    ponytail: a shift in frames, not ms (the file has no times); store frame times if this matters."""
    out = []
    for across in (True, False):
        part = [s for s in sweep if (s.target_y == 0.5) == across]
        if len(part) <= MAX_LAG + 2:
            continue
        eye = np.array([(s.features[0] + s.features[1]) / 2 if across else (s.features[10] + s.features[11]) / 2
                        for s in part])
        dot = np.array([s.target_x if across else s.target_y for s in part])
        lag = max(range(MAX_LAG + 1), key=lambda k: abs(np.nan_to_num(np.corrcoef(eye[k:], dot[:len(dot) - k])[0, 1])))
        out += [CalibrationSample(s.features, d.target_x, d.target_y) for s, d in zip(part[lag:], part)]
    return out


def special_parts(special: Method, recordings: Sequence[CalibrationRecording]
                  ) -> tuple[list[CalibrationSample], list[CalibrationSample]] | None:
    """(head-step frames, step-1 training samples) from the newest special recording that passes its
    check (core/quality.check_special), or None. Sweep: the pursuit frames, lag-matched; Points / Line
    model: the 14 line dots, glances away removed."""
    for rec in recordings:
        report = check_special(rec, special.kind)
        if report.ok:
            step1 = list(report.kept.sweep)
            return list(report.kept.head), pursuit_samples(step1) if special.kind == "sweep" else step1
        print(f"Skipping a {special.name} calibration: {report.summary()}")
    return None


def train(model: Model, grids: Sequence[CalibrationRecording], specials: Sequence[CalibrationRecording] = (),
          screen: ScreenGeometry | None = None, older_weight: float = 0.5) -> tuple[TrainedModel | None, int]:
    """Fit a model: the grid recordings (newest first; the newest weighs 1.0, older ones
    `older_weight`), and for a paired model the newest usable special recording (its step 1 at
    EXTRA_WEIGHT, its head step for the matrix). (model, how many samples). The regressions see every
    frame turned back to the NEWEST grid recording's posture, so older sittings and the special's
    sitting add places to look without adding their head pose. None without usable data of both."""
    kept = kept_recordings(grids)
    if not kept:
        return None, 0
    dots = [s for rec in kept for s in rec.dots]
    weights = [1.0 if i == 0 else older_weight for i, rec in enumerate(kept) for _ in rec.dots]
    head, extra = [], []
    if model.special is not None:
        parts = special_parts(model.special, specials)
        if parts is None:
            return None, 0
        head, extra = parts
    weights += [EXTRA_WEIGHT] * len(extra)
    if model.kind == "line":
        return TrainedModel(model, LineModel(screen).fit(dots + extra, weights)), len(dots) + len(extra)
    K, source = (fit_k(head), "its own head step") if head else (None, "")
    if K is None:
        K, source = DEFAULT_K, "the default"
    ref = reference_angles(kept[0].dots)
    samples = [CalibrationSample(corrected(s.features, K, ref), s.target_x, s.target_y) for s in dots + extra]
    return TrainedModel(model, GazeModel().fit(samples, weights), K, ref, source), len(samples)


if __name__ == "__main__":
    assert [m.dots for m in METHODS[:6]] == [9, 16, 25] * 2 and [m.dots for m in SPECIALS] == [0, 0, 0]
    assert [m.row for m in METHODS] == ["In order"] * 3 + ["Random order"] * 3 + ["Special steps"] * 3
    assert [m.id for m in GRIDS[3:]] == ["random-3x3", "random-4x4", "random-5x5"] and GRIDS[3].name == "Random 3×3"
    assert len(MODELS) == 24 and len({m.id for m in MODELS}) == 24 and not set(MODEL_BY_ID) & set(LEGACY)
    assert model_for("random-4x4", "sweep").name == "Sweep + Random 4×4" and model_for("original-3x3").kind == "ridge"
    assert {m.kind for m in MODELS} == {"ridge", "profile", "line"} and model_name("random-points") == "Random Points (old)"
    assert [p.id for p in partners(METHOD_BY_ID["points"])] == ["line-model"] and partners(METHOD_BY_ID["sweep"]) == []
    # the pursuit lag: eyes that trail the dot by 3 frames are matched to where it was 3 frames before
    xs = 0.05 + 0.9 * (1 - np.cos(np.linspace(0, 6 * np.pi, 120))) / 2
    trail = [CalibrationSample((x, x, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), t, 0.5) for x, t in zip(np.r_[[xs[0]] * 3, xs[:-3]], xs)]
    fixed = pursuit_samples(trail)
    assert len(fixed) == 117 and all(abs(s.features[0] - s.target_x) < 1e-12 for s in fixed)
    print("\n".join(f"{m.id:<26} {m.kind:<8} {m.name}" for m in MODELS))
