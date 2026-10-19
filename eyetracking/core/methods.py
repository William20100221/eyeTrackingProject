"""
The calibration types the lab offers, and the gaze model each one trains.

    METHODS                 every type, in the order the app lists them (12)
    METHOD_BY_ID["sweep"]   one by its id (the id is also its data folder and file prefix)
    train(method, recordings, screen) -> (TrainedModel | None, samples used)

A type says how the calibration runs (kind) and what learns from it (model):

  kind   "grid"    the dots only: the Original method, on a 3x3, 4x4 or 5x5 grid
         "sweep"   a dot glides edge to edge (the eye range), the head step, then the 4x4 grid
         "points"  still dots along a line instead of the moving dot, the head step,
                   then the 4x4 grid
  model  "ridge"   a straight-line fit (ridge) on the 6 eye features of the dots, each
                   frame first turned back to the calibration's posture with the default
                   head-rotation matrix (core/head_correction.py, core/gaze_model.py)
         "profile" the same, with the matrix measured by this calibration's own head step
         "line"    a 3D line from each eye through the screen (vergence 6 degrees),
                   blended with the fitted line (core/line_model.py)
Until 18 Oct 2026 "ridge" was a degree-2 curve on all 12 raw features and "profile" the
curve on eye numbers scaled by the sweep; both fell apart on the next sitting (bench_*.py:
2800% and 437% of the screen), see docs/model_improvements.md.

Two collections of six:
  In order      Original 3x3 / 4x4 / 5x5, Sweep, Points, Line model: the final grid row by
                row, the same order for all, so the order is not one more difference
  Random order  a twin of each (randomised()): the same steps and model, the final grid
                shuffled anew each time, so a type vs its twin measures what the order changes
Points and Line model (and their random twins) share their recordings: a calibration of one is saved
as the other too (partner()), so the two models train on the same data, as the reused 4x4 dots do for
the grid. Sweep, Points and Line model end on 4x4 on this branch (5x5 on their own branches), so
Original 4x4 and those three share the same 16 dot positions.

Every type records the same 12 features per frame (core/features.py), so the files of
different types can be compared, and every file carries its type and grid size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

from eyetracking.core.calibration import training_set
from eyetracking.core.gaze_model import GazeModel
from eyetracking.core.head_correction import DEFAULT_K, corrected, fit_k, reference_angles
from eyetracking.core.line_model import LineModel, ScreenGeometry
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.core.quality import check_recording
from eyetracking.core.smoothing import GazeSmoother


@dataclass(frozen=True)
class Method:
    id: str                       # its data folder (data/[<person>/]<id>/), and the file prefix
    name: str
    kind: str                     # grid | sweep | points
    model: str                    # ridge | profile | line
    grid: int = 5                 # dots per side of the final grid
    random_order: bool = False    # the grid shuffled for each calibration (the "Random order" collection)
    about: str = ""               # one or two sentences for the page
    steps: tuple[str, ...] = ()   # what the user does, in order

    @property
    def collection(self) -> str:
        return "Random order" if self.random_order else "In order"

    @property
    def dots(self) -> int:
        return self.grid * self.grid


GRID = "Look at each dot until it moves on, head still."
HEAD = "Keep looking at the centre dot and move your head as far as the bar asks: turn, nod, tilt, lean."

METHODS = (
    Method("original-3x3", "Original 3×3", "grid", "ridge", 3,
           about="The plain method on a 3×3 grid: 9 dots row by row, then a straight-line fit on the eye features. "
                 "The quickest, but 9 dots can only fit roughly.",
           steps=(GRID + " 9 dots, about 25 seconds.",)),
    Method("original-4x4", "Original 4×4", "grid", "ridge", 4,
           about="The plain method on a 4×4 grid: 16 dots row by row. A middle ground between speed and fit.",
           steps=(GRID + " 16 dots, about 40 seconds.",)),
    Method("original-5x5", "Original 5×5", "grid", "ridge", 5,
           about="The original method: 25 dots row by row, then a straight-line fit (ridge) on the 6 eye features, "
                 "each frame turned back to the calibration's posture with the built-in head-rotation matrix. "
                 "The baseline every other method is compared with.",
           steps=(GRID + " 25 dots, about a minute.",)),
    # ponytail: Sweep / Points / Line model use a 4x4 grid on this branch only (5x5 on their own branches)
    Method("sweep", "Sweep", "sweep", "profile", 4,
           about="Three steps: a dot glides edge to edge while you follow it (your eye range, a check), the head "
                 "step (how YOUR head turn shifts the eye numbers: it replaces the built-in matrix), then 16 dots "
                 "for the same straight-line fit as the Original.",
           steps=("Follow the gliding dot with your eyes, head still: across, then down.", HEAD, GRID + " 16 dots.")),
    Method("points", "Points", "points", "profile", 4,
           about="Like Sweep, but with still dots along the line instead of a moving dot (the eye rests on each, so "
                 "no frame is taken mid-movement), then 16 dots.",
           steps=("Look at each dot along a line across, then down the middle: 14 dots.", HEAD, GRID + " 16 dots.")),
    Method("line-model", "Line model", "points", "line", 4,
           about="The Points procedure, but the model is geometric: a 3D line from each eye through the screen "
                 "(the eyes converge at about 6 degrees), blended with the fitted line by a leave-one-dot-out check.",
           steps=("Look at each dot along a line across, then down the middle: 14 dots.", HEAD, GRID + " 16 dots.")),
)


def randomised(m: Method) -> Method:
    """The "Random order" twin of an in-order type: the same steps and model, its final dot grid
    shuffled anew for each calibration (a sweep / line-dot step and the head step stay as they are)."""
    first = "" if m.kind == "grid" else " Its first steps stay as they are; only the dots at the end are shuffled."
    return replace(m, id="random-" + m.id.removeprefix("original-"), name="Random " + m.name.removeprefix("Original "),
                   random_order=True, steps=m.steps[:-1] + (m.steps[-1].rstrip(".") + ", in random order.",),
                   about=f"{m.name} with its {m.dots} dots in random order, new for each calibration, so the eyes "
                         f"can't move to the next dot before it appears.{first} Compare it with {m.name} to see what "
                         "the order changes.")


# every in-order type gets a random twin, added after the six so saved choices (indexes) keep their type
METHODS = METHODS + tuple(randomised(m) for m in METHODS)
METHOD_BY_ID = {m.id: m for m in METHODS}
# Points and Line model run the same procedure (line dots, head step, 4x4 dots): one recording trains both,
# so a calibration of one is saved as the other too, and the two models are compared on the same data
SHARED = {"points": "line-model", "line-model": "points", "random-points": "random-line-model",
          "random-line-model": "random-points"}


def partner(m: Method) -> Method | None:
    """The type whose model trains on the same recordings as this one (Points <-> Line model), or None."""
    return METHOD_BY_ID.get(SHARED.get(m.id, ""))
DEFAULT_METHOD = "original-5x5"


class TrainedModel:
    """Answers predict(features) for one method: the fitted regression on the 6 eye features
    turned back to the calibration's posture (K, ref: core/head_correction.py), or the line
    model on the raw features (K is None)."""

    def __init__(self, method: Method, inner, K: np.ndarray | None = None, ref: np.ndarray | None = None,
                 k_source: str = "") -> None:
        self.method, self.inner, self.K, self.ref, self.k_source = method, inner, K, ref, k_source

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


def train(method: Method, recordings: Sequence[CalibrationRecording], screen: ScreenGeometry | None = None,
          older_weight: float = 0.5) -> tuple[TrainedModel | None, int]:
    """Fit the method's model on its recordings (newest first; the newest weighs 1.0, older ones
    `older_weight`). (model, how many samples). The regression models see every frame turned back
    to the NEWEST recording's posture, so older sittings add dots without adding their head pose."""
    if method.model == "line":
        samples, weights, _ = training_set(recordings, scaled=False)
        if not samples:
            return None, 0
        return TrainedModel(method, LineModel(screen).fit(samples, weights)), len(samples)
    kept = kept_recordings(recordings)
    if not kept:
        return None, 0
    newest = kept[0]
    K, source = (fit_k(newest.head), "its own head step") if method.model == "profile" else (None, "")
    if K is None:
        K, source = DEFAULT_K, "the default"
    ref = reference_angles(newest.dots)
    samples = [CalibrationSample(corrected(s.features, K, ref), s.target_x, s.target_y) for rec in kept for s in rec.dots]
    weights = [1.0 if i == 0 else older_weight for i, rec in enumerate(kept) for _ in rec.dots]
    return TrainedModel(method, GazeModel().fit(samples, weights), K, ref, source), len(samples)


if __name__ == "__main__":
    assert [m.dots for m in METHODS[:3]] == [9, 16, 25] and METHOD_BY_ID[DEFAULT_METHOD].kind == "grid"
    assert [m.random_order for m in METHODS] == [False] * 6 + [True] * 6
    assert all((a.kind, a.model, a.grid) == (b.kind, b.model, b.grid) for a, b in zip(METHODS[:6], METHODS[6:]))
    assert [m.id for m in METHODS[6:]] == ["random-3x3", "random-4x4", "random-5x5", "random-sweep", "random-points",
                                           "random-line-model"] and METHODS[9].name == "Random Sweep"
    assert {m.kind for m in METHODS} == {"grid", "sweep", "points"} and {m.model for m in METHODS} == {"ridge", "profile", "line"}
    print("\n".join(f"{m.id:<14} {m.kind:<7} {m.model:<8} {m.dots:>2} dots  {m.collection}" for m in METHODS))
