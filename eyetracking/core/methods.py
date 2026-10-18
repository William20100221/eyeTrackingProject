"""
The calibration methods the lab offers, and the gaze model each one trains.

    METHODS                 every method, in the order the app lists them
    METHOD_BY_ID["sweep"]   one by its id (the id is also its data folder and file prefix)
    train(method, recordings, screen) -> (TrainedModel | None, samples used)

A method says how the calibration runs (kind) and what learns from it (model):

  kind   "grid"    the dots only: the Original method, on a 3x3, 4x4 or 5x5 grid
         "sweep"   a dot glides edge to edge (the eye range), the head step, then the 5x5 grid
         "points"  still dots along a line instead of the moving dot, the head step,
                   then the grid in random order
  model  "ridge"   a curve (squares + pairs, ridge) fitted on the raw features of the
                   dots: the Original model (core/gaze_model.py)
         "profile" the same curve, on eye numbers scaled by this user's own sweep +
                   head step (GazeProfile, core/calibration.py)
         "line"    a 3D line from each eye through the screen (vergence 6 degrees),
                   blended with the fitted line (core/line_model.py)

Every method records the same 12 features per frame (core/features.py), so the
files of different methods can be compared, and every file carries its method.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from eyetracking.core.calibration import GazeProfile, training_set
from eyetracking.core.gaze_model import GazeModel
from eyetracking.core.line_model import LineModel, ScreenGeometry
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.core.quality import check_recording


@dataclass(frozen=True)
class Method:
    id: str                       # its data folder (data/[<person>/]<id>/), and the file prefix
    name: str
    kind: str                     # grid | sweep | points
    model: str                    # ridge | profile | line
    grid: int = 5                 # dots per side of the final grid
    random_order: bool = False    # the grid in random order (the eye can't guess the next dot)
    about: str = ""               # one or two sentences for the page
    steps: tuple[str, ...] = ()   # what the user does, in order

    @property
    def dots(self) -> int:
        return self.grid * self.grid


GRID = "Look at each dot until it moves on, head still."
HEAD = "Keep looking at the centre dot and move your head as far as the bar asks: turn, nod, tilt, lean."

METHODS = (
    Method("original-3x3", "Original 3×3", "grid", "ridge", 3,
           about="The plain method on a 3×3 grid: 9 dots row by row, then a curve is fitted on their raw features. "
                 "The quickest, but 9 dots can only fit a rough curve.",
           steps=(GRID + " 9 dots, about 25 seconds.",)),
    Method("original-4x4", "Original 4×4", "grid", "ridge", 4,
           about="The plain method on a 4×4 grid: 16 dots row by row. A middle ground between speed and fit.",
           steps=(GRID + " 16 dots, about 40 seconds.",)),
    Method("original-5x5", "Original 5×5", "grid", "ridge", 5,
           about="The original method: 25 dots row by row, then a curve (squares and pairs, ridge) is fitted on "
                 "their raw features. The baseline every other method is compared with.",
           steps=(GRID + " 25 dots, about a minute.",)),
    Method("sweep", "Sweep", "sweep", "profile", 5,
           about="Three steps: a dot glides edge to edge while you follow it (your eye range), the head step (how "
                 "the head shifts the eye numbers), then the 25 dots. The curve is fitted on eye numbers scaled by "
                 "your own range, so sessions from different seats land on the same scale.",
           steps=("Follow the gliding dot with your eyes, head still: across, then down.", HEAD, GRID + " 25 dots.")),
    Method("points", "Points", "points", "profile", 5, random_order=True,
           about="Like Sweep, but with still dots along the line instead of a moving dot (the eye rests on each, so "
                 "no frame is taken mid-movement), and the 25 dots in random order.",
           steps=("Look at each dot along a line across, then down the middle: 14 dots.", HEAD,
                  GRID + " 25 dots, in random order.")),
    Method("line-model", "Line model", "points", "line", 5, random_order=True,
           about="The Points procedure, but the model is geometric: a 3D line from each eye through the screen "
                 "(the eyes converge at about 6 degrees), blended with the fitted line by a leave-one-dot-out check.",
           steps=("Look at each dot along a line across, then down the middle: 14 dots.", HEAD,
                  GRID + " 25 dots, in random order.")),
)
METHOD_BY_ID = {m.id: m for m in METHODS}
DEFAULT_METHOD = "original-5x5"


class TrainedModel:
    """Answers predict(features) for one method: the fitted model, plus the profile
    that scales live frames first (profile methods)."""

    def __init__(self, method: Method, inner, profile: GazeProfile | None = None) -> None:
        self.method, self.inner, self.profile = method, inner, profile

    def predict(self, features: Sequence[float] | None) -> tuple[float, float] | None:
        if features is None:
            return None
        if self.profile is not None:
            features = self.profile.scale(features)
        try:
            return self.inner.predict(features)
        except ValueError:                      # features of another size (an older file's)
            return None


def plain_training_set(recordings: Sequence[CalibrationRecording], older_weight: float = 0.5
                       ) -> tuple[list[CalibrationSample], list[float]]:
    """Recordings, NEWEST FIRST -> the dots' raw features (cleaned by the quality
    check) and a weight per sample: the newest recording 1.0, older ones `older_weight`."""
    samples: list[CalibrationSample] = []
    weights: list[float] = []
    for rec in recordings:
        report = check_recording(rec)
        if not report.ok:
            print(f"Skipping a calibration: {report.summary()}")
            continue
        w = older_weight if samples else 1.0
        samples += list(report.kept.dots)
        weights += [w] * len(report.kept.dots)
    return samples, weights


def train(method: Method, recordings: Sequence[CalibrationRecording], screen: ScreenGeometry | None = None
          ) -> tuple[TrainedModel | None, int]:
    """Fit the method's model on its recordings (newest first). (model, how many samples)."""
    profile = None
    if method.model == "ridge":
        samples, weights = plain_training_set(recordings)
    elif method.model == "profile":
        samples, weights, profile = training_set(recordings, scaled=True)
    else:
        samples, weights, _ = training_set(recordings, scaled=False)
    if not samples:
        return None, 0
    inner = LineModel(screen).fit(samples, weights) if method.model == "line" else GazeModel().fit(samples, weights)
    return TrainedModel(method, inner, profile), len(samples)


if __name__ == "__main__":
    assert [m.dots for m in METHODS[:3]] == [9, 16, 25] and METHOD_BY_ID[DEFAULT_METHOD].kind == "grid"
    assert {m.kind for m in METHODS} == {"grid", "sweep", "points"} and {m.model for m in METHODS} == {"ridge", "profile", "line"}
    print("\n".join(f"{m.id:<14} {m.kind:<7} {m.model:<8} {m.dots} dots" for m in METHODS))
