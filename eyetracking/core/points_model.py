"""
The gaze model: "Points + Random 4x4", the best of calibration-lab's 24 models on its eleven testers
(ported 25 Oct 2026 from the lab's core/methods.py: train(), TrainedModel, special_parts()).

    train(recordings) -> (TrainedModel | None, samples used)
    TrainedModel.predict(features) -> (x, y) screen fractions, or None

What it learns from (every calibration is one sitting of core/calibration.py's three steps):
  * the grid's dots of every calibration that passes the quality check (core/quality.py), the
    newest weighing 1.0, older ones 0.5: older sittings add places to look;
  * the newest usable calibration's 14 line dots (glances away removed) at EXTRA_WEIGHT;
  * that calibration's head step: the user's OWN head-rotation matrix K (core/head_correction.fit_k),
    or the built-in one if the head step is too short.
A straight-line ridge (core/gaze_model.py) on the 6 eye features, every frame (trained or live)
turned back to the newest grid's posture with K, so a head turn isn't read as a gaze move.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from eyetracking.core.gaze_model import GazeModel
from eyetracking.core.head_correction import DEFAULT_K, corrected, fit_k, reference_angles
from eyetracking.core.models import CalibrationRecording, CalibrationSample
from eyetracking.core.quality import check_recording, check_special

EXTRA_WEIGHT = 0.5       # the line dots: another step than the grid, weighted like an older calibration


class TrainedModel:
    """Answers predict(features): the fitted regression on the 6 eye features turned back to the
    calibration's posture (K, ref: core/head_correction.py)."""

    def __init__(self, inner: GazeModel, K: np.ndarray, ref: np.ndarray, k_source: str) -> None:
        self.inner, self.K, self.ref, self.k_source = inner, K, ref, k_source

    def predict(self, features: Sequence[float] | None) -> tuple[float, float] | None:
        if features is None:
            return None
        try:
            return self.inner.predict(corrected(features, self.K, self.ref))
        except (ValueError, IndexError):        # features of another size (an older file's)
            return None


def kept_recordings(recordings: Sequence[CalibrationRecording]) -> tuple[list[CalibrationRecording], list[str]]:
    """The recordings whose grid passes the quality check, cleaned, newest first; and why the others didn't."""
    kept, skipped = [], []
    for rec in recordings:
        report = check_recording(rec)
        if report.ok:
            kept.append(report.kept)
        else:
            skipped.append(report.summary())
    return kept, skipped


def points_parts(recordings: Sequence[CalibrationRecording]
                 ) -> tuple[list[CalibrationSample], list[CalibrationSample]] | None:
    """(head-step frames, line-dot samples) from the newest recording whose steps 1-2 pass their check
    (core/quality.check_special), or None."""
    for rec in recordings:
        report = check_special(rec, "points")
        if report.ok:
            return list(report.kept.head), list(report.kept.sweep)
        print(f"Skipping a calibration's line dots and head step: {report.summary()}")
    return None


def train(recordings: Sequence[CalibrationRecording], older_weight: float = 0.5
          ) -> tuple[TrainedModel | None, int]:
    """Fit the model on the recordings (newest first). (model, how many samples); None without a
    calibration whose grid passes the check. Without usable line dots / head step it still trains on
    the grid, with the built-in head matrix."""
    kept, _ = kept_recordings(recordings)
    if not kept:
        return None, 0
    dots = [s for rec in kept for s in rec.dots]
    weights = [1.0 if i == 0 else older_weight for i, rec in enumerate(kept) for _ in rec.dots]
    head, extra = points_parts(recordings) or ([], [])
    weights += [EXTRA_WEIGHT] * len(extra)
    K, source = (fit_k(head), "your own head step") if head else (None, "")
    if K is None:
        K, source = DEFAULT_K, "the built-in head matrix"
    ref = reference_angles(kept[0].dots)
    samples = [CalibrationSample(corrected(s.features, K, ref), s.target_x, s.target_y) for s in dots + extra]
    return TrainedModel(GazeModel().fit(samples, weights), K, ref, source), len(samples)


if __name__ == "__main__":
    # a simulated person (core/line_model.features_for): line dots, a head step with the head moving,
    # a 4x4 grid; the model learns their own K and finds held-out points on the screen
    from eyetracking.core.calibration import point_grid, sweep_points
    from eyetracking.core.line_model import ScreenGeometry, features_for
    rng = np.random.default_rng(1)
    screen = ScreenGeometry(27.7, 18.9)
    head_now = lambda sd=1.0: [rng.normal(0, 2 * sd), rng.normal(0, 2 * sd), rng.normal(0, sd), rng.normal(0, 1),
                               rng.normal(-5, 1), rng.normal(-65, 1)]
    frames = lambda t, n, sd=1.0: [CalibrationSample(features_for(screen, t, head_now(sd), (2, -1), 0.9, rng), *t)
                                   for _ in range(n)]
    rec = CalibrationRecording(tuple(s for t in sweep_points() for s in frames(t, 20)),
                               tuple(frames((0.5, 0.5), 200, sd=6)),
                               tuple(s for t in point_grid(4) for s in frames(t, 25)))
    model, n = train([rec])
    assert model is not None and model.k_source == "your own head step" and n > 400, (model, n)
    test = [(0.3, 0.4), (0.7, 0.6)]
    err = [np.hypot(*(np.array(model.predict(frames(t, 1)[0].features)) - t)) for t in test for _ in range(10)]
    print(f"{n} samples, K from {model.k_source}, held-out error {np.mean(err):.1%} of the screen")
    assert np.mean(err) < 0.1
    no_head = CalibrationRecording((), (), rec.dots)                   # grid only: the built-in matrix
    assert train([no_head])[0].k_source == "the built-in head matrix"
    assert train([CalibrationRecording((), (), ())]) == (None, 0)
    print("ok")
