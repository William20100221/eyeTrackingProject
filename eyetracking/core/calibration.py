"""
Collecting data example for ML in gaze_model.py

Only answers: "which dot are we on, and what did we collect for it?"
No Qt, no camera, so it can be tested with hand-written numbers.
Screen positions are fractions 0..1 (x across, y down).
"""
from __future__ import annotations

from typing import Sequence

from eyetracking.core.models import CalibrationSample


def point_grid(n: int = 5, margin: float = 0.05) -> list[tuple[float, float]]:
    """n x n grid of (x, y) screen fractions, row by row.
    9 dots is too few to fit a curve AND test it: use 5x5 (25) at least.
    `margin` keeps the dots off the very edge of the screen."""
    positions = [round(margin + (1 - 2 * margin) * i / (n - 1), 4) for i in range(n)]
    return [(x, y) for y in positions for x in positions]


class CalibrationSession:
    def __init__(self, targets: Sequence[tuple[float, float]]) -> None:
        self._targets = list(targets)
        self._index = 0
        self._samples: list[CalibrationSample] = []

    def current_target(self) -> tuple[float, float] | None:
        if self.is_complete():
            return None
        return self._targets[self._index]

    def add_sample(self, features: Sequence[float]) -> None:
        target = self.current_target()
        if target is None:
            return
        # tuple of plain floats: works with a list or a numpy array, keeps the
        # frozen dataclass comparable, and saves to JSON easily later
        values = tuple(float(v) for v in features)
        self._samples.append(CalibrationSample(values, target[0], target[1]))

    def next_target(self) -> None:
        self._index += 1

    def is_complete(self) -> bool:
        return self._index >= len(self._targets)

    def progress(self) -> tuple[int, int]:
        """(current dot number starting at 1, total dots)"""
        total = len(self._targets)
        return min(self._index + 1, total), total

    def samples(self) -> list[CalibrationSample]:
        return list(self._samples)
