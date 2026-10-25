"""
Calibration logic: what the user is shown, and what the recordings mean.
No Qt, no camera, so all of it can be tested with hand-written numbers.
Screen positions are fractions 0..1 (x across, y down).

One calibration is the testers' best model, "Points + Random 4x4" (ported from calibration-lab,
25 Oct 2026: the best of its 24 models on the eleven testers), three steps in one sitting
(ui/calibration_window.py draws them, core/points_model.py trains on them):

  1. POINTS  14 still dots along a line across the middle, then down the middle, head still.
             Every frame is a fixation: the eye rests on each dot, none is mid-movement.
  2. HEAD    eyes stay on the centre dot while the head turns, nods, tilts and leans, as far as
             a bar asks. The gaze doesn't move, so any change in the eye numbers is the HEAD's:
             this measures the user's own head-rotation matrix (core/head_correction.py).
  3. DOTS    a 4x4 grid in random order, new for each calibration, head still: the eyes can't
             move to the next dot before it appears.
In steps 1 and 3 a frame whose head turned more than HEAD_STILL_DEG from the dot's first frame
is left out ("keep your head still"), and a dot waits for enough good frames ("keep looking").
"""
from __future__ import annotations

import random
from typing import Sequence

from eyetracking.core.models import CalibrationSample

GRID = 4                 # the grid's dots per side: Points + Random 4x4 was the best model on the testers


def grid_positions(n: int = 5, margin: float = 0.05) -> list[float]:
    """The n positions used across AND down, as screen fractions."""
    return [round(margin + (1 - 2 * margin) * i / (n - 1), 4) for i in range(n)]


def point_grid(n: int = 5, margin: float = 0.05) -> list[tuple[float, float]]:
    """n x n grid of (x, y) screen fractions, row by row.
    `margin` keeps the dots off the very edge of the screen."""
    positions = grid_positions(n, margin)
    return [(x, y) for y in positions for x in positions]


def random_grid(n: int = GRID) -> list[tuple[float, float]]:
    """The calibration's grid: n x n dots in a new random order each time."""
    grid = point_grid(n)
    return random.sample(grid, len(grid))


def describe_target(x: float, y: float, n: int = 5) -> str:
    """A dot position in words: "dot (2, 1) of the 5x5 grid" (column, row,
    counted from the top left), or "the point 43% across, 61% down" when it
    isn't on the grid (or there is no grid: the line dots and the head step, n = 0)."""
    positions = grid_positions(n) if n >= 2 else []
    if round(x, 4) in positions and round(y, 4) in positions:
        return f"dot ({positions.index(round(x, 4)) + 1}, {positions.index(round(y, 4)) + 1}) of the {n}x{n} grid"
    return f"the point {x:.0%} across, {y:.0%} down"


BLINK_LOOKBACK = 3   # samples held back per dot: dropped if a skipped frame follows,
                     # because the lids were probably already closing for a blink


class CalibrationSession:
    def __init__(self, targets: Sequence[tuple[float, float]]) -> None:
        self._targets = list(targets)
        self._index = 0
        self._samples: list[CalibrationSample] = []
        self._pending: list[CalibrationSample] = []   # the last BLINK_LOOKBACK samples, not yet kept
        self._kept = 0                                 # samples kept for the current dot

    def current_target(self) -> tuple[float, float] | None:
        if self.is_complete():
            return None
        return self._targets[self._index]

    def add_sample(self, features: Sequence[float] | None) -> None:
        """Record one frame for the current dot. None = a skipped frame (blink,
        eyes reopening, no face): the samples held back just before it are
        dropped too, so the frames where the lids were closing never get saved."""
        target = self.current_target()
        if target is None:
            return
        if features is None:
            self._pending.clear()
            return
        # tuple of plain floats: works with a list or a numpy array, keeps the
        # frozen dataclass comparable, and saves to JSON easily later
        values = tuple(float(v) for v in features)
        self._pending.append(CalibrationSample(values, target[0], target[1]))
        if len(self._pending) > BLINK_LOOKBACK:        # old enough: no blink came after it
            self._samples.append(self._pending.pop(0))
            self._kept += 1

    def current_count(self) -> int:
        """Good samples KEPT for the current dot so far: only frames already
        confirmed blink-free (BLINK_LOOKBACK good frames came after them)."""
        return self._kept

    def next_target(self) -> None:
        # drop, don't keep, the held-back frames: a blink starting just after
        # the dot changes would otherwise never be noticed for them
        self._pending.clear()
        self._kept = 0
        self._index += 1

    def is_complete(self) -> bool:
        return self._index >= len(self._targets)

    def progress(self) -> tuple[int, int]:
        """(current dot number starting at 1, total dots)"""
        total = len(self._targets)
        return min(self._index + 1, total), total

    def samples(self) -> list[CalibrationSample]:
        return list(self._samples)


# =====================================================================
# Step 1: the line dots
# =====================================================================

def sweep_points(n: int = 7, margin: float = 0.05) -> list[tuple[float, float]]:
    """n dots left -> right across the middle (y = 0.5), then n dots top ->
    bottom down the middle (x = 0.5). Same margin as the grid, so they cover
    what the dots cover. In path order, not random: neighbouring dots keep each
    eye jump short. The centre (0.5, 0.5) is on both lines, so it's shown twice."""
    positions = [round(margin + (1 - 2 * margin) * i / (n - 1), 4) for i in range(n)]
    return [(p, 0.5) for p in positions] + [(0.5, p) for p in positions]


# =====================================================================
# Step 2: the head step. Eyes on the centre, head moves.
# =====================================================================

HEAD_SETTLE_MS = 1500    # "look at the dot" first: the head's rest position is measured here, nothing recorded
HEAD_MOVES = (           # (what the user is told, feature index, target either way, too far) - the 6 head numbers
    ("Turn your head slowly left and right", 4, 15.0, 35.0),    # yaw, degrees
    ("Nod slowly up and down", 5, 10.0, 30.0),                   # pitch
    ("Tilt your head toward each shoulder", 6, 10.0, 30.0),      # roll
    ("Lean closer to the screen, then back", 9, 7.0, 20.0),      # z, cm (4 was too small a lean)
)
HEAD_MOVE_MIN_MS = 2500  # a move lasts at least this long...
HEAD_MOVE_MAX_MS = 8000  # ...and ends here at the latest, targets reached or not
# ponytail: targets are guesses (a comfortable turn that still gives MediaPipe a good face); tune on real users


class HeadStep:
    """Step 2, driven by the head itself. After the settle, each move asks one head
    number (yaw, pitch, roll, distance) to go past a target BOTH ways; a bar on
    screen (gauge) shows how far it is. The move ends once both sides were reached
    and HEAD_MOVE_MIN_MS passed, or at HEAD_MOVE_MAX_MS. Frames past "too far"
    are not recorded: MediaPipe's face gets unreliable there."""

    def __init__(self) -> None:
        self.move = 0
        self.rest: list[float] | None = None      # the head numbers at the end of the settle
        self._settle: list[Sequence[float]] = []
        self.value = 0.0                          # this move's number, relative to rest
        self.reached = [False, False]             # the minus side, the plus side
        self._move_start_ms = 0.0

    def feed(self, features: Sequence[float] | None, elapsed_ms: float) -> tuple[str, bool] | None:
        """One frame -> (instruction, record this frame?), or None when the step is over."""
        if elapsed_ms < HEAD_SETTLE_MS or not self._settle:
            if features is not None:
                self._settle.append(features)
            return "Look at the dot", False
        if self.rest is None:                     # the settle just ended
            self.rest = [sum(f[i] for f in self._settle) / len(self._settle) for i in range(len(self._settle[0]))]
            self._move_start_ms = elapsed_ms
        if self.move >= len(HEAD_MOVES):
            return None
        text, i, target, too_far = HEAD_MOVES[self.move]
        if features is not None:
            self.value = features[i] - self.rest[i]
            self.reached[0] |= self.value <= -target
            self.reached[1] |= self.value >= target
        t = elapsed_ms - self._move_start_ms
        if (all(self.reached) and t >= HEAD_MOVE_MIN_MS) or t >= HEAD_MOVE_MAX_MS:
            self.move += 1
            self.reached, self.value, self._move_start_ms = [False, False], 0.0, elapsed_ms
            if self.move >= len(HEAD_MOVES):
                return None
            text = HEAD_MOVES[self.move][0]
        return f"Keep looking at the dot. {text}", abs(self.value) <= too_far

    def gauge(self) -> tuple[float, float, float, list[bool]] | None:
        """(value, target, too far, reached) for the bar on screen; None outside a move."""
        if self.rest is None or self.move >= len(HEAD_MOVES):
            return None
        _, _, target, too_far = HEAD_MOVES[self.move]
        return self.value, target, too_far, list(self.reached)


HEAD_STILL_DEG = 3.0     # yaw / pitch may drift this much from a dot's first frame; further = the frame is dropped


class HeadStill:
    """Steps 1 and 3 want the head still while the eyes work. A frame whose yaw or
    pitch is more than HEAD_STILL_DEG from the first frame of the current dot has
    'moved' and is left out (the dot waits, like for a blink). reset() per dot."""

    def __init__(self) -> None:
        self.ref: tuple[float, float] | None = None

    def reset(self) -> None:
        self.ref = None

    def moved(self, features: Sequence[float]) -> bool:
        if self.ref is None:
            self.ref = (features[4], features[5])
            return False
        return abs(features[4] - self.ref[0]) > HEAD_STILL_DEG or abs(features[5] - self.ref[1]) > HEAD_STILL_DEG


if __name__ == "__main__":
    # a blink drops the frames held back just before it; the grid is shuffled; the head step's moves
    s = CalibrationSession([(0.1, 0.1)])
    for f in [(1,), (2,), (3,), (4,), (5,), None, (6,)]:
        s.add_sample(f)
    assert [x.features for x in s.samples()] == [(1.0,), (2.0,)] and s.current_count() == 2
    assert sorted(random_grid()) == sorted(point_grid(GRID)) and len(sweep_points()) == 14
    head = HeadStep()
    assert head.feed(None, 0) == ("Look at the dot", False)
    rest = [0.0] * 12
    head.feed(rest, 0)
    text, record = head.feed([0.0] * 4 + [20.0] + [0.0] * 7, 2000)
    assert text.startswith("Keep looking at the dot. Turn") and record and head.reached == [False, True]
    still = HeadStill()
    assert not still.moved(rest) and still.moved([0.0] * 4 + [5.0] + [0.0] * 7)
    print("ok")
