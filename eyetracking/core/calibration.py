"""
Calibration logic: what the user is shown, and what the recordings mean.
No Qt, no camera, so all of it can be tested with hand-written numbers.
Screen positions are fractions 0..1 (x across, y down).

A calibration has three steps (ui/calibration_window.py draws them):

  1. SWEEP  still dots along a horizontal then a vertical line through the
            screen centre (the path a moving dot used to sweep), one at a
            time, head still. Measures how far the eyes move to cover the
            screen: the user's EYE RANGE. Still dots instead of a moving
            one: the eye rests on each, so no frame is taken mid-movement.
  2. HEAD   eyes stay on a centre dot while the user turns, nods, tilts
            and leans. The gaze doesn't move, so any change in the eye
            numbers is caused by the HEAD alone.
  3. DOTS   the 25-dot grid in random order: the training data for the
            gaze model. Random, so the eye can't anticipate the next dot
            and start moving early (a fixed row-by-row order invites that).

GazeProfile turns steps 1 + 2 into a correction applied to every frame:

    eye numbers  -> minus what head movement does to them    (step 2)
                 -> rescaled: 0 = screen left/top, 1 = right/bottom   (step 1)
    model input  =  just those 4 corrected eye numbers

After step 2's correction the eye numbers no longer change when only the
head moves, so they already say where on the screen the user looks. The
raw head numbers are NOT given to the model as well: during the 25 dots the
head barely moves, so the model would learn noise from them and then
extrapolate badly when the head does move (tested on fake data: held-out
error 61 px with head numbers, 17 px without).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from eyetracking.core.features import EYE, HEAD, N_FEATURES
from eyetracking.core.models import CalibrationRecording, CalibrationSample


def grid_positions(n: int = 5, margin: float = 0.05) -> list[float]:
    """The n positions used across AND down, as screen fractions."""
    return [round(margin + (1 - 2 * margin) * i / (n - 1), 4) for i in range(n)]


def point_grid(n: int = 5, margin: float = 0.05) -> list[tuple[float, float]]:
    """n x n grid of (x, y) screen fractions, row by row.
    9 dots is too few to fit a curve AND test it: use 5x5 (25) at least.
    `margin` keeps the dots off the very edge of the screen."""
    positions = grid_positions(n, margin)
    return [(x, y) for y in positions for x in positions]


def describe_target(x: float, y: float, n: int = 5) -> str:
    """A dot position in words: "dot (2, 1) of the 5x5 grid" (column, row,
    counted from the top left), or "the point 43% across, 61% down" when it
    isn't on the grid."""
    positions = grid_positions(n)
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

    def add_sample(self, features: Sequence[float] | None,
                   at: tuple[float, float] | None = None) -> None:
        """Record one frame for the current dot. None = a skipped frame (blink,
        eyes reopening, no face): the samples held back just before it are
        dropped too, so the frames where the lids were closing never get saved.
        `at` = record at this position instead (a moving dot, e.g. the sweep)."""
        target = at or self.current_target()
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
# Step 1: the sweep line. Still dots along the old sweep path.
# =====================================================================

def sweep_points(n: int = 7, margin: float = 0.05) -> list[tuple[float, float]]:
    """n dots left -> right across the middle (y = 0.5), then n dots top ->
    bottom down the middle (x = 0.5): the path the moving sweep dot took.
    Same margin as the grid, so the measured range covers what the dots cover.
    In path order, not random: neighbouring dots keep each eye jump short.
    The centre (0.5, 0.5) is on both lines, so it's shown twice."""
    # ponytail: 7 per line (~35 s); fewer if calibration feels long, more if the range is noisy
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
    ("Lean closer to the screen, then back", 9, 4.0, 15.0),      # z, cm
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


# =====================================================================
# The profile: steps 1 + 2 turned into a per-frame correction
# =====================================================================

MIN_SWEEP_FRAMES = 30    # per axis: fewer means the face was lost most of the time
MIN_HEAD_FRAMES = 60
MIN_EYE_RANGE = 1e-3     # eye numbers that barely moved = the eyes didn't follow the dot
HEAD_RIDGE = 0.01        # small ridge in the head fit: a head number that hardly moved
                         # in the head step can't get a huge (noise-amplifying) weight


@dataclass(frozen=True, eq=False)
class GazeProfile:
    eye_min: np.ndarray   # (4,) eye numbers when looking at the left/top edge...
    eye_max: np.ndarray   # (4,) ...and at the right/bottom edge (after head correction)
    head_ref: np.ndarray  # (6,) the head position the correction is measured from
    head_k: np.ndarray    # (6, 4) how much each head number shifts each eye number

    @classmethod
    def from_recording(cls, rec: CalibrationRecording) -> GazeProfile:
        """Work out the correction from one calibration's sweep + head step.
        Raises ValueError if the recording can't give a usable profile."""
        if not rec.head or not rec.sweep or len(rec.head[0].features) != N_FEATURES:
            raise ValueError("missing sweep/head data, or features of another size")
        head = np.array([s.features for s in rec.head])
        if len(head) < MIN_HEAD_FRAMES:
            raise ValueError(f"only {len(head)} head-step frames (need {MIN_HEAD_FRAMES})")

        # --- head step: fit  eye change = head change @ k  ----------------
        # The gaze stayed on the centre dot, so every eye change here came
        # from the head. Head numbers are standardised first (degrees and cm
        # on the same footing), then a ridge least-squares fit.
        # ponytail: straight-line head correction; make it curved if big head turns still drift
        H, E = head[:, HEAD], head[:, EYE]
        head_ref = H.mean(axis=0)
        sd = np.maximum(H.std(axis=0), 1e-6)
        Hs = (H - head_ref) / sd
        lam = HEAD_RIDGE * len(Hs)
        k = np.linalg.solve(Hs.T @ Hs + lam * np.eye(Hs.shape[1]), Hs.T @ (E - E.mean(axis=0)))
        head_k = k / sd[:, None]                     # back to raw head units

        # --- sweep: the eye range, after removing the head's share --------
        sweep = np.array([s.features for s in rec.sweep])
        eye = sweep[:, EYE] - (sweep[:, HEAD] - head_ref) @ head_k
        across = np.array([s.target_y == 0.5 for s in rec.sweep])   # horizontal pass
        down = np.array([s.target_x == 0.5 for s in rec.sweep])     # vertical pass
        if across.sum() < MIN_SWEEP_FRAMES or down.sum() < MIN_SWEEP_FRAMES:
            raise ValueError(f"too few sweep frames ({across.sum()} across, {down.sum()} down)")
        # 5th/95th percentile, not min/max: one glitchy frame can't stretch the range
        lo = np.concatenate([np.percentile(eye[across, :2], 5, axis=0),
                             np.percentile(eye[down, 2:], 5, axis=0)])
        hi = np.concatenate([np.percentile(eye[across, :2], 95, axis=0),
                             np.percentile(eye[down, 2:], 95, axis=0)])
        if np.any(hi - lo < MIN_EYE_RANGE):
            raise ValueError("the eyes barely moved during the sweep")
        return cls(lo, hi, head_ref, head_k)

    def scale(self, features: Sequence[float]) -> tuple[float, ...]:
        """Raw features (gaze_features(), 10 numbers) -> model input: the 4
        head-corrected eye numbers on a 0..1 scale (can go past 0/1 when
        looking beyond the sweep). The head is used here, for the
        correction, and not passed on (see the module docstring)."""
        f = np.asarray(features, dtype=float)
        eye = f[EYE] - (f[HEAD] - self.head_ref) @ self.head_k
        return tuple(float(v) for v in (eye - self.eye_min) / (self.eye_max - self.eye_min))


def training_set(recordings: Sequence[CalibrationRecording], older_weight: float = 0.5
                 ) -> tuple[list[CalibrationSample], list[float], GazeProfile | None]:
    """Recordings, NEWEST FIRST -> (model-ready dot samples, a weight per sample,
    the newest profile). Each recording is scaled by ITS OWN profile, so
    sessions from different seats/cameras land on the same 0..1 scale. The
    newest weighs 1.0, older ones `older_weight`. Live frames must be scaled
    with the returned (newest) profile. Unusable recordings are skipped."""
    samples: list[CalibrationSample] = []
    weights: list[float] = []
    newest: GazeProfile | None = None
    for rec in recordings:
        try:
            profile = GazeProfile.from_recording(rec)
        except ValueError as e:
            print(f"Skipping a calibration: {e}")
            continue
        newest = newest or profile
        w = older_weight if samples else 1.0          # first one kept = newest
        samples += [CalibrationSample(profile.scale(s.features), s.target_x, s.target_y)
                    for s in rec.dots]
        weights += [w] * len(rec.dots)
    return samples, weights, newest


if __name__ == "__main__":
    # self-check on fake data: eyes move with the gaze AND with the head;
    # after scaling, the same gaze must give the same eye numbers whatever the head does
    rng = np.random.default_rng(0)

    def frame(gx, gy, head):
        eye = [0.3 + 0.3 * gx, 0.35 + 0.3 * gx, 0.4 + 0.2 * gy, 0.45 + 0.2 * gy]
        eye = np.array(eye) + head @ np.full((6, 4), 0.002)   # head leaks into the eye numbers
        return tuple(np.concatenate([eye + rng.normal(0, 0.003, 4), head]))

    still = np.array([0, 0, 0, 0, 0, -50.0])
    sweep = [CalibrationSample(frame(x, y, still), x, y)            # ~20 frames per line dot
             for x, y in sweep_points() for _ in range(20)]
    head = [CalibrationSample(frame(0.5, 0.5, still + rng.normal(0, [10, 8, 5, 3, 3, 5])), 0.5, 0.5)
            for _ in range(400)]
    dots = [CalibrationSample(frame(x, y, still), x, y) for x, y in point_grid()]
    profile = GazeProfile.from_recording(CalibrationRecording(tuple(sweep), tuple(head), tuple(dots)))

    edge = np.array(profile.scale(frame(0.05, 0.05, still)))
    moved = np.array(profile.scale(frame(0.5, 0.5, still + [15, 10, 5, 3, 3, 8])))
    assert len(edge) == 4                                  # model input = the 4 eye numbers only
    print(f"left/top edge -> {edge.round(2)}  (want ~0)")
    print(f"centre gaze, head moved -> {moved.round(2)}  (want ~0.5)")
    assert np.all(np.abs(edge) < 0.1) and np.all(np.abs(moved - 0.5) < 0.1)

    samples, weights, newest = training_set([CalibrationRecording((), (), tuple(dots)),   # unusable
                                             CalibrationRecording(tuple(sweep), tuple(head), tuple(dots))])
    assert len(samples) == 25 and weights == [1.0] * 25 and newest is not None
    assert HeadStep().feed(None, 0) == ("Look at the dot", False)
    print("ok")
