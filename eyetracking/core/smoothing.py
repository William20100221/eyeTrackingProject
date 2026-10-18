"""
The 1€ filter (Casiez, Roussel & Vogel, CHI 2012): a low-pass filter for the live gaze
point whose cut-off rises with speed. Still eyes -> heavy smoothing (the jitter of one
frame's landmarks, ~1-2% of the screen, averages away); a saccade -> almost no smoothing
(no lag when the gaze really moves). Two knobs, nothing to train. Ported from
calibration-lab ("Models, step 1", 18 Oct 2026; measurements in docs/model_improvements.md),
where it replaced the plain exponential average on the live point.

    python -m eyetracking.core.smoothing      (self-check)

Paper and reference code: https://cristal.univ-lille.fr/~casiez/1euro/
"""
from __future__ import annotations

import math
import time

MIN_CUTOFF_HZ = 1.0    # smoothing at rest: lower = steadier, more lag (the paper's starting value)
BETA = 0.5             # how fast speed lifts the cut-off: higher = less lag on fast moves
D_CUTOFF_HZ = 1.0      # smoothing of the speed estimate itself (the paper's default)
Point = tuple[float, float]


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    """One filtered value. feed(x, t) -> the smoothed x."""

    def __init__(self, min_cutoff: float = MIN_CUTOFF_HZ, beta: float = BETA, d_cutoff: float = D_CUTOFF_HZ) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: float | None = None
        self._dx = 0.0
        self._t: float | None = None

    def feed(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t <= self._t:
            self._x, self._t = x, t                     # the first sample (or a clock that went back) is taken as is
            return x
        dt = t - self._t
        dx = (x - self._x) / dt
        self._dx += _alpha(self.d_cutoff, dt) * (dx - self._dx)      # the speed, smoothed
        cutoff = self.min_cutoff + self.beta * abs(self._dx)         # faster -> higher cut-off -> less smoothing
        self._x += _alpha(cutoff, dt) * (x - self._x)
        self._t = t
        return self._x


class GazeSmoother:
    """The live gaze point, smoothed: feed(point or None, t_seconds) -> point or None.
    t defaults to now (time.perf_counter(): one clock for every live frame). None (a blink,
    no face) leaves the filter where it is; the next point carries on from there."""

    def __init__(self, **kw) -> None:
        self._x, self._y = OneEuro(**kw), OneEuro(**kw)

    def feed(self, point: Point | None, t: float | None = None) -> Point | None:
        if point is None:
            return None
        t = time.perf_counter() if t is None else t
        return self._x.feed(point[0], t), self._y.feed(point[1], t)


if __name__ == "__main__":
    import random
    rng = random.Random(0)
    s = GazeSmoother()
    # 1. still eyes with 2% jitter (sd): the smoothed point jitters by less than half of that
    out = [s.feed((0.5 + rng.gauss(0, 0.02), 0.5 + rng.gauss(0, 0.02)), i / 30) for i in range(90)]
    rest = out[30:]
    sd = (sum((x - 0.5) ** 2 for x, _ in rest) / len(rest)) ** 0.5
    assert sd < 0.01 and max(abs(x - 0.5) for x, _ in rest) < 0.025, (sd, rest[-3:])
    print(f"at rest: jitter 0.020 -> {sd:.4f} of the screen (sd)")
    # 2. a jump to the other side of the screen (a saccade): caught up within a few frames, not seconds
    jump = [s.feed((0.9 + rng.gauss(0, 0.02), 0.5), 3 + i / 30) for i in range(10)]
    assert jump[5][0] > 0.85, jump[5]                           # 5 frames (170 ms) later: there
    print(f"after a jump: {[round(x, 2) for x, _ in jump[:6]]}")
    # 3. a blink in the middle: None passes through, the filter carries on from where it was
    assert s.feed(None, 4.0) is None and abs(s.feed((0.9, 0.5), 4.03)[0] - 0.9) < 0.02
    # 4. no time given: the live clock, so a second feed a moment later is still smoothed, not reset
    live = GazeSmoother()
    assert live.feed((0.5, 0.5)) == (0.5, 0.5) and live.feed((0.6, 0.5)) != (0.6, 0.5)
    print("ok")
