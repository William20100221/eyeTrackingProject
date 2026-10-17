"""
Self-check for the head step and the head-still rule (no webcam):
    python -m eyetracking.testing.test_head_step

1. HeadStep: the settle measures the head at rest; a move ends once the head went
   past the target both ways (and the minimum time passed); frames too far out are
   not recorded; the gauge reports where the head is.
2. HeadStill: a frame turned more than HEAD_STILL_DEG from the dot's first frame is "moved".
"""
import eyetracking.core.calibration as cal
from eyetracking.core.calibration import HeadStep, HeadStill

cal.HEAD_SETTLE_MS, cal.HEAD_MOVE_MIN_MS, cal.HEAD_MOVE_MAX_MS = 100, 200, 1000


def frame(yaw=0.0, pitch=0.0, roll=0.0, z=60.0):
    return (0.5, 0.5, 0.5, 0.5, yaw, pitch, roll, 0.0, 0.0, z, 0.0, 0.0)


step = HeadStep()
assert step.feed(frame(yaw=2.0), 0) == ("Look at the dot", False) and step.gauge() is None
assert step.feed(frame(yaw=4.0), 50) == ("Look at the dot", False)
text, record = step.feed(frame(yaw=3.0), 100)                  # settle over: rest yaw = 3
assert step.rest[4] == 3.0 and "Turn your head" in text and record
assert step.gauge() == (0.0, 15.0, 35.0, [False, False])
assert step.feed(frame(yaw=-20.0), 150)[1] and step.reached == [True, False]      # -23 from rest: left target
assert step.feed(frame(yaw=50.0), 200)[1] is False                                # +47: too far, not recorded
assert step.feed(frame(yaw=20.0), 250)[1] and step.reached == [True, True]
assert step.move == 0 and step.feed(frame(), 280)[0].startswith("Keep looking")   # both reached, too early
assert step.feed(frame(), 320)[0].endswith("Nod slowly up and down") and step.move == 1   # min time: next move
for t in range(400, 1500, 100):                                                   # never nods: the max time ends it
    out = step.feed(frame(), t)
assert step.move == 2 and out[0].endswith("Tilt your head toward each shoulder"), (step.move, out)

still = HeadStill()
assert not still.moved(frame(yaw=10.0)) and not still.moved(frame(yaw=12.5))
assert still.moved(frame(yaw=13.5)) and still.moved(frame(pitch=-4.0))
still.reset()
assert not still.moved(frame(yaw=13.5))                                          # a new dot: a new reference
print("head step ok")
