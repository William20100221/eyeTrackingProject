"""
Self-check for record mode (no webcam):  python -m eyetracking.testing.test_record_mode

1. The Recorder writes the frames it is given as a video, with their times, and the presses;
   VideoFrames reads them back with the same times; nothing is kept without a press
2. replay_presses: at each press, every model's latest point (held through blinks), as the
   live window records it; a press before any point records nothing; a TrainedModel.tracker()
   (predict + the 1-euro filter) is what runs here, the same as live
3. replay() over a real (black) video runs the real tracker: no face, so no samples, no crash
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from eyetracking.services.recording import Recorder, VideoFrames, replay, replay_presses


class FakeCamera:
    def __init__(self):
        self.n = 0

    def read_bgr(self):
        self.n += 1
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = self.n                                   # frame n is bluer: tells them apart
        return frame


# ---- 1. the recorder and the video ----------------------------------------------
tmp = Path(tempfile.mkdtemp())
rec = Recorder(FakeCamera(), tmp / "gaze-videos", 6)
for _ in range(5):
    rec.frame()
rec.press(0, 0.05, 0.05)
rec.press(1, 0.23, 0.05)
meta = rec.finish()
assert meta is not None and meta.name.startswith("gaze-video_20") and meta.suffix == ".json"
data = json.loads(meta.read_text())
assert data["grid"] == 6 and len(data["frames_ms"]) == 5 and data["frames_ms"] == sorted(data["frames_ms"])
assert [p["index"] for p in data["presses"]] == [0, 1] and data["presses"][1]["target_x"] == 0.23
video = meta.with_name(data["video"])
assert video.exists() and video.suffix in (".mp4", ".avi")
source = VideoFrames(video, data["frames_ms"])
frames = []
while True:
    f = source.read_rgb()
    if f is None:
        break
    frames.append((source.timestamp_ms, f.shape))
assert source.done and [ms for ms, _ in frames] == data["frames_ms"] and frames[0][1] == (48, 64, 3), frames
source.release()
nothing = Recorder(FakeCamera(), tmp / "gaze-videos", 4)
nothing.frame()
assert nothing.finish() is None and not nothing.video.exists()    # no press: no files

# ---- 2. the presses pick the latest prediction ------------------------------------
stream = [(0, (0.1, 0.1)), (33, None), (66, (0.3, 0.3)), (99, (0.5, 0.5))]
presses = [{"index": 0, "t_ms": 20, "target_x": 0.05, "target_y": 0.05},    # after frame 0
           {"index": 1, "t_ms": 40, "target_x": 0.23, "target_y": 0.05},    # during a blink: frame 0's point held
           {"index": 2, "t_ms": 200, "target_x": 0.41, "target_y": 0.05}]   # after the last frame
first = lambda f, t: None if f is None else (f[0], f[1])      # predictors take (features or None, seconds)
double = lambda f, t: None if f is None else (f[0] * 2, f[1] * 2)
never = lambda f, t: None
logs = replay_presses(stream, presses, {"first": first, "double": double, "never": never})
assert logs["first"] == [(0, 0.1, 0.1, 0.05, 0.05), (1, 0.1, 0.1, 0.23, 0.05), (2, 0.5, 0.5, 0.41, 0.05)], logs["first"]
assert logs["double"] == [(0, 0.2, 0.2, 0.05, 0.05), (1, 0.2, 0.2, 0.23, 0.05), (2, 1.0, 1.0, 0.41, 0.05)]
assert logs["never"] == []
early = replay_presses(stream, [{"index": 0, "t_ms": -1, "target_x": 0.5, "target_y": 0.5}], {"first": first})
assert early["first"] == []                                       # a press before any prediction: nothing

# ---- 3. the real tracker over the (faceless) recording -----------------------------
seen = []
from eyetracking.core.methods import METHOD_BY_ID, TrainedModel
class Fixed:                                                     # a "model" that always says the centre
    def predict(self, f):
        return (0.5, 0.5)
track = TrainedModel(METHOD_BY_ID["original-3x3"], Fixed()).tracker()
assert track(None, 0.0) is None and track((1.0,) * 12, 0.03) == (0.5, 0.5)   # a blink passes, a frame is smoothed
logs = replay(meta, {"first": first}, progress=seen.append)
assert logs == {"first": []} and seen == []                       # 5 frames: no face, no progress tick yet (every 10)
print("record mode ok")
