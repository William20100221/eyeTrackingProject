"""
Record mode for the gaze check: the webcam is recorded, with the time of every Enter, so
any model can be run over it later exactly as if it had been live. One recording, every
model, now or in the future: the fairest comparison there is.

    data/<person>/gaze-videos/gaze-video_<date>_<time>.mp4    the frames (.avi if this PC has no mp4 codec)
    data/<person>/gaze-videos/gaze-video_<date>_<time>.json   {"video", "grid", "frames_ms": [...], "presses": [...]}

    Recorder        writes the frames the gaze window asks for, and the presses (index, time, lit ring)
    VideoFrames     reads a recording back as if it were the camera, frame times included
    replay()        every frame through the tracker and the given models; at each press the
                    latest prediction, held through blinks, like the live dot -> {type: samples}

The video is a recording of your face: it is kept only in that folder, and only when record
mode is on. Delete it any time.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2 as cv
import numpy as np

from eyetracking.services.storage import STAMP, Check
from eyetracking.services.tracker import EyeTracker

FPS = 30               # nominal: the real frame times are in the json
Point = tuple[float, float]


class Recorder:
    def __init__(self, camera, folder: Path, grid: int) -> None:
        self._camera = camera                   # anything with read_bgr(): the tracker's Camera
        self.grid = grid
        folder.mkdir(parents=True, exist_ok=True)
        self.stamp = f"{datetime.now():{STAMP}}"
        self.video = folder / f"gaze-video_{self.stamp}.mp4"
        self.meta = folder / f"gaze-video_{self.stamp}.json"
        self._writer: cv.VideoWriter | None = None
        self.frames_ms: list[int] = []          # when each written frame was taken
        self.presses: list[dict] = []           # {"index", "t_ms", "target_x", "target_y"}
        self._t0 = time.perf_counter()

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def frame(self) -> None:
        """Take one camera frame and write it, with its time (the gaze window calls this every tick)."""
        bgr = self._camera.read_bgr()
        if bgr is None:
            return
        if self._writer is None:
            h, w = bgr.shape[:2]
            self._writer = cv.VideoWriter(str(self.video), cv.VideoWriter.fourcc(*"mp4v"), FPS, (w, h))
            if not self._writer.isOpened():     # no mp4 codec on this PC: motion-JPEG in an .avi
                self.video = self.video.with_suffix(".avi")
                self._writer = cv.VideoWriter(str(self.video), cv.VideoWriter.fourcc(*"MJPG"), FPS, (w, h))
        self._writer.write(bgr)
        self.frames_ms.append(self.elapsed_ms())

    def press(self, index: int, target_x: float, target_y: float) -> None:
        """Enter in the gaze window: the lit ring and the moment."""
        self.presses.append({"index": index, "t_ms": self.elapsed_ms(), "target_x": target_x, "target_y": target_y})

    def finish(self) -> Path | None:
        """Close the video and write the json. None (and no files) without a press."""
        if self._writer is not None:
            self._writer.release()
        if not self.presses or not self.frames_ms:
            self.video.unlink(missing_ok=True)
            return None
        self.meta.write_text(json.dumps({"video": self.video.name, "grid": self.grid, "frames_ms": self.frames_ms,
                                         "presses": self.presses}), encoding="utf-8")
        return self.meta


class VideoFrames:
    """A recording read back as if it were the camera: read_rgb() gives the next frame and
    sets timestamp_ms to when it was taken; None at the end (done = True)."""

    def __init__(self, video: Path, frames_ms: Sequence[int]) -> None:
        self._cap = cv.VideoCapture(str(video))
        self._ms = list(frames_ms)
        self._i = 0
        self.timestamp_ms = 0
        self.done = False

    def read_rgb(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok or self._i >= len(self._ms):
            self.done = True
            return None
        self.timestamp_ms = self._ms[self._i]
        self._i += 1
        return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    def release(self) -> None:
        self._cap.release()


def replay_presses(stream: Iterable[tuple[int, Sequence[float] | None]], presses: list[dict],
                   predictors: dict[str, Callable[[Sequence[float]], Point | None]]) -> dict[str, list[Check]]:
    """(ms, features or None) frames in order + the presses -> per model, the samples a live
    window would have recorded: at each press, the model's latest prediction, held through
    blinks and lost faces like the live dot. A press before any prediction records nothing."""
    held: dict[str, Point | None] = {name: None for name in predictors}
    logs: dict[str, list[Check]] = {name: [] for name in predictors}
    todo = sorted(presses, key=lambda p: p["t_ms"])
    k = 0

    def take(before_ms: float) -> None:          # every press before this frame, with what the models said last
        nonlocal k
        while k < len(todo) and todo[k]["t_ms"] < before_ms:
            p = todo[k]
            for name, point in held.items():
                if point is not None:
                    logs[name].append((p["index"], *point, p["target_x"], p["target_y"]))
            k += 1

    for ms, features in stream:
        take(ms)
        if features is not None:
            for name, predict in predictors.items():
                point = predict(features)
                if point is not None:
                    held[name] = point
    take(math.inf)
    return logs


def read_recording(meta: Path) -> dict:
    return json.loads(Path(meta).read_text(encoding="utf-8"))


def replay(meta: Path, predictors: dict[str, Callable[[Sequence[float]], Point | None]],
           progress: Callable[[float], None] | None = None) -> dict[str, list[Check]]:
    """Run models over a recording: every frame through the tracker (blink handling and all),
    the presses picking the latest prediction. `progress` gets 0..1 now and then."""
    data = read_recording(meta)
    source = VideoFrames(Path(meta).with_name(data["video"]), data["frames_ms"])
    tracker = EyeTracker(source=source)
    try:
        def frames():
            n = 0
            while True:
                features = tracker.read_features()
                if source.done:
                    return
                n += 1
                if progress is not None and n % 10 == 0:
                    progress(n / max(1, len(data["frames_ms"])))
                yield source.timestamp_ms, features
        return replay_presses(frames(), data["presses"], predictors)
    finally:
        tracker.close()
