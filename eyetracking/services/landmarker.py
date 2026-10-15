"""
MediaPipe wrapper: camera frames in, FrameResult out.

This is the ONLY module in the project that imports mediapipe. Everything
downstream works with eyetracking.core.models types.
"""
from __future__ import annotations

from typing import Self
import mediapipe as mp
import numpy as np
import time
import math
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from pathlib import Path

from eyetracking.core.models import FrameResult, HeadPose
from path import find_model



class FaceLandmarker:
    def __init__(self, MODEL_PATH: Path | None = None) -> None:
        # use the path you pass in; otherwise find models/face_landmarker.task
        self.MODEL_PATH = str(MODEL_PATH or find_model())
        base_options = BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1,
                                               running_mode=vision.RunningMode.VIDEO)
        self._detector = vision.FaceLandmarker.create_from_options(options)
        self._start = time.perf_counter()

    def get_last_timestamp(self) -> int:
        return int((time.perf_counter() - self._start)*1000)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()   # safe even if close() was already called

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None


    def detect(self, rgb_frame: np.ndarray) -> FrameResult:
        # ONE timestamp per frame: MediaPipe needs it to go up on every call,
        # and the FrameResult should carry the same time MediaPipe saw
        timestamp_ms = self.get_last_timestamp()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        raw = self._detector.detect_for_video(mp_image, timestamp_ms)

        # face found = landmarks came back (blendshapes only exist if that option is on)
        if not raw.face_landmarks:
            return FrameResult(timestamp_ms = timestamp_ms,
                               face_found=False,
                               head_pose=None,
                               blink_score_left = 0.0,
                               blink_score_right = 0.0,
                               landmarks = None)

        blink_score_left = blink_score_right = 0.0
        if raw.face_blendshapes:
            for category in raw.face_blendshapes[0]:
                if category.category_name == "eyeBlinkLeft":
                    blink_score_left = float(category.score)
                elif category.category_name == "eyeBlinkRight":
                    blink_score_right = float(category.score)

        raw_landmark_lst = []
        for landmark in raw.face_landmarks[0]:
            raw_landmark_lst.append([landmark.x, landmark.y, landmark.z])
        raw_landmark_lst = np.array(raw_landmark_lst)

        m = raw.facial_transformation_matrixes[0]
        # clamp: rounding can push the value a hair past 1, and asin(1.0000001) crashes
        yaw = math.degrees(math.asin(max(-1.0, min(1.0, -m[2][0]))))
        pitch = math.degrees(math.atan2(m[2][1], m[2][2]))
        roll = math.degrees(math.atan2(m[1][0], m[0][0]))
        return FrameResult(timestamp_ms = timestamp_ms,
                           face_found=True,
                           head_pose=HeadPose(yaw, pitch, roll),
                           blink_score_left=blink_score_left,
                           blink_score_right=blink_score_right,
                           landmarks = raw_landmark_lst)




