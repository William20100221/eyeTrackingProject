"""MediaPipe wrapper: camera frames in, FrameResult out.

This is the ONLY module in the project that imports mediapipe. Everything
downstream works with eyetracking.core.models types.
"""
from __future__ import annotations

from typing import Self
import mediapipe as mp
import numpy as np
import time
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from eyetracking.core.models import FrameResult, HeadPose
from path import find_model, Path



class FaceLandmarker:
    def __init__(self, MODEL_PATH: Path | None = None) -> None:
        self.MODEL_PATH = str(find_model())
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
        self._detector.close()

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None


    # def detect(self, rgb_frame: np.ndarray, timestamp_ms: int) -> FrameResult:
    #
    #     video = mp.Image.create_from_file(rgb_frame)
    #     detection_result = detector.detect(video)
    #
    #     annotated_image = drawing.draw_landmarks_on_image(video.numpy_view(), detection_result)

    def detect(self, rgb_frame: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        raw = self._detector.detect_for_video(mp_image, self.get_last_timestamp())
        return raw

