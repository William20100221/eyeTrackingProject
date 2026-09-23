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

    def detect(self, rgb_frame: np.ndarray) -> FrameResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        raw = self._detector.detect_for_video(mp_image, self.get_last_timestamp())
        face_blendshapes = raw.face_blendshapes[0]

        for category in face_blendshapes:
            if category.category_name == "eyeBlinkLeft":
                blink_score_left = float(category.score)
                #Debug
                print(blink_score_left)
            if category.category_name == "eyeBlinkRight":
                blink_score_right = float(category.score)
                #Debug
                print(blink_score_right)

        if not raw.face_landmarks:
            return FrameResult(timestamp_ms = self.get_last_timestamp(),
                               face_found=False,
                               head_pose=None,
                               blink_score_left = blink_score_left,
                               blink_score_right = blink_score_right,
                               landmarks = None)

        else:
            landmarks = raw.face_landmarks[0]
            raw_landmark_lst = []
            for landmark in landmarks:
                 raw_landmark_lst.append([landmark.x, landmark.y, landmark.z])

            m = raw.facial_transformation_matrixes[0]
            a = math.degrees(math.asin(-m[2][0]))
            b = math.degrees(math.atan2(m[2][1], m[2][2]))
            c = math.degrees(math.atan2(m[1][0], m[0][0]))
            frame_result = FrameResult(timestamp_ms = self.get_last_timestamp(),
                                       face_found=True,
                                       head_pose=HeadPose(a,b,c),
                                       blink_score_left=blink_score_left,
                                       blink_score_right=blink_score_right,
                                       landmarks = np.array(raw_landmark_lst))


        return frame_result




