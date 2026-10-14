"""
Draw face mesh and irises onto an image.
Function should be used after/when landmarker.py
"""


from eyetracking.core.models import FrameResult
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result: FrameResult | None):
    if detection_result is None:
        print("drawing.py draw_landmarks_on_image() : No detection result, no drawing")
        return rgb_image
    face_landmarks_list = detection_result.landmarks
    annotated_image = np.copy(rgb_image)

    if detection_result.landmarks is None:
        return annotated_image

    # Loop through the detected faces to visualize.
    face_landmarks = [NormalizedLandmark(x=x, y=y, z=z) for x,y,z in face_landmarks_list]



    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image
