from eyetracking.core.models import FrameResult
from eyetracking.services import landmarker

#testing/capture.py
import cv2 as cv
import numpy as np
from eyetracking.services.camera import Camera
from eyetracking.services.landmarker import FaceLandmarker


def open_camera_and_detect(lm : FaceLandmarker):
    cam = Camera()
    while True:
        new_frame = cam.read_rgb()

        if cv.waitKey(1) == ord('q'):
            cam.release()
            break

        if new_frame is None:
            print("Can't receive frame (stream end?). Exiting ...")
            cam.release()
            return None

        cv.imshow("preview", cv.cvtColor(new_frame, cv.COLOR_RGB2BGR))
        raw = lm.detect(new_frame)
        print(raw)


    return raw


if __name__ == "__main__":
    lm = FaceLandmarker()
    open_camera_and_detect(lm = lm)
    cv.destroyAllWindows()

# When everything done, release the capture