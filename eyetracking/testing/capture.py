#testing/capture.py
import cv2 as cv
import numpy as np
from eyetracking.services.camera import Camera


def open_camera():
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

    return new_frame

if __name__ == "__main__":
    open_camera()
    cv.destroyAllWindows()

# When everything done, release the capture