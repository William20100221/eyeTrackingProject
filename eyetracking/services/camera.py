#camera.py
#Webcam frames get the image + BGR -> RGB
#No Mediapipe and Qt, pure OpenCV, cv2

import cv2 as cv
from pygrabber.dshow_graph import FilterGraph


def camera_names() -> list[str]:
    """Webcams Windows knows about, in OpenCV's index order (list position = index)."""
    return FilterGraph().get_input_devices()


class Camera:
    def __init__(self, index: int | None = None):
        # no index given: list the cameras and ask in the console (as before)
        if index is None:
            devices = FilterGraph().get_input_devices()
            print(devices)
            index = int(input("Choose device(s) to capture: "))-1

        self._cap = cv.VideoCapture(index)   # does NOT run at import
        if not self._cap.isOpened():
            print("Cannot open camera")
            raise RuntimeError("Cannot open camera")
    def read_bgr(self):
        """One frame as OpenCV gives it (BGR): what a video recorder writes."""
        ret, frame = self._cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None
        return frame

    def read_rgb(self):
        frame = self.read_bgr()
        return None if frame is None else cv.cvtColor(frame, cv.COLOR_BGR2RGB)


    def release(self):
        self._cap.release()