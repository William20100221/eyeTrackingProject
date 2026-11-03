#camera.py
#Webcam frames get the image + BGR -> RGB
#No Mediapipe and Qt, pure OpenCV, cv2

import sys

import cv2 as cv

PROBE = 5   # off Windows: the camera indexes tried when listing (OpenCV can't name them)


def camera_names() -> list[str]:
    """Webcams in OpenCV's index order (list position = index). Windows names them (DirectShow, through
    pygrabber); macOS and Linux have no portable way to name them, so the indexes that open are listed
    as "Camera 1", "Camera 2"..."""
    if sys.platform == "win32":
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    names = []
    for i in range(PROBE):
        cap = cv.VideoCapture(i)
        if not cap.isOpened():
            cap.release()
            break                                   # indexes run 0, 1, 2...: the first gap ends the list
        cap.release()
        names.append(f"Camera {i + 1}")
    return names


class Camera:
    def __init__(self, index: int | None = None):
        # no index given: list the cameras and ask in the console (as before)
        if index is None:
            devices = camera_names()
            print(devices)
            index = int(input("Choose device(s) to capture: "))-1

        self._cap = cv.VideoCapture(index)   # does NOT run at import
        if not self._cap.isOpened():
            print("Cannot open camera")
            raise RuntimeError("Cannot open camera")
    def read_rgb(self):
        ret, frame = self._cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return None

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        return rgb


    def release(self):
        self._cap.release()
