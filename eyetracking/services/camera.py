#camera.py
#Webcam frames get the image + BGR -> RGB
#No Mediapipe and Qt, pure OpenCV, cv2

import cv2 as cv
import numpy as np
from numpy.ma.core import choose
from pygrabber.dshow_graph import FilterGraph


class Camera:
    def __init__(self):
        devices = FilterGraph().get_input_devices()
        print(devices)
        choice = int(input("Choose device(s) to capture: "))-1

        self._cap = cv.VideoCapture(choice)   # does NOT run at import
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