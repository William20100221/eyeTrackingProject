"""
The face mesh window: a small self-view, like a meeting app's, while you Go Reading.

The camera picture, mirrored, with the face mesh drawing itself on (video/face_mesh.py MeshFx); then
the 3D model alone (the picture fades out) or the mesh on your face, as Settings > Face mesh window says.
Always on top, rounded, drag it anywhere. App opens it with the camera's frames and closes it when
the reading ends.
"""
from __future__ import annotations

import dataclasses
import time

import cv2
from PySide6.QtCore import QPoint, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication, QWidget

from eyetracking.core.models import NO_FRAME
from eyetracking.video.face_mesh import MeshFx

WIDTH = 360              # the window; its height follows the camera's shape
DRAW_WIDTH = 640         # the effects are drawn this wide, then scaled down: fine lines, readable text
RADIUS, MARGIN = 14, 24  # rounded corners; gap to the screen's edge


class MeshWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.Tool)   # Tool: no taskbar button
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Face mesh")
        self.resize(WIDTH, WIDTH * 3 // 4)
        self.image: QImage | None = None
        self.get = None
        self.fx = MeshFx()
        self._drag: QPoint | None = None
        self._placed = False
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)

    def start(self, get, show_face: bool = False, reveal_s: float = 2.0, frame_on: bool = False) -> None:
        """get: a FrameFeed reader of "picture" ((FrameResult, RGB picture) or NO_FRAME)."""
        self.get, self.fx, self.image = get, MeshFx(show_face, reveal_s, frame_on), None
        if not self._placed:   # bottom-right the first time; after that where it was dragged
            area = QApplication.primaryScreen().availableGeometry()
            self.move(area.right() - self.width() - MARGIN, area.bottom() - self.height() - MARGIN)
            self._placed = True
        self.show()
        self.timer.start()

    def stop(self) -> None:
        self.timer.stop()
        self.get = None
        self.hide()

    def tick(self) -> None:
        got = self.get() if self.get else NO_FRAME
        if got is NO_FRAME or got[1] is None:
            return
        result, rgb = got
        # ponytail: the effects run here, on the GUI thread (~20 ms a 640x480 frame); move them onto the
        # camera's thread (FrameFeed) if the app stutters while this window is up
        h, w = rgb.shape[:2]
        size = (DRAW_WIDTH, DRAW_WIDTH * h // w)
        frame = cv2.resize(cv2.cvtColor(cv2.flip(rgb, 1), cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_AREA)
        if result is not None and result.face_found:   # mirrored like the picture: x and the turn flip
            landmarks = result.landmarks.copy()
            landmarks[:, 0] = 1 - landmarks[:, 0]
            pose = dataclasses.replace(result.head_pose, yaw=-result.head_pose.yaw, roll=-result.head_pose.roll)
            result = dataclasses.replace(result, landmarks=landmarks, head_pose=pose)
        out = cv2.cvtColor(self.fx.draw(frame, result, time.perf_counter()), cv2.COLOR_BGR2RGB)
        self.image = QImage(out.data, size[0], size[1], 3 * size[0], QImage.Format.Format_RGB888).copy()
        if self.height() != WIDTH * h // w:
            self.resize(WIDTH, WIDTH * h // w)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), RADIUS, RADIUS)
        p.setClipPath(path)
        p.fillRect(self.rect(), QColor("black"))
        if self.image is None:
            p.setPen(QColor("#9AA0A6"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Starting the camera…")
        else:
            p.drawImage(QRectF(self.rect()), self.image)

    # drag it anywhere
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.globalPosition().toPoint() - self.pos()

    def mouseMoveEvent(self, event) -> None:
        if self._drag is not None:
            self.move(event.globalPosition().toPoint() - self._drag)

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None
