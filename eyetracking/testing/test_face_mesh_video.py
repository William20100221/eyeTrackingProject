"""
Self-check for the face mesh video and window (no camera, no model):  python -m eyetracking.testing.test_face_mesh_video

1. The reveal goes like writing: top line first, left to right, then the next line down; nothing is
   drawn before it starts, a few points part way, every point at the end. Sparks die in half a second.
2. MeshFx: the picture shows while the mesh draws; then the 3D model fades it out and turns plainer
   (low poly + outlines), "your face" keeps the picture and the full mesh.
3. The face mesh window draws each new camera frame, waits on none, and stop() hides it.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtWidgets import QApplication

from eyetracking.core.models import NO_FRAME, FrameResult, HeadPose
from eyetracking.ui.mesh_window import MeshWindow
from eyetracking.video.face_mesh import (EDGE_ARRAY, FADE_S, LOCK_S, ROWS, MeshFx, draw_mesh, emit_sparks, low_poly,
                                        reveal_rank, step_sparks)

rng = np.random.default_rng(0)
landmarks = rng.uniform(0.2, 0.8, (478, 3))
rank = reveal_rank(landmarks)
assert sorted(rank) == list(range(478)), rank
order = np.argsort(rank)
row = ((landmarks[order, 1] - 0.2) / 0.6 * ROWS).astype(int)               # ~ the line each point is on
assert landmarks[order[0], 1] < 0.2 + 0.6 / ROWS                           # starts on the top line...
assert landmarks[order[-1], 1] > 0.8 - 0.6 / ROWS                          # ...ends on the bottom one
assert (np.diff(row) >= -1).all()                                          # never jumps back up a line
same = np.diff(row) == 0
assert (np.diff(landmarks[order, 0])[same] >= 0).mean() > 0.9              # left to right within a line


def lit(k):
    canvas = np.zeros((200, 200, 3), np.uint8)
    draw_mesh(canvas, landmarks, rank, k)
    return np.count_nonzero(canvas.any(axis=2))


assert lit(0) == 0                                                         # nothing before the reveal
assert 0 < lit(20) < lit(478)                                              # part way: some of it

sparks = emit_sparks(np.empty((0, 5)), np.array([[100, 100], [50, 60]]), rng)
assert sparks.shape == (6, 5) and (sparks[:, 4] == 1).all()                # 3 sparks per point, full life
canvas = np.zeros((200, 200, 3), np.uint8)
sparks = step_sparks(sparks, canvas, dt=1 / 30)
assert len(sparks) == 6 and canvas.any()                                   # alive and drawn...
for _ in range(15):
    sparks = step_sparks(sparks, canvas, dt=1 / 30)
assert len(sparks) == 0                                                    # ...gone after half a second
print("1. reveal ok")

# ---- 2. MeshFx ------------------------------------------------------------------------------
face = FrameResult(0, True, HeadPose(1.0, 2.0, 3.0), 0.0, 0.0, rng.uniform(0.3, 0.7, (478, 3)))
picture = np.full((240, 320, 3), 200, np.uint8)
corner = (5, 5)                                                            # far from the mesh and its glow
for show_face, after in ((False, 0), (True, 110)):                         # 200 * DIM = 110
    fx = MeshFx(show_face, reveal_s=1.0)
    assert fx.draw(picture, None, 0.0)[corner].max() == 110                # no face yet: the dimmed picture
    fx.draw(picture, face, 0.1)
    assert fx.draw(picture, face, 0.6)[corner].max() == 110                # drawing: still the picture
    assert fx.draw(picture, face, 1.0 + FADE_S + 0.1)[corner].max() == after, show_face
for show_face in (False, True):
    fx = MeshFx(show_face, reveal_s=1.0)
    for t in (0.0, 0.5, 1.0, 1.0 + LOCK_S - 0.05):
        fx.draw(picture, face, t)
    assert fx.simple is None                                               # the full mesh through the flash...
    fx.draw(picture, face, 1.0 + LOCK_S + 0.05)
    assert (fx.simple is not None) != show_face                            # ...then plainer: the 3D model only
edges, points = low_poly(face.landmarks)
assert len(points) == 100 and points.max() < 468 and len(edges) < len(EDGE_ARRAY) / 3   # no iris, far fewer lines
print("2. MeshFx ok")

# ---- 3. the window --------------------------------------------------------------------------
qt = QApplication.instance() or QApplication([])
frames = iter([(face, picture[:, :, ::-1].copy()), NO_FRAME, (None, picture.copy())])
w = MeshWindow()
w.start(lambda: next(frames, NO_FRAME))
assert w.isVisible() and w.image is None                                   # "Starting the camera…"
w.tick()
assert w.image is not None and w.image.width() == 640 and w.height() == 360 * 240 // 320
first = w.image
w.tick()
assert w.image is first                                                    # NO_FRAME: nothing new, no repaint
w.tick()
assert w.image is not first                                                # a frame without a face still shows
w.stop()
assert not w.isVisible() and not w.timer.isActive()
print("3. window ok")
