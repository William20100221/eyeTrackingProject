"""
Video evidence: the 3D face mesh drawn onto a video of the user, or live (ui/mesh_window.py).

The 478 landmarks appear one by one, like writing (top-left to the right, then
the next line), over the first --reveal seconds (2 by default), then the whole
mesh stays and follows the face. The look, as in face-scan shorts:
  * neon glow, cyan near the camera fading to violet far away (the depth, z, shows)
  * a laser line sweeping down the face as it draws, sparks flying off the pen
  * "SCANNING 213/478" while it draws; when it's done a flash, a short RGB glitch
    and "FACE LOCKED" with the live head pose (--frame: corner brackets snap onto the face too)
  * a faint hologram flicker
  * 3D model (the default): the person's picture fades out after the reveal, the mesh stays alone

    python -m eyetracking.video.face_mesh me.mp4                 # the 3D model
    python -m eyetracking.video.face_mesh me.mp4 --mode face     # the mesh stays on the (dimmed) video
    python -m eyetracking.video.face_mesh me.mp4 --frame         # with the lock-on brackets

Writes me_mesh_model.mp4 / me_mesh_face.mp4 next to the input (no audio).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks.python import vision
from scipy.spatial import Delaunay

from eyetracking.core.models import FrameResult

_C = vision.FaceLandmarksConnections
EDGE_ARRAY = np.array([(c.start, c.end) for c in _C.FACE_LANDMARKS_TESSELATION])
OUTLINES = np.array([(c.start, c.end) for group in (_C.FACE_LANDMARKS_LEFT_EYE, _C.FACE_LANDMARKS_RIGHT_EYE,
                     _C.FACE_LANDMARKS_LEFT_EYEBROW, _C.FACE_LANDMARKS_RIGHT_EYEBROW, _C.FACE_LANDMARKS_LIPS,
                     _C.FACE_LANDMARKS_FACE_OVAL) for c in group])
LOW_POLY = 100                             # 3D model alone: landmarks kept for its plainer mesh
ROWS = 20                                  # "lines of writing" the face is split into for the reveal
NEAR = np.array([255, 255, 60.0])          # BGR cyan: nearest the camera
FAR = np.array([255, 60, 190.0])           # BGR violet: farthest
NEWEST = 12                                # the last few points drawn pop in white
GLOW = 1.8                                 # strength of the neon bloom
DIM = 0.55                                 # the picture is dimmed so the mesh glows
LOCK_S = 0.6                               # seconds of flash + brackets snapping in after the reveal
GLITCH_S = 0.15                            # seconds of RGB split at the lock
FADE_S = 0.8                               # 3D model: seconds for the picture to fade out after the reveal
FONT = cv2.FONT_HERSHEY_SIMPLEX


def reveal_rank(landmarks: np.ndarray) -> np.ndarray:
    """rank[i] = when landmark i appears: like writing, left to right along a line, then the next line down."""
    y = landmarks[:, 1]
    row = ((y - y.min()) / (np.ptp(y) or 1) * ROWS).astype(int)
    order = np.lexsort((landmarks[:, 0], row))   # by row first, then x within the row
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))
    return rank


def pixels(landmarks: np.ndarray, w: int, h: int) -> np.ndarray:
    return (landmarks[:, :2] * (w, h)).astype(int)


def low_poly(landmarks: np.ndarray, n: int = LOW_POLY) -> tuple[np.ndarray, np.ndarray]:
    """The plainer mesh for the 3D model alone: (edges, the landmarks drawn as dots). The full mesh crowds
    the eyes and mouth (and the 10 iris points) into bright blobs, so: n landmarks spread evenly (each the
    farthest from those picked so far), triangulated, and the eyes, brows, lips and face edge as outlines."""
    p = landmarks[:468, :2]                                  # 468+: the irises
    keep = [int(np.argmin(p[:, 1]))]
    far = np.linalg.norm(p - p[keep[0]], axis=1)
    for _ in range(n - 1):
        keep.append(int(np.argmax(far)))
        far = np.minimum(far, np.linalg.norm(p - p[keep[-1]], axis=1))
    keep = np.array(keep)
    tri = keep[Delaunay(p[keep]).simplices]
    edges = np.unique(np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]]), axis=1), axis=0)
    return np.vstack([edges, OUTLINES]), keep


def draw_mesh(canvas: np.ndarray, landmarks: np.ndarray, rank: np.ndarray, k: float, boost: float = 1.0,
              edges: np.ndarray = EDGE_ARRAY, points: np.ndarray | None = None) -> None:
    """Draw the landmarks with rank < k and the mesh edges between them (boost > 1 = the lock flash);
    edges / points: another mesh (low_poly), points None = every landmark."""
    h, w = canvas.shape[:2]
    pts = pixels(landmarks, w, h)
    z = landmarks[:, 2]
    near = ((z.max() - z) / (np.ptp(z) or 1))[:, None]   # MediaPipe: smaller z = nearer; 1 = nearest
    colors = np.minimum(255, (FAR + (NEAR - FAR) * near) * (0.45 + 0.55 * near) * boost)
    dots, lines, pts = colors.tolist(), (colors * 0.6).tolist(), pts.tolist()   # plain lists: fast for cv2
    shown = rank < k
    for a, b in edges[shown[edges[:, 0]] & shown[edges[:, 1]]].tolist():
        cv2.line(canvas, pts[a], pts[b], lines[a], 1, cv2.LINE_AA)
    if points is not None:
        shown = shown & np.isin(np.arange(len(rank)), points)
    for i in np.flatnonzero(shown).tolist():
        if rank[i] >= k - NEWEST and k < len(rank):
            cv2.circle(canvas, pts[i], 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, pts[i], 2, dots[i], -1, cv2.LINE_AA)


def emit_sparks(sparks: np.ndarray, at: np.ndarray, rng: np.random.Generator, each: int = 3) -> np.ndarray:
    """Add `each` sparks at every point in `at` (pixels). A spark row: x, y, vx, vy, life (1 -> 0)."""
    v = rng.normal(0, 2.5, (len(at) * each, 2)) - (0, 1.5)   # mostly up and out
    return np.vstack([sparks, np.column_stack([np.repeat(at, each, axis=0), v, np.ones(len(v))])])


def step_sparks(sparks: np.ndarray, canvas: np.ndarray, dt: float) -> np.ndarray:
    """Move, fade and draw the sparks (dt: seconds since the last frame); drop the dead ones."""
    # ponytail: speeds are per frame, so a 60 fps video gets sparks half as fast; scale by dt if that shows
    sparks[:, :2] += sparks[:, 2:4]
    sparks[:, 3] += 0.25                  # gravity
    sparks[:, 4] -= 2 * dt                # half a second to live
    sparks = sparks[sparks[:, 4] > 0]
    colors = (NEAR * sparks[:, 4:5] + 60).clip(0, 255).tolist()   # all at once: hundreds of sparks a frame
    for xy, color in zip(sparks[:, :2].astype(int).tolist(), colors):
        cv2.circle(canvas, xy, 1, color, -1, cv2.LINE_AA)
    return sparks


def brackets(canvas: np.ndarray, box: np.ndarray, color) -> None:
    """Four corner brackets (the "lock-on" frame) around box = x0, y0, x1, y1."""
    x0, y0, x1, y1 = box.astype(int)
    n = (x1 - x0) // 5
    for x, y, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(canvas, (x, y), (x + dx * n, y), color, 2, cv2.LINE_AA)
        cv2.line(canvas, (x, y), (x, y + dy * n), color, 2, cv2.LINE_AA)


class MeshFx:
    """The effects, frame after frame: draw(picture, its FrameResult, its time) -> the picture with them.
    The reveal starts on the first frame with a face; a new MeshFx plays it again."""

    def __init__(self, show_face: bool = False, reveal_s: float = 2.0, frame_on: bool = False) -> None:
        self.show_face, self.reveal_s, self.frame_on = show_face, reveal_s, frame_on
        self.rng = np.random.default_rng(0)   # same video in, same video out
        self.sparks = np.empty((0, 5))
        self.rank, self.start, self.before, self.k_before = None, 0.0, None, 0.0
        self.simple: tuple[np.ndarray, np.ndarray] | None = None   # low_poly(), made at the lock

    def draw(self, frame: np.ndarray, result: FrameResult | None, now: float) -> np.ndarray:
        """frame: BGR picture (any size: the landmarks are fractions of it); now: seconds, any clock."""
        h, w = frame.shape[:2]
        dt, self.before = (0.0 if self.before is None else now - self.before), now
        text_size = h / 900
        fx = np.zeros_like(frame)   # everything that glows
        lock = -1.0                 # seconds since the reveal finished (< 0: not finished)
        if result is not None and result.face_found:
            if self.rank is None:
                self.rank, self.start = reveal_rank(result.landmarks), now
            rank, pts = self.rank, pixels(result.landmarks, w, h)
            t = min(1.0, (now - self.start) / max(0.01, self.reveal_s))
            k = len(rank) * t * t * (3 - 2 * t)   # ease in-out
            lock = now - self.start - self.reveal_s
            mesh = ()
            if not self.show_face and lock >= LOCK_S:   # the 3D model alone, after the flash: the plainer mesh
                self.simple = self.simple or low_poly(result.landmarks)
                mesh = self.simple
            draw_mesh(fx, result.landmarks, rank, k, 1 + 1.5 * max(0.0, 1 - lock / LOCK_S) if lock >= 0 else 1, *mesh)
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            if lock < 0:
                new = (rank >= self.k_before) & (rank < k)
                self.sparks = emit_sparks(self.sparks, pts[new], self.rng)
                if new.any():   # the laser: across the face at the line being written
                    y = pts[rank < k, 1].max()
                    cv2.line(fx, (x0 - 20, y), (x1 + 20, y), (255, 255, 200), 2, cv2.LINE_AA)
                label = f"SCANNING  {int(k)}/{len(rank)}"
            else:
                pad = 0.08 + 0.4 * max(0.0, 1 - lock / LOCK_S) ** 2   # brackets snap in from wide
                d = np.array([x1 - x0, y1 - y0]) * pad
                if self.frame_on:
                    brackets(fx, np.array([x0, y0, x1, y1]) + np.r_[-d, d], (255, 255, 255))
                p = result.head_pose
                label = f"FACE LOCKED  {len(rank)} LANDMARKS"
                cv2.putText(fx, f"YAW {p.yaw:+5.1f}  PITCH {p.pitch:+5.1f}  ROLL {p.roll:+5.1f}",
                            (int(x0 - d[0]), int(y1 + d[1] + 30 * text_size)),
                            FONT, 0.6 * text_size, (255, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(fx, label, (x0 - 20, int(y0 - 0.12 * (y1 - y0))), FONT, 0.8 * text_size,
                        (255, 255, 160), 2, cv2.LINE_AA)
            if 0 <= lock < GLITCH_S:   # RGB split
                shift = max(2, w // 150)
                fx[:, :, 0] = np.roll(fx[:, :, 0], shift, axis=1)
                fx[:, :, 2] = np.roll(fx[:, :, 2], -shift, axis=1)
            self.k_before = k
        elif self.rank is not None and not self.show_face:
            lock = FADE_S                       # face lost after the reveal: the 3D model stays black
        self.sparks = step_sparks(self.sparks, fx, dt)
        # the bloom blurred at a quarter of the size: the same glow, 16x less work (live needs ~30 ms a frame)
        small = cv2.resize(fx, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        bloom = cv2.resize(cv2.GaussianBlur(small, (0, 0), w / 600), (w, h), interpolation=cv2.INTER_LINEAR)
        picture = DIM if self.show_face else DIM * min(1.0, max(0.0, 1 - lock / FADE_S))
        out = cv2.addWeighted(frame, picture, fx, self.rng.uniform(0.85, 1.0), 0)   # + hologram flicker
        return cv2.addWeighted(out, 1.0, bloom, GLOW, 0)                            # uint8: saturates at 255


def render(src: Path, dst: Path, mode: str = "model", reveal_s: float = 2.0, frame_on: bool = False) -> None:
    from eyetracking.services.landmarker import FaceLandmarker   # here: the live window needs no second model
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"can't open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    fx, i = MeshFx(mode == "face", reveal_s, frame_on), 0
    with FaceLandmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.write(fx.draw(frame, landmarker.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), i / fps))
            i += 1
    cap.release()
    out.release()
    print(f"{i} frames -> {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw the 3D face mesh onto a video.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--mode", choices=["model", "face"], default="model",
                        help="model = the picture fades out after the reveal, face = it stays")
    parser.add_argument("--reveal", type=float, default=2.0, help="seconds to draw every landmark")
    parser.add_argument("--frame", action="store_true", help="corner brackets snap onto the face at the lock")
    parser.add_argument("-o", "--out", type=Path, help="output .mp4 (default: next to the input)")
    args = parser.parse_args()
    render(args.video, args.out or args.video.with_name(f"{args.video.stem}_mesh_{args.mode}.mp4"),
           args.mode, args.reveal, args.frame)
