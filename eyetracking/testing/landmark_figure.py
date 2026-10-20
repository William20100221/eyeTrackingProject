"""
The landmark figure for the report: one frame, the Face Landmarker's 478 points, the eleven the app
uses, the lines its features are measured along, and the 12 feature values that frame gives.

    python -m eyetracking.testing.landmark_figure [--image PATH | --camera N] [--person NAME] [--screen-cm WxH]

Grabs one frame (a photo, or camera N: by default the camera chosen in the lab's Settings), runs the Face
Landmarker (services/landmarker.py) and core/features.py's gaze_features on it, and writes into
docs/img/report/ (docs/img/report/<person>/ with --person), each figure with the numbers it plots:

    landmarks.png              the frame, every landmark as a faint grey dot, the eleven used points large
                               and labelled (index and role), the four ratio lines (iris between the
                               corners, iris between the lids, each eye) with their values, and the nose
                               ruler (corner midpoint to nose tip, 3.75 cm)
    landmarks_points.csv       every landmark it draws: index, role (if used), x, y in pixels, x, y, z as
                               MediaPipe gives them (fractions of the image)
    landmarks_eye.png          a zoomed crop of each eye, the same labels, readable in a report
    landmarks_eye_points.csv   the points in those crops, per panel
    line_model.png             the same frame as core/line_model.py sees it, in cm, a top and a side view: each
                               eye's centre of rotation and iris (IRIS_RADIUS_CM along its measured direction),
                               the two gaze rays and where they meet the screen plane, the straight middle line (an
                               uncalibrated LineModel's prediction), the screen (ScreenGeometry: --screen-cm, or
                               the size Windows reports); the vergence angle core/vergence.py gives, written on it
    line_model.csv             those points in cm, the angles, the middle line's screen fractions
    landmarks.csv              the 12 feature values of the frame, with their names (also printed)

The indices and the ruler length are imported from core/features.py, never copied here. PNGs at 200 dpi.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                         # no window: files only (and headless tests)
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402

from eyetracking.core.features import (CORNER_TO_NOSE_CM, FEATURE_NAMES, L_INNER, L_IRIS, L_LOWER,   # noqa: E402
                                       L_OUTER, L_UPPER, NOSE_TIP, R_INNER, R_IRIS, R_LOWER, R_OUTER, R_UPPER,
                                       gaze_features)
from eyetracking.core.line_model import IRIS_RADIUS_CM, LineModel, ScreenGeometry, eye_lines             # noqa: E402
from eyetracking.core.models import FrameResult                                                    # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "docs" / "img" / "report"
DPI = 200
# (index, role, which eye) for the eleven points the features are measured from
EYES = {"R": (R_OUTER, R_INNER, R_UPPER, R_LOWER, R_IRIS), "L": (L_OUTER, L_INNER, L_UPPER, L_LOWER, L_IRIS)}
PARTS = ("outer", "inner", "upper", "lower", "iris")
USED = [(i, f"{eye} {part}", eye) for eye, idx in EYES.items() for i, part in zip(idx, PARTS)] + \
       [(NOSE_TIP, "nose tip", "")]
ROLE = {i: role for i, role, _ in USED}
GREY, LINE, POINT, RULER = "#9E9E9E", "#1E88E5", "#E5483B", "#43A047"


FACE_WAIT_S = 5.0           # how long a camera is watched for a face before giving up


def chosen_camera() -> int:
    """The camera the lab's Settings chose (by name, as the app finds it), else 0."""
    from PySide6.QtCore import QSettings
    from eyetracking.services.camera import camera_names
    name = QSettings("StudyAssistant", "CalibrationLab").value("camera", "")   # read only
    names = camera_names()
    return names.index(name) if name in names else 0


def grab(image: str | None = None, camera: int | None = None) -> tuple[np.ndarray, FrameResult, str]:
    """(one frame as RGB, its FrameResult, where it came from). A photo: that photo. A camera (the one chosen
    in Settings unless --camera says): frames are read until one shows a face with open eyes, for up to
    FACE_WAIT_S, which also lets the webcam's exposure settle. No face: the cameras are listed to pick from.
    (20 Oct: --camera 0 is the Surface's REAR camera, which saw no face; Settings had the front one.)"""
    import time
    import cv2 as cv
    from eyetracking.services.landmarker import FaceLandmarker
    with FaceLandmarker() as landmarker:
        if image:
            bgr = cv.imread(image)
            if bgr is None:
                raise SystemExit(f"Can't read the image {image}")
            rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
            return rgb, landmarker.detect(rgb, 0), str(Path(image).resolve())
        from eyetracking.services.camera import Camera, camera_names
        names = camera_names()
        index = chosen_camera() if camera is None else camera
        label = f"camera {index} ({names[index] if index < len(names) else 'unknown'})"
        print(f"using {label}: look at the screen")
        cam = Camera(index)
        start, rgb = time.perf_counter(), None
        try:
            while time.perf_counter() - start < FACE_WAIT_S:
                frame = cam.read_rgb()
                if frame is None:
                    continue
                rgb = frame
                result = landmarker.detect(frame, int((time.perf_counter() - start) * 1000))
                if result.face_found and gaze_features(result) is not None:
                    return frame, result, f"{label}, frame grabbed {datetime.now():%Y-%m-%d %H:%M:%S}"
        finally:
            cam.release()
    listed = ", ".join(f"--camera {i} = {n}" for i, n in enumerate(names))
    raise SystemExit(f"No face with open eyes in {FACE_WAIT_S:.0f} s from {label}"
                     + (" (no frame at all)" if rgb is None else "") + f". Cameras: {listed}.")

def _px(landmarks: np.ndarray, shape) -> np.ndarray:
    """Landmarks (fractions of the image) -> pixels."""
    h, w = shape[:2]
    return landmarks[:, :2] * (w, h)


def _per_point(ax) -> float:
    """How many data units (image pixels, down) one typographic point is in `ax`, at its current limits."""
    inv = ax.transData.inverted()
    return abs(inv.transform((0, 0))[1] - inv.transform((0, ax.figure.dpi / 72))[1])


def _draw(ax, p: np.ndarray, features, fs: float = 6.5, ruler: bool = True, out: float = 0.45,
          eyes=tuple(EYES)) -> list[tuple]:
    """The used points, the four ratio lines and the nose ruler on `ax` (pixels; set its limits first). Each
    line is labelled with the value gaze_features gave for it (not measured again here: the left eye's x runs
    from its inner corner, the right eye's from its outer one, and only features.py says so). The labels stand
    in a column beside each eye, `out` eye-widths outside its corners, spaced by the font so they never cover
    each other or the face, with a leader line to their point: the person's right eye (image left) on the
    left, the left eye on the right. The column's order is fixed, top to bottom: upper lid, the x ratio, inner
    corner, OUTER corner (the middle row, level with the eye: its leader is short and runs along no line),
    iris, the y ratio, lower lid. `eyes`: which eyes to label (a zoomed panel: its own). Returns what it
    plotted (for the CSV)."""
    # ponytail: one fixed column per eye; a face filling the photo pushes the columns past its edge (still drawn, outside the frame)
    drawn, step = [], 2.1 * fs * _per_point(ax)
    value_of = {("R", "x"): features[0], ("L", "x"): features[1], ("R", "y"): features[2], ("L", "y"): features[3]}
    for eye in eyes:
        outer, inner, upper, lower, iris = EYES[eye]
        items = []                                              # (anchor, text, colour)
        for a, b, axis, what in ((outer, inner, 0, "x"), (upper, lower, 1, "y")):
            ax.plot(*p[[a, b]].T, color=LINE, lw=fs / 4.5)
            value = float(value_of[(eye, what)])
            drawn.append(("line", f"{eye} iris {what}", a, b, value))
            items.append((p[[a, b]].mean(axis=0), f"{eye} iris {what} = {value:.2f}", LINE))
        items += [(p[i], f"{i} {eye} {part}", "black") for i, part in zip((outer, inner, upper, lower, iris), PARTS)]
        x_ratio, y_ratio, o, n, u, lo, ir = items
        items = [u, x_ratio, n, o, ir, y_ratio, lo]             # the order above
        pts = p[[outer, inner, upper, lower, iris]]
        width = np.ptp(pts[:, 0])
        left = eye == ("R" if p[R_OUTER, 0] < p[L_OUTER, 0] else "L")   # this eye is on the image's left
        x = pts[:, 0].min() - out * width if left else pts[:, 0].max() + out * width
        y0 = pts[:, 1].mean() - step * (len(items) - 1) / 2
        for k, (anchor, text, colour) in enumerate(items):
            ax.annotate(text, anchor, (x, y0 + k * step), fontsize=fs, color=colour, ha="right" if left else "left",
                        va="center", fontweight="bold" if colour == LINE else "normal",
                        arrowprops=dict(arrowstyle="-", color="#555555", lw=fs / 13, shrinkA=1, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    used = [i for i, _, _ in USED if i != NOSE_TIP or ruler]
    ax.scatter(*p[used].T, s=3.4 * fs, color=POINT, edgecolor="white", linewidth=0.6, zorder=3)
    if ruler:
        top = p[[R_OUTER, R_INNER, L_INNER, L_OUTER]].mean(axis=0)   # the corners' midpoint (features.py's ruler)
        ax.plot(*np.array([top, p[NOSE_TIP]]).T, color=RULER, lw=fs / 4, ls="--")
        gap = 1.5 * fs * _per_point(ax)
        ax.annotate(f"{NOSE_TIP} nose tip", p[NOSE_TIP], p[NOSE_TIP] + (gap, 0), fontsize=fs, va="center",
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=fs / 13),
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        # under the nose, away from both eyes' columns
        ax.annotate(f"ruler (dashed): corner midpoint to nose tip = {CORNER_TO_NOSE_CM:.2f} cm", p[NOSE_TIP] + (0, 2 * gap),
                    color=RULER, fontsize=fs, fontweight="bold", va="center", ha="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        drawn.append(("ruler", "corner midpoint to nose tip", -1, NOSE_TIP, CORNER_TO_NOSE_CM))
    return drawn


def _hit(origin: np.ndarray, line: np.ndarray) -> np.ndarray | None:
    """Where a ray meets the screen plane (z = 0): LineModel.predict's own step, for each eye's ray."""
    return origin + (-origin[2] / line[2]) * line if line[2] > 1e-6 else None


def line_model_figure(features, screen: ScreenGeometry, out_dir: Path, who: str = "") -> list[Path]:
    """line_model.png: the frame as core/line_model.py sees it, in cm (camera space: the camera at 0, the user
    at z < 0, y up): each eye's centre of rotation and its iris (IRIS_RADIUS_CM along the eye's measured
    direction), each eye's ray to the screen plane, the straight middle line the model starts from (its hit =
    an uncalibrated LineModel's prediction), the screen (ScreenGeometry), in a top and a side view; the vergence
    angle core/vergence.py gives for the eyes' distance (eye_lines), and the angle between the measured rays."""
    centres, measured, _, _, vergence = eye_lines(features)
    irises = centres + IRIS_RADIUS_CM * measured
    hits = [_hit(c, d) for c, d in zip(centres, measured)]
    middle = centres.mean(axis=0)
    fx, fy = LineModel(screen).predict(features) or (np.nan, np.nan)   # untrained: w = 1, the straight line
    mid_hit = screen.point(fx, fy)
    between = math.degrees(math.acos(float(np.clip(measured[0] @ measured[1], -1, 1))))
    corners = np.array([screen.point(0, 0), screen.point(1, 0), screen.point(1, 1), screen.point(0, 1)])
    distance = -float(centres[:, 2].mean())
    fig, (top, side) = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (a, b), (xl, yl) in ((top, (0, 2), ("x (cm): the user's left is on the left", "z (cm): from the camera")),
                                 (side, (2, 1), ("z (cm): from the camera", "y (cm): up"))):
        ax.plot(*corners[[0, 1, 2, 3, 0]][:, [a, b]].T, color="black", lw=2, label="the screen (ScreenGeometry)")
        ax.plot(0, 0, marker="s", color="black", ms=6)
        ax.annotate("camera", (0, 0), (6, 6), textcoords="offset points", fontsize=8)
        for c, i, h, colour, name in zip(centres, irises, hits, (LINE, POINT), ("right eye", "left eye")):
            ax.plot(c[a], c[b], "o", color=colour, ms=6, label=f"{name}: centre of rotation")
            ax.plot(i[a], i[b], "x", color=colour, ms=7, label=f"{name}: iris ({IRIS_RADIUS_CM:g} cm out)")
            if h is not None:
                ax.plot([c[a], h[a]], [c[b], h[b]], color=colour, lw=1, label=f"{name}: its ray")
                ax.plot(h[a], h[b], "o", mfc="none", color=colour, ms=7)
        ax.plot([middle[a], mid_hit[a]], [middle[b], mid_hit[b]], color=RULER, ls="--", lw=1.2,
                label="the straight middle line (vergence taken out)")
        ax.plot(mid_hit[a], mid_hit[b], "*", color=RULER, ms=12)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    top.invert_xaxis()                                          # the camera's +x is the user's left: seen from behind
    top.set_title("Top view", fontsize=10)
    side.set_title("Side view", fontsize=10)
    zoom = top.inset_axes([0.62, 0.05, 0.36, 0.36])            # the eyes, where 1 cm is visible
    for c, i, colour in zip(centres, irises, (LINE, POINT)):
        zoom.plot([c[0], i[0]], [c[2], i[2]], color=colour, lw=1)
        zoom.plot(c[0], c[2], "o", color=colour, ms=5)
        zoom.plot(i[0], i[2], "x", color=colour, ms=6)
    zoom.set_aspect("equal", adjustable="datalim")
    zoom.invert_xaxis()
    zoom.set_title("the eyes, zoomed", fontsize=7)
    zoom.tick_params(labelsize=6)
    top.legend(fontsize=7, loc="lower left")                    # the screen and the camera are at the top
    side.text(0.02, 0.02, f"vergence (core/vergence.py) at {distance:.0f} cm: {math.degrees(vergence):.2f}°\n"
                          f"angle between the two measured rays: {between:.2f}°\n"
                          f"the middle line meets the screen at {fx:.0%} across, {fy:.0%} down",
              transform=side.transAxes, fontsize=9, va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY))
    fig.suptitle(f"{who}the line model on this frame, in cm (core/line_model.py; screen "
                 f"{screen.width_cm:.1f} x {screen.height_cm:.1f} cm, camera {screen.camera_above_cm:g} cm above it)",
                 fontsize=10)
    png, table = out_dir / "line_model.png", out_dir / "line_model.csv"
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    eyes = ("right eye", "left eye")
    rows = [(f"{n} centre of rotation", *c) for n, c in zip(eyes, centres)]
    rows += [(f"{n} iris", *i) for n, i in zip(eyes, irises)]
    rows += [(f"{n} ray meets the screen", *(h if h is not None else (None,) * 3)) for n, h in zip(eyes, hits)]
    rows += [("middle line meets the screen", *mid_hit), ("screen corner top left", *corners[0]),
             ("screen corner bottom right", *corners[2])]
    with open(table, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["what", "x_cm", "y_cm", "z_cm", "value"])
        wr.writerows([r[0], *("" if v is None else f"{v:.3f}" for v in r[1:]), ""] for r in rows)
        wr.writerows([("vergence_deg", "", "", "", f"{math.degrees(vergence):.3f}"),
                      ("eye_distance_cm", "", "", "", f"{distance:.2f}"),
                      ("angle_between_measured_rays_deg", "", "", "", f"{between:.3f}"),
                      ("middle_line_screen_x", "", "", "", f"{fx:.4f}"),
                      ("middle_line_screen_y", "", "", "", f"{fy:.4f}")])
    print(f"line model: vergence {math.degrees(vergence):.2f} deg at {distance:.0f} cm; measured rays {between:.2f} deg "
          f"apart; middle line at ({fx:.2f}, {fy:.2f}) of the screen")
    return [png, table]


def make_figures(rgb: np.ndarray, result: FrameResult, out_dir: Path, source: str, person: str = "",
                 screen: ScreenGeometry | None = None) -> list[Path]:
    """The figures, their CSVs and the feature CSV, written into out_dir (line_model.png only with the screen's
    size: the model needs it). Returns the files written."""
    if not result.face_found or result.landmarks is None:
        raise SystemExit(f"No face in {source}")
    out_dir.mkdir(parents=True, exist_ok=True)
    features = gaze_features(result)
    if features is None:
        raise SystemExit("The eyes are closed (or a degenerate eye) in this frame: no features")
    p = _px(result.landmarks, rgb.shape)
    who = f"{person}: " if person else ""
    written = []

    # 1. the whole frame
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(rgb)
    ax.scatter(*p.T, s=1.5, color=GREY, alpha=0.6, linewidths=0)
    _draw(ax, p, features)
    ax.set_title(f"{who}the {len(p)} Face Landmarker points; the {len(USED)} the features use", fontsize=9)
    ax.axis("off")
    written.append(out_dir / "landmarks.png")
    fig.savefig(written[-1], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    written.append(out_dir / "landmarks_points.csv")
    with open(written[-1], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["index", "role", "x_px", "y_px", "x", "y", "z"])
        wr.writerows([i, ROLE.get(i, ""), f"{p[i, 0]:.2f}", f"{p[i, 1]:.2f}", *(f"{v:.5f}" for v in result.landmarks[i])]
                     for i in range(len(p)))

    # 2. each eye, zoomed: the eye in the middle, room beside it for its column of labels (1.5 eye widths a
    # side, 1.2 above and below; past the photo's edge the panel is white)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    rows = []
    for ax, (eye, idx) in zip(axes, EYES.items()):
        pts = p[list(idx)]
        width = np.ptp(pts[:, 0])
        x0, x1 = pts[:, 0].min() - 1.5 * width, pts[:, 0].max() + 1.5 * width
        y0, y1 = pts[:, 1].mean() - 1.2 * width, pts[:, 1].mean() + 1.2 * width
        cx0, cx1, cy0, cy1 = int(max(0, x0)), int(min(w, x1)), int(max(0, y0)), int(min(h, y1))
        ax.imshow(rgb[cy0:cy1, cx0:cx1], extent=(cx0, cx1, cy1, cy0))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        near = [i for i in range(len(p)) if x0 <= p[i, 0] <= x1 and y0 <= p[i, 1] <= y1]
        ax.scatter(*p[near].T, s=6, color=GREY, alpha=0.7, linewidths=0)
        drawn = _draw(ax, p, features, fs=10, ruler=False, eyes=(eye,))
        ax.set_title(f"{'Right' if eye == 'R' else 'Left'} eye (the person's; image {'left' if eye == 'R' else 'right'})",
                     fontsize=10)
        ax.axis("off")
        rows += [[eye, i, ROLE.get(i, ""), f"{p[i, 0]:.2f}", f"{p[i, 1]:.2f}", ""] for i in near]
        rows += [[eye, f"{a}-{b}", what, "", "", f"{value:.4f}"] for kind, what, a, b, value in drawn if kind == "line"
                 and what.startswith(eye)]
    fig.suptitle(f"{who}the ratios the features measure, each eye", fontsize=11)
    written.append(out_dir / "landmarks_eye.png")
    fig.savefig(written[-1], dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    written.append(out_dir / "landmarks_eye_points.csv")
    with open(written[-1], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["panel", "index", "role", "x_px", "y_px", "ratio"])
        wr.writerows(rows)

    # 3. the line model on this frame
    if screen is not None:
        written += line_model_figure(features, screen, out_dir, who)

    # 4. the 12 features of this frame (the labels above show the first four)
    written.append(out_dir / "landmarks.csv")
    with open(written[-1], "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["feature", "value"])
        wr.writerows((name, f"{v:.5f}") for name, v in zip(FEATURE_NAMES, features))
    print(f"source: {source}")
    for name, v in zip(FEATURE_NAMES, features):
        print(f"  {name:<40} {v:9.4f}")
    for path in written:
        print(f"wrote {path}")
    return written


def screen_size(text: str | None) -> ScreenGeometry:
    """The screen for the line model: --screen-cm, or Windows' own size of the main screen (the app's
    gaze_window.screen_geometry). Printed, as the figure depends on it."""
    if text:
        w, h = (float(v) for v in text.lower().split("x"))
        screen = ScreenGeometry(w, h)
    else:
        from PySide6.QtWidgets import QApplication
        from eyetracking.ui.gaze_window import screen_geometry
        if QApplication.instance() is None:                    # Qt needs an app to know the screen
            QApplication([])
        screen = screen_geometry()
    print(f"screen: {screen.width_cm:.1f} x {screen.height_cm:.1f} cm" + (" (--screen-cm)" if text else " (from Windows)"))
    return screen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The landmark figure for the report")
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--image", help="a photo of a face instead of the camera")
    src.add_argument("--camera", type=int, help="the camera's index (default: the one chosen in the lab's Settings)")
    parser.add_argument("--person", default="", help="whose face: a sub-folder of docs/img/report/ and the titles")
    parser.add_argument("--screen-cm", help="the screen's size, WxH in cm (default: what Windows reports, via Qt)")
    args = parser.parse_args(argv)
    rgb, result, source = grab(args.image, args.camera)
    out = OUT_DIR / args.person if args.person else OUT_DIR
    make_figures(rgb, result, out, source, args.person, screen_size(args.screen_cm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
