"""
Self-check for the landmark figure (no camera, no model):  python -m eyetracking.testing.test_landmark_figure

A made-up face: 478 landmarks, the eleven the features use placed where they are on a real face, drawn
on a blank frame into a temp folder. The two PNGs and three CSVs must exist; the feature CSV must hold the
12 values gaze_features gives for those landmarks, and the points CSV every landmark with its role.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np

from eyetracking.core.features import FEATURE_NAMES, L_IRIS, NOSE_TIP, R_INNER, R_IRIS, R_OUTER, gaze_features
from eyetracking.core.models import FrameResult, HeadPose
from eyetracking.testing.landmark_figure import USED, make_figures

rng = np.random.default_rng(0)
lm = np.c_[rng.uniform(0.35, 0.65, (478, 2)), rng.normal(0, 0.02, 478)]   # a face's cloud of points
for i, x, y in [(R_OUTER, .33, .42), (R_INNER, .43, .42), (159, .38, .40), (145, .38, .445), (R_IRIS, .39, .42),
                (362, .57, .42), (263, .67, .42), (386, .62, .40), (374, .62, .445), (L_IRIS, .625, .425),
                (NOSE_TIP, .50, .58)]:
    lm[i, :2] = x, y
frame = FrameResult(0, True, HeadPose(2.0, -3.0, 1.0, 0.5, -4.0, -60.0), 0.1, 0.1, lm)
rgb = np.full((480, 640, 3), 200, np.uint8)

out = Path(tempfile.mkdtemp())
written = make_figures(rgb, frame, out, "a made-up face", person="Test")
names = {p.name for p in written}
assert names == {"landmarks.png", "landmarks_points.csv", "landmarks_eye.png", "landmarks_eye_points.csv",
                 "landmarks.csv"}, names
assert all(p.exists() and p.stat().st_size > 0 for p in written)
assert (out / "landmarks.png").read_bytes()[:4] == b"\x89PNG" and (out / "landmarks_eye.png").read_bytes()[:4] == b"\x89PNG"
with open(out / "landmarks.csv", encoding="utf-8") as f:
    rows = list(csv.reader(f))[1:]
assert [r[0] for r in rows] == list(FEATURE_NAMES)
assert np.allclose([float(r[1]) for r in rows], gaze_features(frame), atol=1e-5)    # the numbers the diagram produces
with open(out / "landmarks_points.csv", encoding="utf-8") as f:
    points = list(csv.reader(f))[1:]
assert len(points) == 478 and {r[1] for r in points if r[1]} == {role for _, role, _ in USED}
assert abs(float(points[R_IRIS][2]) - 0.39 * 640) < 0.01                             # pixels, from the fractions
with open(out / "landmarks_eye_points.csv", encoding="utf-8") as f:
    lines = {r[2]: float(r[5]) for r in csv.reader(f) if r[5:] and r[5] and r[0] != "panel"}
f = gaze_features(frame)
assert np.allclose([lines["R iris x"], lines["L iris x"], lines["R iris y"], lines["L iris y"]], f[:4], atol=1e-4), lines   # the figure's labels = the features
print("landmark figure ok:", ", ".join(sorted(names)))
