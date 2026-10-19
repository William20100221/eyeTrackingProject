"""
Self-check for the calibration types and the models (no webcam):  python -m eyetracking.testing.test_methods

1. Every type's calibration window, run quickly on fake features, hands back the right
   recording: a grid = its dots only, row by row (In order) or shuffled (Random order);
   Sweep = a gliding dot + the head step, no dots; Points / Line model = 14 line dots + the head step, no dots
2. train() fits each of the 24 models on a simulated person and predicts near the dots: a grid alone,
   and each grid with Sweep, Points or Line model
3. The bug of 20 Oct: Sweep and Points must not be one model. Paired with the same grid, their models
   differ (their first steps are different data), and differ from the grid alone; a Points recording
   in Sweep's place (a copy from before 20 Oct) trains no Sweep model
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import eyetracking.core.calibration as cal
import eyetracking.ui.calibration_window as cw
from eyetracking.core.features import N_FEATURES
from eyetracking.core.line_model import ScreenGeometry, features_for
from eyetracking.core.methods import GRIDS, METHOD_BY_ID, METHODS, MODELS, TrainedModel, model_for, train
from eyetracking.core.models import CalibrationRecording, CalibrationSample

cal.SWEEP_SETTLE_MS, cal.SWEEP_CROSS_MS = 50, 200                                 # speed everything up
cal.HEAD_SETTLE_MS, cal.HEAD_MOVE_MIN_MS, cal.HEAD_MOVE_MAX_MS = 50, 50, 150    # the fake head never moves: max it is
cw.SETTLE_MS, cw.COLLECT_MS, cw.DOT_SAMPLES, cw.MAX_COLLECT_MS, cw.TICK_MS = 20, 60, 3, 500, 5
qt = QApplication.instance() or QApplication([])


# ---- 1. each window's recording ------------------------------------------------
def run(method) -> CalibrationRecording:
    win = cw.CalibrationWindow(method, get_features=lambda: (0.5,) * N_FEATURES)
    out = []
    win.finished.connect(lambda rec: (out.append(rec), qt.quit()))
    win.resize(400, 300)
    win.show()
    win.start()
    QTimer.singleShot(60000, qt.quit)
    qt.exec()
    assert out, f"{method.id} never finished"
    return out[0]


targets = lambda part: [(s.target_x, s.target_y) for s in part]
assert [m.row for m in METHODS] == ["In order"] * 3 + ["Random order"] * 3 + ["Special steps"] * 3
for m in METHODS:
    rec = run(m)
    if m.kind == "grid":
        dots = list(dict.fromkeys(targets(rec.dots)))                             # in the order they came
        assert len(dots) == m.dots and set(dots) == set(cal.point_grid(m.grid)), (m.id, len(dots))
        if m.random_order:                                                          # the Random row: shuffled
            assert dots != cal.point_grid(m.grid), m.id                             # (9! orders: equal by chance ~never)
        else:
            assert dots == cal.point_grid(m.grid), m.id                             # In order: row by row
        assert rec.sweep == () and rec.head == (), m.id                             # the dots, nothing else
    else:
        assert rec.dots == () and m.grid == 0, m.id                                 # a special records no dots
        assert rec.head and set(targets(rec.head)) == {(0.5, 0.5)}, m.id
        if m.kind == "sweep":
            assert len(set(targets(rec.sweep))) > 50, m.id                          # a moving dot: a target per frame
        else:
            assert set(targets(rec.sweep)) == set(cal.sweep_points()), m.id         # the 14 line dots
    print(f"1. {m.id:<13} {len(rec.sweep):>4} step-1 frames  {len(rec.head):>3} head frames  {len(rec.dots):>3} dot frames")

# ---- 2. training every model -----------------------------------------------------
rng = np.random.default_rng(1)
screen = ScreenGeometry(27.7, 18.9)


def frames(t, n=25, spread=(2, 2, 1, 1, 1, 1)):
    h = lambda: list(rng.normal([0, 0, 0, 0, -5, -65], spread))
    return [CalibrationSample(features_for(screen, t, h(), (2, -1), 0.9, rng), *t) for _ in range(n)]


def grid_recording(g) -> CalibrationRecording:
    return CalibrationRecording((), (), tuple(s for t in cal.point_grid(g.grid) for s in frames(t)))


def special_recording(kind: str) -> CalibrationRecording:
    head = frames((0.5, 0.5), 150, spread=(8, 6, 3, 2, 2, 4))                       # the head really moves here
    if kind == "sweep":                                                             # a gliding dot: one frame per place
        line = np.round(0.05 + 0.9 * (1 - np.cos(np.linspace(0, 4 * np.pi, 160))) / 2, 4)
        step1 = [s for x in line for s in frames((float(x), 0.5), 1)] + [s for y in line for s in frames((0.5, float(y)), 1)]
    else:
        step1 = [s for t in cal.sweep_points() for s in frames(t, 15)]
    return CalibrationRecording(tuple(step1), tuple(head), ())


grids = {g.id: grid_recording(g) for g in GRIDS}
specials = {k: special_recording(k) for k in ("sweep", "points")}
for m in MODELS:
    special = [specials[m.special.kind]] if m.special else []
    model, n = train(m, [grids[m.grid.id]], special, screen)
    assert isinstance(model, TrainedModel) and n >= 0.8 * m.grid.dots * 25, (m.id, n)
    assert (model.k_source == "its own head step") == (m.kind == "profile") and (model.K is None) == (m.kind == "line"), m.id
    if m.special:
        assert n > len(grids[m.grid.id].dots), m.id                                 # step 1 trains too
    err = np.mean([np.hypot(*(np.array(model.predict(s.features)) - (s.target_x, s.target_y)))
                   for s in grids[m.grid.id].dots[::7]])
    # the regressions are a straight line on the 6 eye features, with the head turn taken out by a matrix
    # measured on REAL people; the simulated person (features_for) is a different eye, so allow 20% here
    # (the real-data numbers are in docs/model_improvements.md); the geometric line model fits its own sim best
    assert err < (0.08 if m.kind == "line" else 0.2), (m.id, err)
    assert model.predict(None) is None and model.predict((0.5,) * 10) is None       # a blink; an older file's 10 features
    print(f"2. {m.id:<26} {m.kind:<8} {n} samples, mean error {err:.1%} of the screen")
assert train(MODELS[0], [], [], screen) == (None, 0)
assert train(model_for("original-4x4", "sweep"), [grids["original-4x4"]], [], screen) == (None, 0)   # no Sweep: no model

# ---- 3. Sweep and Points are two models -------------------------------------------
probe = [s.features for s in grids["original-4x4"].dots[::5]]
said = {s: np.array([train(model_for("original-4x4", s), [grids["original-4x4"]], [specials[k]] if k else [], screen)[0]
                     .predict(f) for f in probe]) for s, k in ((None, None), ("sweep", "sweep"), ("points", "points"))}
gap = lambda a, b: float(np.mean(np.hypot(*(said[a] - said[b]).T)))
assert gap("sweep", "points") > 1e-3 and gap("sweep", None) > 1e-3 and gap("points", None) > 1e-3, \
    (gap("sweep", "points"), gap("sweep", None), gap("points", None))
print(f"3. Sweep vs Points + Original 4x4: their points differ by {gap('sweep', 'points'):.1%} of the screen on the same frames")
wrong = train(model_for("original-4x4", "sweep"), [grids["original-4x4"]], [specials["points"]], screen)
assert wrong == (None, 0), "a Points recording must not train a Sweep model"
assert METHOD_BY_ID["line-model"].kind == "points"                                  # the Points procedure
print("methods ok")
