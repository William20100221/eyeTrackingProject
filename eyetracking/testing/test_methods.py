"""
Self-check for the methods (no webcam):  python -m eyetracking.testing.test_methods

1. Every method's calibration window, run quickly on fake features, hands back the right
   recording: Original = dots only, on its own grid, row by row; Sweep = a moving-dot sweep
   + head frames + the 4x4 dots; Points / Line model = 14 line dots + head + the 4x4 dots; every grid row by row
2. train() fits each kind of model on a simulated person and predicts near the dots
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
from eyetracking.core.methods import METHOD_BY_ID, METHODS, TrainedModel, train
from eyetracking.core.models import CalibrationRecording, CalibrationSample

cal.SWEEP_SETTLE_MS, cal.SWEEP_CROSS_MS = 50, 200                                 # speed everything up
cal.HEAD_SETTLE_MS, cal.HEAD_MOVE_MIN_MS, cal.HEAD_MOVE_MAX_MS = 50, 50, 150    # the fake head never moves: max it is
cw.SETTLE_MS, cw.COLLECT_MS, cw.DOT_SAMPLES, cw.MAX_COLLECT_MS, cw.TICK_MS = 20, 60, 3, 500, 5
qt = QApplication.instance() or QApplication([])


# ---- 1. each window's recording ------------------------------------------------
def run(method, **kw) -> CalibrationRecording:
    win = cw.CalibrationWindow(method, get_features=lambda: (0.5,) * N_FEATURES, **kw)
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
for m in METHODS:
    rec = run(m)
    dots = list(dict.fromkeys(targets(rec.dots)))                                 # in the order they came
    assert len(dots) == m.dots and set(dots) == set(cal.point_grid(m.grid)), (m.id, len(dots))
    if m.random_order:                                                              # the Random collection: shuffled
        assert dots != cal.point_grid(m.grid), m.id                                 # (9! orders: equal by chance ~never)
    else:
        assert dots == cal.point_grid(m.grid), m.id                                 # In order: row by row
    assert m.grid == (4 if m.kind != "grid" else m.grid), m.id                       # Sweep / Points / Line: 4x4
    if m.kind == "grid":
        assert rec.sweep == () and rec.head == (), m.id                             # the dots, nothing else
    else:
        assert rec.head and set(targets(rec.head)) == {(0.5, 0.5)}, m.id
        if m.kind == "sweep":
            assert len(set(targets(rec.sweep))) > 50, m.id                          # a moving dot: a target per frame
        else:
            assert set(targets(rec.sweep)) == set(cal.sweep_points()), m.id         # the 14 line dots
    print(f"1. {m.id:<13} {len(rec.sweep):>4} step-1 frames  {len(rec.head):>3} head frames  {len(dots)} dots")

# Sweep with the dots reused from a 4x4 calibration: the window ends after the head step
borrowed = tuple(CalibrationSample((0.4,) * N_FEATURES, x, y) for x, y in cal.point_grid(4))
rec = run(METHOD_BY_ID["sweep"], dots=borrowed, dots_from="original-4x4_2026-10-18_170000.json")
assert rec.dots == borrowed and rec.head and len(set(targets(rec.sweep))) > 50
print("1. sweep with the dots reused: the sweep and head steps only")

# ---- 2. training per model kind --------------------------------------------------
rng = np.random.default_rng(1)
screen = ScreenGeometry(27.7, 18.9)


def frames(t, n=25, spread=(2, 2, 1, 1, 1, 1)):
    h = lambda: list(rng.normal([0, 0, 0, 0, -5, -65], spread))
    return [CalibrationSample(features_for(screen, t, h(), (2, -1), 0.9, rng), *t) for _ in range(n)]


def recording(m) -> CalibrationRecording:
    dots = [s for t in cal.point_grid(m.grid) for s in frames(t)]
    if m.kind == "grid":
        return CalibrationRecording((), (), tuple(dots))
    sweep = [s for t in cal.sweep_points() for s in frames(t, 15)]
    head = frames((0.5, 0.5), 150, spread=(8, 6, 3, 2, 2, 4))                       # the head really moves here
    return CalibrationRecording(tuple(sweep), tuple(head), tuple(dots))


for m in METHODS:
    model, n = train(m, [recording(m)], screen)
    assert isinstance(model, TrainedModel) and n >= 0.8 * m.dots * 25, (m.id, n)
    assert (model.profile is not None) == (m.model == "profile"), m.id
    err = np.mean([np.hypot(*(np.array(model.predict(s.features)) - (s.target_x, s.target_y)))
                   for s in recording(m).dots[::7]])
    # the simulated person's vertical eye RATIO is a weak signal (features_for: corners, no iris height), and
    # the profile models use the 4 ratios only, so they land near 15% there; ridge / line also see iris-down
    assert err < (0.2 if m.model == "profile" else 0.08), (m.id, err)
    assert model.predict(None) is None                                              # a blink
    print(f"2. {m.id:<13} {m.model:<8} {n} samples, mean error {err:.1%} of the screen")
assert train(METHODS[0], [], screen) == (None, 0)
print("methods ok")
