"""Why do ridge models blow up on a new sitting? Per-pair errors + the head features' spread."""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eyetracking.core.features import EYE, HEAD, IRIS_DOWN
from eyetracking.core.line_model import LineModel, ScreenGeometry
from eyetracking.core.quality import check_recording
from eyetracking.services.storage import load_calibration

data = Path(sys.argv[1])
pairs = [("William/original-3x3/original-3x3_2026-10-18_140539.json", "William/original-3x3/original-3x3_2026-10-18_154140.json"),
         ("William/original-5x5/original-5x5_2026-10-18_144205.json", "William/original-5x5/original-5x5_2026-10-18_160424.json"),
         ("William/sweep/sweep_2026-10-18_152112.json", "William/sweep/sweep_2026-10-18_152642.json")]
xy = lambda rec: (np.array([s.features for s in rec.dots]), np.array([(s.target_x, s.target_y) for s in rec.dots]))
err = lambda p, y: float(np.nanmean(np.hypot(*(p - y).T)))
names = ["rx", "lx", "ry", "ly", "yaw", "pitch", "roll", "hx", "hy", "hz", "rdown", "ldown"]
for a, b in pairs:
    A, B = (check_recording(load_calibration(data / p)).kept for p in (a, b))
    Xa, ya = xy(A)
    Xb, yb = xy(B)
    print(f"\n{a.split('/')[1]}: train {Path(a).stem[-6:]} -> test {Path(b).stem[-6:]}")
    print("  feature   train mean   train sd    test mean   shift/sd")
    for i, n in enumerate(names):
        sd = Xa[:, i].std()
        print(f"  {n:<8} {Xa[:, i].mean():10.3f} {sd:10.4f} {Xb[:, i].mean():11.3f} {(Xb[:, i].mean() - Xa[:, i].mean()) / max(sd, 1e-9):10.1f}")
    for label, cols in [("all 12", list(range(12))), ("eye + iris-down (6)", list(range(4)) + [10, 11]),
                        ("eye + iris-down + angles (9)", list(range(7)) + [10, 11]), ("eye 4", list(range(4)))]:
        for deg in (1, 2):
            for alpha in (1.0, 10.0, 100.0):
                m = make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=alpha)).fit(Xa[:, cols], ya)
                print(f"  ridge deg {deg} alpha {alpha:>5} on {label:<30} train {err(m.predict(Xa[:, cols]), ya):6.1%}  new sitting {err(m.predict(Xb[:, cols]), yb):7.1%}")
    for sc in (ScreenGeometry(34.4, 19.4), ScreenGeometry(27.7, 18.9), ScreenGeometry(50, 28)):
        line = LineModel(sc).fit(list(A.dots))
        print(f"  line model screen {sc.width_cm}x{sc.height_cm}: w={line.w} gain={line.gain.round(2)} offset={line.offset.round(3)} "
              f"train {err(line.predict_many(Xa), ya):6.1%}  new sitting {err(line.predict_many(Xb), yb):6.1%}")
