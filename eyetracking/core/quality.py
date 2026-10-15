"""
Calibration quality: drop or reject data a real eye can't produce.

Three layers, cheapest first. Each frame / dot / calibration that fails
gets a reason, so the user can be told what went wrong.

 1. FRAME   physically possible?        iris inside the eye, head in front
            of the screen, both eyes turning together (Hering's law)
 2. DOT     was the user looking at it?  frames whose 3D gaze line is far from the dot's typical
            value are glances away; a dot with too few left is dropped
 3. SESSION did the eyes follow the dots?  eye position must rank with dot
            position; dots that don't fit the rest are dropped; if too many
            fail, the whole calibration is rejected

Sources:
  * Robust outliers: median +- 2.5..3 MAD, not mean +- 3 SD (the mean and SD
    are pulled by the very outliers being looked for). Leys et al. 2013,
    https://dipot.ulb.ac.be/dspace/bitstream/2013/139499/1/Leys_MAD_final-libre.pdf
  * A fixation is a tight cluster of gaze points (~0.5-1 degree); points
    outside it are saccades / glances. Salvucci & Goldberg 2000,
    https://www.cs.drexel.edu/~dds26/publications/Salvucci-ETRA00.pdf
  * Both eyes receive the same command and turn together (only vergence,
    ~3 degrees each, differs). Hering's law,
    https://en.wikipedia.org/wiki/Hering%27s_law_of_equal_innervation
Limits were checked against two real calibrations (15 Oct): iris x 0.42-0.60,
head ~70 cm away, eyes' difference within 0.04 of its own median, dot order
vs eye order |Spearman| = 0.98, 4-12% of frames beyond 3 MAD of their dot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import spearmanr

from eyetracking.core.features import IRIS_DOWN, N_FEATURES
from eyetracking.core.line_model import angles, eye_lines
from eyetracking.core.models import CalibrationRecording, CalibrationSample

# --- 1. one frame ------------------------------------------------------------
IRIS_X_RANGE = (0.15, 0.85)     # iris (~1.2 cm) in a ~2.6 cm eye: its centre can't reach the corners
IRIS_DOWN_MAX_CM = 0.7          # eye turned up/down ~40 deg max x ~1 cm radius
HEAD_DISTANCE_CM = (30, 120)    # nobody uses a laptop from closer or further
HEAD_ANGLE_MAX_DEG = 40         # turned further, the screen is out of view
EYES_APART_X = 0.08             # both eyes turn together: after removing this user's
EYES_APART_DOWN_CM = 0.25       # own fixed difference, they can't disagree by more

# --- 2. one dot --------------------------------------------------------------
MAD_K = 3.0                     # Leys et al.: 2.5 moderate, 3 conservative (keeps more)
MAD_SCALE = 1.4826              # makes MAD comparable to an SD for normal noise
MIN_DOT_FRAMES = 10             # fewer good frames left = the dot can't be trusted

# --- 3. whole calibration ----------------------------------------------------
MIN_FOLLOW_X = 0.7              # |rank correlation| dot x vs eye x across dots (real data: 0.98)
MIN_FOLLOW_Y = 0.4              # vertical eye signal is weaker
DOT_MISFIT_MIN = 0.08           # a dot further than this (screen fraction) from the
                                # straight-line fit of the others is "not looked at"
MAX_BAD_DOTS = 0.25             # more than a quarter of the dots bad -> reject it all

def gaze(features) -> np.ndarray:
    """Where the eyes point, as the 3D model's line (eye turn + head turn),
    [sideways, up/down] radians. Raw eye numbers alone would mislead: when the
    head turns, the eyes counter-rotate to stay on the dot."""
    _, _, d, rot, _ = eye_lines(features)
    return angles(rot @ d)


@dataclass
class QualityReport:
    ok: bool                                   # False = reject the whole calibration
    kept: CalibrationRecording                 # the recording with bad frames / dots removed
    reasons: list[str] = field(default_factory=list)   # what was dropped or why it was rejected

    def summary(self) -> str:
        return "; ".join(self.reasons) if self.reasons else "all data looks real"


def frame_problem(f, eye_gap: tuple[float, float]) -> str | None:
    """Why this frame is physically impossible, or None if it's fine.
    eye_gap = this user's own typical (x, down) difference between the eyes."""
    rx, lx, yaw, pitch, roll, dist = f[0], f[1], f[4], f[5], f[6], -f[9]
    dr, dl = f[IRIS_DOWN]
    if not (IRIS_X_RANGE[0] <= rx <= IRIS_X_RANGE[1] and IRIS_X_RANGE[0] <= lx <= IRIS_X_RANGE[1]):
        return "iris outside the eye"
    if abs(dr) > IRIS_DOWN_MAX_CM or abs(dl) > IRIS_DOWN_MAX_CM:
        return "iris too far up/down"
    if not HEAD_DISTANCE_CM[0] <= dist <= HEAD_DISTANCE_CM[1]:
        return "head too near/far"
    if max(abs(yaw), abs(pitch), abs(roll)) > HEAD_ANGLE_MAX_DEG:
        return "head turned away"
    if abs((rx - lx) - eye_gap[0]) > EYES_APART_X or abs((dr - dl) - eye_gap[1]) > EYES_APART_DOWN_CM:
        return "eyes pointing different ways"
    return None


def _possible(samples, eye_gap, counts: dict) -> list[CalibrationSample]:
    out = []
    for s in samples:
        why = frame_problem(s.features, eye_gap)
        if why is None:
            out.append(s)
        else:
            counts[why] = counts.get(why, 0) + 1
    return out


def _on_target(samples: list[CalibrationSample]) -> tuple[list[CalibrationSample], int]:
    """Layer 2: per dot, keep frames within MAD_K robust SDs of the dot's median
    on every gaze column. Returns (kept frames, dropped count)."""
    kept, dropped = [], 0
    for dot in {(s.target_x, s.target_y) for s in samples}:
        group = [s for s in samples if (s.target_x, s.target_y) == dot]
        g = np.array([gaze(s.features) for s in group])
        med = np.median(g, axis=0)
        mad = MAD_SCALE * np.median(np.abs(g - med), axis=0)
        z = np.abs(g - med) / np.maximum(mad, 1e-6)
        ok = (z <= MAD_K).all(axis=1)
        kept += [s for s, good in zip(group, ok) if good]
        dropped += int((~ok).sum())
    return kept, dropped


def check_recording(rec: CalibrationRecording) -> QualityReport:
    """Run all three layers on one calibration."""
    if not rec.dots or len(rec.dots[0].features) != N_FEATURES:
        return QualityReport(False, rec, ["no dot data, or features of another size"])
    reasons = []
    all_f = np.array([s.features for s in rec.dots])
    d = IRIS_DOWN.start
    eye_gap = (float(np.median(all_f[:, 0] - all_f[:, 1])),            # this user's own fixed
               float(np.median(all_f[:, d] - all_f[:, d + 1])))        # difference between the eyes

    # 1. impossible frames, in every step (sweep and head step too)
    counts: dict[str, int] = {}
    sweep, head, dots = (_possible(part, eye_gap, counts) for part in (rec.sweep, rec.head, rec.dots))
    if counts:
        reasons.append("impossible frames removed: " + ", ".join(f"{n} {why}" for why, n in counts.items()))

    # 2. glances away at each dot (the sweep's still dots too, if it has any)
    dots, n_glance = _on_target(dots)
    if n_glance:
        reasons.append(f"{n_glance} frames where the eyes left the dot removed")
    by_dot: dict[tuple, list] = {}
    for s in dots:
        by_dot.setdefault((s.target_x, s.target_y), []).append(s)
    total = len({(s.target_x, s.target_y) for s in rec.dots})
    thin = [dot for dot, g in by_dot.items() if len(g) < MIN_DOT_FRAMES]

    # 3. did the eyes follow the dots? one median per dot, then rank order + fit
    good = {dot: g for dot, g in by_dot.items() if dot not in thin}
    if len(good) < 5:
        return QualityReport(False, rec, reasons + [f"only {len(good)} of {total} dots had steady gaze"])
    dot_xy = np.array(list(good))
    med = np.array([np.median([gaze(s.features) for s in g], axis=0) for g in good.values()])
    eye_x, eye_down = med[:, 0], med[:, 1]
    # abs(): a mirrored camera flips the sign, but a real eye still ranks with the dots
    follow_x = abs(spearmanr(eye_x, dot_xy[:, 0])[0])
    follow_y = abs(spearmanr(eye_down, dot_xy[:, 1])[0])
    follow_y = 0.0 if np.isnan(follow_y) else follow_y
    if np.isnan(follow_x) or follow_x < MIN_FOLLOW_X or follow_y < MIN_FOLLOW_Y:
        return QualityReport(False, rec, reasons + [
            f"the eyes didn't follow the dots (left/right {follow_x:.2f}, up/down {follow_y:.2f}; "
            f"need {MIN_FOLLOW_X} / {MIN_FOLLOW_Y})"])

    # dots that don't fit the relation of the others: not looked at. A gently
    # curved fit (squares + cross term), because the eye turns in a sphere.
    # Iterative trimming: fit, drop the single worst dot if it's beyond the
    # limit, refit WITHOUT it, repeat. Dropping all misfits from one fit would
    # let one unwatched dot drag the fit and get its good neighbours dropped too.
    a = np.c_[eye_x, eye_down, eye_x ** 2, eye_down ** 2, eye_x * eye_down, np.ones(len(eye_x))]
    keep = np.ones(len(dot_xy), dtype=bool)
    while keep.sum() > a.shape[1] + 2:
        coef = np.linalg.lstsq(a[keep], dot_xy[keep], rcond=None)[0]
        miss = np.hypot(*(a @ coef - dot_xy).T)
        m = miss[keep]
        limit = max(np.median(m) + MAD_K * MAD_SCALE * np.median(np.abs(m - np.median(m))), DOT_MISFIT_MIN)
        worst = int(np.argmax(np.where(keep, miss, -1)))
        if miss[worst] <= limit:
            break
        keep[worst] = False
    misfit = [tuple(dot) for dot, k in zip(dot_xy, keep) if not k]
    bad = thin + misfit
    if bad:
        reasons.append(f"{len(bad)} of {total} dots dropped (not looked at steadily): "
                       + ", ".join(f"({x:.2f}, {y:.2f})" for x, y in sorted(bad)))
    if len(bad) > MAX_BAD_DOTS * total:
        return QualityReport(False, rec, reasons + [f"too many bad dots ({len(bad)} of {total})"])

    dots = [s for s in dots if (s.target_x, s.target_y) not in set(bad)]
    return QualityReport(True, CalibrationRecording(tuple(sweep), tuple(head), tuple(dots)), reasons)


if __name__ == "__main__":
    # self-check on a simulated person (core/line_model.features_for)
    from eyetracking.core.calibration import point_grid
    from eyetracking.core.line_model import ScreenGeometry, features_for
    rng = np.random.default_rng(1)
    screen = ScreenGeometry(27.7, 18.9)
    def frames(t, n=25, look=None):
        h = lambda: [rng.normal(0, 2), rng.normal(0, 2), rng.normal(0, 1), rng.normal(0, 1), rng.normal(-5, 1), rng.normal(-65, 1)]
        return [CalibrationSample(features_for(screen, look or t, h(), (2, -1), 0.9, rng), *t) for _ in range(n)]
    head = frames((0.5, 0.5), 80)
    good = [s for t in point_grid() for s in frames(t)]

    r = check_recording(CalibrationRecording(tuple(good[:50]), tuple(head), tuple(good)))
    print("1. good calibration:", r.ok, "|", r.summary())
    assert r.ok and len(r.kept.dots) > 0.9 * len(good)

    # glances + an impossible frame + one dot the user never looked at
    bad_frame = CalibrationSample(tuple(np.r_[1.3, 1.3, good[0].features[2:]]), *point_grid()[0])
    glances = [CalibrationSample(features_for(screen, (0.9, 0.9), [0, 0, 0, 0, -5, -65]), *point_grid()[3]) for _ in range(4)]
    away = [s for s in frames(point_grid()[7], look=(0.95, 0.05))]
    mixed = [s for s in good if (s.target_x, s.target_y) != point_grid()[7]] + away + glances + [bad_frame]
    r = check_recording(CalibrationRecording(tuple(good[:50]), tuple(head), tuple(mixed)))
    print("2. glances + impossible frame + unwatched dot:", r.ok, "|", r.summary())
    assert r.ok and point_grid()[7] not in {(s.target_x, s.target_y) for s in r.kept.dots}
    assert bad_frame not in r.kept.dots and not any(g in r.kept.dots for g in glances)

    # the user never looked at the dots (eyes on one spot the whole time)
    lazy = [s for t in point_grid() for s in frames(t, look=(0.5, 0.5))]
    r = check_recording(CalibrationRecording(tuple(good[:50]), tuple(head), tuple(lazy)))
    print("3. eyes never followed:", r.ok, "|", r.summary())
    assert not r.ok
    print("ok")
