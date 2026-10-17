"""
Blink scores: what they are, and reading a recording of them.

MediaPipe's face landmarker gives two "blendshapes" per frame, eyeBlinkLeft and
eyeBlinkRight: 0 = that eye wide open, 1 = shut. The app uses them to skip frames
where the eyes are closing or closed (core/analysis.is_blinking, the tracker):
calibration doesn't record those frames, and the live gaze holds its last point
instead of jumping when the iris disappears behind the lid.

Today's cut-off is a guess: 0.5 on both eyes. The blink test (ui/blink_window.py)
records the scores while you hold six known eye states; this module turns such a
recording into words and a suggested cut-off (GitHub issue #5). No Qt in here.
"""
from __future__ import annotations

import statistics
from typing import Sequence

STATES = [("open", "Keep your eyes open and look at the middle of the screen"),
          ("down", "Eyes open, look at the BOTTOM edge of the screen (like reading the last line)"),
          ("half", "Half close your eyes (squint) and look at the screen"),
          ("closed", "Close both eyes"),
          ("wink_left", "Close only your LEFT eye"),
          ("wink_right", "Close only your RIGHT eye")]
NAMES = {"open": "eyes open", "down": "looking down", "half": "half closed", "closed": "closed",
         "wink_left": "left eye shut", "wink_right": "right eye shut"}
SECONDS = 6              # recorded per state
COUNTDOWN = 3            # "get ready" before each state, not recorded
OPEN_STATES = ("open", "down", "half")   # these must stay below the cut-off
Row = tuple[str, int, float, float]      # state, time ms, left score, right score


def spread(values: Sequence[float]) -> tuple[float, float, float]:
    """(5th percentile, median, 95th percentile): the ends ignore the odd glitch frame.
    'inclusive' quantiles never leave the range of the data."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v, v
    q = statistics.quantiles(values, n=20, method="inclusive")
    return q[0], statistics.median(values), q[-1]


def summary(rows: Sequence[Row]) -> dict[str, dict]:
    """Per state: how many frames and each eye's spread. States with no frames are left out."""
    out = {}
    for state, _ in STATES:
        got = [r for r in rows if r[0] == state]
        if got:
            out[state] = {"frames": len(got), "left": spread([r[2] for r in got]), "right": spread([r[3] for r in got])}
    return out


def advice(rows: Sequence[Row]) -> str:
    """The cut-off in words: the gap between the highest 'open' score and the lowest 'closed' one."""
    s = summary(rows)
    opens = [s[k][eye][2] for k in OPEN_STATES if k in s for eye in ("left", "right")]
    if "closed" not in s or not opens:
        return "Not enough states recorded to suggest a cut-off (it needs eyes open and eyes closed at least)."
    open_max = max(opens)
    closed_min = min(s["closed"]["left"][0], s["closed"]["right"][0])
    if closed_min > open_max:
        mid = (open_max + closed_min) / 2
        text = (f"Closed eyes score at least {closed_min:.2f}; open, looking down and half closed never pass "
                f"{open_max:.2f}. A cut-off of {mid:.2f} sits in the middle (the app uses 0.50 today).")
    else:
        text = (f"No single cut-off works: open, looking down and half closed reach {open_max:.2f}, but closed "
                f"eyes start at {closed_min:.2f}. The tracker's per-user level + rise is needed instead.")
    winks = [k for k in ("wink_left", "wink_right") if k in s]
    if winks:
        shut = max(s[k]["left" if k == "wink_left" else "right"][1] for k in winks)
        other = max(s[k]["right" if k == "wink_left" else "left"][1] for k in winks)
        text += (f" A wink closes one eye to about {shut:.2f} while the other stays near {other:.2f}: with the "
                 "both-eyes rule a wink is not a blink, so its frames are kept.")
    return text


def lines(rows: Sequence[Row]) -> list[str]:
    """The recording in words: one line per state, then the advice."""
    out = []
    for state, info in summary(rows).items():
        l, r = info["left"], info["right"]
        out.append(f"{NAMES[state].capitalize()}: {info['frames']} frames; left eye {l[1]:.2f} (from {l[0]:.2f} to "
                   f"{l[2]:.2f}), right eye {r[1]:.2f} (from {r[0]:.2f} to {r[2]:.2f})")
    return out + ["", advice(rows)]


if __name__ == "__main__":
    demo = ([("open", t, 0.05, 0.06) for t in range(0, 300, 33)] + [("down", t, 0.2, 0.22) for t in range(0, 300, 33)]
            + [("closed", t, 0.85, 0.9) for t in range(0, 300, 33)] + [("wink_left", t, 0.9, 0.1) for t in range(0, 300, 33)])
    text = advice(demo)
    assert "at least 0.85" in text and "never pass 0.22" in text and "0.54" in text and "wink" in text, text
    assert "No single cut-off" in advice([("open", 0, 0.6, 0.6), ("closed", 1, 0.55, 0.7)])
    print("\n".join(lines(demo)))
