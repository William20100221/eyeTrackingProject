"""
What the app sees while you read: a debug run of Go Reading, drawn on one screenshot.

    python -m eyetracking.testing.what_it_sees [--seconds 20] [--out docs/img/report/what_the_app_sees.png]
                                               [--camera N | --mouse]

For --seconds it does what Go Reading does (app.py): the camera, the gaze model trained on every saved
calibration (newest weighted most), the 1-euro filter, the point held through blinks; once a second the
text of the window under the gaze (services/screen_text.lines_on_screen: UI Automation, or OCR when the
window gives none) handed to core/reading.ReadingTracker, which every gaze sample is fed to. Then it grabs
ONE picture of the window under the gaze and draws on it:

    every line box the app got, labelled UI Automation or OCR (OCR lines carry word boxes, UIA's don't)
    the gaze samples, a faint trail coloured by time
    the perceptual span (SPAN_LEFT letters left, SPAN_RIGHT right, from core/reading.py) at three fixations
    each line green when most of its words were judged read, red when not, with its read fraction;
    the overall read fraction (ReadingTracker.progress) in the title

and writes the PNG (200 dpi) and a CSV of the samples: time, x, y, the line they were on (its index in
the drawn capture, -1 none or scrolled away) and whether that word ended up read. The screen text is
not written anywhere, and the picture is kept only in memory until it is drawn: the PNG is the only file
(nothing leaves the computer). --mouse: the mouse is the gaze (no camera or calibration needed).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                         # files only: no window of its own over the text
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402

from eyetracking.core.reading import FIXATION_S, SPAN_LEFT, SPAN_RIGHT, Line, ReadingTracker   # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "docs" / "img" / "report" / "what_the_app_sees.png"
DPI = 200
CAPTURE_S = 1.0                               # how often the window's text is read again, as the app's CAPTURE_MS
READ_SHARE = 0.5                              # a line is drawn green once this share of its words was judged read
Sample = tuple[float, float, float, str | None, int | None]   # t (s from the start), x, y px, line key, word index


def source_of(lines: list[Line]) -> str:
    """Where a capture's lines came from: OCR gives a box per word, UI Automation only per line."""
    return "OCR" if any(ln.word_x for ln in lines) else "UI Automation"


def word_under(tracker: ReadingTracker, lines: list[Line], x: float, y: float) -> tuple[str | None, int | None]:
    """(the line's key, the word's index on it) under a gaze point, as the tracker places it now."""
    got = tracker.text_at(x, y)
    if got is None:
        return None, None
    _, key = got
    line = next((ln for ln in lines if ln.text.strip() == key), None)
    if line is None:
        return key, None
    words = line.words()
    i = min(range(len(words)), key=lambda k: 0 if words[k][1] <= x <= words[k][2] else
            min(abs(x - words[k][1]), abs(x - words[k][2])))
    return key, i


def _flags(tracker: ReadingTracker, key: str) -> list[bool]:
    """A line's read flags per word ([] for a line the tracker never got)."""
    try:
        return tracker.read_flags(key)
    except KeyError:
        return []


def draw(image, rect: tuple[int, int, int, int], lines: list[Line], samples: list[Sample], tracker: ReadingTracker,
         png: Path, csv_path: Path, title: str = "") -> dict:
    """The figure and the CSV. image: the window's picture (anything imshow takes, or None: a white page);
    rect: its (left, top, right, bottom) on the screen, in the same pixels as the line boxes and the gaze."""
    left, top, right, bottom = rect
    keys = [ln.text.strip() for ln in lines]
    flags = {k: _flags(tracker, k) for k in keys}
    fig, ax = plt.subplots(figsize=(12, 12 * (bottom - top) / max(right - left, 1)))
    if image is None:
        image = np.full((max(bottom - top, 1), max(right - left, 1), 3), 255, np.uint8)
    ax.imshow(image, extent=(left, right, bottom, top))
    per_line = []
    for k, ln in zip(keys, lines):                                   # the line boxes, read / unread
        f = flags.get(k, [])
        share = sum(f) / len(f) if f else 0.0
        colour = "#2E7D32" if share >= READ_SHARE else "#C62828"
        ax.add_patch(plt.Rectangle((ln.x, ln.y), ln.w, ln.h, fill=False, color=colour, lw=1.2))
        ax.text(ln.x + ln.w + 4, ln.y + ln.h / 2, f"{share:.0%}", color=colour, fontsize=6, va="center")
        per_line.append(share)
    inside = [s for s in samples if left <= s[1] <= right and top <= s[2] <= bottom]
    if inside:                                                       # the gaze trail, coloured by time
        t = np.array([s[0] for s in inside])
        trail = ax.scatter([s[1] for s in inside], [s[2] for s in inside], c=t, cmap="viridis", s=10, alpha=0.35,
                           linewidths=0, zorder=3)
        fig.colorbar(trail, ax=ax, label="seconds", shrink=0.6, pad=0.01)
    on_line = [s for s in inside if s[3] in keys]
    examples = [on_line[int(len(on_line) * q)] for q in (0.2, 0.5, 0.8)] if len(on_line) >= 3 else on_line
    for s in examples:                                               # the perceptual span at three fixations
        ln = lines[keys.index(s[3])]
        letter = ln.letter_px()
        ax.add_patch(plt.Rectangle((s[1] - SPAN_LEFT * letter, ln.y - 2), (SPAN_LEFT + SPAN_RIGHT) * letter, ln.h + 4,
                                   color="#FFB300", alpha=0.35, lw=0, zorder=2))
        ax.plot(s[1], s[2], "o", color="#FF6F00", ms=5, zorder=4)
    if examples:
        ax.plot([], [], "s", color="#FFB300", alpha=0.6,
                label=f"perceptual span: {SPAN_LEFT} letters left, {SPAN_RIGHT} right of the fixation")
    ax.plot([], [], "s", mfc="none", color="#2E7D32", label=f"line read (at least {READ_SHARE:.0%} of its words)")
    ax.plot([], [], "s", mfc="none", color="#C62828", label="line not read")
    ax.legend(fontsize=7, loc="lower right")
    read, seen = tracker.progress()
    source = source_of(lines) if lines else "no text"
    ax.set_title(f"{title}{': ' if title else ''}{len(lines)} lines from {source}; {read} of {seen} words judged read "
                 f"({read / seen:.0%}); a word is read after {FIXATION_S:.2f} s in the span" if seen else
                 f"{title}: no text found", fontsize=9)
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:     # numbers only: no screen text in the file
        w = csv.writer(f)
        w.writerow(["time_s", "x", "y", "line_index", "read"])
        for t, x, y, key, i in samples:
            idx = keys.index(key) if key in keys else -1
            f = _flags(tracker, key) if key else []
            word_read = i is not None and i < len(f) and f[i]
            w.writerow([f"{t:.3f}", f"{x:.1f}", f"{y:.1f}", idx, int(bool(word_read))])
    return {"lines": len(lines), "source": source, "read": read, "seen": seen, "samples": len(samples),
            "line_shares": per_line}


def _cursor() -> tuple[float, float]:
    from ctypes import wintypes
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return float(pt.x), float(pt.y)


def run(seconds: float, camera: int, mouse: bool) -> tuple[list[Sample], ReadingTracker, list[Line], int]:
    """The live part: Go Reading for `seconds`. (samples, the tracker, the last capture's lines, its window)."""
    from eyetracking.services.screen_text import lines_on_screen, make_dpi_aware, window_at
    make_dpi_aware()                                     # physical pixels: the frame of the text boxes
    W, H = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    tracker, model, smoother = None, None, None
    if not mouse:
        from eyetracking.core.points_model import train
        from eyetracking.core.smoothing import GazeSmoother
        from eyetracking.services.storage import load_all_calibrations
        from eyetracking.services.tracker import EyeTracker
        model, _ = train(load_all_calibrations())           # as App._load_model
        if model is None:
            raise SystemExit("No calibration yet: calibrate in the app first (or use --mouse).")
        smoother = GazeSmoother()
        tracker = EyeTracker(camera)
    reader, samples, lines, hwnd = ReadingTracker(), [], [], 0
    pool, job, last_capture, point = ThreadPoolExecutor(1), None, -1e9, None
    start = time.monotonic()
    print(f"Reading for {seconds:.0f} s: look at a window with text and read it.")
    try:
        while (now := time.monotonic()) - start < seconds:
            if mouse:
                x, y = _cursor()
                time.sleep(1 / 30)                       # a camera's pace
            else:
                features = tracker.read_features()       # waits for the camera's next frame
                p = model.predict(features) if features is not None else None
                if p is not None:
                    point = smoother.feed(p, time.perf_counter())
                if point is None:                        # nothing yet (no face): nothing to feed
                    continue
                # a blink: the last point is held, as the app does (issue #9)
                x = min(max(point[0], 0.0), 1.0) * (W - 1)
                y = min(max(point[1], 0.0), 1.0) * (H - 1)
            if job is not None and job.done():           # a capture came back: the tracker gets the new lines
                try:
                    lines = job.result()
                    reader.show(lines)
                except Exception as e:                   # the window went away mid-capture
                    print(f"  a capture failed ({e})")
                job = None
            if job is None and now - last_capture >= CAPTURE_S:
                hwnd = window_at(x, y) or hwnd           # the desktop: keep the window before
                if hwnd:
                    job, last_capture = pool.submit(lines_on_screen, hwnd), now
            reader.feed(x, y, now)
            samples.append((now - start, x, y, *word_under(reader, lines, x, y)))
        if job is not None:
            lines = job.result()
            reader.show(lines)
    finally:
        pool.shutdown(wait=True)
        if tracker is not None:
            tracker.close()
    return samples, reader, lines, hwnd


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Go Reading for a few seconds, drawn on one screenshot")
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mouse", action="store_true", help="the mouse plays the gaze (no camera, no calibration)")
    args = parser.parse_args(argv)
    samples, reader, lines, hwnd = run(args.seconds, args.camera, args.mouse)
    if not hwnd:
        raise SystemExit("The gaze never landed on a window.")
    from PIL import ImageGrab
    from eyetracking.services.screen_text import window_rect, window_title
    rect = window_rect(hwnd)
    image = ImageGrab.grab(rect, all_screens=True)       # in memory only: drawn into the PNG, never saved itself
    title = window_title(hwnd)
    got = draw(np.asarray(image), rect, lines, samples, reader, args.out, args.out.with_suffix(".csv"), title)
    print(f"window: {title!r}, {got['lines']} lines from {got['source']}; {got['samples']} gaze samples; "
          f"{got['read']} of {got['seen']} words judged read")
    print(f"wrote {args.out}\nwrote {args.out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
