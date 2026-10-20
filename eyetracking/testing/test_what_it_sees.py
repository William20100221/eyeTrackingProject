"""
Self-check for the "what the app sees" figure (no camera, no screen, never data/):
    python -m eyetracking.testing.test_what_it_sees

Three made-up lines on a made-up window, as UI Automation would give them; a gaze that reads the first
line slowly, glances at the second, never looks at the third, fed through ReadingTracker as Go Reading
does. draw() goes into a temp folder: the PNG and the CSV must exist, the first line's samples read, the
third line never visited, and the CSV free of any screen text.
"""
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from eyetracking.core.reading import Line, ReadingTracker
from eyetracking.testing.what_it_sees import draw, source_of, word_under

texts = ["Eyes move in quick jumps called saccades across the line",
         "Between jumps they stop for about a quarter of a second",
         "Readers sometimes jump back to reread a word they missed"]
lines = [Line(t, 100, 200 + 40 * i, 600, 24) for i, t in enumerate(texts)]
assert source_of(lines) == "UI Automation" and source_of([Line("a b", 0, 0, 10, 5, ((0, 4), (6, 10)))]) == "OCR"
tracker = ReadingTracker()
tracker.show(lines)
samples, t = [], 0.0
for x in range(110, 700, 8):                         # line 1, slowly: ~0.27 s per 8 px, words stay in the span
    for _ in range(8):
        t += 1 / 30
        tracker.feed(x, 212, t)
        samples.append((t, x, 212, *word_under(tracker, lines, x, 212)))
for x in range(110, 700, 120):                       # line 2: a quick glance across
    t += 1 / 30
    tracker.feed(x, 252, t)
    samples.append((t, x, 252, *word_under(tracker, lines, x, 252)))

out = Path(tempfile.mkdtemp())
got = draw(None, (80, 150, 820, 360), lines, samples, tracker, out / "what_the_app_sees.png",
           out / "what_the_app_sees.csv", "a made-up window")
assert (out / "what_the_app_sees.png").read_bytes()[:4] == b"\x89PNG"
rows = list(csv.reader((out / "what_the_app_sees.csv").open(encoding="utf-8")))
assert rows[0] == ["time_s", "x", "y", "line_index", "read"] and len(rows) - 1 == len(samples)
first = [r for r in rows[1:] if r[3] == "0"]
assert first and sum(r[4] == "1" for r in first) > 0.8 * len(first), "the line read slowly must count as read"
assert not any(r[3] == "2" for r in rows[1:]), "nobody looked at the third line"
assert got["line_shares"][0] >= 0.5 > got["line_shares"][2] == 0.0, got["line_shares"]
words = {w for t_ in texts for w in t_.split()}
assert not any(cell in words for r in rows for cell in r), "no screen text in the CSV"
print(f"what it sees ok: {got['read']} of {got['seen']} words read, lines {[f'{s:.0%}' for s in got['line_shares']]}")
