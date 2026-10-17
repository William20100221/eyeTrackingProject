"""
Self-check for ReadingTracker (no screen, no camera):  python -m eyetracking.testing.test_screen_reading

Three lines of text on a pretend screen. The "eyes":
  1. read line 1 at a normal pace: a fixation every 8 letters, 240 ms each, with
     the gaze estimate bouncing up and down by 0.4 of a line (webcam noise)
  2. skim line 2: three 100 ms glances far apart
  3. never look at line 3
Then: line 1 is read (every word), lines 2 and 3 are not; the word under a
point is right; a scroll keeps what was read; a 2 s pause adds no reading
time; and mark_unread() gets exactly lines 2-3 as <unread>.
"""
from eyetracking.core.reading import FIXATION_S, Line, ReadingTracker, mark_unread, unread_word_count

LETTER, H = 10.0, 30.0                       # px per letter, line height
TEXTS = ["Pseudoliparis swirei, the Mariana snailfish, is a species of snailfish",
         "found at hadal depths in the Mariana Trench in the western Pacific Ocean.",
         "It is known from a depth range of 6,198 to 8,076 m below the surface."]


def lines(top: float) -> list[Line]:
    return [Line(t, 50, top + i * 40, len(t) * LETTER, H) for i, t in enumerate(TEXTS)]


def gaze(tracker, line: Line, letter: float, seconds: float, t: float, wobble: float) -> float:
    """Hold the gaze at `letter` on `line` for `seconds`, 30 samples a second."""
    for n in range(int(seconds * 30)):
        y = line.y + H / 2 + wobble * H * (1 if n % 2 else -1)
        tracker.feed(line.x + letter * LETTER, y, t)
        t += 1 / 30
    return t


tracker = ReadingTracker()
tracker.show(lines(100))
t = 0.0
l1, l2, l3 = lines(100)
for pos in range(3, len(TEXTS[0]), 8):                   # 1. reading: 8-letter saccades, 240 ms fixations
    t = gaze(tracker, l1, pos, FIXATION_S, t, wobble=0.4)
for pos in (5, 30, 55):                                   # 2. skimming: three short glances
    t = gaze(tracker, l2, pos, 0.10, t, wobble=0.0)

assert tracker.text_at(l1.x + 23 * LETTER, l1.y + 5) == ("the", TEXTS[0]), tracker.text_at(l1.x + 23 * LETTER, l1.y + 5)
assert tracker.read_flags(TEXTS[0]) == [True] * len(TEXTS[0].split()), tracker.read_flags(TEXTS[0])
assert not any(tracker.read_flags(TEXTS[1])), "skimming must not count as reading"
assert not any(tracker.read_flags(TEXTS[2]))
assert tracker.read_text() == TEXTS[0]

# a 2 s pause (blink, face lost) must not add 2 s of reading to line 2
t = gaze(tracker, l2, 40, 1 / 30, t + 2.0, wobble=0.0)
assert not any(tracker.read_flags(TEXTS[1])), tracker.read_flags(TEXTS[1])

# 3. scroll: the same lines 60 px higher, plus a new one at the bottom -> order and dwell kept
new = lines(40) + [Line("Various anatomical adaptions help it survive there.", 50, 160, 520, H)]
tracker.show(new)
assert tracker.full_text().split("\n") == TEXTS + ["Various anatomical adaptions help it survive there."]
assert tracker.read_text() == TEXTS[0]
assert tracker.text_at(new[0].x + 23 * LETTER, new[0].y + 5) == ("the", TEXTS[0])

# a skipped short word between read words counts as read
st = tracker._state[TEXTS[0]]
st.dwell[2] = 0.0                                         # "the"
assert tracker.read_flags(TEXTS[0])[2]

marked = mark_unread(tracker.full_text(), tracker.read_text())
assert marked.startswith(TEXTS[0] + "\n<unread>" + TEXTS[1]), marked
assert unread_word_count(marked) == sum(len(x.split()) for x in tracker.full_text().split("\n")[1:])
read, total = tracker.progress()
assert read == len(TEXTS[0].split()) and total == sum(len(x.split()) for x in tracker.full_text().split("\n"))
print("screen reading ok")
