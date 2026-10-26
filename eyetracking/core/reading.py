"""
Find the parts of a text the reader did NOT read.

Two strings come in:
    full_text  - the whole text on screen
    read_text  - the parts the reader looked at, copied from full_text,
                 in any order, joined by spaces or newlines

mark_unread() returns full_text with every unread run wrapped in
<unread>...</unread>, so the AI sees the whole text AND what was skipped.

Where read_text comes from: ReadingTracker (below) turns timed gaze points on
lines of screen text into the words that were read. No Qt, no camera, no
Windows calls in here: hand it numbers and it answers.

    tracker = ReadingTracker()
    tracker.show(lines)                  # Line objects: text + box, from services/screen_text.py
    word = tracker.feed(x, y, t)         # every gaze sample: screen px, seconds -> the word under it
    tracker.text_at(x, y)                # (word, line) under a point, nothing recorded
    tracker.read_text(), tracker.full_text()   # -> mark_unread(full, read)

Self-checks:  python -m eyetracking.core.reading
              python -m eyetracking.testing.test_screen_reading
"""
import re
import string
from dataclasses import dataclass

# A word counts as read when it sits inside a run of this many words that also
# appears in read_text. Shorter runs ("of the", "it is") match by chance, so a
# read piece shorter than this can't be matched on its own (see the pair rule below).
MATCH_WORDS = 4

OPEN, CLOSE = "<unread>", "</unread>"

# Words that carry no information of their own (articles, conjunctions, prepositions, pronouns,
# auxiliaries). A skipped one between two read words counts as read: the reader took it in through
# the perceptual span, and it changes nothing for understanding ("William and Anna": "and" only says
# the two go together). A skipped CONTENT word ("chips") stays unread: it is a key word the reader
# may have missed, and the AI is told so (reading_ai.SYSTEM). English only; in another language the
# rule simply never fires.
FUNCTION_WORDS = frozenset("""
a an the and or but nor so yet for of in on at to by with from as into onto than then that this these
those it its is are was were be been being am do does did has have had not no he she they we you i me
him her them us my your his our their who whom whose which what when where while if because although
though until unless also too very just only such some any each every all both either neither more
most much many few less own same up down out off over under again further here there about above
below between through during before after will would shall should can could may might must
""".split())


def _plain(word: str) -> str:
    """A word as it is read: no punctuation, lower case."""
    return word.strip(string.punctuation + "“”‘’…").lower()


def mark_unread(full_text: str, read_text: str) -> str:
    n = MATCH_WORDS
    spans = [m.span() for m in re.finditer(r"\S+", full_text)]
    words = [full_text[a:b] for a, b in spans]
    read_words = read_text.split()
    read_runs = {tuple(read_words[i:i + n]) for i in range(len(read_words) - n + 1)}
    read_pairs = {tuple(read_words[i:i + 2]) for i in range(len(read_words) - 1)}

    # ponytail: a 4-word phrase that repeats inside a skipped part also counts as read
    is_read = [False] * len(words)
    for i in range(len(words) - n + 1):
        if tuple(words[i:i + n]) in read_runs:
            is_read[i:i + n] = [True] * n
    # the pair rule: read_text leaves a skipped word out, so "buy some chips together" read without
    # "chips" gives the pair (some, together) in read_text. A word that makes such a pair with the
    # nearest read word (either side) was read; the words between them are what the reader missed
    for order in (range(len(words)), range(len(words) - 1, -1, -1)):
        anchor = None
        for i in order:
            if is_read[i]:
                anchor = i
            elif anchor is not None and (words[min(anchor, i)], words[max(anchor, i)]) in read_pairs:
                is_read[i] = True
                anchor = i
    # a skipped function word between two read words: read (the reader took it in, it changes no meaning)
    for i in range(1, len(words) - 1):
        if not is_read[i] and is_read[i - 1] and is_read[i + 1] and _plain(words[i]) in FUNCTION_WORDS:
            is_read[i] = True

    out, pos = [], 0
    for i, (a, b) in enumerate(spans):
        if is_read[i]:
            continue
        if i == 0 or is_read[i - 1]:                 # first word of an unread run
            out.append(full_text[pos:a] + OPEN)
            pos = a
        if i == len(words) - 1 or is_read[i + 1]:    # last word of an unread run
            out.append(full_text[pos:b] + CLOSE)
            pos = b
    out.append(full_text[pos:])
    return "".join(out)


def unread_word_count(marked: str) -> int:
    return sum(len(part.split()) for part in re.findall(f"{OPEN}(.*?){CLOSE}", marked, re.S))


# ---------------------------------------------------------------- reading from gaze
# How people read, from the research (sources and reasoning: docs/screen_reading_dev_log.md).
FIXATION_S = 0.24      # one fixation in silent reading lasts 225-250 ms (Rayner 2009)
SPAN_LEFT = 4          # letters taken in around a fixation: 3-4 to the left...
SPAN_RIGHT = 15        # ...and 14-15 to the right (the perceptual span, Rayner 2009)
MAX_GAP_S = 0.3        # a longer gap between two samples (blink, face lost, capture pause) is not reading time
LINE_SLACK = 1.0       # a sample this many line heights above/below a line still belongs to it: webcam error is about a line
LINE_STICKY = 1.6      # ...and this many to STAY on the line being read: the eye goes line by line, the estimate drifts


@dataclass(frozen=True)
class Line:
    """One visual line of text on screen and its box, in screen pixels.
    word_x: (left, right) per word when the source knows them (OCR does);
    otherwise they are estimated from the character count (UIA gives line boxes only)."""
    text: str
    x: float
    y: float
    w: float
    h: float
    word_x: tuple[tuple[float, float], ...] | None = None

    def words(self) -> list[tuple[str, float, float]]:
        """(word, left, right) for every word, left to right."""
        ws = self.text.split()
        if self.word_x and len(self.word_x) == len(ws):
            return [(w, a, b) for w, (a, b) in zip(ws, self.word_x)]
        # ponytail: every character the same width. Off by a few letters in a proportional
        # font, far below the tracker's error; per-word boxes from UIA if it ever matters.
        letter = self.letter_px()
        out, pos = [], 0
        for w in ws:
            start = self.text.index(w, pos)
            pos = start + len(w)
            out.append((w, self.x + start * letter, self.x + pos * letter))
        return out

    def letter_px(self) -> float:
        return self.w / max(len(self.text), 1)


class _LineState:
    def __init__(self, line: Line) -> None:
        self.line = line
        self.words = line.words()
        self.dwell = [0.0] * len(self.words)     # seconds each word has been inside the perceptual span

    def move(self, line: Line) -> None:         # same text, new place (scrolled): keep the dwell
        self.line, self.words = line, line.words()


class ReadingTracker:
    """Timed gaze points on lines of text -> which words have been read.

    A word is judged read once it has spent one fixation's time (FIXATION_S)
    inside the reader's perceptual span: the few letters left and the ~15
    letters right of where the gaze is. Fast skimming never leaves a word that
    long in the span, so skimmed text stays unread. A single unread FUNCTION word
    between two read ones counts as read: readers skip about a third of the words
    (short ones) and still take them in, through the span. A skipped content word
    stays unread: a key word the reader may have missed.

    Lines that scroll away keep their dwell: show() the new lines and carry on.
    """

    def __init__(self) -> None:
        self._state: dict[str, _LineState] = {}
        self._order: list[str] = []       # every line seen so far, top to bottom
        self._visible: list[str] = []
        self._current: str | None = None  # the line the last sample was on
        self._last_t: float | None = None

    # ---- what is on screen ---------------------------------------------
    def show(self, lines: list[Line]) -> None:
        """The lines on screen now. Call again whenever the screen changed (scroll, new page)."""
        keys = []
        for line in lines:
            key = line.text.strip()          # ponytail: two identical lines share one entry
            if not key:
                continue
            if key in self._state:
                self._state[key].move(line)
            else:
                self._state[key] = _LineState(line)
            keys.append(key)
        for i, key in enumerate(keys):       # a new line goes where it sits among the known ones
            if key in self._order:
                continue
            if i > 0:
                self._order.insert(self._order.index(keys[i - 1]) + 1, key)
            else:
                after = next((k for k in keys[1:] if k in self._order), None)
                self._order.insert(self._order.index(after) if after else len(self._order), key)
        self._visible = keys

    # ---- gaze in ---------------------------------------------------------
    def feed(self, x: float, y: float, t: float) -> str | None:
        """One gaze sample (screen pixels, time in seconds). Returns the word under it, or None."""
        dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), MAX_GAP_S)
        self._last_t = t
        key = self._line_at(x, y)
        self._current = key
        if key is None:
            return None
        st = self._state[key]
        letter = st.line.letter_px()
        lo, hi = x - SPAN_LEFT * letter, x + SPAN_RIGHT * letter
        for i, (_, x0, x1) in enumerate(st.words):
            if x1 >= lo and x0 <= hi:
                st.dwell[i] += dt
        return self._word_at(st, x)

    def text_at(self, x: float, y: float) -> tuple[str, str] | None:
        """(word, whole line) under a screen point, without recording anything."""
        key = self._line_at(x, y)
        if key is None:
            return None
        return self._word_at(self._state[key], x), self._state[key].line.text.strip()

    def _line_at(self, x: float, y: float) -> str | None:
        def rows_away(key: str) -> float:
            line = self._state[key].line
            return abs(line.y + line.h / 2 - y) / max(line.h, 1)

        def within_x(key: str) -> bool:
            line = self._state[key].line
            margin = SPAN_LEFT * line.letter_px()
            return line.x - margin <= x <= line.x + line.w + margin

        cur = self._current
        if cur in self._visible and within_x(cur) and rows_away(cur) <= LINE_STICKY:
            return cur
        best = min((k for k in self._visible if within_x(k)), key=rows_away, default=None)
        return best if best is not None and rows_away(best) <= LINE_SLACK else None

    @staticmethod
    def _word_at(st: _LineState, x: float) -> str:
        return min(st.words, key=lambda w: 0 if w[1] <= x <= w[2] else min(abs(x - w[1]), abs(x - w[2])))[0]

    # ---- what was read ---------------------------------------------------
    def read_flags(self, key: str) -> list[bool]:
        st = self._state[key]
        read = [d >= FIXATION_S for d in st.dwell]
        return [r or (0 < i < len(read) - 1 and read[i - 1] and read[i + 1]       # a skipped function word
                      and _plain(st.words[i][0]) in FUNCTION_WORDS)
                for i, r in enumerate(read)]

    def read_text(self) -> str:
        """The words judged read, in reading order, a line per screen line (partial lines kept)."""
        parts = []
        for key in self._order:
            words = [w for (w, _, _), r in zip(self._state[key].words, self.read_flags(key)) if r]
            if words:
                parts.append(" ".join(words))
        return "\n".join(parts)

    def full_text(self) -> str:
        """Every line seen so far, top to bottom (lines that scrolled away included)."""
        return "\n".join(self._state[key].line.text.strip() for key in self._order)

    def progress(self) -> tuple[int, int]:
        """(words read, words seen)"""
        read = sum(sum(self.read_flags(k)) for k in self._order)
        return read, sum(len(self._state[k].words) for k in self._order)


if __name__ == "__main__":
    full = ("Eyes move in quick jumps called saccades.\n\n"
            "Between jumps they stop for about a quarter of a second.\n\n"
            "Readers sometimes jump back to reread a word.")
    # read parts given out of order, joined by a newline
    read = "Readers sometimes jump back to reread a word.\nEyes move in quick jumps called saccades."
    marked = mark_unread(full, read)
    assert marked == ("Eyes move in quick jumps called saccades.\n\n"
                      "<unread>Between jumps they stop for about a quarter of a second.</unread>\n\n"
                      "Readers sometimes jump back to reread a word."), marked
    assert unread_word_count(marked) == 11
    assert mark_unread(full, "") == f"<unread>{full}</unread>"
    assert mark_unread(full, full) == full
    # a skipped function word between read words is read; a skipped content word is a key word missed
    full = "William and Anna went to buy some chips together."
    assert mark_unread(full, "William Anna went to buy some together.") == \
        "William and Anna went to buy some <unread>chips</unread> together."
    assert mark_unread(full, "William Anna went to buy") == "William and Anna went to buy <unread>some chips together.</unread>"
    assert mark_unread(full, "Anna went to buy some") == "<unread>William and</unread> Anna went to buy some <unread>chips together.</unread>"
    # the tracker: a content word the gaze never rested on stays unread even between read words
    t = ReadingTracker()
    t.show([Line(full, 0, 0, 490, 20)])                     # 10 px a letter
    for i, w in enumerate(full.split()):
        if w not in ("and", "chips"):
            x = sum(len(v) + 1 for v in full.split()[:i]) * 10 + 5
            t.feed(x, 10, i * 10.0)
            t.feed(x, 10, i * 10.0 + FIXATION_S)             # one fixation on the word; the span reaches 15 letters right
    flags = t.read_flags(full)
    assert flags[full.split().index("and")] and t.read_text().startswith("William and Anna"), t.read_text()
    print("ok")
