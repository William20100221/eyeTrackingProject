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
MATCH_CJK = 8      # the same for a Chinese / Japanese / Korean text, in characters: a 4-character name
                   # ("斯瓦尔巴") recurs in a passage, so 4 would mark its later, skipped mentions read
PAIR_GAP = 2       # the pair rule bridges at most this many skipped words ("buy some [chips] together")

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
的 了 着 过 地 得 之 和 与 及 而 也 都 就 把 被 或 并 吗 呢 吧 啊 于 以 其 所 则
""".split())   # the last line: Chinese particles and conjunctions, one character each (a CJK "word" below)

# Chinese, Japanese and Korean are written without spaces between words: each character is one unit
# ("word") here, for tracking, counting and marking. Splitting on spaces found 46 "words" in a 700-character
# Chinese passage (only the spaces around its numbers), so read / unread went by whole sentences.
_CJK = "぀-ヿ㐀-䶿一-鿿가-힯豈-﫿"   # kana, Han, Hangul
_CJK_PUNCT = "，。、；：？！…—·「」『』（）《》〈〉【】“”‘’"
_TOKEN = re.compile(rf"[{_CJK}][{_CJK_PUNCT}]*|[^\s{_CJK}]+")   # a CJK character (+ its punctuation), or a run


def _plain(word: str) -> str:
    """A word as it is read: no punctuation, lower case ("" for punctuation alone)."""
    return word.strip(string.punctuation + "“”‘’…" + _CJK_PUNCT).lower()


def tokens(text: str) -> list[re.Match]:
    """The words of a text, in order, with where each sits: a run between spaces, or one CJK character."""
    return list(_TOKEN.finditer(text))


_CJK_START = re.compile(f"[{_CJK}]")


def is_cjk(word: str) -> bool:
    """A Chinese / Japanese / Korean character (a CJK token starts with one)."""
    return bool(_CJK_START.match(word))


def word_count(text: str) -> int:
    """How many words a text has: punctuation alone is no word; a CJK character is one."""
    return sum(1 for m in tokens(text) if _plain(m.group()))


def count_unit(text: str) -> str:
    """"characters" for a mostly Chinese / Japanese / Korean text, else "words": what word_count counts."""
    words = [m.group() for m in tokens(text) if _plain(m.group())]
    return "characters" if words and sum(map(is_cjk, words)) * 2 > len(words) else "words"


def mark_unread(full_text: str, read_text: str) -> str:
    n = MATCH_CJK if count_unit(full_text) == "characters" else MATCH_WORDS
    spans = [m.span() for m in tokens(full_text)]
    words = [_plain(full_text[a:b]) for a, b in spans]
    content = [i for i, w in enumerate(words) if w]          # punctuation alone takes its neighbour's state, below
    cw = [words[i] for i in content]
    read_words = [w for w in (_plain(m.group()) for m in tokens(read_text)) if w]
    read_runs = {tuple(read_words[i:i + n]) for i in range(len(read_words) - n + 1)}
    read_pairs = {tuple(read_words[i:i + 2]) for i in range(len(read_words) - 1)}

    # ponytail: a 4-word phrase that repeats inside a skipped part also counts as read
    read = [False] * len(cw)
    for i in range(len(cw) - n + 1):
        if tuple(cw[i:i + n]) in read_runs:
            read[i:i + n] = [True] * n
    # the pair rule: read_text leaves a skipped word out, so "buy some chips together" read without
    # "chips" gives the pair (some, together) in read_text. A word that makes such a pair with the
    # nearest read word (either side), at most PAIR_GAP words past it, was read; the words between them are
    # what the reader missed. Not for CJK characters: two characters side by side repeat too often in a
    # text to tell anything
    for order in (range(len(cw)), range(len(cw) - 1, -1, -1)):
        anchor = None
        for i in order:
            if read[i]:
                anchor = i
            elif (anchor is not None and abs(i - anchor) <= PAIR_GAP + 1 and not is_cjk(cw[i]) and not is_cjk(cw[anchor])
                  and (cw[min(anchor, i)], cw[max(anchor, i)]) in read_pairs):
                read[i] = True
                anchor = i
    # a skipped function word between two read words: read (the reader took it in, it changes no meaning)
    for i in range(1, len(cw) - 1):
        if not read[i] and read[i - 1] and read[i + 1] and cw[i] in FUNCTION_WORDS:
            read[i] = True
    is_read = [False] * len(words)
    for i, r in zip(content, read):
        is_read[i] = r
    last = None                                  # punctuation alone: read if the word before it was
    for i, w in enumerate(words):
        if w:
            last = is_read[i]
        else:
            is_read[i] = last if last is not None else bool(read and read[0])

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
    return sum(word_count(part) for part in re.findall(f"{OPEN}(.*?){CLOSE}", marked, re.S))


# ---------------------------------------------------------------- reading from gaze
# How people read, from the research (sources and reasoning: docs/screen_reading_dev_log.md).
FIXATION_S = 0.24      # one fixation in silent reading lasts 225-250 ms (Rayner 2009)
SPAN_LEFT = 4          # letters taken in around a fixation: 3-4 to the left...
SPAN_RIGHT = 15        # ...and 14-15 to the right (the perceptual span, Rayner 2009)
CJK_SPAN_LEFT = 1      # Chinese is denser: the span is 1 character to the left...
CJK_SPAN_RIGHT = 3     # ...and 3 to the right (Inhoff & Liu 1998)
MAX_GAP_S = 0.3        # a longer gap between two samples (blink, face lost, capture pause) is not reading time
LINE_SLACK = 1.0       # a sample this many line heights above/below a line still belongs to it: webcam error is about a line
LINE_STICKY = 1.6      # ...and this many to STAY on the line being read: the eye goes line by line, the estimate drifts
RETURN_SWEEP = 0.4     # a jump left by this much of the line's width is the return sweep to the next line's start
                       # (a regression to reread is a few words); without it the sticky line never let go: the next
                       # line is only ~1 line height down, inside LINE_STICKY, so a whole page was read as its first line


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
        """(word, left, right) for every word (see tokens(): a CJK character is one), left to right."""
        ws = self.text.split()
        if self.word_x and len(self.word_x) == len(ws):
            # OCR's box per space-separated word; a run of CJK characters in one box is shared out by width
            out = []
            for w, (a, b) in zip(ws, self.word_x):
                parts = [m.group() for m in tokens(w)]
                total, x = max(sum(map(_width, parts)), 1), a
                for p in parts:
                    out.append((p, x, x + (b - a) * _width(p) / total))
                    x = out[-1][2]
            return out
        # ponytail: every character the same width (a CJK one twice as wide). Off by a few letters in a
        # proportional font, far below the tracker's error; per-word boxes from UIA if it ever matters.
        letter = self.letter_px()
        at = [0]
        for c in self.text:
            at.append(at[-1] + _width(c))
        return [(m.group(), self.x + at[m.start()] * letter, self.x + at[m.end()] * letter) for m in tokens(self.text)]

    def letter_px(self) -> float:
        """The width of one Latin letter (half a CJK character), from the line's box."""
        return self.w / max(_width(self.text), 1)


def _width(s: str) -> int:
    """How wide a text is, in Latin letters: a CJK character or full-width sign counts 2."""
    return sum(2 if ("⺀" <= c <= "꓏" or "가" <= c <= "힯" or "豈" <= c <= "﫿"
                     or "︰" <= c <= "﹏" or "＀" <= c <= "｠" or "￠" <= c <= "￦") else 1
               for c in s)


class _LineState:
    def __init__(self, line: Line) -> None:
        self.line = line
        self.words = line.words()
        self.dwell = [0.0] * len(self.words)     # seconds each word has been inside the perceptual span
        cjk = count_unit(line.text) == "characters"
        # the perceptual span in letter widths: 4 left / 15 right, or 1 / 3 CJK characters (2 letters each)
        self.span = (CJK_SPAN_LEFT * 2, CJK_SPAN_RIGHT * 2) if cjk else (SPAN_LEFT, SPAN_RIGHT)

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
        self._last_x: float | None = None # where the last sample was across: a big jump left = the next line
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
        gap = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        dt = gap if gap <= MAX_GAP_S else 0.0     # a longer gap is no reading: not even MAX_GAP_S of it (one sample
                                                  # after a blink would otherwise count a whole fixation)
        self._last_t = t
        cur = self._state.get(self._current) if self._current else None
        sweep = cur is not None and self._last_x is not None and x < self._last_x - RETURN_SWEEP * cur.line.w
        self._last_x = x
        key = self._line_at(x, y, sweep)
        self._current = key
        if key is None:
            return None
        st = self._state[key]
        letter = st.line.letter_px()
        lo, hi = x - st.span[0] * letter, x + st.span[1] * letter
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

    def _line_at(self, x: float, y: float, sweep: bool = False) -> str | None:
        """The line a sample is on: the line being read while the gaze stays near it, the next line down
        after a return sweep (`sweep`), else the nearest line."""
        def rows_away(key: str) -> float:
            line = self._state[key].line
            return abs(line.y + line.h / 2 - y) / max(line.h, 1)

        def within_x(key: str) -> bool:
            line = self._state[key].line
            margin = SPAN_LEFT * line.letter_px()
            return line.x - margin <= x <= line.x + line.w + margin

        cur = self._current
        if sweep and cur in self._visible:
            below = [k for k in self._visible if self._state[k].line.y > self._state[cur].line.y]
            nxt = min(below, key=lambda k: self._state[k].line.y, default=None)
            if nxt is not None and within_x(nxt) and rows_away(nxt) <= LINE_STICKY:
                return nxt
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
        """(words read, words seen): punctuation alone is no word, a CJK character is one."""
        words = lambda k: [_plain(w) != "" for w, _, _ in self._state[k].words]
        read = sum(r and w for k in self._order for r, w in zip(self.read_flags(k), words(k)))
        return read, sum(sum(words(k)) for k in self._order)


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
    # a read word pairs only across a short gap, not with a word lines later
    assert mark_unread("snailfish, is a species of snailfish\nfound at sea.\nIt is known.",
                       "snailfish, is a species of snailfish") == \
        "snailfish, is a species of snailfish\n<unread>found at sea.\nIt is known.</unread>"
    # Chinese: a character is a word; a pasted copy with other punctuation and spacing still matches
    zh = "在挪威一个偏远的岛屿上，有一条狭窄的隧道通向一座山的内部。隧道的尽头就是种子库。"
    assert word_count(zh) == 37 and count_unit(zh) == "characters" and count_unit(full) == "words"   # 40, 3 signs
    read_zh = " ".join(m.group() for m in tokens(zh[:24]))                  # the tracker joins read characters with spaces
    marked = mark_unread(zh, read_zh)
    assert marked == zh[:24] + OPEN + zh[24:] + CLOSE, marked
    assert unread_word_count(marked) == word_count(zh[24:]) == 14
    assert mark_unread(zh.replace("，", ", "), read_zh).startswith(zh[:11] + ", ")   # pasted with other punctuation
    line = Line(zh, 0, 0, 40 * 20, 20)                                     # 20 px a character or full-width sign
    ws = line.words()
    assert len(ws) == 37 and ws[0][0] == "在" and abs(ws[1][1] - 20) < 1e-6 and ws[10][0] == "上，"
    t = ReadingTracker()
    t.show([line])
    assert t._state[zh].span == (2, 6)                                    # 1 character left, 3 right
    for k in range(0, 12):                                                # reads the first 12 characters, one at a time
        t.feed(ws[k][1] + 10, 10, k * 0.3)
        t.feed(ws[k][1] + 10, 10, k * 0.3 + 0.25)
    flags = t.read_flags(zh)
    assert all(flags[:11]) and not any(flags[16:]), flags                  # the end of the line was never looked at
    print("ok")
