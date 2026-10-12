"""
Find the parts of a text the reader did NOT read.

Two strings come in:
    full_text  - the whole text on screen
    read_text  - the parts the reader looked at, copied from full_text,
                 in any order, joined by spaces or newlines

mark_unread() returns full_text with every unread run wrapped in
<unread>...</unread>, so the AI sees the whole text AND what was skipped.

Self-check:  python -m eyetracking.core.reading
"""
import re

# A word counts as read when it sits inside a run of this many words that also
# appears in read_text. Shorter runs ("of the", "it is") match by chance, so a
# read piece shorter than this can't be matched.
MATCH_WORDS = 4

OPEN, CLOSE = "<unread>", "</unread>"


def mark_unread(full_text: str, read_text: str) -> str:
    n = MATCH_WORDS
    spans = [m.span() for m in re.finditer(r"\S+", full_text)]
    words = [full_text[a:b] for a, b in spans]
    read_words = read_text.split()
    read_runs = {tuple(read_words[i:i + n]) for i in range(len(read_words) - n + 1)}

    # ponytail: a 4-word phrase that repeats inside a skipped part also counts as read
    is_read = [False] * len(words)
    for i in range(len(words) - n + 1):
        if tuple(words[i:i + n]) in read_runs:
            is_read[i:i + n] = [True] * n

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
    print("ok")