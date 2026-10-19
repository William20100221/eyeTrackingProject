"""
Self-check for the screen capture's caching (no real window, no UI Automation):
    python -m eyetracking.testing.test_screen_capture

The capture keeps each window's document and walks the window's whole UI tree (_find_document) only
when it must: once per page, again when the window's title changes (another tab or page), and, when
the kept document shows almost nothing, at most every NO_DOCUMENT_RETRY_S. Captures run on one daemon
thread whoever asks. An OCR window is OCR'd again only when its picture changed.
"""
import threading

import eyetracking.services.screen_text as st
from eyetracking.core.reading import Line

finds, title, clock = [], ["Article A - Edge"], [100.0]
page = {"doc A": [Line("word " * 30, 0, 0, 100, 20)], "doc B": [Line("other " * 30, 0, 0, 100, 20)], "empty": []}
current = ["doc A"]


def fake_find(hwnd):
    finds.append(current[0])
    return current[0]


st._find_document = fake_find
st._visible = lambda doc: page[doc]
st.window_title = lambda hwnd: title[0]
st.time.monotonic = lambda: clock[0]
words = lambda lines: sum(len(ln.text.split()) for ln in lines)

assert words(st.lines_on_screen(1)) == 30 and finds == ["doc A"]            # the first capture finds the document
for _ in range(5):
    st.lines_on_screen(1)
assert finds == ["doc A"]                                                     # the same page: never walked again
current[0], title[0] = "doc B", "Article B - Edge"                           # another tab: another title
assert st.lines_on_screen(1)[0].text.startswith("other") and finds == ["doc A", "doc B"]
current[0] = "empty"                                                          # the page went blank (a reload)
page["doc B"] = []
st.lines_on_screen(1)
assert finds == ["doc A", "doc B"]                                            # within 2 s: not walked again...
clock[0] += st.NO_DOCUMENT_RETRY_S + 0.1
st.lines_on_screen(1)
assert finds == ["doc A", "doc B", "empty"]                                   # ...after it, once
st.lines_on_screen(1)
assert len(finds) == 3                                                        # and not again straight away
assert st.lines_on_screen(2) == [] and finds[-1] == "empty"                   # another window: its own document

names = []                                                                    # every capture on the one daemon thread
st._find_document = lambda hwnd: names.append(threading.current_thread()) or "doc A"
st.lines_on_screen(3)
st.lines_on_screen(4)
assert len(set(names)) == 1 and names[0].daemon and names[0] is not threading.main_thread()
print("screen capture ok:", len(finds), "tree walks for", 11, "captures")
