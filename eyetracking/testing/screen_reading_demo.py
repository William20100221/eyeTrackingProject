"""
Read-from-screen demo, with the MOUSE as the gaze (no camera):
    python -m eyetracking.testing.screen_reading_demo [seconds]

1. Put a page of text in front (browser, Notepad, PDF...). You have 3 s to
   point the mouse at it.
2. Move the mouse along the lines as if reading, for 20 s (or `seconds`). The
   window's text is re-read once a second on another thread, so scrolling is fine.
3. While you go it prints the word under the mouse; at the end, the text it
   judged read, and the full text with the unread parts marked: exactly what
   the summary / quiz AI would get.
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time
from ctypes import wintypes

from eyetracking.core.reading import ReadingTracker, mark_unread, unread_word_count
from eyetracking.services.screen_text import lines_on_screen, make_dpi_aware, window_title, window_under_mouse

CAPTURE_S = 1.0     # how often the window's text is re-read


def mouse() -> tuple[int, int]:
    pt = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def main() -> int:
    make_dpi_aware()
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    print("Point the mouse at the text you will read...")
    time.sleep(3)
    hwnd = window_under_mouse()
    print(f"Reading {window_title(hwnd)!r} for {seconds:.0f} s: move the mouse along the lines.")

    tracker, lock, stop = ReadingTracker(), threading.Lock(), threading.Event()

    def capture() -> None:               # UIA/OCR take ~1 s: off the sampling loop
        while not stop.is_set():
            try:
                lines = lines_on_screen(hwnd)
            except Exception as e:           # the window closed or changed mid-read: try again next second
                print(f"  (capture failed: {e})")
                lines = None
            if lines:
                with lock:
                    tracker.show(lines)
            stop.wait(CAPTURE_S)

    threading.Thread(target=capture, daemon=True).start()
    end, last = time.monotonic() + seconds, None
    while time.monotonic() < end:
        x, y = mouse()
        with lock:
            word = tracker.feed(x, y, time.monotonic())
        if word != last:
            print(f"  {word or '-':20}", end="\r", flush=True)
            last = word
        time.sleep(1 / 30)
    stop.set()

    with lock:
        full, read = tracker.full_text(), tracker.read_text()
        done, total = tracker.progress()
    marked = mark_unread(full, read)
    print(f"\n\nRead {done} of {total} words seen; {unread_word_count(marked)} words unread.\n")
    print("---- judged read ----\n" + read)
    print("\n---- what the AI would get ----\n" + marked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
