"""
What text is on the screen, and where (Windows only).

    lines_on_screen(hwnd)   -> list[Line]   one Line (text + box, screen px) per visual line
    window_at(x, y)         -> hwnd         the top-level window under a screen point (0 = none)
    window_under_mouse()    -> hwnd
    active_window()         -> hwnd         the window with the focus
    window_title(hwnd)      -> str

Two ways in, tried in this order:
  1. UI Automation: the app hands over its own text and the box of every visible
     line, through the same interface screen readers use. Exact text, a few
     hundred ms, no pixels. Chrome/Edge pages, Notepad, Office, most Electron apps.
  2. OCR of a screenshot of the window with Windows' built-in OCR, when UIA
     exposes (almost) nothing: PDF readers, Kindle, Discord... The screenshot is
     used for one call and dropped; nothing is saved.
Why not record the window: docs/screen_reading_dev_log.md.

All boxes are physical screen pixels, the frame of ctypes GetCursorPos, as long
as the process is DPI-aware: Qt makes it so; a plain script calls make_dpi_aware().
Qt's own coordinates (QCursor.pos(), widget geometry) are scaled pixels: multiply
by the screen's devicePixelRatio before using them with these boxes.

Threads: UIA is COM; lines_on_screen() initialises COM for whatever thread calls it.

Run:  python -m eyetracking.services.screen_text    (3 s to point the mouse at a window, then prints its lines)
"""
from __future__ import annotations

import asyncio
import ctypes
import sys
import time
from ctypes import wintypes

from eyetracking.core.reading import Line

OCR_LANG = "en"        # a Windows OCR language pack (Settings > Time & language); "zh-Hans-CN" for simplified Chinese
MIN_WORDS = 20         # fewer words than this from UIA = probably a toolbar only: try OCR too, keep whichever found more

if sys.platform == "win32":
    import comtypes
    import comtypes.client

    comtypes.client.GetModule("UIAutomationCore.dll")  # writes comtypes.gen.UIAutomationClient on first run
    from comtypes.gen import UIAutomationClient as UIA

    _user32 = ctypes.windll.user32
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.WindowFromPoint.restype = wintypes.HWND
    _user32.WindowFromPoint.argtypes = [wintypes.POINT]
    _user32.GetAncestor.restype = wintypes.HWND
    _user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]


# ---------------------------------------------------------------- windows
def make_dpi_aware() -> None:
    """Physical pixels everywhere. For scripts without Qt (Qt does this itself, its own way)."""
    ctypes.windll.shcore.SetProcessDpiAwareness(2)


def active_window() -> int:
    return _user32.GetForegroundWindow() or 0


def window_at(x: float, y: float) -> int:
    """The top-level window under a screen point; 0 on the desktop. Click-through
    windows (the gaze dot) are skipped, like the mouse skips them."""
    return _user32.GetAncestor(_user32.WindowFromPoint(wintypes.POINT(int(x), int(y))), 2) or 0   # 2 = GA_ROOT


def window_under_mouse() -> int:
    pt = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return window_at(pt.x, pt.y)


def window_title(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    _user32.GetWindowTextW(wintypes.HWND(hwnd), buf, 256)
    return buf.value


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of the window as drawn, screen px (DWM: without the invisible resize borders)."""
    r = wintypes.RECT()
    ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 9, ctypes.byref(r), ctypes.sizeof(r))
    return r.left, r.top, r.right, r.bottom


# ---------------------------------------------------------------- text
def lines_on_screen(hwnd: int | None = None) -> list[Line]:
    """The visible text of a window, one Line per visual line, in reading order."""
    hwnd = hwnd or active_window()
    lines = _uia_lines(hwnd)
    if sum(len(ln.text.split()) for ln in lines) < MIN_WORDS:
        ocr = _ocr_lines(hwnd)
        if sum(len(ln.text.split()) for ln in ocr) > sum(len(ln.text.split()) for ln in lines):
            lines = ocr
    return lines     # in the source's own order: UIA gives document order, which is the reading order


def _uia_lines(hwnd: int) -> list[Line]:
    try:
        comtypes.CoInitialize()                          # COM per thread: the app captures on a worker thread
    except OSError:
        pass                                             # already initialised another way: fine
    uia = comtypes.CoCreateInstance(UIA.CUIAutomation._reg_clsid_, interface=UIA.IUIAutomation,
                                    clsctx=comtypes.CLSCTX_INPROC_SERVER)
    has_text = uia.CreatePropertyCondition(UIA.UIA_IsTextPatternAvailablePropertyId, True)
    try:
        found = uia.ElementFromHandle(wintypes.HWND(hwnd)).FindAll(UIA.TreeScope_Descendants, has_text)
    except Exception:                                    # COMError: the window is gone or refuses (admin app)
        return []
    best = None                                          # the document = the text element with the most text
    for i in range(found.Length):                        # (a page also has an address bar, buttons, tooltips)
        try:
            tp = found.GetElement(i).GetCurrentPattern(UIA.UIA_TextPatternId).QueryInterface(UIA.IUIAutomationTextPattern)
            n = len(tp.DocumentRange.GetText(-1))
        except Exception:
            continue
        if best is None or n > best[0]:
            best = (n, tp)
    if best is None:
        return []
    lines: list[Line] = []
    try:
        ranges = best[1].GetVisibleRanges()              # Chromium: one range per visual line; Notepad: the whole text
        for i in range(ranges.Length):
            rng = ranges.GetElement(i)
            lines += _split(rng.GetText(-1), rng.GetBoundingRectangles())
    except Exception:                                    # the page re-rendered under us: keep what we got, next call gets the rest
        pass
    return lines


def _split(text: str, flat) -> list[Line]:
    """One UIA text range -> Lines. The rectangles come back flat (x, y, w, h, x, y, ...),
    one per visible piece: the inline pieces of one line (links, superscripts) share a
    row and are joined; several rows are matched to the text's "\\n" parts, or the text
    is cut by row width when it wrapped."""
    text = text.replace("￼", "")                    # the "object" character that stands for an image
    rects = [tuple(int(v) for v in flat[i:i + 4]) for i in range(0, len(flat), 4)]
    rects = [r for r in rects if r[2] > 0 and r[3] > 0]
    if not rects or not text.strip():
        return []
    rows: list[list[int]] = []                           # [x, y, right, bottom] per row
    for x, y, w, h in sorted(rects, key=lambda r: (r[1], r[0])):
        if rows and rows[-1][1] <= y + h / 2 <= rows[-1][3]:   # its centre is inside the last row: same line
            rows[-1] = [min(rows[-1][0], x), min(rows[-1][1], y), max(rows[-1][2], x + w), max(rows[-1][3], y + h)]
        else:
            rows.append([x, y, x + w, y + h])
    typical = sorted(r[3] - r[1] for r in rows)[len(rows) // 2]
    rows = [r for r in rows if r[3] - r[1] <= 3 * typical]      # not the pictures
    parts = text.rstrip("\n").split("\n")               # Notepad: an empty line has a rectangle too
    if len(parts) != len(rows):
        parts = [p for p in parts if p.strip()]
    if len(parts) != len(rows):
        # wrapped text, or pieces that don't line up: share the text out over the rows by their
        # width. ponytail: a word can land on the row next to its real one; never a box over several rows
        parts = _cut(" ".join(parts), [r[2] - r[0] for r in rows])
    return [Line(p, x, y, r - x, b - y) for p, (x, y, r, b) in zip(parts, rows) if p.strip()]


def _cut(text: str, widths: list[int]) -> list[str]:
    """Cut `text` into len(widths) pieces whose lengths follow the widths, at the nearest spaces."""
    total, out, pos = sum(widths) or 1, [], 0
    for w in widths[:-1]:
        want = pos + round(len(text) * w / total)
        before, after = text.rfind(" ", pos, want + 1), text.find(" ", want)
        cut = max(before, after) if min(before, after) < 0 else min((before, after), key=lambda c: abs(c - want))
        if cut < 0:
            cut = want
        out.append(text[pos:cut])
        pos = cut
    return out + [text[pos:]]


def _ocr_lines(hwnd: int) -> list[Line]:
    try:
        import winocr                                    # optional: uv pip install winocr
        from PIL import ImageGrab
    except ImportError:
        print("screen_text: no text through UI Automation and OCR isn't installed (uv pip install winocr)")
        return []
    left, top, right, bottom = window_rect(hwnd)
    if right <= left or bottom <= top:
        return []
    image = ImageGrab.grab((left, top, right, bottom), all_screens=True)
    result = asyncio.run(winocr.recognize_pil(image, OCR_LANG))
    lines = []
    for ln in result.lines:
        boxes = [(left + w.bounding_rect.x, left + w.bounding_rect.x + w.bounding_rect.width) for w in ln.words]
        y0 = top + min(w.bounding_rect.y for w in ln.words)
        y1 = top + max(w.bounding_rect.y + w.bounding_rect.height for w in ln.words)
        lines.append(Line(" ".join(w.text for w in ln.words), boxes[0][0], y0, boxes[-1][1] - boxes[0][0], y1 - y0,
                          tuple(boxes)))
    return lines


def main() -> int:
    make_dpi_aware()
    print("Point the mouse at a window with text...")
    time.sleep(3)
    hwnd = window_under_mouse()
    t = time.time()
    lines = lines_on_screen(hwnd)
    print(f"{window_title(hwnd)!r}: {len(lines)} lines, {sum(len(ln.text.split()) for ln in lines)} words, "
          f"{time.time() - t:.2f} s")
    for ln in lines:
        print(f"  ({ln.x:5.0f},{ln.y:5.0f}) {ln.w:4.0f}x{ln.h:<3.0f} {ln.text[:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
