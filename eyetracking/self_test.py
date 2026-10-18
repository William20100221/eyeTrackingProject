"""
Install check:  StudyAssistant.exe --self-test   (or: python main.py --self-test)

Checks, without a camera or a window, that everything the app needs is really
there: the MediaPipe model and its native parts (runs the face landmarker on a
blank frame), the logo, Qt's SVG and Windows styling, the AI and OCR packages,
and that the data folder can be written to. The report goes to the console and
to <data folder>\\..\\self_test.txt (an installed app has no console), and the
exit code is 0 when everything passed.

This is how the frozen build and the installer are tested (installer/build.py),
and what to ask a user to run when the installed app won't start.
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import traceback


def run() -> int:
    from path import BUNDLE_DIR, DATA_DIR, FROZEN, find_model

    results: list[tuple[str, bool, str]] = []

    def check(name: str, fn) -> None:
        try:
            results.append((name, True, fn() or ""))
        except Exception as e:                               # every failure goes in the report, the rest still run
            results.append((name, False, f"{type(e).__name__}: {e}"))

    def landmarker() -> str:
        import numpy as np
        from eyetracking.services.landmarker import FaceLandmarker
        with FaceLandmarker() as lm:
            result = lm.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        return f"ran on a blank frame, face found: {result.face_found}"

    def logo() -> str:
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([sys.argv[0], "-platform", "offscreen"])
        from eyetracking.ui.assistant_overlay import MARK_SVG
        icon = QIcon(str(MARK_SVG))
        assert MARK_SVG.is_file(), f"missing {MARK_SVG}"
        assert not icon.pixmap(64, 64).isNull(), "the SVG icon plugin didn't draw the logo"
        return f"{MARK_SVG.name} drawn ({app.style().name()} style)"

    def data_dir() -> str:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=DATA_DIR, delete=True) as f:
            f.write(b"ok")
        return str(DATA_DIR)

    def package(name: str):
        return lambda: getattr(importlib.import_module(name), "__version__", "imported")

    check("model file", lambda: str(find_model()))
    check("MediaPipe face landmarker", landmarker)
    check("logo + Qt SVG plugin", logo)
    check("data folder writable", data_dir)
    for name in ("cv2", "sklearn", "anthropic", "winocr"):
        check(f"package {name}", package(name))

    lines = [f"Study Assistant self-test ({'installed' if FROZEN else 'from source'}, Python {sys.version.split()[0]})",
             f"bundle: {BUNDLE_DIR}", ""]
    lines += [f"{'PASS' if ok else 'FAIL'}  {name}: {detail}" for name, ok, detail in results]
    passed = all(ok for _, ok, _ in results)
    lines += ["", "all passed" if passed else "SOME CHECKS FAILED"]
    report = "\n".join(lines)
    try:
        out = DATA_DIR.parent / "self_test.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        report += f"\n(report also saved to {out})"
    except OSError:
        report += "\n" + traceback.format_exc()
    if sys.stdout is not None:                               # a windowed .exe has no console: stdout is None
        print(report)
    return 0 if passed else 1


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.exit(run())
