"""
The Windows icon (installer/app.ico) from the logo (eyetracking/ui/app_mark.svg).

    python installer/make_icon.py

Qt draws the SVG at 256 px; Pillow writes one .ico holding the sizes Windows asks
for (16 px in lists, 32 on the desktop, 256 in large icon views). Run by
installer/build.py; the .ico is also committed, so the build works without it.
"""
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PIL import Image
from PySide6.QtCore import QBuffer, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
SVG = ROOT / "eyetracking" / "ui" / "app_mark.svg"
ICO = ROOT / "installer" / "app.ico"
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]


def main() -> int:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    renderer = QSvgRenderer(str(SVG))
    if not renderer.isValid():
        print(f"can't read {SVG}")
        return 1
    image = QImage(256, 256, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter)
    painter.end()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    Image.open(io.BytesIO(bytes(buffer.data()))).save(ICO, sizes=[(s, s) for s in SIZES])
    print(f"wrote {ICO} ({', '.join(map(str, SIZES))} px)")
    del app
    return 0


if __name__ == "__main__":
    sys.exit(main())
