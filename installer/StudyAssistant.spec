# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller recipe for Study Assistant (run by installer/build.py).

One folder (dist/StudyAssistant/), not one file: a one-file .exe unpacks
everything to %TEMP% at every start (slow, and antivirus programs dislike it),
and the installer hides the folder anyway. No console window. See
docs/installer_dev_log.md for why each line is here.
"""
import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

# Files the code finds next to itself: MediaPipe's model (not bundled automatically,
# MediaPipe's guide says so) and the logo (found with Path(__file__).with_name).
datas = [
    (os.path.join(ROOT, "models", "face_landmarker.task"), "models"),
    (os.path.join(ROOT, "eyetracking", "ui", "app_mark.svg"), os.path.join("eyetracking", "ui")),
]
binaries = []
# Imported inside functions, so PyInstaller can't see them: the OCR, the screenshot,
# and the AI providers other than Claude.
hiddenimports = ["mediapipe.tasks.c", "winocr", "PIL.ImageGrab", "openai", "google.genai"]

# MediaPipe: its native libraries and data files, as MediaPipe's guide recommends.
d, b, h = collect_all("mediapipe")
datas += d
binaries += b
hiddenimports += h

# The app uses QtCore/Gui/Widgets/OpenGL/Svg only: leave the rest of Qt out (size),
# and packages that sit in the dev venv but the app never imports. NOT matplotlib:
# MediaPipe imports it when it loads (mediapipe/tasks/python/vision/drawing_utils.py).
excludes = [
    "tkinter", "IPython", "pytest", "PyInstaller",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets", "PySide6.QtQuickControls2",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput", "PySide6.Qt3DLogic", "PySide6.Qt3DExtras",
    "PySide6.Qt3DAnimation", "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtGraphs", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtDesigner", "PySide6.QtUiTools", "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning",
    "PySide6.QtLocation", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtSerialBus", "PySide6.QtSql",
    "PySide6.QtTest", "PySide6.QtRemoteObjects", "PySide6.QtScxml", "PySide6.QtStateMachine",
    "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech", "PySide6.QtHttpServer", "PySide6.QtWebSockets",
]

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StudyAssistant",
    icon=os.path.join(ROOT, "installer", "app.ico"),
    console=False,          # a windowed app; --self-test writes its report to a file
    upx=False,              # UPX slows the start and makes antivirus programs suspicious
)
coll = COLLECT(exe, a.binaries, a.datas, name="StudyAssistant", upx=False)
