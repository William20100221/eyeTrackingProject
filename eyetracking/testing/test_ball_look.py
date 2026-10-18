"""
Self-check for the floating ball's look (no webcam):  python -m eyetracking.testing.test_ball_look

Settings > Floating ball: Logo (the default), Water ball (moving) or Image (the user's
picture), a still ball inside a thin ring of a chosen colour. An unreadable image, or no
OpenGL for the water, falls back to the logo. The choice survives a restart.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

import eyetracking.ui.control_window as cw
from eyetracking.ui.assistant_overlay import MARK_RED, FloatingBall

tmp = Path(tempfile.mkdtemp())
cw.QSettings = lambda *_: QSettings(str(tmp / "settings.ini"), QSettings.IniFormat)   # never the user's settings
qt = QApplication([])

ball = FloatingBall()
assert ball.look == "logo" and not ball.avatar.isNull() and ball.ring == QColor(MARK_RED)   # the default: the logo
ball.set_look("image", str(tmp / "missing.png"), "#ff0000")
assert ball.look == "image" and ball.avatar.width() == ball.BALL * 2 and ball.ring == QColor("#ff0000")   # unreadable: the logo
me = QPixmap(10, 10)
me.fill(Qt.GlobalColor.green)
me.save(str(tmp / "me.png"))
ball.set_look("image", str(tmp / "me.png"), "#00ff00")
assert ball.avatar.width() == 10 and ball.ring == QColor("#00ff00")                      # the user's picture
ball.set_look("water", "", "")
assert ball.look in ("water", "logo") and ball.ring == QColor(MARK_RED)                      # no OpenGL offscreen: the logo
ball.set_look("logo", "", "not a colour")
assert ball.look == "logo" and ball.ring == QColor(MARK_RED)
ball.resize(ball.size())
ball.grab()                                                                                 # paints without error

w = cw.SettingsWindow()
assert w.ball_look == "logo" and not w.image_btn.isEnabled() and w.ring_hex.text() == "Theme"
w.set_ball_image(str(tmp / "me.png"))
w.set_ball_look("image")
w.set_ball_ring("#123456")
assert w.image_btn.isEnabled() and w.image_btn.text() == "me.png" and w.ring_hex.text() == "#123456"
saved = []
w.saved.connect(saved.append)
w.save()
assert saved[-1]["ball_look"] == "image" and saved[-1]["ball_image"] == str(tmp / "me.png") and saved[-1]["ball_ring"] == "#123456"
w.store.sync()
w2 = cw.SettingsWindow()                                                                    # a restart
assert (w2.ball_look, w2.ball_image, w2.ball_ring) == ("image", str(tmp / "me.png"), "#123456")
assert w2.look_group.checkedId() == 2
print("ball look ok")
