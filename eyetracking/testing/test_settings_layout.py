"""
Self-check for issue #14 (no camera):  python -m eyetracking.testing.test_settings_layout

Opening the Theme colour sheet narrows the settings page; the rows must reflow to
the new width instead of being clipped (the subtitles wrap). The sheet starts closed.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QScrollArea

import eyetracking.ui.control_window as cw

cw.QSettings = lambda *_: QSettings(str(Path(tempfile.mkdtemp()) / "settings.ini"), QSettings.IniFormat)
qt = QApplication([])
w = cw.SettingsWindow()
w.resize(1100, 700)
w.pages.setCurrentWidget(w.settings_page)
w.show()
qt.processEvents()
scroll = w.settings_page.findChild(QScrollArea)
assert not w.sheet.isVisible(), "the sheet should start closed"
for step in (w.open_sheet, w.cancel_sheet, w.open_sheet):
    step()
    qt.processEvents()
    viewport, inner = scroll.viewport().width(), scroll.widget().width()
    assert inner <= viewport, f"settings content {inner} px wider than its {viewport} px viewport: rows are clipped"
# the turning arrow (Reset to defaults) turns anticlockwise, the way the arrow points (20 Oct)
for spin in w.findChildren(cw.SpinButton):
    spin._turn.setCurrentTime(275)                                  # halfway through the turn
    assert spin._turn.endValue() == -360.0 and spin.angle < 0, (spin.text(), spin.angle)

# Settings > Font: every window's text at once, light ones too; kept with Save changes, back with Reset
assert [a.text() for a in w.font_menu.actions()] == list(cw.FONTS) and qt.font().weight() == 400
w.set_font_choice("Segoe UI Light")
assert qt.font().families()[0] == "Segoe UI" and qt.font().weight() == 300 and w.font_btn.text().startswith("Segoe UI Light")
assert cw.ICON_FONTS[0] in qt.font().families()                  # the glyphs still draw
w.save()
assert w.store.value("font") == "Segoe UI Light"
w.load(cw.DEFAULTS)
assert qt.font().weight() == 400 and w.font_btn.text().startswith("Segoe UI  ")
print("settings layout ok")
