"""
Self-check for choosing the AI (no network, no popup):  python -m eyetracking.testing.test_ai_choice

The key's prefix says the brand; a key in a .env file counts; on first run the
popup's key becomes Key 1; adding another key switches to it at once.
"""
import os
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import eyetracking.services.reading_ai as ai

assert ai.provider_for_key("sk-ant-abc") == "claude" and ai.provider_for_key("sk-abc") == "openai"
assert ai.provider_for_key("AIzaXYZ") == "gemini" and ai.provider_for_key("hello") is None

for name in ai.KEY_ENV.values():                         # a clean environment for the test
    os.environ.pop(name, None)
os.environ.pop("GOOGLE_API_KEY", None)
with tempfile.TemporaryDirectory() as folder:            # .env next to main.py: read, quotes stripped
    fake = Path(folder) / "eyetracking" / "services" / "reading_ai.py"
    fake.parent.mkdir(parents=True)
    (Path(folder) / ".env").write_text("# keys\nGEMINI_API_KEY='AIza-from-dotenv'\n")
    ai.__file__, real = str(fake), ai.__file__
    assert ai.key_from_environment() == ("gemini", "AIza-from-dotenv")
    ai.__file__ = real
ai.OLLAMA_URL = "http://localhost:9"                      # nothing listens there
models, why = ai.list_models("ollama")
assert models == [] and why

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication
import eyetracking.ui.control_window as cw
settings_file = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(settings_file), QSettings.IniFormat)   # never the real settings
qt = QApplication([])
import eyetracking.app as appmod
offline = lambda provider, key="": (ai.KNOWN_MODELS[provider], "offline test")
appmod.list_models = cw.list_models = offline
appmod.key_from_environment = lambda: None
appmod.pick_ai = lambda: "sk-ant-pasted"                  # the first-run popup, answered
app = appmod.App(mouse=True)
app.timer.stop()
assert app.provider == "claude" and os.environ["ANTHROPIC_API_KEY"] == "sk-ant-pasted"
assert [k["name"] for k in app.control.stored_keys()] == ["Key 1"]
app.control.add_key("AIza-new")                          # another key: used at once
assert app.provider == "gemini" and os.environ["GEMINI_API_KEY"] == "AIza-new"
assert "Key 2 · Gemini (Google)" in app.panel.status.text(), app.panel.status.text()
print("ai choice ok")
