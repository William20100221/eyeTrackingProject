"""
Self-check for several AI keys (no network):  python -m eyetracking.testing.test_ai_keys

Keys are named Key 1, Key 2... and shown only by their last 4 characters; the
brand comes from the prefix; the models come from the provider (here a fake
Models API), or the usual list when it can't be asked; a key's model can be
changed; keys can be renamed and removed; nothing is stored as plain text; the
chat's picker lists every key with its models.
"""
import os
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import anthropic
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

import eyetracking.services.reading_ai as ai

# ---- 1. the models a key can use: asked from the provider, or the usual ones
calls = []


class FakeAnthropic:                                       # what anthropic.Anthropic(...).models.list() gives
    def __init__(self, api_key, **_):
        calls.append(api_key)
        if api_key.endswith("bad"):
            raise PermissionError("invalid x-api-key")       # what a refused key looks like here
        self.models = types.SimpleNamespace(list=lambda: [types.SimpleNamespace(id="claude-sonnet-5-5", display_name="Claude Sonnet 5.5"),
                                                          types.SimpleNamespace(id="claude-haiku-5-5", display_name="Claude Haiku 5.5")])


real_client, anthropic.Anthropic = anthropic.Anthropic, FakeAnthropic
models, why = ai.list_models("claude", "sk-ant-good")
assert models == [("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("claude-haiku-5-5", "Claude Haiku 5.5")] and why == ""
assert ai.default_model("claude", models) == "claude-haiku-5-5"                  # the app's default, the key can use it
assert ai.default_model("claude", [("claude-opus-5-5", "Opus")]) == "claude-opus-5-5"   # else the first listed
models, why = ai.list_models("claude", "sk-ant-bad")
assert models == ai.KNOWN_MODELS["claude"] and why == "PermissionError: invalid x-api-key", why
models, why = ai.list_models("openai", "sk-x")
assert models == ai.KNOWN_MODELS["openai"] and why

# ---- 2. Settings > AI > Keys
import eyetracking.ui.control_window as cw
ini = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(ini), QSettings.IniFormat)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)            # confirm removals
cw.QInputDialog.getText = staticmethod(lambda *a, **k: ("Work key", True))      # the rename dialog, answered
qt = QApplication([])
w = cw.SettingsWindow()
changed = []
w.ai_changed.connect(lambda: changed.append(True))
assert w.stored_keys() == [] and w.active_entry() is None
assert w.add_key("hello") is None and "doesn't look like" in w.keys_note.text()

k1 = w.add_key("sk-ant-first-1111")
k2 = w.add_key("AIza-second-2222")
assert "usual ones" in w.keys_note.text(), w.keys_note.text()                   # Gemini isn't asked live: said so
k3 = w.add_key("sk-ant-third-3333")
assert [k["name"] for k in w.stored_keys()] == ["Key 1", "Key 2", "Key 3"] and changed
assert [k["provider"] for k in w.stored_keys()] == ["claude", "gemini", "claude"]
assert k1["models"] == [["claude-sonnet-5-5", "Claude Sonnet 5.5"], ["claude-haiku-5-5", "Claude Haiku 5.5"]]
assert k2["models"] == [list(m) for m in ai.KNOWN_MODELS["gemini"]]
assert w.active_entry()["id"] == k3["id"]                                       # a new key is used at once
assert w.add_key("sk-ant-first-1111")["id"] == k1["id"] and len(w.stored_keys()) == 3   # the same key twice: no copy
w.store.sync()
text = ini.read_text()
assert all(s not in text for s in ("first-1111", "second-2222", "third-3333")), "a key was stored as plain text"

w.use_entry(k1["id"], "claude-sonnet-5-5")                                      # Key 1, with Sonnet
assert w.active_entry()["id"] == k1["id"] and w.active_entry()["model"] == "claude-sonnet-5-5"
assert cw.model_label(w.active_entry()) == "Claude Sonnet 5.5"
assert "Key 1 · Claude Sonnet 5.5" in w.chat.model_btn.text()
rows = [w.keys_list.item(i).text() for i in range(w.keys_list.count())]
assert rows[0].startswith("●  Key 1 — Claude (Anthropic)   (in use)") and "key ending …1111" in rows[0], rows[0]
assert all("first-1111" not in r for r in rows)                                  # only the last 4 are shown
w.rename_key(k1["id"])
assert w.stored_keys()[0]["name"] == "Work key"
w.remove_key(k1["id"])
assert [k["name"] for k in w.stored_keys()] == ["Key 2", "Key 3"] and w.active_entry()["id"] == k2["id"]
k4 = w.add_key("sk-ant-fourth-4444")
assert k4["name"] == "Key 4"                                                     # a removed key's number isn't reused

w._fill_ai_menu()                                                                # the chat's picker
titles = [a.text() for a in w.chat.model_btn.menu().actions()]
assert titles[:3] == ["Key 2  ·  Gemini (Google)", "Key 3  ·  Claude (Anthropic)", "Key 4  ·  Claude (Anthropic)"], titles
assert titles[-1] == "Manage keys…"
sub = w.chat.model_btn.menu().actions()[2].menu()
sub.actions()[0].trigger()                                                       # Key 4 -> Claude Sonnet 5.5
assert w.active_entry()["id"] == k4["id"] and w.active_entry()["model"] == "claude-sonnet-5-5"

w.set_extra_entries([{"id": "ollama", "name": "Ollama", "provider": "ollama", "last4": "",
                      "models": [["llama4", "llama4"], ["qwen3", "qwen3"]], "model": "llama4", "added": ""}])
w.use_entry("ollama", "qwen3")
w2 = cw.SettingsWindow()                                                         # a restart: Ollama's choice remembered
w2.set_extra_entries([{"id": "ollama", "name": "Ollama", "provider": "ollama", "last4": "",
                       "models": [["llama4", "llama4"], ["qwen3", "qwen3"]], "model": "llama4", "added": ""}])
assert w2.active_entry()["id"] == "ollama" and w2.active_entry()["model"] == "qwen3"
anthropic.Anthropic = real_client
print("ai keys ok:", len(calls), "provider calls")
