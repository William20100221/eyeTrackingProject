"""
Self-check for Settings > AI > Providers (no network):  python -m eyetracking.testing.test_ai_providers

Claude, ChatGPT, Gemini and Ollama are always listed; each takes its own key (stored
encrypted, shown only by its last 4), API host and models; the models come from the
provider's Models API (faked here) or the usual list when it can't be asked; a provider
can be turned off; a custom provider speaks the OpenAI or Anthropic API at its own host;
older "Key 1, Key 2" settings migrate; the chat's picker lists every usable provider.
"""
import json
import os
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import anthropic
import openai
from google import genai
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox

import eyetracking.services.reading_ai as ai
from eyetracking.services.secret_store import protect

# ---- 1. the models a key can use: asked from the provider at its host, or the usual ones
calls = []


def fake_models(*pairs):
    return types.SimpleNamespace(list=lambda: [types.SimpleNamespace(**p) for p in pairs])


class FakeAnthropic:                                       # anthropic.Anthropic(...).models.list()
    def __init__(self, api_key, base_url, **_):
        calls.append(("claude", api_key, base_url))
        if api_key.endswith("bad"):
            raise PermissionError("invalid x-api-key")       # what a refused key looks like here
        self.models = fake_models({"id": "claude-sonnet-5-5", "display_name": "Claude Sonnet 5.5"},
                                  {"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5"})


class FakeOpenAI:                                          # OpenAI(...).models.list(): ids only
    def __init__(self, api_key, base_url, **_):
        calls.append(("openai", api_key, base_url))
        if "openrouter" in base_url:
            self.models = fake_models({"id": "deepseek/deepseek-v4"}, {"id": "anthropic/claude-haiku-5-5"})
        elif "lmstudio" in base_url:
            raise ConnectionError("nothing listens there")
        else:
            self.models = fake_models({"id": "gpt-6-luna"})


class FakeGemini:                                          # genai.Client(...).models.list(): "models/..." names
    def __init__(self, api_key, http_options):
        calls.append(("gemini", api_key, http_options.base_url))
        self.models = fake_models({"name": "models/gemini-3.5-flash-lite", "display_name": "Gemini 3.5 Flash-Lite",
                                   "supported_actions": ["generateContent"]},
                                  {"name": "models/embedding-001", "display_name": "Embedding", "supported_actions": ["embedContent"]})


real = anthropic.Anthropic, openai.OpenAI, genai.Client
anthropic.Anthropic, openai.OpenAI, genai.Client = FakeAnthropic, FakeOpenAI, FakeGemini
models, why = ai.list_models("claude", "sk-ant-good")
assert models == [("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("claude-haiku-5-5", "Claude Haiku 5.5")] and why == ""
assert calls[-1] == ("claude", "sk-ant-good", ai.DEFAULT_HOSTS["claude"])                # at the usual host
assert ai.default_model("claude", models) == "claude-haiku-5-5"                  # the app's default, the key can use it
assert ai.default_model("claude", [("claude-opus-5-5", "Opus")]) == "claude-opus-5-5"   # else the first listed
models, why = ai.list_models("claude", "sk-ant-bad")
assert models == ai.KNOWN_MODELS["claude"] and why == "PermissionError: invalid x-api-key", why
assert ai.list_models("gemini", "AQ.x") == ([("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite")], "")   # only chat models
models, why = ai.list_models("openai", "or-key", "https://openrouter.ai/api/v1/")        # another host, trailing / dropped
assert models == [("anthropic/claude-haiku-5-5", "anthropic/claude-haiku-5-5"), ("deepseek/deepseek-v4", "deepseek/deepseek-v4")]
assert calls[-1] == ("openai", "or-key", "https://openrouter.ai/api/v1") and why == ""

# ---- 2. Settings > AI > Providers
import eyetracking.ui.control_window as cw
ini = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(ini), QSettings.IniFormat)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)            # confirm removals
cw.QInputDialog.getText = staticmethod(lambda *a, **k: ("my/local-model", True))   # the add-a-model / rename dialog, answered
qt = QApplication([])
w = cw.SettingsWindow()
changed = []
w.ai_changed.connect(lambda: changed.append(True))
names = lambda: [w.provider_list.item(i).text() for i in range(w.provider_list.count())]
assert [p["id"] for p in w.providers()] == ["claude", "openai", "gemini", "ollama"] and w.usable() == []
assert w.active() is None and w.keys_note.isHidden() and names() == ["Claude", "ChatGPT", "Gemini", "Ollama"]
assert w.add_key("hello") is None and "doesn't look like" in w.keys_note.text()   # the first-run popup's paste

claude = w.add_key("sk-ant-first-1111")                                          # the brand gets the key, and answers
assert claude["id"] == "claude" and claude["last4"] == "1111" and w.active()["id"] == "claude" and changed
assert claude["models"] == [["claude-sonnet-5-5", "Claude Sonnet 5.5"], ["claude-haiku-5-5", "Claude Haiku 5.5"]]
assert claude["model"] == "claude-haiku-5-5" and "key saved, 2 models" in w.keys_note.text()
gemini = w.set_key("gemini", "AIza-second-2222")                                 # typed in Gemini's own field
assert gemini["models"] == [["gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite"]] and w.active()["id"] == "gemini"
w.store.sync()
text = ini.read_text()
assert "first-1111" not in text and "second-2222" not in text, "a key was stored as plain text"
assert w.set_key("gemini", "AIza-second-2222") is gemini or w.set_key("gemini", "AIza-second-2222")["last4"] == "2222"
assert len(calls) == 6, calls                                                     # the same key again: not asked again
assert w.set_key("gemini", "") ["last4"] == "2222"                               # an empty field: nothing happens
w.set_key("openai", "sk-ant-looks-wrong")                                        # another brand's key on ChatGPT: kept, said
assert "looks like a Claude key" in w.keys_note.text(), w.keys_note.text()

w.use("claude", "claude-sonnet-5-5")
assert w.active()["id"] == "claude" and w.active()["model"] == "claude-sonnet-5-5"
assert cw.model_label(w.active()) == "Claude Sonnet 5.5" and "Claude Sonnet 5.5" in w.chat.model_btn.text()
assert "·" not in w.chat.model_btn.text()                                       # the model only, not its provider
assert names()[0] == "Claude  ·  in use"
w._show_provider("claude")
rows = [w.model_list.item(i).text() for i in range(w.model_list.count())]
assert rows == ["○  Claude Sonnet 5.5    claude-sonnet-5-5", "●  Claude Haiku 5.5    claude-haiku-5-5"] or rows[0].startswith("●"), rows
assert w.key_field.text() == "" and "ending …1111" in w.key_field.placeholderText()   # the key itself is never shown
assert not w.forget_link.isHidden() and w.host_field.text() == ai.DEFAULT_HOSTS["claude"] and w.host_link.isHidden()
w.set_host("claude", "https://proxy.example.com/anthropic/")                      # a proxy: used for the model list too
assert w.provider("claude")["host"] == "https://proxy.example.com/anthropic" and not w.host_link.isHidden()
w.sync_models("claude")
assert calls[-1] == ("claude", "sk-ant-first-1111", "https://proxy.example.com/anthropic")
w.set_host("claude", "")
assert w.provider("claude")["host"] == ai.DEFAULT_HOSTS["claude"]

w.set_on("claude", False)                                                        # off: can't answer, says so
assert w.active()["id"] == "openai" and names()[0] == "Claude  (off)" and "claude" not in [p["id"] for p in w.usable()]
w.set_on("claude", True)
w.forget_key("openai")
assert w.provider("openai")["secret"] == "" and w.active()["id"] == "claude"
w.add_model("claude")                                                            # a model typed by its id
assert w.active()["model"] == "my/local-model" and ["my/local-model", "my/local-model"] in w.provider("claude")["models"]
w.remove_model("claude", "my/local-model")
assert w.provider("claude")["model"] == "claude-sonnet-5-5"

# a custom provider: OpenRouter speaks the OpenAI API at its own host
cw.add_provider_dialog = lambda *_: {"name": "OpenRouter", "key": "or-key-3333", "host": "https://openrouter.ai/api/v1", "kind": "openai"}
w.add_provider()
custom = w.provider("custom1")
assert custom["custom"] and custom["kind"] == "openai" and custom["last4"] == "3333" and w.active()["id"] == "custom1"
assert custom["model"] == "anthropic/claude-haiku-5-5" and names()[-1] == "OpenRouter  ·  in use"
w._show_provider("custom1")
assert w.key_link.isHidden() and w.host_link.isHidden() and "OpenAI API" in w.provider_maker.text()
cw.add_provider_dialog = lambda *_: {"name": "LM Studio", "key": "", "host": "http://lmstudio.local/v1", "kind": "openai"}
w.add_provider()                                                                 # no key, no models API: still usable
assert w.provider("custom2")["models"] == [] and w.active()["id"] == "custom2" and "couldn't ask" in w.keys_note.text()
w.remove_provider("custom2")
assert w.provider("custom2") is None and w.active()["id"] == "claude" and w.shown_id == "claude"
cw.add_provider_dialog = lambda *_: {"name": "Again", "key": "", "host": "http://x/v1", "kind": "claude"}
w.add_provider()
assert w.provider("custom2")["kind"] == "claude"                                 # the next free number
w.remove_provider("custom2")

w._fill_ai_menu()                                                                # the chat's picker
actions = w.chat.model_btn.menu().actions()
titles = [a.text() for a in actions if a.text()]
assert titles == ["Claude", "Gemini", "OpenRouter", "Manage providers…"], titles  # on and with a key / custom
sub = next(a for a in actions if a.text() == "OpenRouter").menu()
sub.actions()[0].trigger()
assert w.active()["id"] == "custom1" and w.active()["model"] == "anthropic/claude-haiku-5-5"

w.set_found(None, [["llama4", "llama4"], ["qwen3", "qwen3"]])                     # Ollama running, two models
w.use("ollama", "qwen3")
w2 = cw.SettingsWindow()                                                         # a restart: Ollama's choice remembered
w2.set_found(("openai", "sk-env-4444"), [["llama4", "llama4"], ["qwen3", "qwen3"]])
assert w2.active()["id"] == "ollama" and w2.active()["model"] == "qwen3"
assert w2.provider("openai")["last4"] == "4444" and w2.secret_of(w2.provider("openai")) == "sk-env-4444"   # from the environment
w2.store.sync()
assert "sk-env-4444" not in ini.read_text()                                      # ...and never saved

# ---- 3. older settings: "Key 1, Key 2…" become the brands' keys; a second key of a brand, a custom provider
ini2 = Path(tempfile.mkdtemp()) / "settings.ini"
cw.QSettings = lambda *_: QSettings(str(ini2), QSettings.IniFormat)
old = QSettings(str(ini2), QSettings.IniFormat)
old.setValue("ai_keys", json.dumps([
    {"id": "key1", "name": "Key 1", "provider": "claude", "secret": protect("sk-ant-old-1"), "last4": "ld-1",
     "models": [["claude-opus-5-5", "Claude Opus 5.5"]], "model": "claude-opus-5-5"},
    {"id": "key2", "name": "Work key", "provider": "claude", "secret": protect("sk-ant-old-2"), "last4": "ld-2",
     "models": [], "model": "claude-haiku-5-5"}]))
old.setValue("ai_active", "key2")
old.setValue("ai_extra_models", json.dumps({"ollama": "qwen3"}))
old.sync()
w3 = cw.SettingsWindow()
assert w3.secret_of(w3.provider("claude")) == "sk-ant-old-1" and w3.provider("claude")["model"] == "claude-opus-5-5"
second = w3.provider("custom1")
assert second["name"] == "Work key (Claude)" and second["kind"] == "claude" and w3.secret_of(second) == "sk-ant-old-2"
assert w3.store.value("ai_active") == "custom1" and w3.provider("ollama")["model"] == "qwen3"
assert all(w3.store.value(k) is None for k in ("ai_keys", "ai_extra_models"))
anthropic.Anthropic, openai.OpenAI, genai.Client = real
print("ai providers ok:", len(calls), "provider calls")
