"""
Ask an AI model about a text the user has partly read.

    summarize(full_text, read_text, mode)
        mode "unread":  summary of the skipped parts only
        mode "outline": summary of the whole text + outline of the skipped parts
    make_questions(full_text, read_text, scope)  -> [{"question": ..., "model_answer": ...}, ...]
        scope "unread": questions about what was skipped
        scope "whole":  questions about the whole text
    grade_answer(full_text, question, user_answer) -> {"score": 0-10, "explanation": ...}

Every function takes provider="claude" | "gemini" | "openai" | "ollama".
Install only the one you use, and set that provider's API key:
    claude   uv add anthropic      ANTHROPIC_API_KEY   (keys start with sk-ant-)
    gemini   uv add google-genai   GEMINI_API_KEY      (AIza...)
    openai   uv add openai         OPENAI_API_KEY      (sk-...)
    ollama   nothing to install: a free local AI from ollama.com, no key
The app works the provider out from the key (provider_for_key), looks for a key
in the environment or a .env file first (key_from_environment), and uses Ollama
when it runs on this PC and there is no key (ollama_model).
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from eyetracking.core.reading import mark_unread, unread_word_count

# Fast & cheap model per provider (checked 2026-10-08). Change one line to switch.
MODELS = {
    "claude": "claude-haiku-5-5",  # the default for a new Claude key; each key can pick its own
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-6-luna",
    "ollama": "",                  # filled in at start with the first model Ollama has installed
}
OLLAMA_URL = "http://localhost:11434"   # Ollama's default address
KEY_ENV = {"claude": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
KEY_PREFIXES = (("sk-ant-", "claude"), ("AIza", "gemini"), ("sk-", "openai"))   # the longer one first


PROVIDER_NAMES = {"claude": "Claude (Anthropic)", "gemini": "Gemini (Google)", "openai": "ChatGPT (OpenAI)",
                  "ollama": "Ollama (on this PC)"}
# Used when the provider can't be asked which models a key can use (offline, or its package isn't installed).
KNOWN_MODELS = {
    "claude": [("claude-haiku-5-5", "Claude Haiku 5.5"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
               ("claude-opus-5-5", "Claude Opus 5.5"), ("claude-fable-5-1", "Claude Fable 5.1")],
    "gemini": [("gemini-3.5-flash-lite", "gemini-3.5-flash-lite")],
    "openai": [("gpt-6-luna", "gpt-6-luna")],
    "ollama": [],
}


def list_models(provider: str, key: str = "") -> tuple[list[tuple[str, str]], str]:
    """Ask the provider which models this key can use: ([(model id, display name)], "").
    If it can't be asked, ([the usual models], why). A key says nothing about models by
    itself (any key can call any model its account may use), so this is how a key's
    models are found. Claude: the Models API through the anthropic package; Ollama: its
    local /api/tags."""
    try:
        if provider == "claude":
            import anthropic
            client = anthropic.Anthropic(api_key=key, timeout=10.0, max_retries=0)
            return [(m.id, m.display_name) for m in client.models.list()], ""
        if provider == "ollama":
            with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2) as r:
                return [(m["name"], m["name"]) for m in json.load(r).get("models", [])], ""
        # ponytail: Gemini / OpenAI use the built-in list; ask them live once their package is installed and tested
        return KNOWN_MODELS[provider], "the live model list is only built for Claude and Ollama"
    except ImportError:
        return KNOWN_MODELS[provider], f"the {provider} package isn't installed"
    except Exception as e:                      # offline, key refused, Ollama not running...
        return KNOWN_MODELS[provider], f"{type(e).__name__}: {e}"


def default_model(provider: str, models: list) -> str:
    """The model a new key starts on: the app's default if the key can use it, else the first one listed."""
    ids = [m[0] for m in models]
    return MODELS[provider] if MODELS[provider] in ids or not ids else ids[0]


def provider_for_key(key: str) -> str | None:
    """Which provider a pasted key belongs to, from its prefix; None if it looks like none."""
    return next((p for prefix, p in KEY_PREFIXES if key.startswith(prefix)), None)


def key_from_environment() -> tuple[str, str] | None:
    """(provider, key) from the environment or a .env file next to main.py, if any."""
    env = dict(os.environ)
    dotenv = Path(__file__).resolve().parents[2] / ".env"
    if dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() and not k.lstrip().startswith("#"):
                env.setdefault(k.strip(), v.strip().strip("'\""))
    env.setdefault("GEMINI_API_KEY", env.get("GOOGLE_API_KEY", ""))
    return next(((p, env[name]) for p, name in KEY_ENV.items() if env.get(name)), None)


def ollama_model() -> str | None:
    """The first model installed in a running Ollama, or None (not running, nothing installed)."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=1) as r:
            models = json.load(r).get("models", [])
    except (OSError, ValueError):
        return None
    return models[0]["name"] if models else None

SYSTEM = (
    "You help a student who is reading a text on screen. An eye tracker recorded what "
    "they read. In the text, the parts they did NOT read are wrapped in "
    "<unread>...</unread>; they read everything else. Talk to the student as 'you'. "
    "Always write in the same language as the text."
)
LENGTH = "Keep it as short as you can, but never drop an important point just to keep it short."

WORDS_PER_QUESTION = 150   # 5 questions for ~750 words
MAX_QUESTIONS = 10

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "model_answer": {"type": "string"},
                },
                "required": ["question", "model_answer"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["questions"],
    "additionalProperties": False,
}

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "explanation": {"type": "string"},
    },
    "required": ["score", "explanation"],
    "additionalProperties": False,
}


def summarize(full_text: str, read_text: str, mode: str = "unread",
              provider: str = "claude", words: int | None = None) -> str:
    marked = mark_unread(full_text, read_text)
    if mode == "unread":
        if unread_word_count(marked) == 0:
            return "You read the whole text: nothing to summarize."
        task = ("Summarize only the <unread> parts, in bullet points, so the student can "
                "catch up without reading them. Use the read parts only as context.")
    elif mode == "outline":
        task = ("First summarize the whole text in bullet points. Then add the heading "
                "'What you skipped' with one bullet per <unread> part: what it covers and how "
                "it connects to what the student read. If nothing is unread, say so.")
    else:
        raise ValueError(f"mode must be 'unread' or 'outline', not {mode!r}")
    length = f"Use about {words} words." if words else LENGTH   # words: Settings > Summary length
    return _ask(provider, f"<text>\n{marked}\n</text>\n\n{task} {length}")


def question_count(words: int) -> int:
    return max(1, min(MAX_QUESTIONS, round(words / WORDS_PER_QUESTION)))


def make_questions(full_text: str, read_text: str, scope: str = "unread",
                   provider: str = "claude") -> list[dict]:
    marked = mark_unread(full_text, read_text)
    if scope == "unread":
        words = unread_word_count(marked)
        if words == 0:
            return []
        about = ("about the <unread> parts, so the student realizes what they missed. Ask "
                 "about ideas that matter for understanding the whole text, not small details.")
    elif scope == "whole":
        words = len(full_text.split())
        about = "about the whole text, testing its most important ideas."
    else:
        raise ValueError(f"scope must be 'unread' or 'whole', not {scope!r}")
    n = question_count(words)
    task = (f"Write {n} questions {about} Write fewer if the text doesn't have {n} important "
            "points. Every question must be answerable from the text. Give each one a model "
            "answer of 1-3 sentences.")
    reply = _ask(provider, f"<text>\n{marked}\n</text>\n\n{task}", QUESTIONS_SCHEMA)
    return json.loads(reply)["questions"]


def grade_answer(full_text: str, question: dict, user_answer: str,
                 provider: str = "claude") -> dict:
    task = (f"Question: {question['question']}\n"
            f"Model answer: {question['model_answer']}\n"
            f"Student's answer: {user_answer}\n\n"
            "Score the student's answer from 0 to 10, using the text and the model answer: "
            "10 = every key point, 0 = wrong or off-topic. Judge the ideas, not the wording "
            "or spelling. In 1-2 sentences, explain what they got right and what they missed.")
    result = json.loads(_ask(provider, f"<text>\n{full_text}\n</text>\n\n{task}", GRADE_SCHEMA))
    # not every provider's schema can enforce 0-10, so clamp here
    result["score"] = max(0, min(10, result["score"]))
    return result


def _ask(provider: str, prompt: str, schema: dict | None = None) -> str:
    """Send one prompt and return the reply text (JSON text when a schema is given).
    Each SDK is imported here, so you only need the one you use."""
    if provider not in MODELS:
        raise ValueError(f"provider must be one of {list(MODELS)}, not {provider!r}")
    model = MODELS[provider]

    if provider == "ollama":
        body = {"model": model or ollama_model(), "stream": False,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        if schema:
            body["format"] = schema               # Ollama's structured output takes a JSON schema
        req = urllib.request.Request(OLLAMA_URL + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:   # a local model can be slow
            return json.load(r)["message"]["content"]

    if provider == "claude":
        import anthropic
        extra = {"output_config": {"format": {"type": "json_schema", "schema": schema}}} if schema else {}
        reply = anthropic.Anthropic().messages.create(
            model=model, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}], **extra)
        if reply.stop_reason in ("max_tokens", "refusal"):
            raise RuntimeError(f"Claude stopped early: {reply.stop_reason}")
        return next(b.text for b in reply.content if b.type == "text")

    if provider == "gemini":
        from google import genai
        from google.genai import types
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json" if schema else None,
            response_json_schema=schema)
        return genai.Client().models.generate_content(model=model, contents=prompt, config=config).text

    from openai import OpenAI
    extra = {"text": {"format": {"type": "json_schema", "name": "result",
                                 "schema": schema, "strict": True}}} if schema else {}
    return OpenAI().responses.create(model=model, instructions=SYSTEM, input=prompt, **extra).output_text