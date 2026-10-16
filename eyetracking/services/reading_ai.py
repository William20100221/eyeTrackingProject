"""
Ask an AI model about a text the user has partly read.

    summarize(full_text, read_text, mode)
        mode "unread":  summary of the skipped parts only
        mode "outline": summary of the whole text + outline of the skipped parts
    make_questions(full_text, read_text, scope)  -> [{"question": ..., "model_answer": ...}, ...]
        scope "unread": questions about what was skipped
        scope "whole":  questions about the whole text
    grade_answer(full_text, question, user_answer) -> {"score": 0-10, "explanation": ...}

Every function takes provider="claude" | "gemini" | "openai".
Install only the ones you use, and set that provider's API key:
    claude   uv add anthropic      ANTHROPIC_API_KEY
    gemini   uv add google-genai   GEMINI_API_KEY
    openai   uv add openai         OPENAI_API_KEY
"""
from __future__ import annotations

import json

from eyetracking.core.reading import mark_unread, unread_word_count

# Fast & cheap model per provider (checked 2026-10-08). Change one line to switch.
MODELS = {
    "claude": "claude-haiku-4-5",
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-6-luna",
}

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