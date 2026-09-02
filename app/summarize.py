"""Summarization via Groq's chat completions endpoint.

Takes a (timestamped) transcript, returns the structured JSON summary defined in
prompts.py. Defensive JSON parsing: even in json_object mode, we guard against a
malformed response rather than crashing the request.
"""

from __future__ import annotations

import json
from typing import Any

from . import config, llm, prompts


class SummarizationError(RuntimeError):
    """Raised when the LLM call or its output cannot be used."""


# The keys we always hand back to the frontend, so rendering never sees a missing field.
# Explicitly Any-valued: the template mixes a string with lists of objects.
_EMPTY_SUMMARY: dict[str, Any] = {
    "tldr": "",
    "key_decisions": [],
    "action_items": [],
    "open_questions": [],
    "topics": [],
}


def summarize(title: str, timestamped_transcript: str) -> dict:
    """Generate a structured summary from a transcript."""
    if not config.has_api_key():
        raise SummarizationError("No GROQ_API_KEY configured for summarization.")
    if not timestamped_transcript.strip():
        raise SummarizationError("Transcript is empty; nothing to summarize.")

    # temperature 0.2 -> faithful, less inventive (grounding matters here).
    try:
        content = llm.chat(
            prompts.build_messages(title, timestamped_transcript),
            json_mode=True,
            temperature=0.2,
        )
    except llm.LLMError as exc:
        raise SummarizationError(str(exc)) from exc

    return normalize_summary(_parse_json(content))


def _parse_json(content: str) -> dict:
    """Parse the model's JSON, tolerating stray markdown fences if they slip in."""
    content = content.strip()
    if content.startswith("```"):
        # Strip ```json ... ``` fences defensively.
        content = content.strip("`")
        content = content[content.find("{") : content.rfind("}") + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        # Last resort: extract the outermost {...} block.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise SummarizationError("LLM did not return valid JSON.") from exc


def normalize_summary(data: Any) -> dict[str, Any]:
    """Ensure every expected key exists with the right type, so the UI is robust.

    Takes `Any` rather than `dict` on purpose: the input is whatever an LLM returned,
    so the non-dict case is a real code path, not a type error to be silenced.
    """
    out: dict[str, Any] = dict(_EMPTY_SUMMARY)
    if not isinstance(data, dict):
        return out
    out["tldr"] = str(data.get("tldr", "") or "")
    out["open_questions"] = [str(q) for q in _as_list(data.get("open_questions"))]
    out["key_decisions"] = [
        {"decision": str(d.get("decision", "")), "timestamp": str(d.get("timestamp", ""))}
        for d in _as_list(data.get("key_decisions"))
        if isinstance(d, dict)
    ]
    out["action_items"] = [
        {
            "task": str(a.get("task", "")),
            "owner": str(a.get("owner", "Unassigned") or "Unassigned"),
            "due": str(a.get("due", "Not specified") or "Not specified"),
            "timestamp": str(a.get("timestamp", "")),
        }
        for a in _as_list(data.get("action_items"))
        if isinstance(a, dict)
    ]
    out["topics"] = [
        {
            "title": str(t.get("title", "")),
            "summary": str(t.get("summary", "")),
            "timestamp": str(t.get("timestamp", "")),
        }
        for t in _as_list(data.get("topics"))
        if isinstance(t, dict)
    ]
    return out


def _as_list(value) -> list:
    return value if isinstance(value, list) else []
