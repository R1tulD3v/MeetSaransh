"""Summarization via Groq's chat completions endpoint.

Takes a (timestamped) transcript, returns the structured JSON summary defined in
prompts.py. Defensive JSON parsing: even in json_object mode, we guard against a
malformed response rather than crashing the request.
"""

from __future__ import annotations

import json

import httpx

from . import config, prompts


class SummarizationError(RuntimeError):
    """Raised when the LLM call or its output cannot be used."""


# The keys we always hand back to the frontend, so rendering never sees a missing field.
_EMPTY_SUMMARY = {
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

    url = f"{config.GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.LLM_MODEL,
        "messages": prompts.build_messages(title, timestamped_transcript),
        "temperature": 0.2,  # low -> faithful, less inventive (grounding matters here)
        "response_format": {"type": "json_object"},
    }

    try:
        resp = httpx.post(url, headers=headers, json=body, timeout=config.HTTP_TIMEOUT_SECONDS)
    except httpx.HTTPError as exc:
        raise SummarizationError(f"Could not reach the LLM provider: {exc}") from exc

    if resp.status_code != 200:
        raise SummarizationError(_explain_http_error(resp))

    try:
        content = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise SummarizationError(f"Unexpected LLM response shape: {exc}") from exc

    return normalize_summary(_parse_json(content))


def _parse_json(content: str) -> dict:
    """Parse the model's JSON, tolerating stray markdown fences if they slip in."""
    content = content.strip()
    if content.startswith("```"):
        # Strip ```json ... ``` fences defensively.
        content = content.strip("`")
        content = content[content.find("{"): content.rfind("}") + 1]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Last resort: extract the outermost {...} block.
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(content[start:end + 1])
            except json.JSONDecodeError:
                pass
        raise SummarizationError("LLM did not return valid JSON.")


def normalize_summary(data: dict) -> dict:
    """Ensure every expected key exists with the right type, so the UI is robust."""
    out = dict(_EMPTY_SUMMARY)
    if not isinstance(data, dict):
        return out
    out["tldr"] = str(data.get("tldr", "") or "")
    out["open_questions"] = [str(q) for q in _as_list(data.get("open_questions"))]
    out["key_decisions"] = [
        {"decision": str(d.get("decision", "")), "timestamp": str(d.get("timestamp", ""))}
        for d in _as_list(data.get("key_decisions")) if isinstance(d, dict)
    ]
    out["action_items"] = [
        {
            "task": str(a.get("task", "")),
            "owner": str(a.get("owner", "Unassigned") or "Unassigned"),
            "due": str(a.get("due", "Not specified") or "Not specified"),
            "timestamp": str(a.get("timestamp", "")),
        }
        for a in _as_list(data.get("action_items")) if isinstance(a, dict)
    ]
    out["topics"] = [
        {"title": str(t.get("title", "")), "summary": str(t.get("summary", "")), "timestamp": str(t.get("timestamp", ""))}
        for t in _as_list(data.get("topics")) if isinstance(t, dict)
    ]
    return out


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _explain_http_error(resp: httpx.Response) -> str:
    code = resp.status_code
    try:
        detail = resp.json().get("error", {}).get("message", resp.text)
    except Exception:
        detail = resp.text
    if code == 401:
        return "LLM provider rejected the API key (401). Check GROQ_API_KEY in your .env."
    if code == 429:
        return "LLM provider rate limit hit (429). Wait a moment and retry."
    return f"LLM provider error {code}: {detail}"
