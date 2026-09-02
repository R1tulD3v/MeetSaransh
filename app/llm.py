"""Thin, shared client for Groq's chat completions endpoint (OpenAI-compatible).

Both the summarizer and the RAG chat call the same endpoint, so the transport,
error mapping, and retry live here in one place.
"""

from __future__ import annotations

import time

import httpx

from . import config, observability
from .observability import get_logger

log = get_logger("meetsaransh.llm")


def _record(outcome: str, elapsed: float) -> None:
    """Count and time every provider call, so cost and latency are observable."""
    if config.METRICS_ENABLED:
        observability.PROVIDER_CALLS.labels("llm", outcome).inc()
        observability.PROVIDER_DURATION.labels("llm").observe(elapsed)


class LLMError(RuntimeError):
    """Raised when the chat completion call fails or returns an unusable response."""


def chat(
    messages: list[dict], *, json_mode: bool = False, temperature: float = 0.2, max_retries: int = 2
) -> str:
    """Send a chat completion and return the assistant message content.

    Retries with exponential backoff on 429 (rate limit) and transient network errors,
    which is exactly the "production thinking" the free-tier rate limits demand.
    """
    if not config.has_api_key():
        raise LLMError("No GROQ_API_KEY configured.")

    url = f"{config.GROQ_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}", "Content-Type": "application/json"}
    body: dict = {"model": config.LLM_MODEL, "messages": messages, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    last_err: str = "unknown error"
    for attempt in range(max_retries + 1):
        started = time.monotonic()
        try:
            resp = httpx.post(url, headers=headers, json=body, timeout=config.HTTP_TIMEOUT_SECONDS)
        except httpx.HTTPError as exc:
            _record("network_error", time.monotonic() - started)
            last_err = f"network error: {exc}"
            log.warning("LLM network error", extra={"attempt": attempt, "error": str(exc)})
            _backoff(attempt, max_retries)
            continue
        elapsed = time.monotonic() - started

        if resp.status_code == 200:
            _record("success", elapsed)
            try:
                return resp.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError) as exc:
                raise LLMError(f"Unexpected LLM response shape: {exc}") from exc

        if resp.status_code == 429 and attempt < max_retries:
            _record("rate_limited", elapsed)
            last_err = "rate limited (429)"
            log.warning("LLM rate limited; backing off", extra={"attempt": attempt})
            _backoff(attempt, max_retries)
            continue

        _record("error", elapsed)
        raise LLMError(_explain_http_error(resp))

    raise LLMError(f"LLM call failed after {max_retries + 1} attempts: {last_err}")


def _backoff(attempt: int, max_retries: int) -> None:
    if attempt < max_retries:
        time.sleep(1.5 * (2**attempt))  # 1.5s, 3s, ...


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
