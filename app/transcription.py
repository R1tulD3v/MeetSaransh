"""Automatic Speech Recognition via Groq's Whisper endpoint (OpenAI-compatible).

We request `verbose_json` with segment timestamps so downstream code can:
  - show a timestamped, click-to-seek transcript, and
  - feed [mm:ss] markers into the summarizer for grounded citations.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx

from . import config, observability
from .observability import get_logger

log = get_logger("meetsaransh.asr")


class TranscriptionError(RuntimeError):
    """Raised when the ASR provider call fails."""


def _fmt_timestamp(seconds: float) -> str:
    """Seconds -> mm:ss (or h:mm:ss for long meetings)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def transcribe(audio_path: Path) -> dict:
    """Transcribe an audio file.

    Returns a dict:
        {
          "text": "<full plain transcript>",
          "segments": [{"start": float, "end": float, "text": str}, ...],
          "timestamped_text": "[mm:ss] line\n[mm:ss] line ...",
          "duration": float,   # seconds, best-effort
        }
    """
    if not config.has_api_key():
        raise TranscriptionError(
            "No GROQ_API_KEY configured. Set one in .env, or use 'Load sample meeting' "
            "to try the app without a key."
        )

    url = f"{config.GROQ_BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {config.GROQ_API_KEY}"}

    with audio_path.open("rb") as fh:
        files = {"file": (audio_path.name, fh, "application/octet-stream")}
        data = {
            "model": config.ASR_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        }
        started = time.monotonic()
        try:
            resp = httpx.post(
                url,
                headers=headers,
                files=files,
                data=data,
                timeout=config.HTTP_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:  # network-level failure
            _record("network_error", time.monotonic() - started)
            raise TranscriptionError(f"Could not reach the ASR provider: {exc}") from exc

    elapsed = time.monotonic() - started
    if resp.status_code != 200:
        _record("error", elapsed)
        raise TranscriptionError(_explain_http_error(resp))

    _record("success", elapsed)
    payload = resp.json()
    result = _normalize(payload)
    # The ratio of these two numbers is the app's headline performance claim, so it is
    # logged on every transcription rather than measured by hand once.
    log.info(
        "transcription complete",
        extra={
            "audio_seconds": round(result["duration"], 1),
            "wall_seconds": round(elapsed, 2),
            "segments": len(result["segments"]),
        },
    )
    return result


def _record(outcome: str, elapsed: float) -> None:
    if config.METRICS_ENABLED:
        observability.PROVIDER_CALLS.labels("asr", outcome).inc()
        observability.PROVIDER_DURATION.labels("asr").observe(elapsed)


def _normalize(payload: dict) -> dict:
    """Shape Groq's verbose_json response into our internal transcript dict."""
    raw_segments = payload.get("segments") or []
    segments: list[dict[str, Any]] = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip(),
        }
        for s in raw_segments
    ]
    full_text = str(payload.get("text") or " ".join(str(s["text"]) for s in segments)).strip()
    timestamped = build_timestamped_text(segments) if segments else full_text
    duration = segments[-1]["end"] if segments else float(payload.get("duration", 0.0) or 0.0)
    return {
        "text": full_text,
        "segments": segments,
        "timestamped_text": timestamped,
        "duration": duration,
    }


def build_timestamped_text(segments: list[dict]) -> str:
    """Render segments as '[mm:ss] text' lines for the LLM and the UI."""
    return "\n".join(f"[{_fmt_timestamp(s['start'])}] {s['text']}" for s in segments if s["text"])


def _explain_http_error(resp: httpx.Response) -> str:
    """Turn provider HTTP errors into actionable messages."""
    code = resp.status_code
    try:
        detail = resp.json().get("error", {}).get("message", resp.text)
    except Exception:
        detail = resp.text
    if code == 401:
        return "ASR provider rejected the API key (401). Check GROQ_API_KEY in your .env."
    if code == 429:
        return "ASR provider rate limit hit (429). Wait a moment and retry."
    if code == 413:
        return "Audio file too large for the ASR provider (413). Keep files under 25 MB."
    return f"ASR provider error {code}: {detail}"
