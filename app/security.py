"""Security layer: upload content validation, rate limiting, and response headers.

Three concerns live here because they share one motivation -- this service spends real
money per request (paid ASR + LLM APIs) and accepts arbitrary binary uploads from the
internet, so both the input and the request rate have to be bounded.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from . import config

# ------------------------------------------------------------------ magic-byte sniffing
# An extension check alone is worthless: anything can be renamed to `.mp3`. These are the
# container signatures we actually accept, matched against the first bytes of the upload.
#
# Format notes:
#   MP3   - either an ID3 tag ("ID3") or a raw MPEG frame sync (11 set bits).
#   WAV   - RIFF....WAVE  (the 4-byte size field sits between the two markers).
#   M4A / MP4 - an ISO-BMFF box: bytes 4..8 are "ftyp".
#   WebM  - the Matroska EBML header.
#   OGG   - "OggS" page header.
#   FLAC  - "fLaC" stream marker.

_AUDIO_MIN_BYTES = 12  # enough to cover every signature below


def sniff_audio_format(head: bytes) -> str | None:
    """Identify an audio/video container from its leading bytes, or None if unknown."""
    if len(head) < _AUDIO_MIN_BYTES:
        return None

    if head[:3] == b"ID3":
        return "mp3"
    # MPEG audio frame sync: 0xFF followed by three set bits in the next byte.
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[4:8] == b"ftyp":
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"fLaC":
        return "flac"
    return None


# Which sniffed formats are acceptable for a given uploaded extension. Deliberately
# permissive within a family (`.m4a`/`.mp4`/`.mpeg` are all ISO-BMFF or MPEG streams,
# and `.ogg` may carry an Opus or Vorbis payload) but never across families.
_EXT_TO_FORMATS: dict[str, set[str]] = {
    ".mp3": {"mp3"},
    ".mpga": {"mp3"},
    ".mpeg": {"mp3", "mp4"},
    ".wav": {"wav"},
    ".m4a": {"mp4"},
    ".mp4": {"mp4"},
    ".webm": {"webm"},
    ".ogg": {"ogg"},
    ".flac": {"flac"},
}


class UnsupportedAudioError(ValueError):
    """Raised when an upload's actual content is not an accepted audio container."""


def validate_audio_content(head: bytes, ext: str) -> str:
    """Verify that `head` really is audio, and that it matches the claimed extension.

    Returns the detected format name. Raises UnsupportedAudioError otherwise.
    """
    detected = sniff_audio_format(head)
    if detected is None:
        raise UnsupportedAudioError(
            "That file does not look like audio. Upload a real audio recording "
            "(mp3, wav, m4a, ogg, flac, webm)."
        )
    allowed = _EXT_TO_FORMATS.get(ext.lower(), set())
    if detected not in allowed:
        raise UnsupportedAudioError(
            f"File content is '{detected}' but the name claims '{ext}'. "
            "Rename the file to match its real format and try again."
        )
    return detected


# ------------------------------------------------------------------------ rate limiting
class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    A sliding window (rather than a fixed window) because a fixed window lets a client
    send 2x the limit across a window boundary -- which, on an endpoint that triggers a
    paid transcription, is a real cost bug rather than a theoretical one.

    In-process by design: this app currently runs as a single process. The interface is
    the part that matters -- swapping the deques for a Redis sorted set is a drop-in
    change once there is more than one worker.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float) -> tuple[bool, int, float]:
        """Record a hit for `key`. Returns (allowed, remaining, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - window
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(0.0, bucket[0] + window - now)
                return False, 0, retry_after
            bucket.append(now)
            return True, limit - len(bucket), 0.0

    def reset(self) -> None:
        """Drop all state (used between tests)."""
        with self._lock:
            self._hits.clear()

    def prune(self, older_than: float) -> int:
        """Drop buckets with no hits inside `older_than` seconds. Returns keys removed.

        Without this, the dict grows one entry per distinct client forever -- a slow
        memory leak that only shows up in production.
        """
        cutoff = time.monotonic() - older_than
        with self._lock:
            stale = [k for k, b in self._hits.items() if not b or b[-1] <= cutoff]
            for k in stale:
                del self._hits[k]
        return len(stale)


limiter = RateLimiter()


def limit_for_path(method: str, path: str) -> int:
    """Per-endpoint request budget. Costly endpoints get the tightest caps."""
    if method == "POST" and path.rstrip("/").endswith("/meetings"):
        return config.RATE_LIMIT_UPLOAD  # runs ASR + LLM: slowest and most expensive
    if method == "POST" and (path.endswith("/chat") or path.endswith("/meetings/sample")):
        return config.RATE_LIMIT_CHAT
    return config.RATE_LIMIT_DEFAULT


# ---------------------------------------------------------------------- response headers
def security_headers(*, is_production: bool) -> dict[str, str]:
    """Baseline hardening headers applied to every response.

    The CSP can afford to be strict because the frontend has no inline script and no
    inline style -- everything is an external file under /static.
    """
    headers = {
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "media-src 'self' blob:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'"
        ),
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), interest-cohort=()",
        "Cross-Origin-Opener-Policy": "same-origin",
    }
    if is_production:
        # Only meaningful over HTTPS, and actively harmful on a local http:// dev server.
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return headers
