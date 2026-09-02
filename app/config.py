"""Central configuration. All tunables live here so nothing is hard-coded elsewhere.

Every value is overridable by environment variable, which is what makes the same image
runnable in dev, CI, and production without a code change. `validate()` runs at startup
so a misconfiguration fails at boot with a clear message instead of at first request.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from this file's package).
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# --- Environment ---
# "development" relaxes CORS and enables docs; "production" tightens both.
ENVIRONMENT: str = _env_str("ENVIRONMENT", "development").lower()
IS_PRODUCTION: bool = ENVIRONMENT == "production"

# --- Provider (Groq is OpenAI-compatible; swapping providers = swap these two URLs) ---
GROQ_API_KEY: str = _env_str("GROQ_API_KEY")
GROQ_BASE_URL: str = _env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

ASR_MODEL: str = _env_str("ASR_MODEL", "whisper-large-v3-turbo")
LLM_MODEL: str = _env_str("LLM_MODEL", "openai/gpt-oss-120b")

# --- RAG ("Ask your meetings") ---
EMBED_MODEL: str = _env_str("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_TARGET_WORDS: int = _env_int("CHUNK_TARGET_WORDS", 130)  # ~170-220 tokens
CHUNK_OVERLAP_WORDS: int = _env_int("CHUNK_OVERLAP_WORDS", 30)  # overlap keeps context
RAG_TOP_K: int = _env_int("RAG_TOP_K", 6)  # excerpts fed to the LLM per question
# Cheap pre-filter threshold. bge-small can't cleanly separate loosely-related from
# off-topic in the ~0.5-0.65 band, so this gate only hard-refuses the CLEARLY unrelated
# (France/pizza score <0.50). Anything above it, OR sharing a keyword with a meeting, is
# passed to the LLM, whose grounded prompt is the real "not discussed" guard.
RAG_MIN_SCORE: float = _env_float("RAG_MIN_SCORE", 0.50)

# --- Upload limits ---
# Groq's transcription endpoint caps a single file at 25 MB. We reject earlier with a
# clear message rather than letting the provider return an opaque error.
MAX_UPLOAD_BYTES: int = _env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
ALLOWED_AUDIO_EXTS: set[str] = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".webm",
    ".ogg",
    ".flac",
}

# --- Authentication ---
# There is deliberately NO hardcoded default secret. A shipped default is a shipped
# forgery key: anyone who reads the source can mint tokens for any account. In
# production a missing secret is a hard startup failure; in development an ephemeral
# random one is generated, which merely means tokens do not survive a restart.
JWT_SECRET: str = _env_str("JWT_SECRET") or secrets.token_urlsafe(48)
JWT_SECRET_WAS_GENERATED: bool = not _env_str("JWT_SECRET")
# Short access token, long refresh token: a leaked access token expires quickly, and a
# refresh token can be revoked server-side because its id is stored.
ACCESS_TOKEN_MINUTES: int = _env_int("ACCESS_TOKEN_MINUTES", 30)
REFRESH_TOKEN_DAYS: int = _env_int("REFRESH_TOKEN_DAYS", 7)
MIN_JWT_SECRET_LENGTH: int = 32

# --- Background jobs ---
# Transcription runs off the request thread. Worker count is small on purpose: each job
# is dominated by a network call to the ASR provider, and the provider rate-limits us
# long before local CPU becomes the constraint.
JOB_WORKERS: int = _env_int("JOB_WORKERS", 2)
# A per-user ceiling on queued/running jobs. Without it one account can fill the queue
# and spend the whole provider budget.
MAX_ACTIVE_JOBS_PER_USER: int = _env_int("MAX_ACTIVE_JOBS_PER_USER", 3)

# --- Rate limiting ---
# These endpoints call *paid* provider APIs, so they are capped per client. Limits are
# expressed as "N requests per WINDOW seconds".
RATE_LIMIT_ENABLED: bool = _env_bool("RATE_LIMIT_ENABLED", True)
RATE_LIMIT_WINDOW_SECONDS: int = _env_int("RATE_LIMIT_WINDOW_SECONDS", 60)
RATE_LIMIT_UPLOAD: int = _env_int("RATE_LIMIT_UPLOAD", 5)  # transcription: slow + costly
RATE_LIMIT_CHAT: int = _env_int("RATE_LIMIT_CHAT", 20)  # chat: cheap but LLM-billed
RATE_LIMIT_DEFAULT: int = _env_int("RATE_LIMIT_DEFAULT", 120)  # everything else
# Login and registration are brute-force targets, and each login runs an intentionally
# expensive KDF -- so an uncapped login endpoint is both a credential risk and a CPU DoS.
RATE_LIMIT_AUTH: int = _env_int("RATE_LIMIT_AUTH", 10)
# X-Forwarded-For is trivially spoofable unless a proxy you control overwrites it, so
# honouring it is opt-in: enable this only when the app really sits behind one.
TRUST_PROXY_HEADERS: bool = _env_bool("TRUST_PROXY_HEADERS", False)

# --- CORS ---
# Comma-separated allowlist. Empty means same-origin only (the default, and correct for
# the bundled frontend, which is served by this same app).
CORS_ORIGINS: list[str] = [o for o in _env_str("CORS_ORIGINS").split(",") if o.strip()]

# --- Observability ---
LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO").upper()
# JSON logs in production (machine-parseable); human-readable lines in development.
LOG_JSON: bool = _env_bool("LOG_JSON", IS_PRODUCTION)
METRICS_ENABLED: bool = _env_bool("METRICS_ENABLED", True)

# --- Storage paths ---
# DATA_DIR is env-overridable so a container can mount a volume and tests can use tmp.
DATA_DIR: Path = Path(_env_str("DATA_DIR") or (BASE_DIR / "data"))
AUDIO_DIR: Path = DATA_DIR / "audio"
DB_PATH: Path = Path(_env_str("DB_PATH") or (DATA_DIR / "meetsaransh.db"))
SAMPLE_DIR: Path = Path(_env_str("SAMPLE_DIR") or (BASE_DIR / "data" / "sample"))
STATIC_DIR: Path = BASE_DIR / "static"

# Network timeout for provider calls (transcription of long audio can take a while).
HTTP_TIMEOUT_SECONDS: float = _env_float("HTTP_TIMEOUT_SECONDS", 300.0)


class ConfigError(RuntimeError):
    """Raised at startup when the configuration cannot produce a working app."""


def has_api_key() -> bool:
    """True when a Groq key is configured. Drives the app's offline/sample fallback."""
    return bool(GROQ_API_KEY)


def ensure_dirs() -> None:
    """Create the runtime data directories if they don't exist yet."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def validate() -> list[str]:
    """Fail fast on unusable config; return non-fatal warnings for the startup log.

    Raises ConfigError for anything that makes the app incorrect (e.g. a production
    deployment with wildcard CORS). Everything merely degraded -- such as a missing
    API key, which still leaves sample mode working -- comes back as a warning.
    """
    if CHUNK_OVERLAP_WORDS >= CHUNK_TARGET_WORDS:
        raise ConfigError(
            f"CHUNK_OVERLAP_WORDS ({CHUNK_OVERLAP_WORDS}) must be smaller than "
            f"CHUNK_TARGET_WORDS ({CHUNK_TARGET_WORDS}), or chunking cannot advance."
        )
    if RAG_TOP_K < 1:
        raise ConfigError("RAG_TOP_K must be at least 1.")
    if MAX_UPLOAD_BYTES < 1024:
        raise ConfigError("MAX_UPLOAD_BYTES is implausibly small.")
    if IS_PRODUCTION and "*" in CORS_ORIGINS:
        raise ConfigError("CORS_ORIGINS must not be '*' in production.")
    if IS_PRODUCTION and JWT_SECRET_WAS_GENERATED:
        raise ConfigError(
            "JWT_SECRET must be set in production. A generated secret changes on every "
            "restart, which would log every user out, and cannot be shared across "
            'replicas. Generate one with: python -c "import secrets; '
            'print(secrets.token_urlsafe(48))"'
        )
    if not JWT_SECRET_WAS_GENERATED and len(JWT_SECRET) < MIN_JWT_SECRET_LENGTH:
        raise ConfigError(
            f"JWT_SECRET must be at least {MIN_JWT_SECRET_LENGTH} characters; "
            f"a short secret is brute-forceable offline."
        )
    if ACCESS_TOKEN_MINUTES < 1 or REFRESH_TOKEN_DAYS < 1:
        raise ConfigError("Token lifetimes must be at least one minute / one day.")
    if JOB_WORKERS < 1:
        raise ConfigError("JOB_WORKERS must be at least 1.")

    warnings: list[str] = []
    if not has_api_key():
        warnings.append(
            "GROQ_API_KEY is not set - transcription and written answers are disabled; "
            "sample mode and retrieval-only chat still work."
        )
    if IS_PRODUCTION and not RATE_LIMIT_ENABLED:
        warnings.append("Rate limiting is disabled in production - paid endpoints are uncapped.")
    if JWT_SECRET_WAS_GENERATED:
        warnings.append(
            "JWT_SECRET is not set - a random one was generated, so all sessions end "
            "when this process restarts. Set JWT_SECRET in .env to keep them."
        )
    return warnings
