"""Central configuration. All tunables live here so nothing is hard-coded elsewhere."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from this file's package).
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# --- Provider (Groq is OpenAI-compatible; swapping providers = swap these two URLs) ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

ASR_MODEL: str = os.getenv("ASR_MODEL", "whisper-large-v3-turbo").strip()
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile").strip()

# --- RAG ("Ask your meetings") ---
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5").strip()
CHUNK_TARGET_WORDS: int = 130     # ~170-220 tokens; small enough for precise citations
CHUNK_OVERLAP_WORDS: int = 30     # overlap keeps context across chunk boundaries
RAG_TOP_K: int = 6                # excerpts fed to the LLM per question
# Cheap pre-filter threshold. bge-small can't cleanly separate loosely-related from
# off-topic in the ~0.5-0.65 band, so this gate only hard-refuses the CLEARLY unrelated
# (France/pizza score <0.50). Anything above it, OR sharing a keyword with a meeting, is
# passed to the LLM, whose grounded prompt is the real "not discussed" guard.
RAG_MIN_SCORE: float = 0.50

# --- Upload limits ---
# Groq's transcription endpoint caps a single file at 25 MB. We reject earlier with a
# clear message rather than letting the provider return an opaque error.
MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm", ".ogg", ".flac"}

# --- Storage paths ---
DATA_DIR: Path = BASE_DIR / "data"
AUDIO_DIR: Path = DATA_DIR / "audio"
DB_PATH: Path = DATA_DIR / "meetsaransh.db"
SAMPLE_DIR: Path = DATA_DIR / "sample"
STATIC_DIR: Path = BASE_DIR / "static"

# Network timeout for provider calls (transcription of long audio can take a while).
HTTP_TIMEOUT_SECONDS: float = 300.0


def has_api_key() -> bool:
    """True when a Groq key is configured. Drives the app's offline/sample fallback."""
    return bool(GROQ_API_KEY)


def ensure_dirs() -> None:
    """Create the runtime data directories if they don't exist yet."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
