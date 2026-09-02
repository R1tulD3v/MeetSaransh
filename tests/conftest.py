"""Shared fixtures.

Two rules hold across the whole suite:

1. **No network.** Every provider call is mocked with respx, and the embedding model is
   forced unavailable by default (loading it downloads ~90 MB and takes ~50 s). Tests
   that need the dense retrieval path use the `fake_embeddings` fixture, which supplies
   deterministic vectors instead.
2. **No shared state.** Each test gets its own temporary data directory and database,
   and the rate limiter and migration cache are reset between tests, so tests can run in
   any order and in parallel.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import respx
from fastapi.testclient import TestClient

from app import config, embeddings, observability, security, storage
from app.main import app

REPO_ROOT = Path(__file__).resolve().parent.parent
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


# ------------------------------------------------------------------------ isolation
@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point every filesystem path at a per-test temp directory."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "AUDIO_DIR", data_dir / "audio")
    monkeypatch.setattr(config, "DB_PATH", data_dir / "meetsaransh.db")
    # The bundled sample is real fixture data, so keep pointing at the repo copy.
    monkeypatch.setattr(config, "SAMPLE_DIR", REPO_ROOT / "data" / "sample")

    storage.reset_migration_cache()
    security.limiter.reset()
    # Off by default: a suite that hammers the same endpoint would otherwise trip it.
    # tests/test_middleware.py turns it back on explicitly.
    monkeypatch.setattr(config, "RATE_LIMIT_ENABLED", False)
    observability.configure_logging()
    yield
    storage.reset_migration_cache()
    security.limiter.reset()


@pytest.fixture(autouse=True)
def no_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the dense model unavailable so no test downloads or loads it."""
    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_load_failed", True)


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic stand-in vectors, so the dense path is testable without the model.

    The vector is a bag-of-characters histogram: texts sharing vocabulary land near each
    other under cosine similarity, which is the only property the retrieval code needs.
    """

    def _vector(text: str) -> np.ndarray:
        vec = np.zeros(embeddings.DIM, dtype=np.float32)
        for token in text.lower().split():
            vec[hash(token) % embeddings.DIM] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(
        embeddings,
        "embed_texts",
        lambda texts: np.vstack([_vector(t) for t in texts]) if texts else None,
    )
    monkeypatch.setattr(embeddings, "embed_one", lambda text: _vector(text))


# ----------------------------------------------------------------------- API key state
@pytest.fixture
def with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key-not-real")


@pytest.fixture
def without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "GROQ_API_KEY", "")


# ------------------------------------------------------------------------------ client
@pytest.fixture
def client(without_api_key: None) -> Iterator[TestClient]:
    """Test client with NO API key -- the safe default: a test that forgets to mock a
    provider call fails loudly instead of silently trying to reach Groq."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def keyed_client(with_api_key: None) -> Iterator[TestClient]:
    """Test client WITH a (fake) key, for exercising the provider-backed paths."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def groq_mock() -> Iterator[respx.MockRouter]:
    """Intercept outbound Groq calls while letting TestClient's own requests through.

    respx patches httpx globally, and TestClient *is* an httpx client, so its requests
    to http://testserver have to be explicitly passed through or every API test would
    deadlock against the mock router.
    """
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        yield router


# ------------------------------------------------------------------------ audio fixtures
def mp3_bytes(size: int = 2048) -> bytes:
    """A minimal byte string that passes the ID3 magic-byte check."""
    return b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * max(0, size - 10)


def wav_bytes(size: int = 2048) -> bytes:
    """A well-formed-enough RIFF/WAVE header followed by silence."""
    payload = b"\x00" * max(0, size - 44)
    return (
        b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
        b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 16000, 2, 16)
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )


@pytest.fixture
def sample_segments() -> list[dict]:
    """A short, deterministic transcript used by the chunking and retrieval tests."""
    return [
        {"start": 0.0, "end": 5.0, "text": "Welcome everyone to the quarterly planning meeting."},
        {"start": 5.0, "end": 11.0, "text": "Priya will own the payment gateway migration."},
        {"start": 11.0, "end": 18.0, "text": "Rahul raised a concern about the checkout latency."},
        {"start": 18.0, "end": 25.0, "text": "We agreed to ship the release on the fifteenth."},
        {"start": 25.0, "end": 31.0, "text": "Hiring for two backend engineers stays open."},
    ]


# --------------------------------------------------------------------- provider payloads
def chat_completion(content: str) -> dict:
    """The subset of Groq's chat-completions response that the client reads."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def verbose_transcription(segments: list[dict]) -> dict:
    """The subset of Groq's verbose_json ASR response that the client reads."""
    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "duration": segments[-1]["end"] if segments else 0.0,
    }


SUMMARY_JSON = {
    "tldr": "The team planned the quarterly release.",
    "key_decisions": [{"decision": "Ship on the fifteenth", "timestamp": "00:18"}],
    "action_items": [
        {
            "task": "Migrate the payment gateway",
            "owner": "Priya",
            "due": "Next Friday",
            "timestamp": "00:05",
        }
    ],
    "open_questions": ["Who reviews the latency fix?"],
    "topics": [{"title": "Release scope", "summary": "Agreed the cut line.", "timestamp": "00:00"}],
}
