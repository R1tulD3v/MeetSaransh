"""Unhandled-crash behaviour and the embeddings module's graceful degradation."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import embeddings, errors, storage
from app.main import app
from tests.conftest import TEST_EMAIL, TEST_PASSWORD, sign_up


# ------------------------------------------------------------------- unhandled crashes
@pytest.fixture
def crashing_client(without_api_key, monkeypatch):
    """A client that returns the server's 500 instead of re-raising the exception.

    Signed in first, then broken: an unauthenticated request is rejected at the auth
    dependency and never reaches the code under test.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        sign_up(c, TEST_EMAIL, TEST_PASSWORD)
        monkeypatch.setattr(storage, "count_meetings", _boom)
        yield c


def _boom(*_args, **_kwargs):
    raise RuntimeError("database on fire: user=alice password=hunter2")


def test_an_unhandled_crash_returns_the_error_envelope(crashing_client):
    response = crashing_client.get("/api/v1/meetings")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


def test_a_crash_never_leaks_internals_to_the_client(crashing_client):
    """Stack traces and anything they quote go to the logs, not to the browser."""
    body = crashing_client.get("/api/v1/meetings").text
    assert "hunter2" not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body


def test_a_crash_still_yields_a_traceable_request_id(crashing_client):
    """The client gets nothing useful about the failure except the thread to pull."""
    response = crashing_client.get("/api/v1/meetings")
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_a_crash_is_logged_with_its_traceback(crashing_client, caplog):
    with caplog.at_level("ERROR"):
        crashing_client.get("/api/v1/meetings")
    assert any("hunter2" in r.getMessage() or r.exc_info for r in caplog.records)


def test_an_unknown_path_is_a_clean_404_envelope(client):
    response = client.get("/api/v1/no-such-endpoint")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_a_wrong_method_is_reported_as_method_not_allowed(client):
    response = client.put("/api/v1/health")
    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


# --------------------------------------------------------------------------- APIError
def test_api_error_derives_a_code_from_its_status():
    assert errors.APIError("nope", status_code=404).code == "not_found"


def test_an_explicit_code_wins_over_the_status_default():
    assert errors.APIError("nope", status_code=502, code="asr_failed").code == "asr_failed"


def test_an_unmapped_status_gets_a_generic_code():
    assert errors.APIError("odd", status_code=418).code == "error"


# ------------------------------------------------------------------------- embeddings
def test_cosine_scores_rank_the_identical_vector_highest():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
    scores = embeddings.cosine_scores(query, matrix)

    assert np.argmax(scores) == 1
    assert scores[1] == pytest.approx(1.0)
    assert scores[0] == pytest.approx(0.0)


def test_cosine_scores_are_magnitude_invariant():
    """Cosine measures direction; a longer chunk vector must not score higher for it."""
    query = np.array([1.0, 1.0], dtype=np.float32)
    unit = embeddings.cosine_scores(query, np.array([[1.0, 1.0]], dtype=np.float32))
    scaled = embeddings.cosine_scores(query, np.array([[100.0, 100.0]], dtype=np.float32))
    assert unit[0] == pytest.approx(scaled[0])


def test_cosine_scores_on_an_empty_matrix_returns_empty():
    assert embeddings.cosine_scores(np.array([1.0], dtype=np.float32), np.empty((0, 1))).size == 0


def test_a_zero_vector_does_not_divide_by_zero():
    scores = embeddings.cosine_scores(
        np.zeros(3, dtype=np.float32), np.zeros((2, 3), dtype=np.float32)
    )
    assert np.all(np.isfinite(scores))


def test_embedding_calls_return_none_when_the_model_is_unavailable():
    """The whole RAG feature degrades to lexical-only rather than erroring out."""
    assert embeddings.available() is False  # forced off by the autouse fixture
    assert embeddings.embed_texts(["anything"]) is None
    assert embeddings.embed_one("anything") is None


def test_embedding_an_empty_list_is_none_not_a_crash(monkeypatch):
    monkeypatch.setattr(embeddings, "_load_failed", False)
    monkeypatch.setattr(embeddings, "_model", object())
    assert embeddings.embed_texts([]) is None


def test_a_model_that_fails_to_load_is_only_attempted_once(monkeypatch):
    """Retrying a broken model load on every query would make each request slow."""
    attempts = []

    class _Failing:
        def __init__(self, *_args, **_kwargs):
            attempts.append(1)
            raise RuntimeError("no ONNX runtime here")

    monkeypatch.setattr(embeddings, "_model", None)
    monkeypatch.setattr(embeddings, "_load_failed", False)
    monkeypatch.setitem(
        __import__("sys").modules, "fastembed", type("m", (), {"TextEmbedding": _Failing})
    )

    assert embeddings.available() is False
    assert embeddings.available() is False
    assert len(attempts) == 1
