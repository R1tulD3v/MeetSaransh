"""End-to-end API tests through the ASGI stack, with every provider call mocked."""

from __future__ import annotations

import json

import httpx
import pytest

from app import config, storage
from tests.conftest import (
    GROQ_ASR_URL,
    GROQ_CHAT_URL,
    SUMMARY_JSON,
    chat_completion,
    mp3_bytes,
    verbose_transcription,
    wav_bytes,
)

ASR_SEGMENTS = [
    {"start": 0.0, "end": 6.0, "text": "Priya will own the payment gateway migration."},
    {"start": 6.0, "end": 12.0, "text": "Rahul flagged the checkout latency regression."},
]


def _upload(client, data: bytes | None = None, filename: str = "standup.mp3", title: str = ""):
    return client.post(
        "/api/v1/meetings",
        files={"file": (filename, data if data is not None else mp3_bytes(), "audio/mpeg")},
        data={"title": title},
    )


def _mock_full_pipeline(groq_mock) -> None:
    groq_mock.post(GROQ_ASR_URL).mock(
        return_value=httpx.Response(200, json=verbose_transcription(ASR_SEGMENTS))
    )
    groq_mock.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion(json.dumps(SUMMARY_JSON)))
    )


# ------------------------------------------------------------------------------ health
def test_health_reports_real_dependency_state(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["database"] is True  # actually round-tripped, not hard-coded
    assert body["has_api_key"] is False
    assert body["embeddings"] is False  # model forced off in tests
    assert body["environment"] == config.ENVIRONMENT


def test_health_degrades_rather_than_crashing_when_the_database_is_broken(client, monkeypatch):
    monkeypatch.setattr(storage, "healthcheck", lambda: False)
    body = client.get("/api/v1/health").json()
    assert body["status"] == "degraded"
    assert body["database"] is False


def test_the_unversioned_alias_still_works(client):
    """Existing clients pinned to /api must not break when /api/v1 is introduced."""
    assert client.get("/api/health").json() == client.get("/api/v1/health").json()


def test_only_the_versioned_routes_are_documented(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/meetings" in paths
    assert "/api/meetings" not in paths


# ---------------------------------------------------------------------- list & pagination
def test_listing_an_empty_store_returns_an_empty_page(client):
    assert client.get("/api/v1/meetings").json() == {
        "items": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }


def test_pagination_walks_the_full_set_without_gaps_or_repeats(client):
    for i in range(5):
        storage.create_meeting(
            title=f"M{i}",
            filename=None,
            transcript={"text": "", "segments": [], "duration": 0},
            summary={},
        )
    first = client.get("/api/v1/meetings?limit=2&offset=0").json()
    second = client.get("/api/v1/meetings?limit=2&offset=2").json()
    third = client.get("/api/v1/meetings?limit=2&offset=4").json()

    assert first["total"] == 5
    ids = [m["id"] for page in (first, second, third) for m in page["items"]]
    assert len(ids) == 5
    assert len(set(ids)) == 5


@pytest.mark.parametrize("query", ["limit=0", "limit=500", "offset=-1", "limit=abc"])
def test_invalid_pagination_is_rejected_with_the_error_envelope(client, query):
    response = client.get(f"/api/v1/meetings?{query}")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# --------------------------------------------------------------------------- upload flow
def test_a_full_upload_produces_a_summarized_indexed_meeting(keyed_client, groq_mock):
    _mock_full_pipeline(groq_mock)
    response = _upload(keyed_client, title="Q3 Planning")

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Q3 Planning"
    assert body["duration"] == 12.0
    assert len(body["segments"]) == 2
    assert body["summary"]["action_items"][0]["owner"] == "Priya"
    # Indexed for RAG as part of the same request.
    assert storage.count_chunks() > 0


def test_the_filename_is_used_when_no_title_is_given(keyed_client, groq_mock):
    _mock_full_pipeline(groq_mock)
    body = _upload(keyed_client, filename="weekly-standup.mp3").json()
    assert body["title"] == "weekly-standup"


def test_the_audio_file_is_persisted_and_downloadable(keyed_client, groq_mock):
    _mock_full_pipeline(groq_mock)
    mid = _upload(keyed_client, data=wav_bytes(), filename="a.wav").json()["id"]

    assert (config.AUDIO_DIR / f"{mid}.wav").exists()
    assert keyed_client.get(f"/api/v1/meetings/{mid}/audio").status_code == 200


@pytest.mark.parametrize("filename", ["notes.txt", "archive.zip", "script.py", "noextension"])
def test_disallowed_extensions_are_rejected(keyed_client, filename):
    response = _upload(keyed_client, filename=filename)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_extension"


def test_a_renamed_non_audio_file_is_rejected_before_any_provider_call(keyed_client, groq_mock):
    """The extension says mp3; the bytes say PDF. No money is spent finding out."""
    asr = groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(200, json={}))
    response = _upload(keyed_client, data=b"%PDF-1.7\n" + b"\x00" * 100)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_content"
    assert asr.call_count == 0


def test_an_oversized_upload_is_rejected_with_413(keyed_client, monkeypatch, groq_mock):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 4096)
    asr = groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(200, json={}))
    response = _upload(keyed_client, data=mp3_bytes(20_000))

    assert response.status_code == 413
    assert asr.call_count == 0


def test_a_rejected_upload_leaves_no_file_behind(keyed_client, monkeypatch):
    """A partially-written file from an aborted upload would leak disk forever."""
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 4096)
    _upload(keyed_client, data=mp3_bytes(20_000))
    assert list(config.AUDIO_DIR.glob("*")) == []


def test_an_empty_file_is_rejected(keyed_client):
    response = _upload(keyed_client, data=b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_an_asr_failure_returns_502_and_cleans_up_the_audio(keyed_client, groq_mock):
    groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(500, json={}))
    response = _upload(keyed_client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "asr_failed"
    assert list(config.AUDIO_DIR.glob("*")) == []
    assert storage.count_meetings() == 0  # nothing half-saved


def test_a_summarization_failure_after_a_good_transcript_saves_nothing(keyed_client, groq_mock):
    """Transcription succeeded and cost money, but a half-processed meeting is worse
    than none: the response says exactly which stage failed."""
    groq_mock.post(GROQ_ASR_URL).mock(
        return_value=httpx.Response(200, json=verbose_transcription(ASR_SEGMENTS))
    )
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(401, json={}))
    response = _upload(keyed_client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "summarization_failed"
    assert storage.count_meetings() == 0
    assert list(config.AUDIO_DIR.glob("*")) == []


def test_uploading_without_a_key_fails_with_a_helpful_message(client):
    response = _upload(client)
    assert response.status_code == 502
    assert "sample" in response.json()["error"]["message"].lower()


# ---------------------------------------------------------------------------- sample flow
def test_the_sample_meeting_works_with_no_api_key(client):
    """The app must be fully demonstrable offline -- that is the whole fallback story."""
    response = client.post("/api/v1/meetings/sample")
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Q3 Mobile App Planning Sync"
    assert body["summary"]["tldr"]
    assert body["segments"]
    assert storage.count_chunks() > 0


def test_the_sample_falls_back_to_its_bundled_summary_if_the_llm_fails(keyed_client, groq_mock):
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(500, json={}))
    response = keyed_client.post("/api/v1/meetings/sample")
    assert response.status_code == 201
    assert response.json()["summary"]["tldr"]  # bundled summary, not an error


def test_a_missing_sample_file_is_a_clear_server_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SAMPLE_DIR", tmp_path / "nothing-here")
    response = client.post("/api/v1/meetings/sample")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "sample_missing"


# --------------------------------------------------------------------------- read/delete
def test_fetching_a_missing_meeting_returns_the_error_envelope(client):
    response = client.get("/api/v1/meetings/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert error["request_id"]  # traceable back to the logs


def test_deleting_a_meeting_removes_its_row_audio_and_chunks(keyed_client, groq_mock):
    _mock_full_pipeline(groq_mock)
    mid = _upload(keyed_client).json()["id"]
    assert storage.count_chunks() > 0

    assert keyed_client.delete(f"/api/v1/meetings/{mid}").json() == {"deleted": mid}
    assert storage.get_meeting(mid) is None
    assert storage.count_chunks() == 0
    assert list(config.AUDIO_DIR.glob("*")) == []


def test_deleting_twice_is_a_404_not_a_crash(client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    assert client.delete(f"/api/v1/meetings/{mid}").status_code == 200
    assert client.delete(f"/api/v1/meetings/{mid}").status_code == 404


def test_audio_is_404_for_a_meeting_that_has_none(client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]  # sample has no audio file
    assert client.get(f"/api/v1/meetings/{mid}/audio").status_code == 404


def test_audio_is_404_when_the_row_survives_but_the_file_is_gone(keyed_client, groq_mock):
    _mock_full_pipeline(groq_mock)
    mid = _upload(keyed_client).json()["id"]
    (config.AUDIO_DIR / f"{mid}.mp3").unlink()
    assert keyed_client.get(f"/api/v1/meetings/{mid}/audio").status_code == 404


# --------------------------------------------------------------------------------- export
def test_markdown_export_contains_every_summary_section(client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    markdown = client.get(f"/api/v1/meetings/{mid}/export").text

    assert markdown.startswith("# Q3 Mobile App Planning Sync")
    assert "## TL;DR" in markdown
    assert "## Action items" in markdown
    assert "| Task | Owner | Due |" in markdown


def test_exporting_a_missing_meeting_is_a_404(client):
    assert client.get("/api/v1/meetings/nope/export").status_code == 404


# ------------------------------------------------------------------------------ RAG chat
def test_chat_returns_a_grounded_answer_with_citations(keyed_client, groq_mock):
    _mock_full_pipeline(groq_mock)
    _upload(keyed_client)
    groq_mock.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("Priya owns the migration."))
    )

    body = keyed_client.post(
        "/api/v1/chat", json={"question": "Who owns the payment gateway migration?"}
    ).json()

    assert body["mode"] == "answer"
    assert body["answer"] == "Priya owns the migration."
    assert body["citations"][0]["meeting_title"]
    assert body["citations"][0]["timestamp"]


def test_chat_refuses_a_question_the_meetings_never_touch(client):
    client.post("/api/v1/meetings/sample")
    body = client.post("/api/v1/chat", json={"question": "What is the capital of France?"}).json()
    assert body["mode"] == "refused"
    assert body["citations"] == []


def test_chat_without_a_key_returns_excerpts_instead_of_prose(client):
    client.post("/api/v1/meetings/sample")
    body = client.post("/api/v1/chat", json={"question": "payment bug"}).json()
    assert body["mode"] == "retrieval_only"
    assert body["answer"] is None
    assert body["citations"]


def test_chat_reports_a_provider_failure_but_still_returns_the_excerpts(keyed_client, groq_mock):
    # Mocked first: with a key configured, creating the sample also calls the LLM (and
    # falls back to the bundled summary when it fails, which is its own tested behaviour).
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(500, json={}))
    keyed_client.post("/api/v1/meetings/sample")
    body = keyed_client.post("/api/v1/chat", json={"question": "payment bug"}).json()

    assert body["mode"] == "error"
    assert body["citations"]  # degraded, not empty


def test_chat_can_be_scoped_to_a_single_meeting(client):
    a = client.post("/api/v1/meetings/sample").json()["id"]
    storage.create_meeting(
        title="Other",
        filename=None,
        transcript={
            "text": "budget",
            "segments": [{"start": 0.0, "end": 3.0, "text": "The budget review is next week."}],
            "duration": 3.0,
        },
        summary={},
    )
    body = client.post("/api/v1/chat", json={"question": "budget", "meeting_id": a}).json()
    assert all(c["meeting_id"] == a for c in body["citations"])


def test_chat_with_no_meetings_says_so(client):
    body = client.post("/api/v1/chat", json={"question": "anything"}).json()
    assert body["mode"] == "empty"


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}])
def test_malformed_chat_requests_are_rejected_by_the_schema(client, payload):
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# ------------------------------------------------------------------------------- reindex
def test_reindex_picks_up_meetings_stored_outside_the_upload_path(client):
    storage.create_meeting(
        title="Imported",
        filename=None,
        transcript={
            "text": "hello",
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello there"}],
            "duration": 2.0,
        },
        summary={},
    )
    body = client.post("/api/v1/reindex").json()
    assert body["newly_indexed"] == 1
    assert body["total_chunks"] > 0


def test_rag_status_reflects_what_is_indexed(client):
    client.post("/api/v1/meetings/sample")
    body = client.get("/api/v1/rag/status").json()
    assert body["indexed_meetings"] == 1
    assert body["total_chunks"] > 0


# ------------------------------------------------------------------------ static frontend
def test_the_frontend_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "MeetSaransh" in response.text


def test_static_assets_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_the_frontend_and_the_backend_agree_on_the_api_version(client):
    """A cheap guard against the two drifting: the shipped JS must call /api/v1."""
    app_js = client.get("/static/app.js").text
    assert 'const API = "/api/v1"' in app_js
    assert '"/api/meetings"' not in app_js  # no leftover unversioned call sites
