"""End-to-end API tests through the ASGI stack, with every provider call mocked.

Upload *processing* lives in tests/test_jobs.py; what is tested here is the request
surface -- validation, pagination, the error envelope, export, and chat.
"""

from __future__ import annotations

import httpx
import pytest

from app import config, jobs, storage
from tests.conftest import GROQ_ASR_URL, GROQ_CHAT_URL, chat_completion, mp3_bytes, wav_bytes


def _upload(client, data: bytes | None = None, filename: str = "standup.mp3", title: str = ""):
    return client.post(
        "/api/v1/meetings",
        files={"file": (filename, data if data is not None else mp3_bytes(), "audio/mpeg")},
        data={"title": title},
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


def test_the_openapi_schema_marks_protected_routes_as_needing_a_token(client):
    """Auth as a dependency means the contract advertises itself."""
    schema = client.get("/openapi.json").json()
    assert "security" in schema["paths"]["/api/v1/meetings"]["get"]
    assert "security" not in schema["paths"]["/api/v1/health"]["get"]


# ---------------------------------------------------------------------- list & pagination
def test_listing_an_empty_store_returns_an_empty_page(client):
    assert client.get("/api/v1/meetings").json() == {
        "items": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }


def test_pagination_walks_the_full_set_without_gaps_or_repeats(client, user_id):
    for i in range(5):
        storage.create_meeting(user_id=user_id, title=f"M{i}", filename=None)

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


def test_the_list_carries_processing_state(client, user_id):
    """The sidebar renders a spinner from this, so it has to come back in the list."""
    storage.create_meeting(user_id=user_id, title="Working", filename=None, status="queued")
    item = client.get("/api/v1/meetings").json()["items"][0]
    assert item["status"] == "queued"
    assert item["error"] is None


# --------------------------------------------------------------------- upload validation
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


def test_a_rejected_upload_leaves_no_file_and_no_row(keyed_client, monkeypatch):
    """A partial file from an aborted upload would leak disk forever."""
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 4096)
    _upload(keyed_client, data=mp3_bytes(20_000))

    assert list(config.AUDIO_DIR.glob("*")) == []
    assert keyed_client.get("/api/v1/meetings").json()["total"] == 0


def test_an_empty_file_is_rejected(keyed_client):
    response = _upload(keyed_client, data=b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_a_wav_upload_is_accepted_and_keeps_its_extension(keyed_client, groq_mock):
    groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(500, json={}))
    meeting_id = _upload(keyed_client, data=wav_bytes(), filename="a.wav").json()["id"]
    assert keyed_client.get(f"/api/v1/meetings/{meeting_id}").json()["audio_ext"] == ".wav"


# ---------------------------------------------------------------------------- sample flow
def test_the_sample_meeting_works_with_no_api_key(client):
    """The app must be fully demonstrable offline -- that is the whole fallback story."""
    response = client.post("/api/v1/meetings/sample")
    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Q3 Mobile App Planning Sync"
    assert body["status"] == "done"  # nothing to transcribe, so no queue
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


def test_deleting_a_meeting_removes_its_row_and_chunks(client):
    meeting_id = client.post("/api/v1/meetings/sample").json()["id"]
    assert storage.count_chunks() > 0

    assert client.delete(f"/api/v1/meetings/{meeting_id}").json() == {"deleted": meeting_id}
    assert client.get(f"/api/v1/meetings/{meeting_id}").status_code == 404
    assert storage.count_chunks() == 0


def test_deleting_twice_is_a_404_not_a_crash(client):
    meeting_id = client.post("/api/v1/meetings/sample").json()["id"]
    assert client.delete(f"/api/v1/meetings/{meeting_id}").status_code == 200
    assert client.delete(f"/api/v1/meetings/{meeting_id}").status_code == 404


def test_audio_is_404_for_a_meeting_that_has_none(client):
    meeting_id = client.post("/api/v1/meetings/sample").json()["id"]  # sample has no file
    assert client.get(f"/api/v1/meetings/{meeting_id}/audio").status_code == 404


def test_audio_is_404_when_the_row_survives_but_the_file_is_gone(keyed_client, groq_mock):
    """The real scenario: a failed job deletes the recording but keeps the row, so the
    meeting is still listable and still explains itself -- its audio just isn't there."""
    groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(500, json={}))
    meeting_id = _upload(keyed_client).json()["id"]
    jobs.wait_for_idle(timeout=15)

    assert not (config.AUDIO_DIR / f"{meeting_id}.mp3").exists()
    assert keyed_client.get(f"/api/v1/meetings/{meeting_id}").json()["audio_ext"] == ".mp3"
    assert keyed_client.get(f"/api/v1/meetings/{meeting_id}/audio").status_code == 404


# --------------------------------------------------------------------------------- export
def test_markdown_export_contains_every_summary_section(client):
    meeting_id = client.post("/api/v1/meetings/sample").json()["id"]
    markdown = client.get(f"/api/v1/meetings/{meeting_id}/export").text

    assert markdown.startswith("# Q3 Mobile App Planning Sync")
    assert "## TL;DR" in markdown
    assert "## Action items" in markdown
    assert "| Task | Owner | Due |" in markdown


def test_exporting_a_missing_meeting_is_a_404(client):
    assert client.get("/api/v1/meetings/nope/export").status_code == 404


# ------------------------------------------------------------------------------ RAG chat
def test_chat_returns_a_grounded_answer_with_citations(keyed_client, groq_mock):
    groq_mock.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("Priya owns the migration."))
    )
    keyed_client.post("/api/v1/meetings/sample")

    body = keyed_client.post(
        "/api/v1/chat", json={"question": "Who owns the payment bug fix?"}
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


def test_chat_can_be_scoped_to_a_single_meeting(client, user_id):
    scoped_to = client.post("/api/v1/meetings/sample").json()["id"]
    storage.create_meeting(
        user_id=user_id,
        title="Other",
        filename=None,
        transcript={
            "text": "budget",
            "segments": [{"start": 0.0, "end": 3.0, "text": "The budget review is next week."}],
            "duration": 3.0,
        },
        summary={},
    )
    body = client.post("/api/v1/chat", json={"question": "budget", "meeting_id": scoped_to}).json()
    assert all(c["meeting_id"] == scoped_to for c in body["citations"])


def test_scoping_to_a_meeting_that_does_not_exist_is_a_404(client):
    """Rejected rather than silently widened to 'all meetings', which would answer from
    a different set than the caller asked about."""
    client.post("/api/v1/meetings/sample")
    response = client.post("/api/v1/chat", json={"question": "hi", "meeting_id": "nope"})
    assert response.status_code == 404


def test_chat_with_no_meetings_says_so(client):
    body = client.post("/api/v1/chat", json={"question": "anything"}).json()
    assert body["mode"] == "empty"


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}])
def test_malformed_chat_requests_are_rejected_by_the_schema(client, payload):
    response = client.post("/api/v1/chat", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


# ------------------------------------------------------------------------------- reindex
def test_reindex_picks_up_meetings_stored_outside_the_upload_path(client, user_id):
    storage.create_meeting(
        user_id=user_id,
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
def test_the_frontend_is_served_at_the_root(anon_client):
    """Unauthenticated on purpose: the page itself is what renders the sign-in form."""
    response = anon_client.get("/")
    assert response.status_code == 200
    assert "MeetSaransh" in response.text


def test_static_assets_are_served(anon_client):
    assert anon_client.get("/static/app.js").status_code == 200
    assert anon_client.get("/static/style.css").status_code == 200


def test_the_frontend_and_the_backend_agree_on_the_api_version(anon_client):
    """A cheap guard against the two drifting: the shipped JS must call /api/v1."""
    app_js = anon_client.get("/static/app.js").text
    assert 'const API = "/api/v1"' in app_js
    assert '"/api/meetings"' not in app_js  # no leftover unversioned call sites


def test_asset_urls_are_cache_busted_by_version(anon_client):
    """A cached app.js from the previous release against a moved-on API looks exactly
    like a broken deploy, and is invisible to anyone testing with an empty cache."""
    from app import __version__

    html = anon_client.get("/").text
    assert f"/static/app.js?v={__version__}" in html
    assert f"/static/style.css?v={__version__}" in html


def test_the_app_shell_is_always_revalidated(anon_client):
    """The shell carries the versioned asset URLs, so caching it defeats the busting."""
    assert "no-cache" in anon_client.get("/").headers["cache-control"]
