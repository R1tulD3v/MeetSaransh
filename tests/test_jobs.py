"""Background transcription: the state machine, failure handling, and crash recovery.

The provider is mocked, so these run in milliseconds and assert behaviour that would
otherwise only show up under a real multi-minute transcription.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app import config, jobs, storage
from tests.conftest import (
    GROQ_ASR_URL,
    GROQ_CHAT_URL,
    SUMMARY_JSON,
    chat_completion,
    mp3_bytes,
    verbose_transcription,
)

ASR_SEGMENTS = [
    {"start": 0.0, "end": 6.0, "text": "Priya will own the payment gateway migration."},
    {"start": 6.0, "end": 12.0, "text": "Rahul flagged the checkout latency regression."},
]


def _upload(client, filename: str = "standup.mp3", title: str = ""):
    return client.post(
        "/api/v1/meetings",
        files={"file": (filename, mp3_bytes(), "audio/mpeg")},
        data={"title": title},
    )


def _mock_pipeline(groq_mock) -> None:
    groq_mock.post(GROQ_ASR_URL).mock(
        return_value=httpx.Response(200, json=verbose_transcription(ASR_SEGMENTS))
    )
    groq_mock.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion(json.dumps(SUMMARY_JSON)))
    )


def _finish(client, meeting_id: str) -> dict:
    """Wait for the worker, then return the finished meeting."""
    jobs.wait_for_idle(timeout=15)
    return client.get(f"/api/v1/meetings/{meeting_id}").json()


# ------------------------------------------------------------------------ accept flow
def test_an_upload_is_accepted_immediately_with_202(keyed_client, groq_mock):
    """202, not 201: the row exists but the work has not happened yet."""
    _mock_pipeline(groq_mock)
    response = _upload(keyed_client, title="Q3 Planning")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] in {"queued", "processing", "done"}
    assert body["title"] == "Q3 Planning"
    assert body["poll_url"] == f"/api/v1/meetings/{body['id']}"


def test_the_meeting_is_pollable_the_instant_the_upload_returns(keyed_client, groq_mock):
    """No window where a client holds an id that 404s -- that would break polling."""
    _mock_pipeline(groq_mock)
    meeting_id = _upload(keyed_client).json()["id"]
    assert keyed_client.get(f"/api/v1/meetings/{meeting_id}").status_code == 200


def test_processing_completes_the_meeting(keyed_client, groq_mock):
    _mock_pipeline(groq_mock)
    meeting_id = _upload(keyed_client).json()["id"]

    finished = _finish(keyed_client, meeting_id)
    assert finished["status"] == "done"
    assert finished["stage"] is None
    assert finished["error"] is None
    assert finished["duration"] == 12.0
    assert len(finished["segments"]) == 2
    assert finished["summary"]["action_items"][0]["owner"] == "Priya"


def test_a_completed_meeting_is_indexed_for_rag(keyed_client, groq_mock):
    _mock_pipeline(groq_mock)
    meeting_id = _upload(keyed_client).json()["id"]
    _finish(keyed_client, meeting_id)

    assert storage.count_chunks() > 0
    assert keyed_client.get("/api/v1/rag/status").json()["indexed_meetings"] == 1


def test_the_filename_is_used_when_no_title_is_given(keyed_client, groq_mock):
    _mock_pipeline(groq_mock)
    body = _upload(keyed_client, filename="weekly-standup.mp3").json()
    assert body["title"] == "weekly-standup"


def test_the_audio_is_kept_and_streamable_after_processing(keyed_client, groq_mock):
    _mock_pipeline(groq_mock)
    meeting_id = _upload(keyed_client).json()["id"]
    _finish(keyed_client, meeting_id)

    assert (config.AUDIO_DIR / f"{meeting_id}.mp3").exists()
    assert keyed_client.get(f"/api/v1/meetings/{meeting_id}/audio").status_code == 200


# --------------------------------------------------------------------- failure paths
def test_an_asr_failure_marks_the_meeting_failed_with_a_reason(keyed_client, groq_mock):
    groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(500, json={}))
    meeting_id = _upload(keyed_client).json()["id"]

    failed = _finish(keyed_client, meeting_id)
    assert failed["status"] == "error"
    assert failed["error"]
    assert "500" in failed["error"]


def test_a_summarization_failure_is_reported_on_the_meeting(keyed_client, groq_mock):
    groq_mock.post(GROQ_ASR_URL).mock(
        return_value=httpx.Response(200, json=verbose_transcription(ASR_SEGMENTS))
    )
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(401, json={}))
    meeting_id = _upload(keyed_client).json()["id"]

    failed = _finish(keyed_client, meeting_id)
    assert failed["status"] == "error"
    assert "GROQ_API_KEY" in failed["error"]


def test_a_failed_meeting_is_kept_not_deleted(keyed_client, groq_mock):
    """A file that vanishes tells the user nothing; a failed row carries the reason."""
    groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(500, json={}))
    meeting_id = _upload(keyed_client).json()["id"]
    _finish(keyed_client, meeting_id)

    assert keyed_client.get("/api/v1/meetings").json()["total"] == 1


def test_a_failed_job_cleans_up_its_audio(keyed_client, groq_mock):
    """The recording cannot be retried from the server side, so keeping it only
    consumes disk for a file nothing will ever read."""
    groq_mock.post(GROQ_ASR_URL).mock(return_value=httpx.Response(500, json={}))
    meeting_id = _upload(keyed_client).json()["id"]
    _finish(keyed_client, meeting_id)

    assert list(config.AUDIO_DIR.glob("*")) == []


def test_an_unexpected_crash_is_caught_and_reported(keyed_client, groq_mock, monkeypatch):
    """A worker thread that dies silently leaves a permanent spinner."""

    def _explode(*_args, **_kwargs):
        raise RuntimeError("secret internal detail: db password hunter2")

    monkeypatch.setattr(jobs.transcription, "transcribe", _explode)
    meeting_id = _upload(keyed_client).json()["id"]

    failed = _finish(keyed_client, meeting_id)
    assert failed["status"] == "error"
    assert "hunter2" not in failed["error"]  # internals stay in the logs
    assert "unexpectedly" in failed["error"]


def test_uploading_without_an_api_key_fails_the_job_not_the_request(client):
    """The upload itself is still valid; it is the processing that cannot proceed."""
    accepted = _upload(client)
    assert accepted.status_code == 202

    failed = _finish(client, accepted.json()["id"])
    assert failed["status"] == "error"
    assert "sample" in failed["error"].lower()


def test_an_indexing_failure_does_not_fail_a_good_transcription(
    keyed_client, groq_mock, monkeypatch
):
    """Indexing is best-effort: the meeting is already saved and worth showing."""
    _mock_pipeline(groq_mock)
    monkeypatch.setattr(
        jobs.rag, "index_meeting", lambda _mid: (_ for _ in ()).throw(RuntimeError("no model"))
    )
    meeting_id = _upload(keyed_client).json()["id"]

    finished = _finish(keyed_client, meeting_id)
    assert finished["status"] == "done"
    assert finished["summary"]["tldr"]


# --------------------------------------------------------------------------- quotas
def test_a_user_cannot_exceed_their_active_job_quota(keyed_client, monkeypatch, groq_mock):
    """Without a ceiling, one account can fill the queue and spend the whole budget."""
    monkeypatch.setattr(config, "MAX_ACTIVE_JOBS_PER_USER", 1)
    # Never resolves during the test, so the first job stays in flight.
    monkeypatch.setattr(jobs.transcription, "transcribe", _block_forever)

    first = _upload(keyed_client)
    assert first.status_code == 202

    storage.set_meeting_status(first.json()["id"], "processing", stage="transcribing")
    second = _upload(keyed_client)

    assert second.status_code == 429
    assert second.json()["error"]["code"] == "job_quota_exceeded"
    _release_block()
    jobs.wait_for_idle(timeout=10)


def test_a_rejected_upload_leaves_no_row_and_no_file(keyed_client, monkeypatch):
    """The placeholder is rolled back, so a quota rejection is not a phantom meeting."""
    monkeypatch.setattr(config, "MAX_ACTIVE_JOBS_PER_USER", 0)
    response = _upload(keyed_client)

    assert response.status_code == 429
    assert keyed_client.get("/api/v1/meetings").json()["total"] == 0
    assert list(config.AUDIO_DIR.glob("*")) == []


def test_quotas_are_per_user_not_global(keyed_client, second_client, monkeypatch):
    """One busy account must not block everybody else."""
    monkeypatch.setattr(config, "MAX_ACTIVE_JOBS_PER_USER", 0)
    assert _upload(keyed_client).status_code == 429

    monkeypatch.setattr(config, "MAX_ACTIVE_JOBS_PER_USER", 3)
    assert _upload(second_client).status_code == 202
    jobs.wait_for_idle(timeout=10)


import threading  # noqa: E402 - used only by the blocking helpers below

_unblock = threading.Event()


def _block_forever(*_args, **_kwargs):
    _unblock.wait(timeout=10)
    raise RuntimeError("released")


def _release_block() -> None:
    _unblock.set()


@pytest.fixture(autouse=True)
def _reset_block():
    _unblock.clear()
    yield
    _unblock.set()


# ------------------------------------------------------------------ crash recovery
def test_a_job_interrupted_by_a_restart_is_failed_not_left_hanging():
    """A meeting stuck in 'processing' forever is a spinner that never resolves."""
    uid = storage.create_user(email="crash@example.com", password_hash="x")["id"]
    stuck = storage.create_meeting(
        user_id=uid, title="Interrupted", filename="a.mp3", status="processing"
    )

    assert jobs.recover_interrupted_jobs() == 1
    recovered = storage.get_meeting(stuck, uid)
    assert recovered["status"] == "error"
    assert "restart" in recovered["error"]


def test_recovery_does_not_retry_the_job():
    """The previous attempt may already have spent an ASR call; silently re-spending a
    user's provider budget on every restart is worse than telling them it failed."""
    uid = storage.create_user(email="crash@example.com", password_hash="x")["id"]
    storage.create_meeting(user_id=uid, title="Interrupted", filename="a.mp3", status="processing")

    jobs.recover_interrupted_jobs()
    assert storage.count_active_jobs(uid) == 0  # not re-queued


def test_recovery_leaves_queued_and_finished_meetings_alone():
    uid = storage.create_user(email="crash@example.com", password_hash="x")["id"]
    queued = storage.create_meeting(user_id=uid, title="q", filename=None, status="queued")
    done = storage.create_meeting(user_id=uid, title="d", filename=None, status="done")

    jobs.recover_interrupted_jobs()
    assert storage.get_meeting(queued, uid)["status"] == "queued"
    assert storage.get_meeting(done, uid)["status"] == "done"


def test_recovery_on_a_clean_database_is_a_no_op():
    assert jobs.recover_interrupted_jobs() == 0


# ---------------------------------------------------------------------- worker pool
def test_starting_workers_twice_reuses_the_same_pool():
    jobs.start_workers()
    first = jobs._executor
    jobs.start_workers()
    assert jobs._executor is first


def test_stopping_workers_is_safe_when_none_are_running():
    jobs.stop_workers()
    jobs.stop_workers()  # must not raise


def test_the_request_id_follows_the_job_into_the_worker_thread(keyed_client, groq_mock, caplog):
    """A ContextVar is per-thread, so without explicit propagation the job's log lines
    would be orphaned from the upload that caused them."""
    _mock_pipeline(groq_mock)
    with caplog.at_level("INFO"):
        response = _upload(keyed_client)
        request_id = response.headers["x-request-id"]
        jobs.wait_for_idle(timeout=15)

    job_records = [r for r in caplog.records if r.name == "meetsaransh.jobs"]
    assert any(getattr(r, "request_id", None) == request_id for r in job_records)


# ------------------------------------------------------------------------- metrics
def test_job_outcomes_are_counted(keyed_client, groq_mock):
    _mock_pipeline(groq_mock)
    meeting_id = _upload(keyed_client).json()["id"]
    _finish(keyed_client, meeting_id)

    body = keyed_client.get("/metrics").text
    assert 'meetsaransh_jobs_completed_total{outcome="done"}' in body


def test_queue_depth_is_gauged(keyed_client, groq_mock):
    _mock_pipeline(groq_mock)
    _finish(keyed_client, _upload(keyed_client).json()["id"])
    assert "meetsaransh_job_queue_depth" in keyed_client.get("/metrics").text
