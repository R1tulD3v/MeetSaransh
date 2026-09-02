"""Background transcription pipeline.

Uploading a meeting used to run ASR and summarization inside the request, so a 40-minute
recording held an HTTP connection open for minutes and any client timeout destroyed work
that had already been paid for. Now the upload returns immediately with a `queued`
meeting, and a worker pool drains the queue while the client polls for status.

**Why a thread pool and a status column rather than Celery or RQ.** The work is one long
network call to the ASR provider, so it is I/O-bound: threads are the right primitive and
a broker would add an operational dependency for no throughput. The database is the queue
of record, which is what makes crash recovery possible at startup. If this ever needs to
run across several machines, the swap is `submit()` -> `queue.enqueue()`; the state
machine and the recovery logic stay exactly as they are.

State machine:

    queued ---> processing ---> done
                    |
                    +---------> error   (message stored on the meeting)

An interrupted `processing` job is failed explicitly at the next startup, because a
meeting that stays `processing` forever is a spinner that never resolves.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from . import config, observability, rag, storage, summarize, transcription
from .observability import get_logger, request_id_var

log = get_logger("meetsaransh.jobs")

# Stages are surfaced to the UI, so the user sees "Transcribing" rather than a bare
# spinner for two minutes.
STAGE_TRANSCRIBING = "transcribing"
STAGE_SUMMARIZING = "summarizing"
STAGE_INDEXING = "indexing"

_executor: ThreadPoolExecutor | None = None
_lock = threading.Lock()
# Kept only so tests (and a graceful shutdown) can wait for in-flight work.
_pending: set[Future] = set()


class QuotaExceeded(Exception):
    """Raised when a user already has the maximum number of jobs in flight."""


def start_workers() -> None:
    """Create the worker pool. Called from the app lifespan."""
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=config.JOB_WORKERS, thread_name_prefix="meetsaransh-job"
            )
            log.info("job workers started", extra={"workers": config.JOB_WORKERS})


def stop_workers(wait: bool = True) -> None:
    """Shut the pool down, letting in-flight jobs finish so no work is lost."""
    global _executor
    with _lock:
        executor, _executor = _executor, None
    if executor is not None:
        executor.shutdown(wait=wait)
        log.info("job workers stopped")


def recover_interrupted_jobs() -> int:
    """Fail any job left mid-flight by a crash. Returns how many were recovered.

    Deliberately does NOT retry: the previous attempt may have already spent an ASR
    call, and silently re-spending a user's provider budget on restart is worse than
    telling them plainly that it failed and letting them decide.
    """
    stranded = storage.find_interrupted_meetings()
    for meeting in stranded:
        storage.set_meeting_status(
            meeting["id"],
            "error",
            error="Processing was interrupted by a server restart. Please upload again.",
        )
        log.warning("recovered interrupted job", extra={"meeting_id": meeting["id"]})
    if stranded:
        _observe_queue_depth()
    return len(stranded)


def enqueue(meeting_id: str, audio_path: Path, title: str, user_id: str) -> None:
    """Submit a queued meeting for processing.

    The per-user quota is checked here rather than in the route so that every caller
    gets it, and so the check and the submit cannot drift apart.
    """
    active = storage.count_active_jobs(user_id)
    # The meeting being enqueued is itself already 'queued', hence the > rather than >=.
    if active > config.MAX_ACTIVE_JOBS_PER_USER:
        raise QuotaExceeded(
            f"You already have {config.MAX_ACTIVE_JOBS_PER_USER} recordings processing. "
            "Wait for one to finish before uploading another."
        )

    start_workers()
    assert _executor is not None  # start_workers guarantees it
    # Carry the request id into the worker thread: a ContextVar is per-thread, so
    # without this the job's log lines would be orphaned from the upload that caused it.
    request_id = request_id_var.get()
    future = _executor.submit(_run_job, meeting_id, audio_path, title, request_id)
    _pending.add(future)
    future.add_done_callback(_pending.discard)
    _observe_queue_depth()
    log.info("job enqueued", extra={"meeting_id": meeting_id, "user_id": user_id})


def wait_for_idle(timeout: float = 30.0) -> None:
    """Block until every submitted job has finished. For tests and shutdown only."""
    from concurrent.futures import wait as futures_wait

    futures_wait(list(_pending), timeout=timeout)


def _run_job(meeting_id: str, audio_path: Path, title: str, request_id: str) -> None:
    """Transcribe, summarize, and index one meeting. Never raises."""
    token = request_id_var.set(request_id)
    try:
        storage.set_meeting_status(meeting_id, "processing", stage=STAGE_TRANSCRIBING)
        _observe_queue_depth()

        transcript = transcription.transcribe(audio_path)

        storage.set_meeting_status(meeting_id, "processing", stage=STAGE_SUMMARIZING)
        summary = summarize.summarize(title, transcript["timestamped_text"])

        storage.complete_meeting(meeting_id, transcript=transcript, summary=summary)

        # Indexing is best-effort: the meeting is already saved and viewable, so an
        # embedding failure must not turn a successful transcription into an error.
        try:
            storage.set_meeting_status(meeting_id, "done", stage=STAGE_INDEXING)
            rag.index_meeting(meeting_id)
        except Exception:
            log.exception("RAG indexing failed", extra={"meeting_id": meeting_id})
        finally:
            storage.set_meeting_status(meeting_id, "done")

        if config.METRICS_ENABLED:
            observability.JOBS_COMPLETED.labels("done").inc()
        log.info(
            "job complete",
            extra={
                "meeting_id": meeting_id,
                "duration_s": round(transcript.get("duration", 0.0), 1),
            },
        )

    except (transcription.TranscriptionError, summarize.SummarizationError) as exc:
        _fail(meeting_id, audio_path, str(exc), expected=True)
    except Exception as exc:
        log.exception("job crashed", extra={"meeting_id": meeting_id})
        _fail(
            meeting_id,
            audio_path,
            "Processing failed unexpectedly. The error has been logged.",
            expected=False,
            detail=repr(exc),
        )
    finally:
        request_id_var.reset(token)
        _observe_queue_depth()


def _fail(
    meeting_id: str,
    audio_path: Path,
    message: str,
    *,
    expected: bool,
    detail: str | None = None,
) -> None:
    """Mark a job failed and clean up its audio.

    The row is kept rather than deleted: a user who uploaded a file and saw it vanish
    has no idea what happened, whereas a failed row carries the reason.
    """
    audio_path.unlink(missing_ok=True)
    storage.set_meeting_status(meeting_id, "error", error=message)
    if config.METRICS_ENABLED:
        observability.JOBS_COMPLETED.labels("error").inc()
    log.warning(
        "job failed",
        extra={"meeting_id": meeting_id, "expected": expected, "detail": detail or message},
    )


def _observe_queue_depth() -> None:
    if not config.METRICS_ENABLED:
        return
    try:
        with storage._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM meetings "
                "WHERE status IN ('queued', 'processing') GROUP BY status"
            ).fetchall()
        counts = {r["status"]: r["n"] for r in rows}
        observability.JOB_QUEUE_DEPTH.labels("queued").set(counts.get("queued", 0))
        observability.JOB_QUEUE_DEPTH.labels("processing").set(counts.get("processing", 0))
    except Exception:  # a metrics failure must never break the pipeline
        log.debug("could not sample queue depth", exc_info=True)
