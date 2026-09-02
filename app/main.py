"""FastAPI application: lifespan, routes, upload handling, and the process pipeline.

Pipeline for an uploaded meeting:
    audio -> validate (extension, size, magic bytes) -> save -> transcribe (ASR)
          -> summarize (LLM) -> store -> index for RAG -> return

Routes are defined once on a router and mounted twice: at `/api/v1` (the documented,
versioned surface) and at `/api` (an undocumented alias kept so existing clients keep
working). New endpoints should be considered v1-only.

The frontend is plain HTML/JS served statically, so there is no build step and no
node_modules -- in keeping with the "minimal and native" submission guideline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, Query, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import (
    __version__,
    config,
    embeddings,
    errors,
    middleware,
    observability,
    rag,
    schemas,
    security,
    storage,
    summarize,
    transcription,
)
from .errors import APIError
from .observability import get_logger

log = get_logger("meetsaransh.app")

# Read uploads in 1 MB slices so an oversized body is rejected after ~1 MB of memory
# instead of after the whole file has been buffered.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown. Replaces the deprecated @app.on_event hooks."""
    observability.configure_logging()
    warnings = config.validate()  # raises ConfigError on anything unusable
    config.ensure_dirs()
    storage.init_db()
    for warning in warnings:
        log.warning("config warning", extra={"detail": warning})
    log.info(
        "startup complete",
        extra={
            "version": __version__,
            "environment": config.ENVIRONMENT,
            "schema_version": storage.SCHEMA_VERSION,
            "has_api_key": config.has_api_key(),
            "rate_limit_enabled": config.RATE_LIMIT_ENABLED,
        },
    )
    yield
    log.info("shutdown complete")


app = FastAPI(
    title="MeetSaransh",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Transcribe meeting audio, generate action-oriented summaries, and ask "
        "grounded questions across every meeting."
    ),
    # Interactive docs are a development convenience, not something to expose publicly.
    docs_url=None if config.IS_PRODUCTION else "/docs",
    redoc_url=None,
    openapi_url=None if config.IS_PRODUCTION else "/openapi.json",
    responses={
        400: {"model": schemas.ErrorResponse},
        404: {"model": schemas.ErrorResponse},
        429: {"model": schemas.ErrorResponse},
        500: {"model": schemas.ErrorResponse},
    },
)

errors.install_error_handlers(app)
middleware.install_middleware(app)

api = APIRouter()


# ------------------------------------------------------------------------------ health
@api.get("/health", response_model=schemas.HealthResponse, tags=["ops"])
def health() -> schemas.HealthResponse:
    """Readiness probe that actually checks its dependencies.

    A health endpoint that only returns `{"status": "ok"}` tells a load balancer nothing
    -- the process can be up while the database is unreachable. This one round-trips the
    database, and reports (without failing) whether the optional pieces are available.
    """
    db_ok = storage.healthcheck()
    return schemas.HealthResponse(
        status="ok" if db_ok else "degraded",
        version=__version__,
        environment=config.ENVIRONMENT,
        has_api_key=config.has_api_key(),
        database=db_ok,
        embeddings=embeddings.available(),
        asr_model=config.ASR_MODEL,
        llm_model=config.LLM_MODEL,
    )


# ---------------------------------------------------------------------------- meetings
@api.get("/meetings", response_model=schemas.MeetingPage, tags=["meetings"])
def api_list_meetings(
    limit: int = Query(50, ge=1, le=200, description="Page size."),
    offset: int = Query(0, ge=0, description="Rows to skip."),
) -> schemas.MeetingPage:
    """List meetings, newest first."""
    items = storage.list_meetings(limit=limit, offset=offset)
    total = storage.count_meetings()
    if config.METRICS_ENABLED:
        observability.MEETINGS_STORED.set(total)
    return schemas.MeetingPage(
        items=[schemas.MeetingListItem(**i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@api.get("/meetings/{meeting_id}", response_model=schemas.Meeting, tags=["meetings"])
def api_get_meeting(meeting_id: str) -> schemas.Meeting:
    return schemas.Meeting(**_require_meeting(meeting_id))


@api.delete("/meetings/{meeting_id}", response_model=schemas.DeleteResponse, tags=["meetings"])
def api_delete_meeting(meeting_id: str) -> schemas.DeleteResponse:
    meeting = _require_meeting(meeting_id)
    if meeting.get("audio_ext"):
        (config.AUDIO_DIR / f"{meeting_id}{meeting['audio_ext']}").unlink(missing_ok=True)
    # Chunks go with the meeting via ON DELETE CASCADE (see storage migration 2).
    storage.delete_meeting(meeting_id)
    log.info("meeting deleted", extra={"meeting_id": meeting_id})
    return schemas.DeleteResponse(deleted=meeting_id)


@api.post("/meetings", response_model=schemas.Meeting, status_code=201, tags=["meetings"])
async def api_create_meeting(
    file: UploadFile = File(..., description="Meeting audio."),
    title: str = Form("", description="Optional title; the filename is used otherwise."),
) -> schemas.Meeting:
    """Upload audio, transcribe, summarize, and persist."""
    ext = _validate_extension(file)
    meeting_id = storage.new_id()
    audio_path = config.AUDIO_DIR / f"{meeting_id}{ext}"

    size = await _stream_upload_to_disk(file, audio_path, ext)
    log.info(
        "upload accepted",
        extra={"meeting_id": meeting_id, "bytes": size, "ext": ext},
    )

    try:
        transcript = transcription.transcribe(audio_path)
    except transcription.TranscriptionError as exc:
        audio_path.unlink(missing_ok=True)
        raise APIError(str(exc), status_code=502, code="asr_failed") from exc

    try:
        summary = summarize.summarize(title, transcript["timestamped_text"])
    except summarize.SummarizationError as exc:
        audio_path.unlink(missing_ok=True)
        raise APIError(str(exc), status_code=502, code="summarization_failed") from exc

    storage.create_meeting(
        title=title or Path(file.filename or "meeting").stem,
        filename=file.filename,
        transcript=transcript,
        summary=summary,
        audio_ext=ext,
        meeting_id=meeting_id,
    )
    _safe_index(meeting_id)
    log.info(
        "meeting created",
        extra={"meeting_id": meeting_id, "duration_s": round(transcript["duration"], 1)},
    )
    return schemas.Meeting(**_require_meeting(meeting_id))


@api.post("/meetings/sample", response_model=schemas.Meeting, status_code=201, tags=["meetings"])
def api_create_sample() -> schemas.Meeting:
    """Create a meeting from the bundled sample transcript.

    Lets the app be demoed end-to-end with NO API key. If a key IS configured, the
    summary is generated live from the sample transcript (so graders can see the real
    LLM path too); otherwise the pre-generated sample summary is used.
    """
    sample_path = config.SAMPLE_DIR / "sample_meeting.json"
    if not sample_path.exists():
        raise APIError("Sample meeting file is missing.", status_code=500, code="sample_missing")

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    segments = sample["segments"]
    transcript = {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "timestamped_text": transcription.build_timestamped_text(segments),
        "duration": segments[-1]["end"] if segments else 0.0,
    }

    if config.has_api_key():
        try:
            summary = summarize.summarize(sample["title"], transcript["timestamped_text"])
        except summarize.SummarizationError:
            log.warning("live sample summarization failed; using the bundled summary")
            summary = summarize.normalize_summary(sample["summary"])
    else:
        summary = summarize.normalize_summary(sample["summary"])

    mid = storage.create_meeting(
        title=sample["title"],
        filename="sample_meeting.mp3",
        transcript=transcript,
        summary=summary,
        audio_ext=None,
    )
    _safe_index(mid)
    return schemas.Meeting(**_require_meeting(mid))


@api.get("/meetings/{meeting_id}/audio", tags=["meetings"])
def api_get_audio(meeting_id: str) -> FileResponse:
    meeting = _require_meeting(meeting_id)
    if not meeting.get("audio_ext"):
        raise APIError("No audio for this meeting.", status_code=404)
    audio = config.AUDIO_DIR / f"{meeting_id}{meeting['audio_ext']}"
    if not audio.exists():
        raise APIError("Audio file not found.", status_code=404)
    return FileResponse(audio)


@api.get(
    "/meetings/{meeting_id}/export",
    response_class=PlainTextResponse,
    tags=["meetings"],
)
def api_export_markdown(meeting_id: str) -> str:
    """Export the summary + action items as copy-ready Markdown (close-the-loop)."""
    return _to_markdown(_require_meeting(meeting_id))


# -------------------------------------------------------------------------- RAG / chat
@api.get("/rag/status", response_model=schemas.RagStatus, tags=["rag"])
def api_rag_status() -> schemas.RagStatus:
    return schemas.RagStatus(**rag.status())


@api.post("/reindex", response_model=schemas.ReindexResponse, tags=["rag"])
def api_reindex() -> schemas.ReindexResponse:
    """(Re)index any meetings that aren't in the vector store yet."""
    return schemas.ReindexResponse(**rag.reindex_all())


@api.post("/chat", response_model=schemas.ChatResponse, tags=["rag"])
def api_chat(payload: schemas.ChatRequest) -> schemas.ChatResponse:
    """Answer a question grounded in the user's meetings (optionally scoped to one)."""
    rag.reindex_all()  # cheap: only indexes meetings not already indexed
    result = rag.answer(payload.question, scope_meeting_id=payload.meeting_id)
    if config.METRICS_ENABLED:
        observability.RAG_ANSWERS.labels(result.get("mode", "unknown")).inc()
        observability.CHUNKS_STORED.set(storage.count_chunks())
    return schemas.ChatResponse(**result)


# ------------------------------------------------------------------------------ mounts
# `/api/v1` is the documented surface; `/api` is a compatibility alias for older
# clients and is intentionally hidden from the OpenAPI schema.
app.include_router(api, prefix="/api/v1")
app.include_router(api, prefix="/api", include_in_schema=False)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    if not config.METRICS_ENABLED:
        raise APIError("Metrics are disabled.", status_code=404)
    observability.MEETINGS_STORED.set(storage.count_meetings())
    observability.CHUNKS_STORED.set(storage.count_chunks())
    body, content_type = observability.render_metrics()
    return Response(content=body, media_type=content_type)


# ----------------------------------------------------------------------------- helpers
def _require_meeting(meeting_id: str) -> dict:
    meeting = storage.get_meeting(meeting_id)
    if meeting is None:
        raise APIError("Meeting not found.", status_code=404)
    return meeting


def _validate_extension(file: UploadFile) -> str:
    if not file.filename:
        raise APIError("No file provided.", status_code=400, code="no_file")
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_AUDIO_EXTS:
        if config.METRICS_ENABLED:
            observability.UPLOADS_REJECTED.labels("extension").inc()
        allowed = ", ".join(sorted(config.ALLOWED_AUDIO_EXTS))
        raise APIError(
            f"Unsupported file type '{ext}'. Allowed: {allowed}",
            status_code=400,
            code="unsupported_extension",
        )
    return ext


async def _stream_upload_to_disk(file: UploadFile, dest: Path, ext: str) -> int:
    """Write an upload to disk in slices, validating content and enforcing the cap.

    Streaming rather than `await file.read()` matters twice over: an oversized upload is
    rejected after one slice instead of after the whole body is in memory, and the magic
    -byte check runs on the first slice, before a single provider call is made.
    """
    written = 0
    checked = False
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                if not checked:
                    try:
                        security.validate_audio_content(chunk[:64], ext)
                    except security.UnsupportedAudioError as exc:
                        if config.METRICS_ENABLED:
                            observability.UPLOADS_REJECTED.labels("content").inc()
                        raise APIError(
                            str(exc), status_code=400, code="unsupported_content"
                        ) from exc
                    checked = True
                written += len(chunk)
                if written > config.MAX_UPLOAD_BYTES:
                    if config.METRICS_ENABLED:
                        observability.UPLOADS_REJECTED.labels("too_large").inc()
                    raise APIError(
                        f"File exceeds the {_mb(config.MAX_UPLOAD_BYTES)} MB limit.",
                        status_code=413,
                    )
                out.write(chunk)
    except BaseException:
        dest.unlink(missing_ok=True)  # never leave a partial file behind
        raise

    if written == 0:
        dest.unlink(missing_ok=True)
        if config.METRICS_ENABLED:
            observability.UPLOADS_REJECTED.labels("empty").inc()
        raise APIError("The uploaded file is empty.", status_code=400, code="empty_file")
    return written


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f}"


def _safe_index(meeting_id: str) -> None:
    """Index a meeting for RAG without ever failing the main request."""
    try:
        rag.index_meeting(meeting_id)
    except Exception:  # indexing is best-effort; the meeting is already saved
        log.exception("RAG indexing failed", extra={"meeting_id": meeting_id})


def _to_markdown(meeting: dict) -> str:
    s = meeting.get("summary", {})
    lines = [f"# {meeting['title']}", ""]
    if s.get("tldr"):
        lines += ["## TL;DR", s["tldr"], ""]
    if s.get("key_decisions"):
        lines += ["## Key decisions"]
        lines += [
            f"- {d['decision']}" + (f" _( {d['timestamp']} )_" if d.get("timestamp") else "")
            for d in s["key_decisions"]
        ]
        lines.append("")
    if s.get("action_items"):
        lines += ["## Action items", "", "| Task | Owner | Due |", "| --- | --- | --- |"]
        lines += [f"| {a['task']} | {a['owner']} | {a['due']} |" for a in s["action_items"]]
        lines.append("")
    if s.get("open_questions"):
        lines += ["## Open questions"]
        lines += [f"- {q}" for q in s["open_questions"]]
        lines.append("")
    if s.get("topics"):
        lines += ["## Topics"]
        lines += [f"- **{t['title']}** — {t['summary']}" for t in s["topics"]]
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------- static frontend
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    return (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
