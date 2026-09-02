"""FastAPI application: lifespan, routes, upload handling, and the process pipeline.

Pipeline for an uploaded meeting:
    audio -> validate (extension, size, magic bytes) -> save -> create a `queued`
          meeting -> return 202 -> [background worker] transcribe -> summarize
          -> store -> index for RAG

The upload request returns as soon as the file is on disk; everything after that runs in
`app/jobs.py` while the client polls `GET /meetings/{id}` for status.

Every meeting route requires authentication and is scoped to the caller. Ownership is
enforced in SQL inside `app/storage.py`, so an endpoint cannot forget to check it.

Routes are defined once on a router and mounted twice: at `/api/v1` (the documented,
versioned surface) and at `/api` (an undocumented alias kept so existing clients keep
working). New endpoints should be considered v1-only.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, File, Form, Query, Response, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import (
    __version__,
    analytics,
    auth,
    config,
    embeddings,
    errors,
    jobs,
    middleware,
    observability,
    rag,
    schemas,
    security,
    storage,
    summarize,
    transcription,
)
from .deps import CurrentUser
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

    # Housekeeping that only makes sense once, at boot.
    recovered = jobs.recover_interrupted_jobs()
    purged = storage.purge_expired_refresh_tokens()
    jobs.start_workers()

    log.info(
        "startup complete",
        extra={
            "version": __version__,
            "environment": config.ENVIRONMENT,
            "schema_version": storage.SCHEMA_VERSION,
            "has_api_key": config.has_api_key(),
            "rate_limit_enabled": config.RATE_LIMIT_ENABLED,
            "recovered_jobs": recovered,
            "purged_tokens": purged,
        },
    )
    yield
    # Let in-flight transcriptions finish rather than throwing away work already paid for.
    jobs.stop_workers(wait=True)
    log.info("shutdown complete")


app = FastAPI(
    title="MeetSaransh",
    version=__version__,
    lifespan=lifespan,
    description=(
        "Transcribe meeting audio, generate action-oriented summaries, and ask "
        "grounded questions across every meeting. All meeting routes require a bearer "
        "token from /auth/login and only ever return the caller's own data."
    ),
    # Interactive docs are a development convenience, not something to expose publicly.
    docs_url=None if config.IS_PRODUCTION else "/docs",
    redoc_url=None,
    openapi_url=None if config.IS_PRODUCTION else "/openapi.json",
    responses={
        400: {"model": schemas.ErrorResponse},
        401: {"model": schemas.ErrorResponse},
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

    Public and unauthenticated: a load balancer has no credentials, and it reports only
    infrastructure state, never user data.
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


# -------------------------------------------------------------------------------- auth
@api.post("/auth/register", response_model=schemas.TokenPair, status_code=201, tags=["auth"])
def api_register(payload: schemas.RegisterRequest) -> schemas.TokenPair:
    """Create an account and sign in.

    The first account on a database also claims any meetings created before
    authentication existed, so upgrading an existing local install does not strand the
    data that is already there.
    """
    email = auth.normalize_email(payload.email)
    try:
        password_hash = auth.hash_password(payload.password)
    except auth.AuthError as exc:
        raise APIError(str(exc), status_code=422, code="weak_password") from exc

    is_first_account = storage.count_users() == 0
    try:
        user = storage.create_user(email=email, password_hash=password_hash)
    except sqlite3.IntegrityError as exc:
        # The unique index is the real guard; checking first would be a race.
        raise APIError(
            "That email is already registered.", status_code=409, code="email_taken"
        ) from exc

    if is_first_account:
        claimed = storage.claim_unowned_meetings(user["id"])
        if claimed:
            log.info(
                "claimed pre-auth meetings",
                extra={"user_id": user["id"], "claimed": claimed},
            )

    log.info("user registered", extra={"user_id": user["id"]})
    return _issue_tokens(user)


@api.post("/auth/login", response_model=schemas.TokenPair, tags=["auth"])
def api_login(payload: schemas.LoginRequest) -> schemas.TokenPair:
    """Exchange credentials for an access + refresh token pair."""
    email = auth.normalize_email(payload.email)
    user = storage.get_user_by_email(email)

    if user is None or not auth.verify_password(payload.password, user["password_hash"]):
        if config.METRICS_ENABLED:
            observability.LOGINS.labels("failure").inc()
        log.warning("failed login", extra={"email_domain": email.rpartition("@")[2]})
        # One message for both "no such account" and "wrong password". Distinguishing
        # them turns the login form into an account-enumeration oracle.
        raise APIError("Incorrect email or password.", status_code=401, code="invalid_credentials")

    # Transparent upgrade: if the stored hash used weaker parameters than we now
    # require, re-hash it now, while the plaintext is legitimately in hand.
    if auth.needs_rehash(user["password_hash"]):
        storage.update_password_hash(user["id"], auth.hash_password(payload.password))
        log.info("password hash upgraded", extra={"user_id": user["id"]})

    if config.METRICS_ENABLED:
        observability.LOGINS.labels("success").inc()
    return _issue_tokens(user)


@api.post("/auth/refresh", response_model=schemas.TokenPair, tags=["auth"])
def api_refresh(payload: schemas.RefreshRequest) -> schemas.TokenPair:
    """Trade a refresh token for a new pair, rotating the refresh token.

    Rotation matters: a refresh token is spendable exactly once, so a stolen one is
    useful only until the real user next refreshes -- at which point the theft is
    detectable rather than indefinite.
    """
    try:
        claims = auth.decode_token(payload.refresh_token, "refresh")
    except auth.AuthError as exc:
        raise APIError(str(exc), status_code=401, code="invalid_token") from exc

    user_id = str(claims["sub"])
    jti_hash = auth.hash_token_id(str(claims["jti"]))

    # A valid signature is not enough: the token must still be live server-side, which
    # is what makes logout and revocation actually mean something.
    if not storage.refresh_token_is_active(jti_hash, user_id):
        log.warning("refresh with a revoked or unknown token", extra={"user_id": user_id})
        raise APIError("Session has expired. Sign in again.", status_code=401, code="invalid_token")

    user = storage.get_user(user_id)
    if user is None:
        raise APIError("Session has expired. Sign in again.", status_code=401, code="invalid_token")

    storage.revoke_refresh_token(jti_hash)  # single use
    return _issue_tokens(user)


@api.post("/auth/logout", response_model=schemas.LogoutResponse, tags=["auth"])
def api_logout(user: CurrentUser) -> schemas.LogoutResponse:
    """Revoke every refresh token for the caller, ending all their sessions.

    Access tokens are stateless and remain valid until they expire (at most
    ACCESS_TOKEN_MINUTES); that is the trade-off statelessness buys, and the short
    lifetime is what makes it acceptable.
    """
    revoked = storage.revoke_all_refresh_tokens(user["id"])
    log.info("user logged out", extra={"user_id": user["id"], "revoked": revoked})
    return schemas.LogoutResponse(revoked=revoked)


@api.get("/auth/me", response_model=schemas.User, tags=["auth"])
def api_me(user: CurrentUser) -> schemas.User:
    """Who the current token belongs to. Used by the frontend to restore a session."""
    return schemas.User(**{k: user[k] for k in ("id", "email", "role", "created_at")})


# ---------------------------------------------------------------------------- meetings
@api.get("/meetings", response_model=schemas.MeetingPage, tags=["meetings"])
def api_list_meetings(
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=200, description="Page size."),
    offset: int = Query(0, ge=0, description="Rows to skip."),
) -> schemas.MeetingPage:
    """List the caller's meetings, newest first."""
    items = storage.list_meetings(user["id"], limit=limit, offset=offset)
    return schemas.MeetingPage(
        items=[schemas.MeetingListItem(**i) for i in items],
        total=storage.count_meetings(user["id"]),
        limit=limit,
        offset=offset,
    )


@api.get("/meetings/{meeting_id}", response_model=schemas.Meeting, tags=["meetings"])
def api_get_meeting(meeting_id: str, user: CurrentUser) -> schemas.Meeting:
    """Fetch one meeting. Also the polling endpoint while a meeting is processing."""
    return schemas.Meeting(**_require_meeting(meeting_id, user))


@api.delete("/meetings/{meeting_id}", response_model=schemas.DeleteResponse, tags=["meetings"])
def api_delete_meeting(meeting_id: str, user: CurrentUser) -> schemas.DeleteResponse:
    meeting = _require_meeting(meeting_id, user)
    if meeting.get("audio_ext"):
        (config.AUDIO_DIR / f"{meeting_id}{meeting['audio_ext']}").unlink(missing_ok=True)
    # Chunks go with the meeting via ON DELETE CASCADE (see storage migration 2).
    storage.delete_meeting(meeting_id, user["id"])
    log.info("meeting deleted", extra={"meeting_id": meeting_id, "user_id": user["id"]})
    return schemas.DeleteResponse(deleted=meeting_id)


@api.post(
    "/meetings",
    response_model=schemas.MeetingAccepted,
    status_code=202,
    tags=["meetings"],
)
async def api_create_meeting(
    user: CurrentUser,
    file: UploadFile = File(..., description="Meeting audio."),
    title: str = Form("", description="Optional title; the filename is used otherwise."),
) -> schemas.MeetingAccepted:
    """Accept audio for processing and return immediately.

    202, not 201: the meeting row exists but is not finished. Transcribing an hour of
    audio takes minutes, and holding an HTTP request open for that long means any proxy
    or client timeout destroys work that has already been paid for at the provider.
    """
    ext = _validate_extension(file)
    meeting_id = storage.new_id()
    audio_path = config.AUDIO_DIR / f"{meeting_id}{ext}"

    size = await _stream_upload_to_disk(file, audio_path, ext)
    resolved_title = title.strip() or Path(file.filename or "meeting").stem

    storage.create_meeting(
        user_id=user["id"],
        title=resolved_title,
        filename=file.filename,
        audio_ext=ext,
        meeting_id=meeting_id,
        status="queued",
    )

    try:
        jobs.enqueue(meeting_id, audio_path, resolved_title, user["id"])
    except jobs.QuotaExceeded as exc:
        # Roll the placeholder back so a rejected upload leaves nothing behind.
        storage.delete_meeting(meeting_id, user["id"])
        audio_path.unlink(missing_ok=True)
        raise APIError(str(exc), status_code=429, code="job_quota_exceeded") from exc

    log.info(
        "upload accepted",
        extra={"meeting_id": meeting_id, "bytes": size, "ext": ext, "user_id": user["id"]},
    )
    meeting = storage.get_meeting(meeting_id, user["id"])
    assert meeting is not None  # just created in this request
    return schemas.MeetingAccepted(
        id=meeting_id,
        title=meeting["title"],
        status=meeting["status"],
        created_at=meeting["created_at"],
        poll_url=f"/api/v1/meetings/{meeting_id}",
    )


@api.post("/meetings/sample", response_model=schemas.Meeting, status_code=201, tags=["meetings"])
def api_create_sample(user: CurrentUser) -> schemas.Meeting:
    """Create a meeting from the bundled sample transcript.

    Stays synchronous and returns 201: there is no audio to transcribe, so the only
    provider call is an optional summarization that takes a couple of seconds. Making
    the one instant path in the app asynchronous would be ceremony, not engineering.
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
        user_id=user["id"],
        title=sample["title"],
        filename="sample_meeting.mp3",
        transcript=transcript,
        summary=summary,
        audio_ext=None,
        status="done",
    )
    _safe_index(mid)
    return schemas.Meeting(**_require_meeting(mid, user))


@api.get("/meetings/{meeting_id}/audio", tags=["meetings"])
def api_get_audio(meeting_id: str, user: CurrentUser) -> FileResponse:
    meeting = _require_meeting(meeting_id, user)
    if not meeting.get("audio_ext"):
        raise APIError("No audio for this meeting.", status_code=404)
    audio = config.AUDIO_DIR / f"{meeting_id}{meeting['audio_ext']}"
    if not audio.exists():
        raise APIError("Audio file not found.", status_code=404)
    return FileResponse(audio)


@api.get("/meetings/{meeting_id}/export", response_class=PlainTextResponse, tags=["meetings"])
def api_export_markdown(meeting_id: str, user: CurrentUser) -> str:
    """Export the summary + action items as copy-ready Markdown (close-the-loop)."""
    return _to_markdown(_require_meeting(meeting_id, user))


# --------------------------------------------------------------------------- analytics
@api.get("/analytics", response_model=schemas.AnalyticsResponse, tags=["analytics"])
def api_analytics(
    user: CurrentUser,
    days: int = Query(30, ge=1, le=365, description="Trailing window for the time series."),
) -> schemas.AnalyticsResponse:
    """Cross-meeting aggregates for the caller: workload, cadence, topics, loose ends."""
    return schemas.AnalyticsResponse(**analytics.dashboard(user["id"], days=days))


# -------------------------------------------------------------------------- RAG / chat
@api.get("/rag/status", response_model=schemas.RagStatus, tags=["rag"])
def api_rag_status(user: CurrentUser) -> schemas.RagStatus:
    return schemas.RagStatus(**rag.status(user["id"]))


@api.post("/reindex", response_model=schemas.ReindexResponse, tags=["rag"])
def api_reindex(user: CurrentUser) -> schemas.ReindexResponse:
    """(Re)index any of the caller's meetings that aren't in the vector store yet."""
    return schemas.ReindexResponse(**rag.reindex_all(user["id"]))


@api.post("/chat", response_model=schemas.ChatResponse, tags=["rag"])
def api_chat(payload: schemas.ChatRequest, user: CurrentUser) -> schemas.ChatResponse:
    """Answer a question grounded in the caller's own meetings."""
    if payload.meeting_id and storage.get_meeting(payload.meeting_id, user["id"]) is None:
        # Reject an unknown scope rather than silently widening to "all meetings",
        # which would answer from a different set than the caller asked about.
        raise APIError("Meeting not found.", status_code=404)

    rag.reindex_all(user["id"])  # cheap: only indexes meetings not already indexed
    result = rag.answer(payload.question, user["id"], scope_meeting_id=payload.meeting_id)
    if config.METRICS_ENABLED:
        observability.RAG_ANSWERS.labels(result.get("mode", "unknown")).inc()
    return schemas.ChatResponse(**result)


def _sse(event: dict) -> str:
    """Frame one event for Server-Sent Events: `data: <json>` and a blank line.

    The blank line is the frame terminator -- omit it and the browser buffers every
    event forever, waiting for an end that never comes.
    """
    return f"data: {json.dumps(event)}\n\n"


@api.post("/chat/stream", tags=["rag"])
def api_chat_stream(payload: schemas.ChatRequest, user: CurrentUser) -> StreamingResponse:
    """Answer a question as a Server-Sent Events stream.

    SSE rather than WebSockets: this is one-directional, short-lived, and survives
    proxies and reconnects for free. It is also plain HTTP, so it inherits the same
    auth dependency, rate limit and error envelope as every other endpoint instead of
    needing a parallel set of them.

    The client keeps `POST /chat` as a fallback; both return the same modes.
    """
    if payload.meeting_id and storage.get_meeting(payload.meeting_id, user["id"]) is None:
        raise APIError("Meeting not found.", status_code=404)

    rag.reindex_all(user["id"])

    def events() -> Iterator[str]:
        mode = "error"
        try:
            for event in rag.answer_stream(
                payload.question, user["id"], scope_meeting_id=payload.meeting_id
            ):
                mode = event.get("mode", mode)
                yield _sse(event)
        except Exception:
            # A generator raising mid-stream cannot become a 500 -- the response has
            # already started -- so the failure is delivered as a final event instead.
            log.exception("chat stream failed", extra={"user_id": user["id"]})
            failure = {
                "type": "error",
                "mode": "error",
                "message": "The answer stream failed. The error has been logged.",
            }
            yield _sse(failure)
        finally:
            if config.METRICS_ENABLED:
                observability.RAG_ANSWERS.labels(mode).inc()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            # Without this an intermediate proxy will happily buffer the whole stream
            # and deliver it in one lump, which defeats the entire feature.
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


# ------------------------------------------------------------------------------ mounts
# `/api/v1` is the documented surface; `/api` is a compatibility alias for older
# clients and is intentionally hidden from the OpenAPI schema.
app.include_router(api, prefix="/api/v1")
app.include_router(api, prefix="/api", include_in_schema=False)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint.

    Reports service-wide totals with no per-user labels, so scraping it cannot become a
    side channel for one account's activity.
    """
    if not config.METRICS_ENABLED:
        raise APIError("Metrics are disabled.", status_code=404)
    observability.MEETINGS_STORED.set(storage.count_meetings())
    observability.CHUNKS_STORED.set(storage.count_chunks())
    body, content_type = observability.render_metrics()
    return Response(content=body, media_type=content_type)


# ----------------------------------------------------------------------------- helpers
def _issue_tokens(user: dict) -> schemas.TokenPair:
    """Mint an access + refresh pair and record the refresh token so it can be revoked."""
    access, _, _ = auth.create_token(user["id"], "access")
    refresh, jti, expires = auth.create_token(user["id"], "refresh")
    storage.store_refresh_token(
        jti_hash=auth.hash_token_id(jti),
        user_id=user["id"],
        expires_at=expires.isoformat(timespec="seconds"),
    )
    return schemas.TokenPair(
        access_token=access,
        refresh_token=refresh,
        expires_in=config.ACCESS_TOKEN_MINUTES * 60,
        user=schemas.User(**{k: user[k] for k in ("id", "email", "role", "created_at")}),
    )


def _require_meeting(meeting_id: str, user: dict) -> dict:
    """Load a meeting the caller owns, or 404.

    404 rather than 403 for someone else's meeting: a 403 would confirm the id exists,
    letting an attacker enumerate which meetings other accounts hold.
    """
    meeting = storage.get_meeting(meeting_id, user["id"])
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
        lines += [f"- **{t['title']}** - {t['summary']}" for t in s["topics"]]
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------- static frontend
@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    """Serve the app shell with cache-busted asset URLs.

    Without this the browser happily keeps serving last release's `app.js` from cache
    while the API has already moved on -- which looks exactly like a broken deploy and
    is invisible to anyone testing with an empty cache. Stamping the version onto the
    asset URLs makes a new release a new URL, and `no-cache` on the shell itself means
    the browser always revalidates the one document that carries those URLs.
    """
    html = (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("/static/style.css", f"/static/style.css?v={__version__}")
    html = html.replace("/static/app.js", f"/static/app.js?v={__version__}")
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
