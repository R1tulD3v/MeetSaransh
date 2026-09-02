"""Pydantic request/response models.

Every endpoint declares a response model. That is not decoration: it turns the OpenAPI
schema at /docs into an accurate contract, guarantees the shape callers receive even if
an internal dict grows a field, and lets the frontend be written against something
checkable instead of against whatever the handler happened to return that day.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# ------------------------------------------------------------------------------- errors


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable explanation, safe to show a user.")
    request_id: str = Field(description="Correlates this error with the server logs.")


class ErrorResponse(BaseModel):
    """The single error envelope every failing endpoint returns."""

    error: ErrorBody


# ------------------------------------------------------------------------------- health


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    has_api_key: bool
    database: bool = Field(description="True when the database is reachable and migrated.")
    embeddings: bool = Field(description="True when the dense embedding model is loaded.")
    asr_model: str
    llm_model: str


# ----------------------------------------------------------------------------- meetings

# A meeting is created immediately and filled in by a background worker, so its
# lifecycle is part of the public contract rather than an implementation detail.
MeetingStatus = Literal["queued", "processing", "done", "error"]


class ActionItem(BaseModel):
    task: str
    owner: str = "Unassigned"
    due: str = "Not specified"
    timestamp: str = ""


class KeyDecision(BaseModel):
    decision: str
    timestamp: str = ""


class Topic(BaseModel):
    title: str
    summary: str = ""
    timestamp: str = ""


class Summary(BaseModel):
    tldr: str = ""
    key_decisions: list[KeyDecision] = []
    action_items: list[ActionItem] = []
    open_questions: list[str] = []
    topics: list[Topic] = []


class Segment(BaseModel):
    start: float
    end: float
    text: str


class MeetingListItem(BaseModel):
    id: str
    title: str
    created_at: str
    duration: float = 0.0
    status: MeetingStatus = "done"
    stage: str | None = Field(
        default=None, description="What the worker is currently doing, while processing."
    )
    error: str | None = Field(default=None, description="Why processing failed, if it did.")


class MeetingPage(BaseModel):
    """Paginated meeting list. `total` lets a client render a page count without a
    second request, and the limit/offset echo makes responses self-describing."""

    items: list[MeetingListItem]
    total: int
    limit: int
    offset: int


class Meeting(BaseModel):
    id: str
    title: str
    filename: str | None = None
    created_at: str
    updated_at: str | None = None
    duration: float = 0.0
    transcript: str = ""
    audio_ext: str | None = None
    segments: list[Segment] = []
    summary: Summary = Summary()
    status: MeetingStatus = "done"
    stage: str | None = None
    error: str | None = None


class DeleteResponse(BaseModel):
    deleted: str


# --------------------------------------------------------------------------- RAG / chat


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    meeting_id: str | None = Field(
        default=None, description="Scope the answer to one meeting. Null searches all."
    )


class Citation(BaseModel):
    meeting_id: str
    meeting_title: str
    timestamp: str
    start: float
    snippet: str
    score: float


class ChatResponse(BaseModel):
    answer: str | None = Field(
        default=None, description="Null when no key is configured or the provider failed."
    )
    citations: list[Citation] = []
    mode: Literal["answer", "refused", "retrieval_only", "empty", "error"]
    note: str | None = None


class RagStatus(BaseModel):
    embeddings_available: bool
    embed_model: str
    indexed_meetings: int
    total_chunks: int


class ReindexResponse(BaseModel):
    meetings: int
    newly_indexed: int
    total_chunks: int


# -------------------------------------------------------------------------------- auth
# A deliberate sanity check, not RFC 5322 validation. Full email validation is not
# regex-expressible, and the only proof an address is real is sending mail to it -- so
# pydantic's EmailStr (which pulls in email-validator and dnspython) would buy strictness
# we do not actually need for a login identifier. This rejects the obviously-malformed
# and bounds the length; anything subtler is the mail server's problem.
Email = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        to_lower=True,
        min_length=3,
        max_length=254,  # the practical maximum length of an email address
        pattern=r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$",
    ),
]


class RegisterRequest(BaseModel):
    email: Email
    # The lower bound is enforced again in auth.hash_password, because the API is not
    # the only caller and a password policy that lives only in a schema is optional.
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: Email
    password: str = Field(min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class User(BaseModel):
    id: str
    email: str
    role: str
    created_at: str


class TokenPair(BaseModel):
    """The tokens plus the user, so a client needs one round trip to sign in."""

    access_token: str
    refresh_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Access-token lifetime in seconds.")
    user: User


class LogoutResponse(BaseModel):
    revoked: int = Field(description="How many refresh tokens were invalidated.")


# -------------------------------------------------------------------------------- jobs
class MeetingAccepted(BaseModel):
    """202 response for an upload: the meeting exists, processing has not finished."""

    id: str
    title: str
    status: MeetingStatus
    created_at: str
    poll_url: str = Field(description="Poll this until status is 'done' or 'error'.")


# --------------------------------------------------------------------------- analytics
class AnalyticsOverview(BaseModel):
    meetings: int
    total_seconds: float
    action_items: int
    decisions: int
    open_questions: int


class OwnerLoad(BaseModel):
    owner: str
    total: int
    with_due_date: int


class DayPoint(BaseModel):
    day: str
    meetings: int
    seconds: float


class TopicCount(BaseModel):
    title: str
    mentions: int
    meetings: int


class UnassignedItem(BaseModel):
    task: str
    timestamp: str = ""
    meeting_id: str
    meeting_title: str


class AnalyticsResponse(BaseModel):
    """The whole dashboard in one payload -- the page renders as a unit."""

    overview: AnalyticsOverview
    by_owner: list[OwnerLoad]
    over_time: list[DayPoint]
    top_topics: list[TopicCount]
    unassigned: list[UnassignedItem]
    window_days: int
