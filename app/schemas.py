"""Pydantic request/response models.

Every endpoint declares a response model. That is not decoration: it turns the OpenAPI
schema at /docs into an accurate contract, guarantees the shape callers receive even if
an internal dict grows a field, and lets the frontend be written against something
checkable instead of against whatever the handler happened to return that day.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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
    duration: float = 0.0
    transcript: str = ""
    audio_ext: str | None = None
    segments: list[Segment] = []
    summary: Summary = Summary()


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
