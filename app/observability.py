"""Structured logging, request correlation, and Prometheus metrics.

Logging is stdlib `logging` with a JSON formatter rather than a logging framework: the
only thing a framework would add here is a dependency. Every log line carries the
request id from a ContextVar, so a single request's lines can be grepped out of a busy
log even though the handlers never pass the id around explicitly.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from . import config

# Set once per request by the middleware; read by the log filter and error handlers.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


# ------------------------------------------------------------------------------ logging
_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
}


class _RequestIdFilter(logging.Filter):
    """Attach the current request id to every record, including third-party ones."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line -- what a log aggregator expects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        # Anything passed via logger.info("...", extra={...}) rides along as a field.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} "
            f"[{getattr(record, 'request_id', '-')}] {record.name}: {record.getMessage()}"
        )
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_") and k != "request_id"
        }
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging() -> None:
    """Install the root handler. Idempotent, so tests can call it repeatedly."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if config.LOG_JSON else ConsoleFormatter())
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    # uvicorn installs its own access log with a different shape; ours already records
    # method/path/status/duration with a request id, so silence the duplicate.
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False
    logging.getLogger("uvicorn.error").propagate = True

    # httpx logs every outbound call at INFO. Our own provider logs already record the
    # calls we care about (with latency and outcome), so this is pure duplication.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ------------------------------------------------------------------------------ metrics
# Label values are route TEMPLATES ("/api/meetings/{meeting_id}"), never raw paths, or
# every meeting id would create its own time series and blow up cardinality.
REQUESTS = Counter(
    "meetsaransh_http_requests_total",
    "HTTP requests handled.",
    ["method", "route", "status"],
)
REQUEST_DURATION = Histogram(
    "meetsaransh_http_request_duration_seconds",
    "Request latency.",
    ["method", "route"],
    # Buckets span a static file (~5ms) to a full transcription (minutes); the defaults
    # top out at 10s and would put every upload in +Inf.
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
RATE_LIMITED = Counter(
    "meetsaransh_rate_limited_total",
    "Requests rejected by the rate limiter.",
    ["route"],
)
UPLOADS_REJECTED = Counter(
    "meetsaransh_uploads_rejected_total",
    "Uploads rejected before reaching the provider.",
    ["reason"],
)
PROVIDER_CALLS = Counter(
    "meetsaransh_provider_calls_total",
    "Outbound calls to the ASR/LLM provider.",
    ["provider", "outcome"],
)
PROVIDER_DURATION = Histogram(
    "meetsaransh_provider_duration_seconds",
    "Latency of outbound provider calls.",
    ["provider"],
    buckets=(0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
)
RAG_ANSWERS = Counter(
    "meetsaransh_rag_answers_total",
    "Chat answers by outcome mode.",
    ["mode"],
)
JOBS_COMPLETED = Counter(
    "meetsaransh_jobs_completed_total",
    "Background transcription jobs that reached a terminal state.",
    ["outcome"],
)
JOB_QUEUE_DEPTH = Gauge(
    "meetsaransh_job_queue_depth",
    "Meetings waiting for or undergoing processing.",
    ["state"],
)
LOGINS = Counter(
    "meetsaransh_logins_total",
    "Authentication attempts by outcome.",
    ["outcome"],
)
AUTH_REJECTIONS = Counter(
    "meetsaransh_auth_rejections_total",
    "Requests rejected for a missing or invalid credential.",
    ["reason"],
)
MEETINGS_STORED = Gauge("meetsaransh_meetings_stored", "Meetings currently stored.")
CHUNKS_STORED = Gauge("meetsaransh_chunks_indexed", "RAG chunks currently indexed.")


def render_metrics() -> tuple[bytes, str]:
    """Serialize the registry in Prometheus text exposition format."""
    return generate_latest(), CONTENT_TYPE_LATEST
