"""One error envelope for the whole API, plus the handlers that produce it.

Every failure -- a raised HTTPException, a request that fails validation, or an
unhandled crash -- comes back as:

    {"error": {"code": "...", "message": "...", "request_id": "..."}}

A stable `code` gives clients something to branch on that is not a prose string, and
`request_id` is the thread back to the server logs for whoever is debugging.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .observability import get_logger, request_id_var

log = get_logger("meetsaransh.errors")

# Default code per status, used when a raiser didn't supply one.
_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "provider_error",
    503: "unavailable",
}


class APIError(Exception):
    """Application error carrying an HTTP status and a stable machine-readable code."""

    def __init__(self, message: str, *, status_code: int = 400, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code or _STATUS_CODES.get(status_code, "error")


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    """Build the envelope, and echo the request id as a header.

    The header is set here rather than in the middleware because a 500 is rendered by
    Starlette's outermost error middleware, which sits *outside* our stack -- so the
    middleware never sees that response. Setting it here covers every error path.
    """
    request_id = request_id_var.get()
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
        headers={"X-Request-ID": request_id},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def _api_error(_: Request, exc: APIError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        code = _STATUS_CODES.get(exc.status_code, "error")
        return error_response(exc.status_code, code, detail)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Summarize as "field: reason" rather than dumping pydantic's nested structure,
        # which is unreadable in a toast and leaks internal model layout.
        parts = []
        for err in exc.errors()[:5]:
            field = ".".join(str(p) for p in err.get("loc", ()) if p not in ("body", "query"))
            parts.append(f"{field or 'request'}: {err.get('msg', 'invalid')}")
        return error_response(422, "validation_error", "; ".join(parts) or "Invalid request.")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the full traceback server-side; return a generic message to the client so
        # stack traces and internal paths never reach the browser.
        log.exception(
            "unhandled exception",
            extra={"path": request.url.path, "method": request.method, "error": repr(exc)},
        )
        return error_response(
            500,
            "internal_error",
            "Something went wrong on our side. The request id can be used to trace it.",
        )
