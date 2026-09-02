"""ASGI middleware: request correlation + access logging + metrics, rate limiting,
and security headers.

Installation order (see `install_middleware`) is load-bearing and documented there.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from . import config, observability, security
from .observability import get_logger, new_request_id, request_id_var

log = get_logger("meetsaransh.http")

_Next = Callable[[Request], Awaitable[Response]]

# A caller-supplied correlation id is useful for tracing, but it lands in log files, so
# only a conservative character set is accepted -- otherwise a caller could inject
# newlines and forge log lines.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _incoming_request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    return supplied if _SAFE_REQUEST_ID.match(supplied) else new_request_id()


def route_template(request: Request, original_path: str) -> str:
    """The matched route pattern, e.g. '/api/v1/meetings/{meeting_id}'.

    Metrics label on the template, never the raw path: labelling on the path would mint
    a new time series per meeting id and eventually take the metrics endpoint down.

    `scope["route"]` is only populated once the router has run, which is NOT the case
    for a response a middleware short-circuits -- a 429 from the rate limiter, for one.
    Falling back to "unmatched" there would throw away the single most useful fact
    about that 429: which endpoint is being hammered. So when the scope carries no
    route, the path is matched against the router directly.

    `original_path` must be captured BEFORE the request is passed downstream: entering
    a Mount rewrites scope["path"] to the remainder ("/app.js" rather than
    "/static/app.js"), so re-reading it afterwards matches nothing.
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path or _resolve_route(request, original_path)


def _resolve_route(request: Request, original_path: str) -> str:
    """Find the route template a path would match, without invoking the endpoint."""
    from starlette.routing import Match

    app = request.scope.get("app")
    if app is None:
        return "unmatched"

    scope = {**request.scope, "path": original_path, "root_path": ""}
    partial: str | None = None
    for route in app.routes:
        match, _ = route.matches(scope)
        if match is Match.FULL:
            # For a Mount ("/static") this is the mount path, which is exactly right:
            # every asset under it shares one time series instead of one per file.
            return str(route.path)
        if match is Match.PARTIAL and partial is None:
            # The path matched but the method did not -- i.e. a 405. Still worth
            # attributing to the endpoint the caller was aiming at.
            partial = str(getattr(route, "path", "")) or None
    return partial or "unmatched"


def client_key(request: Request) -> str:
    """Identify the caller for rate-limiting purposes.

    Today that is an IP address. Once authentication lands this becomes the user id,
    which is strictly better -- an IP is shared by everyone behind one NAT.
    """
    if config.TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request, log it, and record metrics."""

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        rid = _incoming_request_id(request)
        # Captured before anything downstream can rewrite scope["path"] (see
        # route_template's docstring).
        original_path = request.url.path
        token = request_id_var.set(rid)
        started = time.perf_counter()
        reset_token = True
        try:
            try:
                response = await call_next(request)
            except Exception:
                # The exception handler turns this into a 500; record it here too so a
                # crashing endpoint still appears in the metrics and the access log.
                elapsed = time.perf_counter() - started
                self._record(request, original_path, 500, elapsed)
                log.exception(
                    "request failed",
                    extra={"method": request.method, "path": request.url.path},
                )
                # Leave the context var set: the 500 body is rendered by Starlette's
                # error middleware, which sits OUTSIDE this one and runs after this
                # frame unwinds. Resetting here would strip the request id from exactly
                # the response a developer most needs to trace. The var lives in the
                # per-request task context, so nothing leaks between requests.
                reset_token = False
                raise

            elapsed = time.perf_counter() - started
            self._record(request, original_path, response.status_code, elapsed)
            response.headers["X-Request-ID"] = rid
            # Handy in dev, and the number a load test wants without scraping metrics.
            response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.1f}"

            level = log.warning if response.status_code >= 500 else log.info
            level(
                "request",
                extra={
                    "method": request.method,
                    "path": original_path,
                    "route": route_template(request, original_path),
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                    "client": client_key(request),
                },
            )
            return response
        finally:
            # Reset only AFTER the access log line above is written -- resetting any
            # earlier would strip the request id off the very line whose job is to
            # carry it.
            if reset_token:
                request_id_var.reset(token)

    @staticmethod
    def _record(request: Request, original_path: str, status: int, elapsed: float) -> None:
        if not config.METRICS_ENABLED:
            return
        route = route_template(request, original_path)
        observability.REQUESTS.labels(request.method, route, str(status)).inc()
        observability.REQUEST_DURATION.labels(request.method, route).observe(elapsed)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Cap requests per client per window, with the tightest caps on paid endpoints."""

    #  Prune the limiter's bucket dict every N requests so it cannot grow unbounded.
    _PRUNE_EVERY = 500

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        self._since_prune = 0

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        if not config.RATE_LIMIT_ENABLED or request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        # Health and metrics are what a load balancer and Prometheus poll constantly.
        if path.endswith(("/health", "/metrics")):
            return await call_next(request)

        self._maybe_prune()

        limit = security.limit_for_path(request.method, path)
        window = float(config.RATE_LIMIT_WINDOW_SECONDS)
        # Key on method+path so a burst of uploads cannot exhaust the chat budget.
        key = f"{client_key(request)}|{request.method}|{path}"
        allowed, remaining, retry_after = security.limiter.check(key, limit, window)

        if not allowed:
            if config.METRICS_ENABLED:
                observability.RATE_LIMITED.labels(route_template(request, path)).inc()
            log.warning(
                "rate limited",
                extra={"path": path, "limit": limit, "client": client_key(request)},
            )
            retry_seconds = max(1, int(retry_after + 0.999))
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": (
                            f"Too many requests. This endpoint allows {limit} per "
                            f"{config.RATE_LIMIT_WINDOW_SECONDS}s. "
                            f"Retry in {retry_seconds}s."
                        ),
                        "request_id": request_id_var.get(),
                    }
                },
                headers={
                    "Retry-After": str(retry_seconds),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response

    def _maybe_prune(self) -> None:
        self._since_prune += 1
        if self._since_prune >= self._PRUNE_EVERY:
            self._since_prune = 0
            security.limiter.prune(older_than=config.RATE_LIMIT_WINDOW_SECONDS * 10)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach the hardening headers to every response, including error responses."""

    def __init__(self, app: FastAPI) -> None:
        super().__init__(app)
        self._headers = security.security_headers(is_production=config.IS_PRODUCTION)

    async def dispatch(self, request: Request, call_next: _Next) -> Response:
        response = await call_next(request)
        for name, value in self._headers.items():
            response.headers.setdefault(name, value)
        return response


def install_middleware(app: FastAPI) -> None:
    """Install the stack.

    Starlette runs middleware in reverse registration order, so the LAST one added is
    the OUTERMOST. Registering in this order gives an execution order of:

        RequestContext -> SecurityHeaders -> RateLimit -> CORS -> route handler

    which is what we want: every response gets a request id and hardening headers,
    including the 429s produced by the limiter itself.
    """
    if config.CORS_ORIGINS:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
        )

    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)
