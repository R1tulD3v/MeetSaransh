"""FastAPI dependencies -- principally `get_current_user`.

Authentication is a dependency rather than middleware on purpose. Middleware would have
to carry a list of public paths and stay in sync with the routes, and the failure mode
of that drifting is an endpoint that is silently unauthenticated. As a dependency, an
endpoint is protected exactly when its signature says so, which is visible in review and
in the generated OpenAPI schema.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import auth, config, observability, storage
from .errors import APIError
from .observability import get_logger

log = get_logger("meetsaransh.auth")

# auto_error=False so a missing header reaches our own handler and produces the standard
# error envelope, rather than HTTPBearer's own differently-shaped 403.
_bearer = HTTPBearer(auto_error=False, description="JWT access token from /auth/login.")


def _reject(reason: str, message: str) -> APIError:
    if config.METRICS_ENABLED:
        observability.AUTH_REJECTIONS.labels(reason).inc()
    return APIError(message, status_code=401, code="unauthorized")


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict[str, Any]:
    """Resolve the caller from their bearer token, or raise 401.

    The user row is loaded on every request rather than trusted from the token body, so
    deleting an account takes effect immediately instead of when its access token
    happens to expire.
    """
    if credentials is None or not credentials.credentials:
        raise _reject("missing", "Authentication required. Sign in and retry.")

    try:
        payload = auth.decode_token(credentials.credentials, "access")
    except auth.AuthError as exc:
        raise _reject("invalid", str(exc)) from exc

    user = storage.get_user(str(payload["sub"]))
    if user is None:
        # The signature was valid but the account is gone. Same generic message: a
        # distinct one would let an attacker probe which user ids still exist.
        raise _reject("unknown_user", "Authentication required. Sign in and retry.")

    # Stashed so the rate limiter can key on the user instead of an IP that a whole
    # office might share.
    request.state.user_id = user["id"]
    return user


CurrentUser = Annotated[dict[str, Any], Depends(get_current_user)]


def require_admin(user: CurrentUser) -> dict[str, Any]:
    """Gate an endpoint to administrators.

    403 rather than 401: the caller proved who they are, they simply are not allowed.
    Collapsing the two makes a client retry a login that will never help.
    """
    if user.get("role") != "admin":
        raise APIError(
            "This action requires an administrator account.",
            status_code=403,
            code="forbidden",
        )
    return user


AdminUser = Annotated[dict[str, Any], Depends(require_admin)]
