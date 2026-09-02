"""Password hashing and JWT issuing/verification.

Two deliberate choices, both worth defending:

**Passwords use `hashlib.scrypt` (stdlib), not bcrypt or argon2.** scrypt is a
memory-hard KDF designed for exactly this, it ships with Python, and calling it
correctly is a matter of picking parameters rather than of trusting a wrapper. The
stored format carries its own parameters, so the cost can be raised later and old
hashes still verify (and are transparently upgraded on the next successful login).

**Tokens use PyJWT, not a hand-rolled HMAC.** Signing a JWT is easy; the ways JWT
verification goes wrong (accepting `alg: none`, algorithm confusion, missing `exp`
checks) are subtle and well-documented. Auth is the last place to save a dependency.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import jwt

from . import config

# --------------------------------------------------------------------------- passwords
# n=2^15, r=8, p=1 costs ~32 MB and ~50-100 ms per hash on a modern CPU: slow enough to
# make offline cracking expensive, fast enough that a login is not a user-visible wait.
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_MAXMEM = 128 * _SCRYPT_N * _SCRYPT_R * 2  # headroom over the exact requirement
_SALT_BYTES = 16
_KEY_BYTES = 32

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128  # bounded so a huge password cannot be used as a CPU DoS


class AuthError(Exception):
    """Raised when a credential or token cannot be accepted."""


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    """Hash a password into a self-describing string: scrypt$n$r$p$salt$key."""
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        raise AuthError(
            f"Password must be between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters."
        )
    salt = secrets.token_bytes(_SALT_BYTES)
    key = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        maxmem=_SCRYPT_MAXMEM,
        dklen=_KEY_BYTES,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${_b64(salt)}${_b64(key)}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash. False on any malformed input."""
    try:
        scheme, n, r, p, salt_b64, key_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        salt, expected = _unb64(salt_b64), _unb64(key_b64)
        candidate = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            maxmem=128 * int(n) * int(r) * 2,
            dklen=len(expected),
        )
    except (ValueError, TypeError, MemoryError):
        # A corrupt or foreign hash format is a failed login, never a 500.
        return False
    # Constant time: a byte-by-byte `==` leaks how much of the hash matched.
    return hmac.compare_digest(candidate, expected)


def needs_rehash(stored: str) -> bool:
    """True when a stored hash used weaker parameters than we now require.

    Lets the cost be raised over time: the next successful login re-hashes silently.
    """
    try:
        scheme, n, r, p, _, _ = stored.split("$")
    except ValueError:
        return True
    return scheme != "scrypt" or (int(n), int(r), int(p)) != (_SCRYPT_N, _SCRYPT_R, _SCRYPT_P)


# ------------------------------------------------------------------------------ tokens
TokenType = Literal["access", "refresh"]
_ALGORITHM = "HS256"


def _now() -> datetime:
    return datetime.now(UTC)


def create_token(user_id: str, token_type: TokenType) -> tuple[str, str, datetime]:
    """Mint a signed token. Returns (encoded, jti, expires_at).

    The `jti` is what makes a refresh token revocable: its hash is stored server-side,
    so a stolen token can be invalidated without waiting for it to expire.
    """
    lifetime = (
        timedelta(minutes=config.ACCESS_TOKEN_MINUTES)
        if token_type == "access"
        else timedelta(days=config.REFRESH_TOKEN_DAYS)
    )
    issued = _now()
    expires = issued + lifetime
    jti = uuid.uuid4().hex
    payload: dict[str, Any] = {
        "sub": user_id,
        "typ": token_type,
        "jti": jti,
        "iat": int(issued.timestamp()),
        "exp": int(expires.timestamp()),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=_ALGORITHM), jti, expires


def decode_token(token: str, expected_type: TokenType) -> dict[str, Any]:
    """Verify a token's signature, expiry, and type. Raises AuthError otherwise.

    `algorithms` is pinned to a single value: accepting whatever the token's own header
    claims is the classic JWT vulnerability, and passing a list of one closes it.
    """
    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET,
            algorithms=[_ALGORITHM],
            options={"require": ["exp", "iat", "sub", "typ", "jti"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("Token is invalid.") from exc

    if payload.get("typ") != expected_type:
        # Without this an access token would work as a refresh token and vice versa,
        # quietly turning a 30-minute credential into a 7-day one.
        raise AuthError("Token is not valid for this operation.")
    return payload


def hash_token_id(jti: str) -> str:
    """Hash a token id for storage.

    The refresh-token table is a credential store: if it leaked, raw ids would let an
    attacker forge nothing (they cannot sign) but would confirm live sessions. Hashing
    costs nothing here because a jti is already high-entropy, so a fast hash is enough.
    """
    return hashlib.sha256(jti.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------- email
def normalize_email(email: str) -> str:
    """Lowercase and trim, so `Alice@Example.com` and `alice@example.com` are one account."""
    return email.strip().lower()
