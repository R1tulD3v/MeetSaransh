"""Authentication: password hashing, token handling, the endpoints, and isolation.

The isolation tests at the bottom are the point of the whole feature: they assert that
one account cannot read, delete, or ask questions about another account's meetings.
"""

from __future__ import annotations

import time
from datetime import UTC

import pytest
from fastapi.testclient import TestClient

from app import auth, config, storage
from app.main import app
from tests.conftest import TEST_EMAIL, TEST_PASSWORD, sign_up


# ------------------------------------------------------------------ password hashing
def test_a_password_verifies_against_its_own_hash():
    stored = auth.hash_password("correct-horse-battery")
    assert auth.verify_password("correct-horse-battery", stored) is True


def test_a_wrong_password_does_not_verify():
    stored = auth.hash_password("correct-horse-battery")
    assert auth.verify_password("correct-horse-batteryy", stored) is False
    assert auth.verify_password("", stored) is False


def test_the_same_password_hashes_differently_every_time():
    """Per-password salt: identical passwords must not produce identical hashes, or a
    leaked table would reveal which accounts share one."""
    a = auth.hash_password("same-password")
    b = auth.hash_password("same-password")
    assert a != b
    assert auth.verify_password("same-password", a)
    assert auth.verify_password("same-password", b)


def test_the_hash_never_contains_the_password():
    assert "hunter2hunter2" not in auth.hash_password("hunter2hunter2")


def test_the_stored_format_carries_its_own_parameters():
    """Self-describing, so the cost can be raised later without invalidating old hashes."""
    scheme, n, r, p, salt, key = auth.hash_password("a-password").split("$")
    assert scheme == "scrypt"
    assert int(n) > 0 and int(r) > 0 and int(p) > 0
    assert salt and key


@pytest.mark.parametrize("stored", ["", "notahash", "scrypt$bad", "bcrypt$1$2$3$4$5", "$$$$$"])
def test_a_malformed_stored_hash_is_a_failed_login_not_a_crash(stored):
    assert auth.verify_password("anything", stored) is False


def test_passwords_that_are_too_short_or_too_long_are_rejected():
    with pytest.raises(auth.AuthError):
        auth.hash_password("short")
    with pytest.raises(auth.AuthError):
        # Bounded so a huge password cannot be used to burn CPU through the KDF.
        auth.hash_password("x" * 5000)


def test_production_scrypt_parameters_are_expensive_enough(monkeypatch):
    """The suite lowers the cost for speed; the shipped parameters are asserted here.

    n=2^15 with r=8 is ~32 MB of memory per hash, which is what makes large-scale
    offline cracking uneconomical.
    """
    import importlib

    module = importlib.reload(auth)
    try:
        assert module._SCRYPT_N == 2**15
        assert module._SCRYPT_R == 8
        assert 128 * module._SCRYPT_N * module._SCRYPT_R >= 32 * 1024 * 1024
    finally:
        importlib.reload(auth)


def test_a_hash_with_weaker_parameters_is_flagged_for_upgrade(monkeypatch):
    weak = auth.hash_password("a-password")
    monkeypatch.setattr(auth, "_SCRYPT_N", 2**10)  # requirements just went up
    assert auth.needs_rehash(weak) is True
    assert auth.verify_password("a-password", weak) is True  # still verifies meanwhile


def test_a_current_hash_is_not_flagged_for_upgrade():
    assert auth.needs_rehash(auth.hash_password("a-password")) is False


# --------------------------------------------------------------------------- tokens
def test_a_token_round_trips_its_claims():
    token, jti, expires = auth.create_token("user-123", "access")
    claims = auth.decode_token(token, "access")

    assert claims["sub"] == "user-123"
    assert claims["typ"] == "access"
    assert claims["jti"] == jti
    assert claims["exp"] == int(expires.timestamp())


def test_a_refresh_token_cannot_be_used_as_an_access_token():
    """Without the type check a 7-day credential would silently work as a 30-minute one."""
    refresh, _, _ = auth.create_token("user-123", "refresh")
    with pytest.raises(auth.AuthError, match="not valid for this operation"):
        auth.decode_token(refresh, "access")


def test_an_access_token_cannot_be_used_as_a_refresh_token():
    access, _, _ = auth.create_token("user-123", "access")
    with pytest.raises(auth.AuthError, match="not valid for this operation"):
        auth.decode_token(access, "refresh")


def test_a_token_signed_with_another_secret_is_rejected(monkeypatch):
    token, _, _ = auth.create_token("user-123", "access")
    monkeypatch.setattr(config, "JWT_SECRET", "a-completely-different-secret-value-here")
    with pytest.raises(auth.AuthError, match="invalid"):
        auth.decode_token(token, "access")


def test_a_tampered_token_is_rejected():
    token, _, _ = auth.create_token("user-123", "access")
    header, payload, signature = token.split(".")
    forged = f"{header}.{payload}.{signature[:-4]}AAAA"
    with pytest.raises(auth.AuthError):
        auth.decode_token(forged, "access")


def test_an_alg_none_token_is_rejected():
    """The classic JWT attack: an unsigned token claiming it needs no signature.

    Pinning `algorithms=["HS256"]` at the verify call is what closes it.
    """
    import base64
    import json as jsonlib

    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(jsonlib.dumps(obj).encode()).decode().rstrip("=")

    forged = (
        seg({"alg": "none", "typ": "JWT"})
        + "."
        + seg({"sub": "victim", "typ": "access", "jti": "x", "iat": 0, "exp": 9999999999})
        + "."
    )
    with pytest.raises(auth.AuthError):
        auth.decode_token(forged, "access")


def test_an_expired_token_is_rejected(monkeypatch):
    """Minted in the past so it is already expired, rather than sleeping for 30 minutes."""
    from datetime import datetime, timedelta

    real_now = auth._now
    long_ago = datetime.now(UTC) - timedelta(days=2)
    monkeypatch.setattr(auth, "_now", lambda: long_ago)
    token, _, _ = auth.create_token("user-123", "access")
    monkeypatch.setattr(auth, "_now", real_now)  # restore only the clock

    with pytest.raises(auth.AuthError, match="expired"):
        auth.decode_token(token, "access")


def test_garbage_is_rejected_without_raising_something_unexpected():
    for junk in ["", "not.a.token", "a.b.c", "Bearer x"]:
        with pytest.raises(auth.AuthError):
            auth.decode_token(junk, "access")


def test_token_ids_are_hashed_before_storage():
    digest = auth.hash_token_id("some-jti")
    assert digest != "some-jti"
    assert len(digest) == 64  # sha256 hex
    assert auth.hash_token_id("some-jti") == digest  # deterministic


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Alice@Example.COM", "alice@example.com"), ("  bob@x.io  ", "bob@x.io")],
)
def test_emails_are_normalized(raw, expected):
    assert auth.normalize_email(raw) == expected


# ------------------------------------------------------------------------ endpoints
def test_registering_returns_tokens_and_the_user(anon_client):
    response = anon_client.post(
        "/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["access_token"] and body["refresh_token"]
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == config.ACCESS_TOKEN_MINUTES * 60
    assert body["user"]["email"] == TEST_EMAIL
    assert body["user"]["role"] == "user"


def test_a_password_is_never_echoed_back(anon_client):
    response = anon_client.post(
        "/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    )
    assert TEST_PASSWORD not in response.text


def test_registering_the_same_email_twice_is_a_conflict(anon_client):
    payload = {"email": TEST_EMAIL, "password": TEST_PASSWORD}
    anon_client.post("/api/v1/auth/register", json=payload)
    response = anon_client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "email_taken"


def test_email_case_does_not_create_a_second_account(anon_client):
    anon_client.post("/api/v1/auth/register", json={"email": "A@x.com", "password": TEST_PASSWORD})
    response = anon_client.post(
        "/api/v1/auth/register", json={"email": "a@X.com", "password": TEST_PASSWORD}
    )
    assert response.status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "not-an-email", "password": "longenough1"},
        {"email": "a@b.com", "password": "short"},
        {"email": "", "password": "longenough1"},
        {"password": "longenough1"},
        {"email": "a@b.com"},
    ],
)
def test_malformed_registrations_are_rejected(anon_client, payload):
    assert anon_client.post("/api/v1/auth/register", json=payload).status_code == 422


def test_login_returns_a_working_access_token(anon_client):
    anon_client.post("/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    response = anon_client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    )
    assert response.status_code == 200

    token = response.json()["access_token"]
    me = anon_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["email"] == TEST_EMAIL


def test_login_accepts_a_differently_cased_email(anon_client):
    anon_client.post("/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    response = anon_client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL.upper(), "password": TEST_PASSWORD}
    )
    assert response.status_code == 200


def test_a_wrong_password_and_an_unknown_account_are_indistinguishable(anon_client):
    """Distinguishing them turns the login form into an account-enumeration oracle."""
    anon_client.post("/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    wrong_password = anon_client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": "wrong-password"}
    )
    no_such_user = anon_client.post(
        "/api/v1/auth/login", json={"email": "ghost@example.com", "password": TEST_PASSWORD}
    )

    assert wrong_password.status_code == no_such_user.status_code == 401
    assert wrong_password.json()["error"]["message"] == no_such_user.json()["error"]["message"]
    assert wrong_password.json()["error"]["code"] == no_such_user.json()["error"]["code"]


def test_a_weak_hash_is_upgraded_on_the_next_successful_login(anon_client, monkeypatch):
    anon_client.post("/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    before = storage.get_user_by_email(TEST_EMAIL)["password_hash"]

    monkeypatch.setattr(auth, "_SCRYPT_N", 2**9)  # requirements just went up
    monkeypatch.setattr(auth, "_SCRYPT_MAXMEM", 128 * (2**9) * 8 * 2)
    assert (
        anon_client.post(
            "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
        ).status_code
        == 200
    )

    after = storage.get_user_by_email(TEST_EMAIL)["password_hash"]
    assert after != before
    assert after.split("$")[1] == str(2**9)


def test_me_requires_a_token(anon_client):
    response = anon_client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.parametrize(
    "header",
    ["", "Bearer", "Bearer ", "Basic abc", "Bearer not.a.real.token", "token abc"],
)
def test_malformed_authorization_headers_are_rejected(anon_client, header):
    response = anon_client.get("/api/v1/auth/me", headers={"Authorization": header})
    assert response.status_code == 401


def test_a_token_for_a_deleted_account_stops_working(client):
    """The user row is loaded per request, so deletion takes effect immediately rather
    than whenever the access token happens to expire."""
    user_id = client.get("/api/v1/auth/me").json()["id"]
    with storage._connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))

    assert client.get("/api/v1/auth/me").status_code == 401


# ---------------------------------------------------------------------- refresh flow
def test_refreshing_returns_a_new_working_pair(anon_client):
    original = anon_client.post(
        "/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    ).json()

    time.sleep(1.1)  # JWT `iat`/`exp` have one-second resolution
    refreshed = anon_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]}
    )
    assert refreshed.status_code == 200
    body = refreshed.json()
    assert body["access_token"] != original["access_token"]
    assert body["refresh_token"] != original["refresh_token"]

    me = anon_client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"}
    )
    assert me.status_code == 200


def test_a_refresh_token_is_single_use(anon_client):
    """Rotation: replaying a spent token is what makes a theft detectable."""
    original = anon_client.post(
        "/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    ).json()

    anon_client.post("/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]})
    replay = anon_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": original["refresh_token"]}
    )
    assert replay.status_code == 401
    assert replay.json()["error"]["code"] == "invalid_token"


def test_an_access_token_cannot_be_spent_as_a_refresh_token(anon_client):
    tokens = anon_client.post(
        "/api/v1/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    ).json()
    response = anon_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": tokens["access_token"]}
    )
    assert response.status_code == 401


def test_logout_revokes_every_refresh_token(client):
    """Access tokens stay valid until they expire; that is the cost of statelessness,
    and the short lifetime is what makes it acceptable."""
    tokens = client.post(
        "/api/v1/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD}
    ).json()

    assert client.post("/api/v1/auth/logout").json()["revoked"] >= 1
    replay = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert replay.status_code == 401


def test_logout_requires_authentication(anon_client):
    assert anon_client.post("/api/v1/auth/logout").status_code == 401


def test_one_users_refresh_token_cannot_be_replayed_by_another(anon_client):
    alice = anon_client.post(
        "/api/v1/auth/register", json={"email": "alice@example.com", "password": TEST_PASSWORD}
    ).json()
    # Alice logs out, revoking her token; the signature is still perfectly valid.
    anon_client.headers["Authorization"] = f"Bearer {alice['access_token']}"
    anon_client.post("/api/v1/auth/logout")
    del anon_client.headers["Authorization"]

    response = anon_client.post(
        "/api/v1/auth/refresh", json={"refresh_token": alice["refresh_token"]}
    )
    assert response.status_code == 401


# ------------------------------------------------------------------ route protection
PROTECTED = [
    ("get", "/api/v1/meetings"),
    ("post", "/api/v1/meetings/sample"),
    ("get", "/api/v1/meetings/anything"),
    ("delete", "/api/v1/meetings/anything"),
    ("get", "/api/v1/meetings/anything/audio"),
    ("get", "/api/v1/meetings/anything/export"),
    ("post", "/api/v1/chat"),
    ("post", "/api/v1/reindex"),
    ("get", "/api/v1/rag/status"),
    ("get", "/api/v1/auth/me"),
    ("post", "/api/v1/auth/logout"),
]


@pytest.mark.parametrize(("method", "path"), PROTECTED)
def test_every_data_route_requires_authentication(anon_client, method, path):
    kwargs = {"json": {"question": "hi"}} if method == "post" else {}
    response = getattr(anon_client, method)(path, **kwargs)
    # 401 before validation: an unauthenticated caller must not learn whether their
    # body was well-formed, or whether the id they guessed exists.
    assert response.status_code == 401, f"{method.upper()} {path} was not protected"


PUBLIC = [("get", "/api/v1/health"), ("get", "/metrics"), ("get", "/")]


@pytest.mark.parametrize(("method", "path"), PUBLIC)
def test_infrastructure_routes_stay_public(anon_client, method, path):
    """A load balancer and a Prometheus scraper have no credentials."""
    assert getattr(anon_client, method)(path).status_code == 200


# ------------------------------------------------------------------- data isolation
def test_one_user_cannot_read_anothers_meeting(client, second_client):
    mine = client.post("/api/v1/meetings/sample").json()["id"]

    response = second_client.get(f"/api/v1/meetings/{mine}")
    assert response.status_code == 404
    # 404 and not 403: a 403 would confirm the id exists, letting an attacker
    # enumerate which meetings other accounts hold.
    assert response.json()["error"]["code"] == "not_found"


def test_one_user_cannot_delete_anothers_meeting(client, second_client):
    mine = client.post("/api/v1/meetings/sample").json()["id"]

    assert second_client.delete(f"/api/v1/meetings/{mine}").status_code == 404
    assert client.get(f"/api/v1/meetings/{mine}").status_code == 200  # still mine


def test_one_user_cannot_export_anothers_meeting(client, second_client):
    mine = client.post("/api/v1/meetings/sample").json()["id"]
    assert second_client.get(f"/api/v1/meetings/{mine}/export").status_code == 404


def test_one_user_cannot_stream_anothers_audio(client, second_client):
    mine = client.post("/api/v1/meetings/sample").json()["id"]
    assert second_client.get(f"/api/v1/meetings/{mine}/audio").status_code == 404


def test_the_meeting_list_only_shows_your_own(client, second_client):
    client.post("/api/v1/meetings/sample")

    assert client.get("/api/v1/meetings").json()["total"] == 1
    assert second_client.get("/api/v1/meetings").json() == {
        "items": [],
        "total": 0,
        "limit": 50,
        "offset": 0,
    }


def test_rag_chat_never_retrieves_from_another_users_meetings(client, second_client):
    """The headline isolation guarantee. A leak here would be invisible in the UI and
    would come complete with citations, which is what makes it worth its own test."""
    client.post("/api/v1/meetings/sample")

    mine = client.post("/api/v1/chat", json={"question": "payment bug"}).json()
    theirs = second_client.post("/api/v1/chat", json={"question": "payment bug"}).json()

    assert mine["citations"]
    assert theirs["mode"] == "empty"
    assert theirs["citations"] == []


def test_scoping_a_question_to_someone_elses_meeting_is_a_404(client, second_client):
    """Rejected rather than silently widened to 'all meetings', which would answer
    from a different set than the caller asked about."""
    mine = client.post("/api/v1/meetings/sample").json()["id"]

    response = second_client.post(
        "/api/v1/chat", json={"question": "payment bug", "meeting_id": mine}
    )
    assert response.status_code == 404


def test_rag_status_counts_only_your_own(client, second_client):
    client.post("/api/v1/meetings/sample")
    client.post("/api/v1/chat", json={"question": "payment"})  # forces indexing

    assert client.get("/api/v1/rag/status").json()["indexed_meetings"] == 1
    assert second_client.get("/api/v1/rag/status").json()["indexed_meetings"] == 0


def test_reindex_only_touches_your_own(client, second_client):
    client.post("/api/v1/meetings/sample")
    assert second_client.post("/api/v1/reindex").json()["meetings"] == 0
    assert client.post("/api/v1/reindex").json()["meetings"] == 1


# --------------------------------------------------------------- first-account claim
def test_the_first_account_claims_meetings_created_before_auth_existed(anon_client):
    """Upgrading an existing local install must not strand the data already in it."""
    with storage._connect() as conn:
        conn.execute(
            "INSERT INTO meetings (id, title, created_at, status) "
            "VALUES ('legacy', 'Pre-auth meeting', '2024-01-01', 'done')"
        )

    sign_up(anon_client, TEST_EMAIL, TEST_PASSWORD)
    assert anon_client.get("/api/v1/meetings/legacy").status_code == 200


def test_a_later_account_claims_nothing(anon_client):
    """Only the first registration claims; otherwise anyone could sign up and inherit
    another user's data."""
    with storage._connect() as conn:
        conn.execute(
            "INSERT INTO meetings (id, title, created_at, status) "
            "VALUES ('legacy', 'Pre-auth meeting', '2024-01-01', 'done')"
        )
    sign_up(anon_client, "first@example.com", TEST_PASSWORD)

    second = sign_up(TestClient(app), "second@example.com", TEST_PASSWORD)
    assert second.get("/api/v1/meetings/legacy").status_code == 404
