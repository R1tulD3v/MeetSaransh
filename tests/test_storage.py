"""Storage layer: migrations, ownership, CRUD, cascade deletes, pagination, vectors."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from app import config, storage


def _transcript(segments: list[dict] | None = None) -> dict:
    segments = segments or [{"start": 0.0, "end": 3.0, "text": "hello"}]
    return {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
        "timestamped_text": "[00:00] hello",
        "duration": segments[-1]["end"],
    }


def _user(email: str = "owner@example.com") -> str:
    """Create a user and return its id. Every meeting needs an owner."""
    return storage.create_user(email=email, password_hash="scrypt$fake")["id"]


def _make(title: str = "Meeting", user_id: str | None = None) -> str:
    return storage.create_meeting(
        user_id=user_id or _user(f"{title.replace(' ', '-').lower()}@example.com"),
        title=title,
        filename=f"{title}.mp3",
        transcript=_transcript(),
        summary={"tldr": "x"},
    )


# ------------------------------------------------------------------------- migrations
def test_fresh_database_is_migrated_to_head():
    storage.init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION


def test_migrations_are_idempotent():
    """Re-opening an already-migrated file must not re-run or fail on any migration."""
    storage.init_db()
    storage.reset_migration_cache()
    storage.init_db()  # would raise "table chunks_v2 already exists" if v2 re-ran
    assert storage.healthcheck() is True


def test_migration_upgrades_a_v1_database_in_place():
    """A database created before migration 2 must upgrade without losing rows."""
    config.ensure_dirs()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.executescript(storage._MIGRATIONS[0][1])
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO meetings (id, title, created_at) VALUES ('m1', 'Old', '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO chunks (id, meeting_id, ord, text) VALUES ('c1', 'm1', 0, 'legacy')"
        )

    storage.reset_migration_cache()
    storage.init_db()

    assert storage.healthcheck() is True
    # The legacy meeting predates ownership, so it is claimed the way a real upgrade
    # claims it: by the first account to register.
    uid = _user()
    assert storage.claim_unowned_meetings(uid) == 1
    assert [c["text"] for c in storage.get_chunks(uid, "m1")] == ["legacy"]


def test_a_pre_auth_database_upgrades_to_head_on_first_connection():
    """A real v3 file -- built by running only migrations 1-3 -- must upgrade in place.

    Built genuinely rather than by rewinding `user_version` on a current database: the
    columns really are absent, which is what makes this a test of the upgrade rather
    than of the version counter.
    """
    config.ensure_dirs()
    with sqlite3.connect(config.DB_PATH) as conn:
        for version, script in storage._MIGRATIONS[:3]:
            conn.executescript(script)
            conn.execute(f"PRAGMA user_version = {version}")
        conn.execute(
            "INSERT INTO meetings (id, title, created_at) VALUES ('m1', 'Old', '2024-01-01')"
        )
    storage.reset_migration_cache()

    assert storage.healthcheck() is True
    with sqlite3.connect(config.DB_PATH) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION

    # Migration 5 backfills `status`: an existing row is a finished meeting by
    # definition, so it must not come back as 'queued' and get re-processed.
    uid = _user()
    storage.claim_unowned_meetings(uid)
    assert storage.get_meeting("m1", uid)["status"] == "done"


def test_healthcheck_false_when_the_database_is_unusable(monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", config.DATA_DIR / "audio")  # a directory
    storage.reset_migration_cache()
    assert storage.healthcheck() is False


# ------------------------------------------------------------------------------- users
def test_create_and_fetch_a_user():
    created = storage.create_user(email="a@example.com", password_hash="hashed")
    assert storage.get_user(created["id"])["email"] == "a@example.com"
    assert storage.get_user_by_email("a@example.com")["id"] == created["id"]
    assert created["role"] == "user"


def test_email_uniqueness_is_enforced_by_the_database():
    """The unique index is the real guard -- a check-then-insert would be a race."""
    storage.create_user(email="dup@example.com", password_hash="h")
    with pytest.raises(sqlite3.IntegrityError):
        storage.create_user(email="dup@example.com", password_hash="h2")


def test_unknown_users_are_none_not_an_error():
    assert storage.get_user("nope") is None
    assert storage.get_user_by_email("nobody@example.com") is None


def test_password_hash_can_be_upgraded_in_place():
    uid = _user()
    storage.update_password_hash(uid, "scrypt$stronger")
    assert storage.get_user(uid)["password_hash"] == "scrypt$stronger"


def test_deleting_a_user_cascades_to_their_meetings():
    uid = _user()
    _make(user_id=uid)
    with storage._connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    assert storage.count_meetings() == 0


# -------------------------------------------------------------------- claiming legacy
def test_claiming_assigns_only_ownerless_meetings():
    owner = _user("first@example.com")
    mine = _make(user_id=owner)
    with storage._connect() as conn:
        conn.execute(
            "INSERT INTO meetings (id, title, created_at, status) "
            "VALUES ('legacy', 'Pre-auth', '2024-01-01', 'done')"
        )

    claimer = _user("second@example.com")
    assert storage.claim_unowned_meetings(claimer) == 1
    assert storage.get_meeting("legacy", claimer) is not None
    assert storage.get_meeting(mine, claimer) is None  # someone else's stays theirs


def test_claiming_on_a_fresh_database_is_a_no_op():
    assert storage.claim_unowned_meetings(_user()) == 0


# ---------------------------------------------------------------------- refresh tokens
def test_a_stored_refresh_token_is_active_until_revoked():
    uid = _user()
    storage.store_refresh_token(jti_hash="h1", user_id=uid, expires_at="2099-01-01T00:00:00")

    assert storage.refresh_token_is_active("h1", uid) is True
    assert storage.revoke_refresh_token("h1") is True
    assert storage.refresh_token_is_active("h1", uid) is False


def test_an_unknown_token_is_never_active():
    assert storage.refresh_token_is_active("never-issued", _user()) is False


def test_a_token_cannot_be_replayed_against_another_account():
    """A signature can be perfectly valid while the token belongs to somebody else."""
    alice, mallory = _user("alice@example.com"), _user("mallory@example.com")
    storage.store_refresh_token(jti_hash="h1", user_id=alice, expires_at="2099-01-01T00:00:00")
    assert storage.refresh_token_is_active("h1", mallory) is False


def test_revoking_all_tokens_logs_a_user_out_everywhere():
    uid = _user()
    for i in range(3):
        storage.store_refresh_token(jti_hash=f"h{i}", user_id=uid, expires_at="2099-01-01T00:00:00")
    assert storage.revoke_all_refresh_tokens(uid) == 3
    assert storage.revoke_all_refresh_tokens(uid) == 0  # already revoked
    assert all(not storage.refresh_token_is_active(f"h{i}", uid) for i in range(3))


def test_expired_tokens_are_purged_but_live_ones_are_kept():
    uid = _user()
    storage.store_refresh_token(jti_hash="old", user_id=uid, expires_at="2000-01-01T00:00:00")
    storage.store_refresh_token(jti_hash="new", user_id=uid, expires_at="2099-01-01T00:00:00")

    assert storage.purge_expired_refresh_tokens() == 1
    assert storage.refresh_token_is_active("new", uid) is True


# ------------------------------------------------------------------------------- CRUD
def test_create_and_get_meeting_round_trip():
    uid = _user()
    segments = [
        {"start": 0.0, "end": 4.0, "text": "first"},
        {"start": 4.0, "end": 9.0, "text": "second"},
    ]
    mid = storage.create_meeting(
        user_id=uid,
        title="Planning",
        filename="planning.mp3",
        transcript=_transcript(segments),
        summary={"tldr": "we planned"},
        audio_ext=".mp3",
    )
    got = storage.get_meeting(mid, uid)
    assert got is not None
    assert got["title"] == "Planning"
    assert got["audio_ext"] == ".mp3"
    assert got["duration"] == 9.0
    assert got["status"] == "done"
    assert [s["text"] for s in got["segments"]] == ["first", "second"]
    assert got["summary"] == {"tldr": "we planned"}


def test_get_missing_meeting_returns_none():
    assert storage.get_meeting("does-not-exist", _user()) is None


def test_blank_title_falls_back_to_a_placeholder():
    uid = _user()
    mid = storage.create_meeting(
        user_id=uid, title="", filename=None, transcript=_transcript(), summary={}
    )
    stored = storage.get_meeting(mid, uid)
    assert stored is not None
    assert stored["title"] == "Untitled meeting"


def test_delete_meeting_reports_whether_a_row_was_removed():
    uid = _user()
    mid = _make(user_id=uid)
    assert storage.delete_meeting(mid, uid) is True
    assert storage.delete_meeting(mid, uid) is False  # second delete is a no-op


def test_deleting_a_meeting_cascades_to_its_chunks():
    """The FK carries ON DELETE CASCADE and PRAGMA foreign_keys is on, so no orphans."""
    uid = _user()
    mid = _make(user_id=uid)
    storage.replace_chunks(mid, [{"ord": 0, "start": 0.0, "end": 3.0, "text": "c", "segs": []}])
    assert storage.count_chunks() == 1

    storage.delete_meeting(mid, uid)
    assert storage.count_chunks() == 0


def test_chunks_cannot_reference_a_missing_meeting():
    """Foreign keys are enforced, not merely declared."""
    with pytest.raises(sqlite3.IntegrityError):
        storage.replace_chunks("ghost-meeting", [{"ord": 0, "text": "orphan", "segs": []}])


# ----------------------------------------------------------------- ownership isolation
def test_one_user_cannot_read_anothers_meeting():
    alice, mallory = _user("alice@example.com"), _user("mallory@example.com")
    mid = _make("Private", user_id=alice)

    assert storage.get_meeting(mid, alice) is not None
    assert storage.get_meeting(mid, mallory) is None


def test_one_user_cannot_delete_anothers_meeting():
    alice, mallory = _user("alice@example.com"), _user("mallory@example.com")
    mid = _make("Private", user_id=alice)

    assert storage.delete_meeting(mid, mallory) is False
    assert storage.get_meeting(mid, alice) is not None  # still there


def test_listing_and_counting_only_ever_see_your_own():
    alice, mallory = _user("alice@example.com"), _user("mallory@example.com")
    _make("A1", user_id=alice)
    _make("A2", user_id=alice)
    _make("M1", user_id=mallory)

    assert {m["title"] for m in storage.list_meetings(alice)} == {"A1", "A2"}
    assert {m["title"] for m in storage.list_meetings(mallory)} == {"M1"}
    assert storage.count_meetings(alice) == 2
    assert storage.count_meetings() == 3  # unscoped, for metrics only


def test_retrieval_chunks_are_scoped_to_their_owner():
    """The single most important ownership filter: RAG reads straight out of here."""
    alice, mallory = _user("alice@example.com"), _user("mallory@example.com")
    a_meeting = _make("Alices", user_id=alice)
    m_meeting = _make("Mallorys", user_id=mallory)
    storage.replace_chunks(a_meeting, [{"ord": 0, "text": "alice secret", "segs": []}])
    storage.replace_chunks(m_meeting, [{"ord": 0, "text": "mallory secret", "segs": []}])

    assert [c["text"] for c in storage.get_chunks(alice)] == ["alice secret"]
    assert [c["text"] for c in storage.get_chunks(mallory)] == ["mallory secret"]
    assert storage.count_chunks(alice) == 1
    assert storage.indexed_meeting_ids(alice) == {a_meeting}


def test_scoping_to_another_users_meeting_id_returns_nothing():
    """Naming someone else's meeting must not widen access to it."""
    alice, mallory = _user("alice@example.com"), _user("mallory@example.com")
    a_meeting = _make("Alices", user_id=alice)
    storage.replace_chunks(a_meeting, [{"ord": 0, "text": "alice secret", "segs": []}])

    assert storage.get_chunks(mallory, a_meeting) == []


# ------------------------------------------------------------------------- job status
def test_a_queued_meeting_starts_empty_and_completes_later():
    uid = _user()
    mid = storage.create_meeting(user_id=uid, title="Recording", filename="r.mp3", status="queued")
    queued = storage.get_meeting(mid, uid)
    assert queued["status"] == "queued"
    assert queued["segments"] == []

    storage.complete_meeting(
        mid,
        transcript=_transcript([{"start": 0.0, "end": 8.0, "text": "spoken words"}]),
        summary={"tldr": "done"},
    )
    finished = storage.get_meeting(mid, uid)
    assert finished["status"] == "done"
    assert finished["duration"] == 8.0
    assert finished["summary"]["tldr"] == "done"
    assert finished["error"] is None


def test_completing_a_meeting_preserves_its_title_when_none_is_given():
    uid = _user()
    mid = storage.create_meeting(user_id=uid, title="Chosen name", filename=None, status="queued")
    storage.complete_meeting(mid, transcript=_transcript(), summary={})
    assert storage.get_meeting(mid, uid)["title"] == "Chosen name"


def test_status_transitions_record_stage_and_error():
    uid = _user()
    mid = storage.create_meeting(user_id=uid, title="R", filename=None, status="queued")

    storage.set_meeting_status(mid, "processing", stage="transcribing")
    assert storage.get_meeting(mid, uid)["stage"] == "transcribing"

    storage.set_meeting_status(mid, "error", error="provider exploded")
    failed = storage.get_meeting(mid, uid)
    assert failed["status"] == "error"
    assert failed["error"] == "provider exploded"
    assert failed["stage"] is None


def test_active_job_count_only_counts_unfinished_work():
    uid = _user()
    storage.create_meeting(user_id=uid, title="a", filename=None, status="queued")
    storage.create_meeting(user_id=uid, title="b", filename=None, status="processing")
    storage.create_meeting(user_id=uid, title="c", filename=None, status="done")
    storage.create_meeting(user_id=uid, title="d", filename=None, status="error")

    assert storage.count_active_jobs(uid) == 2
    assert storage.count_active_jobs(_user("other@example.com")) == 0


def test_interrupted_jobs_are_findable_after_a_crash():
    uid = _user()
    stuck = storage.create_meeting(user_id=uid, title="stuck", filename=None, status="processing")
    storage.create_meeting(user_id=uid, title="fine", filename=None, status="done")

    assert [m["id"] for m in storage.find_interrupted_meetings()] == [stuck]


def test_the_worker_can_read_a_meeting_without_knowing_its_owner():
    """The background worker acts for the system, not for a request."""
    mid = _make(user_id=_user())
    assert storage.get_meeting_unscoped(mid) is not None
    assert storage.get_meeting_unscoped("nope") is None


# ------------------------------------------------------------------------- pagination
def test_list_meetings_paginates_and_counts():
    uid = _user()
    ids = [_make(f"Meeting {i}", user_id=uid) for i in range(5)]
    assert storage.count_meetings(uid) == 5

    page = storage.list_meetings(uid, limit=2, offset=0)
    second = storage.list_meetings(uid, limit=2, offset=2)
    assert len(page) == 2
    assert len(second) == 2
    assert {m["id"] for m in page}.isdisjoint({m["id"] for m in second})
    assert {m["id"] for m in storage.list_meetings(uid)} == set(ids)


def test_list_meetings_omits_the_heavy_columns():
    """The sidebar query must not drag full transcripts across the wire."""
    uid = _user()
    _make(user_id=uid)
    row = storage.list_meetings(uid)[0]
    assert set(row) == {"id", "title", "created_at", "duration", "status", "stage", "error"}


# ------------------------------------------------------------------------ chunks (RAG)
def test_embeddings_survive_the_blob_round_trip():
    uid = _user()
    mid = _make(user_id=uid)
    vector = np.array([0.5, -0.25, 0.125, 2.0], dtype=np.float32)
    storage.replace_chunks(
        mid,
        [{"ord": 0, "start": 1.0, "end": 2.0, "text": "hello", "segs": [], "embedding": vector}],
    )
    stored = storage.get_chunks(uid, mid)[0]
    assert np.array_equal(stored["embedding"], vector)
    assert stored["meeting_title"] == "Meeting"


def test_chunks_without_embeddings_round_trip_as_none():
    """Lexical-only mode stores NULL vectors; retrieval must see None, not zeros."""
    uid = _user()
    mid = _make(user_id=uid)
    storage.replace_chunks(mid, [{"ord": 0, "text": "no vector", "segs": []}])
    assert storage.get_chunks(uid, mid)[0]["embedding"] is None


def test_replace_chunks_is_a_replacement_not_an_append():
    uid = _user()
    mid = _make(user_id=uid)
    storage.replace_chunks(mid, [{"ord": 0, "text": "old", "segs": []}])
    storage.replace_chunks(mid, [{"ord": 0, "text": "new", "segs": []}])
    assert [c["text"] for c in storage.get_chunks(uid, mid)] == ["new"]


def test_get_chunks_scopes_to_one_meeting():
    uid = _user()
    a, b = _make("A", user_id=uid), _make("B", user_id=uid)
    storage.replace_chunks(a, [{"ord": 0, "text": "from a", "segs": []}])
    storage.replace_chunks(b, [{"ord": 0, "text": "from b", "segs": []}])

    assert [c["text"] for c in storage.get_chunks(uid, a)] == ["from a"]
    assert len(storage.get_chunks(uid)) == 2
    assert storage.indexed_meeting_ids(uid) == {a, b}


def test_segments_are_preserved_on_chunks_for_citation_precision():
    uid = _user()
    mid = _make(user_id=uid)
    segs = [{"start": 3.0, "text": "the precise line"}]
    storage.replace_chunks(mid, [{"ord": 0, "text": "chunk text", "segs": segs}])
    assert storage.get_chunks(uid, mid)[0]["segs"] == segs
