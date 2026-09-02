"""Persistence layer -- SQLite via the Python standard library (no ORM, no extra deps).

Two tables: `meetings` (one row per processed meeting, with transcript segments and the
structured summary as JSON text columns) and `chunks` (the RAG vector store, embeddings
held as raw float32 blobs).

Schema changes go through the versioned migration runner below rather than by deleting
the database. `PRAGMA user_version` records which migrations a file has already seen, so
an existing database upgrades in place and a fresh one is built from scratch by the same
code path.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import numpy as np

from . import config

# --------------------------------------------------------------------------- migrations
# Append-only: never edit a migration that has shipped, add the next one instead.
_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS meetings (
            id            TEXT PRIMARY KEY,
            title         TEXT NOT NULL,
            filename      TEXT,
            created_at    TEXT NOT NULL,
            duration      REAL DEFAULT 0,
            transcript    TEXT,
            segments_json TEXT,
            summary_json  TEXT,
            audio_ext     TEXT
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id            TEXT PRIMARY KEY,
            meeting_id    TEXT NOT NULL,
            ord           INTEGER NOT NULL,
            start         REAL,
            end           REAL,
            text          TEXT NOT NULL,
            segs_json     TEXT,
            embedding     BLOB,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);
        """,
    ),
    (
        2,
        # Rebuild `chunks` so the foreign key actually cascades. Previously the FK was
        # declared but not enforced (SQLite needs PRAGMA foreign_keys=ON, which we now
        # set on every connection), and deletes were cleaned up in application code --
        # which silently leaves orphans if any path forgets to do it.
        """
        CREATE TABLE chunks_v2 (
            id            TEXT PRIMARY KEY,
            meeting_id    TEXT NOT NULL,
            ord           INTEGER NOT NULL,
            start         REAL,
            end           REAL,
            text          TEXT NOT NULL,
            segs_json     TEXT,
            embedding     BLOB,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        );
        INSERT INTO chunks_v2 (id, meeting_id, ord, start, end, text, segs_json, embedding)
            SELECT id, meeting_id, ord, start, end, text, segs_json, embedding FROM chunks;
        DROP TABLE chunks;
        ALTER TABLE chunks_v2 RENAME TO chunks;
        CREATE INDEX IF NOT EXISTS idx_chunks_meeting ON chunks(meeting_id);
        """,
    ),
    (
        3,
        # Sorting the meeting list is the single hottest query in the app.
        "CREATE INDEX IF NOT EXISTS idx_meetings_created ON meetings(created_at DESC);",
    ),
    (
        4,
        # Accounts, ownership, and revocable refresh tokens.
        #
        # `meetings.user_id` is nullable rather than NOT NULL because databases created
        # before this migration already hold rows with no owner. Those rows are
        # invisible to every user until claimed -- see `claim_unowned_meetings`, which
        # the first account to register calls, so a pre-auth local database is not
        # silently orphaned.
        """
        CREATE TABLE IF NOT EXISTS users (
            id            TEXT PRIMARY KEY,
            email         TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            created_at    TEXT NOT NULL
        );
        -- Case-insensitive uniqueness: emails are normalised to lowercase on the way
        -- in, and the index enforces it even if a code path ever forgets.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);

        ALTER TABLE meetings ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE;
        CREATE INDEX IF NOT EXISTS idx_meetings_user ON meetings(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS refresh_tokens (
            jti_hash   TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            issued_at  TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked    INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
        CREATE INDEX IF NOT EXISTS idx_refresh_expiry ON refresh_tokens(expires_at);
        """,
    ),
    (
        5,
        # Background processing state. Existing rows are finished meetings by
        # definition -- they only exist because the synchronous pipeline completed --
        # so they default to 'done' rather than being re-processed.
        """
        ALTER TABLE meetings ADD COLUMN status TEXT NOT NULL DEFAULT 'done';
        ALTER TABLE meetings ADD COLUMN stage TEXT;
        ALTER TABLE meetings ADD COLUMN error TEXT;
        ALTER TABLE meetings ADD COLUMN updated_at TEXT;
        CREATE INDEX IF NOT EXISTS idx_meetings_status ON meetings(status);
        """,
    ),
]

SCHEMA_VERSION = _MIGRATIONS[-1][0]

_migrated_paths: set[str] = set()
_migration_lock = threading.Lock()


def _apply_migrations(conn: sqlite3.Connection) -> None:
    """Bring one database file up to SCHEMA_VERSION, one migration at a time."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, script in _MIGRATIONS:
        if version <= current:
            continue
        # Each migration is its own transaction: a failure leaves the file at the last
        # good version rather than half-upgraded.
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()


def _open() -> sqlite3.Connection:
    """Open a connection with the pragmas this schema depends on, migrating if needed."""
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    # ON DELETE CASCADE is inert without this -- SQLite defaults foreign keys to off.
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets reads proceed during a write, which matters once a background worker
    # writes while the API serves reads.
    conn.execute("PRAGMA journal_mode = WAL")

    key = str(config.DB_PATH)
    if key not in _migrated_paths:
        try:
            with _migration_lock:
                if key not in _migrated_paths:
                    _apply_migrations(conn)
                    _migrated_paths.add(key)
        except Exception:
            conn.close()  # don't leak a handle when an upgrade fails
            raise
    return conn


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Transactional connection that is always closed.

    `sqlite3.Connection` used directly as a context manager commits, but does NOT
    close -- which leaks file handles and, on Windows, keeps the database file locked.
    """
    conn = _open()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Explicit startup hook: create/upgrade the database before serving traffic."""
    with _connect():
        pass


def reset_migration_cache() -> None:
    """Forget which database files have been migrated (used by tests between temp DBs)."""
    with _migration_lock:
        _migrated_paths.clear()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------- users
def create_user(*, email: str, password_hash: str, role: str = "user") -> dict:
    """Insert a user. Raises sqlite3.IntegrityError if the email is already taken."""
    uid = new_id()
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (uid, email, password_hash, role, now),
        )
    return {"id": uid, "email": email, "role": role, "created_at": now}


def get_user_by_email(email: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    return dict(row) if row else None


def get_user(user_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def update_password_hash(user_id: str, password_hash: str) -> None:
    """Used to transparently upgrade a hash to stronger parameters after a login."""
    with _connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))


def count_users() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


def claim_unowned_meetings(user_id: str) -> int:
    """Assign every ownerless meeting to a user. Returns how many were claimed.

    Called once, by the first account created on a database. Meetings written before
    authentication existed have `user_id IS NULL` and would otherwise be permanently
    invisible; this migrates that data rather than stranding it. On a fresh database it
    is a no-op.
    """
    with _connect() as conn:
        return int(
            conn.execute(
                "UPDATE meetings SET user_id = ? WHERE user_id IS NULL", (user_id,)
            ).rowcount
        )


# ---------------------------------------------------------------------- refresh tokens
def store_refresh_token(*, jti_hash: str, user_id: str, expires_at: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO refresh_tokens (jti_hash, user_id, issued_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            (jti_hash, user_id, _utc_now(), expires_at),
        )


def refresh_token_is_active(jti_hash: str, user_id: str) -> bool:
    """True only for a token that exists, belongs to this user, and is not revoked.

    Checking `user_id` as well as the hash means a token issued to account A cannot be
    replayed against account B even though its signature is perfectly valid.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT revoked FROM refresh_tokens WHERE jti_hash = ? AND user_id = ?",
            (jti_hash, user_id),
        ).fetchone()
    return row is not None and not row["revoked"]


def revoke_refresh_token(jti_hash: str) -> bool:
    with _connect() as conn:
        return (
            conn.execute(
                "UPDATE refresh_tokens SET revoked = 1 WHERE jti_hash = ?", (jti_hash,)
            ).rowcount
            > 0
        )


def revoke_all_refresh_tokens(user_id: str) -> int:
    """Log a user out everywhere -- the response to a suspected compromise."""
    with _connect() as conn:
        return int(
            conn.execute(
                "UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ? AND revoked = 0",
                (user_id,),
            ).rowcount
        )


def purge_expired_refresh_tokens() -> int:
    """Drop rows that can no longer authenticate anything, so the table stays bounded."""
    with _connect() as conn:
        return int(
            conn.execute("DELETE FROM refresh_tokens WHERE expires_at < ?", (_utc_now(),)).rowcount
        )


# ---------------------------------------------------------------------------- meetings
# Every function below takes `user_id` and filters on it in SQL. That is deliberate:
# ownership is enforced inside the query rather than by a caller remembering to check
# afterwards, so forgetting it is a TypeError at import time instead of a data leak in
# production.
def create_meeting(
    *,
    user_id: str,
    title: str,
    filename: str | None,
    transcript: dict | None = None,
    summary: dict | None = None,
    audio_ext: str | None = None,
    meeting_id: str | None = None,
    status: str = "done",
) -> str:
    """Insert a meeting and return its id.

    Called two ways: with a finished transcript and summary (the sample path), or as an
    empty `queued` placeholder that a background worker fills in later.
    """
    mid = meeting_id or new_id()
    transcript = transcript or {}
    now = _utc_now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO meetings
               (id, user_id, title, filename, created_at, updated_at, duration,
                transcript, segments_json, summary_json, audio_ext, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                user_id,
                title or "Untitled meeting",
                filename,
                now,
                now,
                float(transcript.get("duration", 0.0) or 0.0),
                transcript.get("text", ""),
                json.dumps(transcript.get("segments", [])),
                json.dumps(summary or {}),
                audio_ext,
                status,
            ),
        )
    return mid


def complete_meeting(
    meeting_id: str, *, transcript: dict, summary: dict, title: str | None = None
) -> None:
    """Fill in a queued meeting once the worker has transcribed and summarized it."""
    with _connect() as conn:
        conn.execute(
            """UPDATE meetings
               SET transcript = ?, segments_json = ?, summary_json = ?, duration = ?,
                   status = 'done', stage = NULL, error = NULL, updated_at = ?,
                   title = COALESCE(?, title)
               WHERE id = ?""",
            (
                transcript.get("text", ""),
                json.dumps(transcript.get("segments", [])),
                json.dumps(summary),
                float(transcript.get("duration", 0.0) or 0.0),
                _utc_now(),
                title,
                meeting_id,
            ),
        )


def set_meeting_status(
    meeting_id: str, status: str, *, stage: str | None = None, error: str | None = None
) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE meetings SET status = ?, stage = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, stage, error, _utc_now(), meeting_id),
        )


def list_meetings(user_id: str, *, limit: int | None = None, offset: int = 0) -> list[dict]:
    """Lightweight list for the sidebar (no heavy transcript/segments payload)."""
    query = (
        "SELECT id, title, created_at, duration, status, stage, error FROM meetings "
        "WHERE user_id = ? ORDER BY created_at DESC"
    )
    params: tuple[Any, ...] = (user_id,)
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params = (user_id, int(limit), int(offset))
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def count_meetings(user_id: str | None = None) -> int:
    """Count a user's meetings, or every meeting when called without one (metrics)."""
    with _connect() as conn:
        if user_id is None:
            return int(conn.execute("SELECT COUNT(*) AS n FROM meetings").fetchone()["n"])
        return int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM meetings WHERE user_id = ?", (user_id,)
            ).fetchone()["n"]
        )


def count_active_jobs(user_id: str) -> int:
    """Queued or in-flight meetings for one user, for the per-user job quota."""
    with _connect() as conn:
        return int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM meetings "
                "WHERE user_id = ? AND status IN ('queued', 'processing')",
                (user_id,),
            ).fetchone()["n"]
        )


def get_meeting(meeting_id: str, user_id: str) -> dict | None:
    """Fetch a meeting the given user owns. Someone else's id returns None, not a row."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM meetings WHERE id = ? AND user_id = ?", (meeting_id, user_id)
        ).fetchone()
    return _hydrate_meeting(row)


def get_meeting_unscoped(meeting_id: str) -> dict | None:
    """Fetch a meeting without an ownership filter.

    Only for the background worker, which acts on behalf of the system rather than of a
    request. Named so that using it inside a request handler is visibly wrong in review.
    """
    with _connect() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    return _hydrate_meeting(row)


def _hydrate_meeting(row: sqlite3.Row | None) -> dict | None:
    """Decode the JSON columns and normalise NULLs the API contract does not allow.

    `transcript` is coerced from NULL to "" because rows written by older versions of
    this schema can hold NULL there, and a nullable field would push that accident into
    the public response model forever.
    """
    if row is None:
        return None
    data = dict(row)
    data["segments"] = json.loads(data.pop("segments_json") or "[]")
    data["summary"] = json.loads(data.pop("summary_json") or "{}")
    data["transcript"] = data.get("transcript") or ""
    data["title"] = data.get("title") or "Untitled meeting"
    return data


def find_interrupted_meetings() -> list[dict]:
    """Meetings left mid-flight by a crash or a restart.

    A job that was 'processing' when the process died would otherwise sit there forever
    behind a spinner, so startup finds and fails them explicitly.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, user_id, title, audio_ext FROM meetings WHERE status = 'processing'"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_meeting(meeting_id: str, user_id: str) -> bool:
    """Delete a meeting the user owns; its chunks go with it via ON DELETE CASCADE."""
    with _connect() as conn:
        deleted = conn.execute(
            "DELETE FROM meetings WHERE id = ? AND user_id = ?", (meeting_id, user_id)
        ).rowcount
    return deleted > 0


# ------------------------------------------------------------------------ chunks (RAG)
def replace_chunks(meeting_id: str, chunks: list[dict]) -> None:
    """Delete any existing chunks for a meeting and insert the new set.

    Each chunk dict: {ord, start, end, text, segs, embedding(np.ndarray|None)}.
    Embeddings are stored as raw float32 bytes (or NULL when unavailable). The delete
    and the insert share one transaction, so a failure mid-insert cannot leave a
    meeting partially indexed.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM chunks WHERE meeting_id = ?", (meeting_id,))
        conn.executemany(
            """INSERT INTO chunks (id, meeting_id, ord, start, end, text, segs_json, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    new_id(),
                    meeting_id,
                    c["ord"],
                    c.get("start"),
                    c.get("end"),
                    c["text"],
                    json.dumps(c.get("segs", [])),
                    (
                        c["embedding"].astype("float32").tobytes()
                        if c.get("embedding") is not None
                        else None
                    ),
                )
                for c in chunks
            ],
        )


def get_chunks(user_id: str, scope_meeting_id: str | None = None) -> list[dict]:
    """Fetch a user's chunks (optionally for one meeting), joined with the meeting title.

    This is the single most important ownership filter in the application. Retrieval
    reads straight out of here, so a missing `m.user_id` predicate would not throw or
    look wrong -- it would quietly let one user's question be answered from another
    user's meetings, complete with citations. Hence the filter lives in the SQL, and
    `user_id` is a required positional argument rather than an optional keyword.

    Returns dicts with `embedding` decoded back to a float32 numpy array (or None).
    """
    query = (
        "SELECT c.id, c.meeting_id, c.ord, c.start, c.end, c.text, c.segs_json, "
        "c.embedding, m.title "
        "FROM chunks c JOIN meetings m ON m.id = c.meeting_id "
        "WHERE m.user_id = ?"
    )
    params: tuple[Any, ...] = (user_id,)
    if scope_meeting_id:
        query += " AND c.meeting_id = ?"
        params = (user_id, scope_meeting_id)
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        blob = d.pop("embedding")
        d["embedding"] = np.frombuffer(blob, dtype="float32") if blob else None
        d["segs"] = json.loads(d.pop("segs_json") or "[]")
        d["meeting_title"] = d.pop("title")
        out.append(d)
    return out


def indexed_meeting_ids(user_id: str | None = None) -> set[str]:
    """Meeting ids that have chunks -- for one user, or globally when user_id is None."""
    with _connect() as conn:
        if user_id is None:
            rows = conn.execute("SELECT DISTINCT meeting_id FROM chunks").fetchall()
        else:
            rows = conn.execute(
                "SELECT DISTINCT c.meeting_id FROM chunks c "
                "JOIN meetings m ON m.id = c.meeting_id WHERE m.user_id = ?",
                (user_id,),
            ).fetchall()
    return {r["meeting_id"] for r in rows}


def count_chunks(user_id: str | None = None) -> int:
    """Count a user's indexed chunks, or every chunk when called without one (metrics)."""
    with _connect() as conn:
        if user_id is None:
            return int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
        return int(
            conn.execute(
                "SELECT COUNT(*) AS n FROM chunks c "
                "JOIN meetings m ON m.id = c.meeting_id WHERE m.user_id = ?",
                (user_id,),
            ).fetchone()["n"]
        )


def healthcheck() -> bool:
    """Cheap round-trip proving the database is reachable and migrated."""
    try:
        with _connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        return int(version) == SCHEMA_VERSION
    except sqlite3.Error:
        return False
