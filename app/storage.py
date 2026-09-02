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


# ---------------------------------------------------------------------------- meetings
def create_meeting(
    *,
    title: str,
    filename: str | None,
    transcript: dict,
    summary: dict,
    audio_ext: str | None = None,
    meeting_id: str | None = None,
) -> str:
    """Insert a fully-processed meeting and return its id."""
    mid = meeting_id or new_id()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO meetings
               (id, title, filename, created_at, duration, transcript, segments_json,
                summary_json, audio_ext)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                title or "Untitled meeting",
                filename,
                datetime.now(UTC).isoformat(timespec="seconds"),
                float(transcript.get("duration", 0.0) or 0.0),
                transcript.get("text", ""),
                json.dumps(transcript.get("segments", [])),
                json.dumps(summary),
                audio_ext,
            ),
        )
    return mid


def list_meetings(*, limit: int | None = None, offset: int = 0) -> list[dict]:
    """Lightweight list for the sidebar (no heavy transcript/segments payload)."""
    query = "SELECT id, title, created_at, duration FROM meetings ORDER BY created_at DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        query += " LIMIT ? OFFSET ?"
        params = (int(limit), int(offset))
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def count_meetings() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM meetings").fetchone()["n"])


def get_meeting(meeting_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["segments"] = json.loads(data.pop("segments_json") or "[]")
    data["summary"] = json.loads(data.pop("summary_json") or "{}")
    return data


def delete_meeting(meeting_id: str) -> bool:
    """Delete a meeting; its chunks go with it via ON DELETE CASCADE."""
    with _connect() as conn:
        deleted = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,)).rowcount
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


def get_chunks(scope_meeting_id: str | None = None) -> list[dict]:
    """Fetch chunks (optionally for one meeting), joined with the meeting title.

    Returns dicts with `embedding` decoded back to a float32 numpy array (or None).
    """
    query = (
        "SELECT c.id, c.meeting_id, c.ord, c.start, c.end, c.text, c.segs_json, "
        "c.embedding, m.title "
        "FROM chunks c JOIN meetings m ON m.id = c.meeting_id"
    )
    params: tuple[Any, ...] = ()
    if scope_meeting_id:
        query += " WHERE c.meeting_id = ?"
        params = (scope_meeting_id,)
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


def indexed_meeting_ids() -> set[str]:
    with _connect() as conn:
        rows = conn.execute("SELECT DISTINCT meeting_id FROM chunks").fetchall()
    return {r["meeting_id"] for r in rows}


def count_chunks() -> int:
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])


def healthcheck() -> bool:
    """Cheap round-trip proving the database is reachable and migrated."""
    try:
        with _connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
        return int(version) == SCHEMA_VERSION
    except sqlite3.Error:
        return False
