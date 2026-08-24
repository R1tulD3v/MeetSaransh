"""Persistence layer — SQLite via the Python standard library (no ORM, no extra deps).

One table, `meetings`, holds each processed meeting. Transcript segments and the
structured summary are stored as JSON text columns; SQLite is schemaless enough for
this and keeps the footprint to a single file with zero external services.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

from . import config

_SCHEMA = """
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
"""


_initialized = False


def _connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the table once per process, so storage never depends on the startup hook."""
    global _initialized
    if not _initialized:
        conn.execute(_SCHEMA)
        _initialized = True


def init_db() -> None:
    with _connect() as conn:  # _connect already ensures the schema
        conn.execute(_SCHEMA)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def create_meeting(
    *, title: str, filename: Optional[str], transcript: dict, summary: dict,
    audio_ext: Optional[str] = None, meeting_id: Optional[str] = None,
) -> str:
    """Insert a fully-processed meeting and return its id."""
    mid = meeting_id or new_id()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO meetings
               (id, title, filename, created_at, duration, transcript, segments_json, summary_json, audio_ext)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mid,
                title or "Untitled meeting",
                filename,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                float(transcript.get("duration", 0.0) or 0.0),
                transcript.get("text", ""),
                json.dumps(transcript.get("segments", [])),
                json.dumps(summary),
                audio_ext,
            ),
        )
    return mid


def list_meetings() -> list[dict]:
    """Lightweight list for the sidebar (no heavy transcript/segments payload)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, duration FROM meetings ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_meeting(meeting_id: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["segments"] = json.loads(data.pop("segments_json") or "[]")
    data["summary"] = json.loads(data.pop("summary_json") or "{}")
    return data


def delete_meeting(meeting_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    return cur.rowcount > 0
