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

import numpy as np

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
"""


_initialized = False


def _connect() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the tables once per process, so storage never depends on the startup hook."""
    global _initialized
    if not _initialized:
        conn.executescript(_SCHEMA)  # executescript: _SCHEMA has multiple statements
        _initialized = True


def init_db() -> None:
    with _connect() as conn:  # _connect already ensures the schema
        conn.executescript(_SCHEMA)


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
        conn.execute("DELETE FROM chunks WHERE meeting_id = ?", (meeting_id,))
        cur = conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
    return cur.rowcount > 0


# ------------------------------------------------------------------ chunks (RAG)
def replace_chunks(meeting_id: str, chunks: list[dict]) -> None:
    """Delete any existing chunks for a meeting and insert the new set.

    Each chunk dict: {ord, start, end, text, embedding(np.ndarray|None)}.
    Embeddings are stored as raw float32 bytes (or NULL when unavailable).
    """
    with _connect() as conn:
        conn.execute("DELETE FROM chunks WHERE meeting_id = ?", (meeting_id,))
        conn.executemany(
            """INSERT INTO chunks (id, meeting_id, ord, start, end, text, segs_json, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    new_id(), meeting_id, c["ord"], c.get("start"), c.get("end"), c["text"],
                    json.dumps(c.get("segs", [])),
                    (c["embedding"].astype("float32").tobytes() if c.get("embedding") is not None else None),
                )
                for c in chunks
            ],
        )


def get_chunks(scope_meeting_id: Optional[str] = None) -> list[dict]:
    """Fetch chunks (optionally for one meeting), joined with the meeting title.

    Returns dicts with `embedding` decoded back to a float32 numpy array (or None).
    """
    query = (
        "SELECT c.id, c.meeting_id, c.ord, c.start, c.end, c.text, c.segs_json, c.embedding, m.title "
        "FROM chunks c JOIN meetings m ON m.id = c.meeting_id"
    )
    params: tuple = ()
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
        return conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
