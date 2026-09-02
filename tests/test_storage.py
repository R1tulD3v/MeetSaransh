"""Storage layer: migrations, CRUD, cascade deletes, pagination, vector round-trips."""

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


def _make(title: str = "Meeting") -> str:
    return storage.create_meeting(
        title=title, filename=f"{title}.mp3", transcript=_transcript(), summary={"tldr": "x"}
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
    chunks = storage.get_chunks("m1")
    assert [c["text"] for c in chunks] == ["legacy"]  # data survived the table rebuild


def test_healthcheck_migrates_a_stale_database_and_then_reports_healthy():
    """A file left at an older version is upgraded by the first connection, not rejected."""
    storage.init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("PRAGMA user_version = 1")  # pretend this file predates migration 2
    storage.reset_migration_cache()

    assert storage.healthcheck() is True
    with sqlite3.connect(config.DB_PATH) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == storage.SCHEMA_VERSION


def test_healthcheck_false_when_the_database_is_unusable(monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", config.DATA_DIR / "audio")  # a directory
    storage.reset_migration_cache()
    assert storage.healthcheck() is False


# ------------------------------------------------------------------------------- CRUD
def test_create_and_get_meeting_round_trip():
    segments = [
        {"start": 0.0, "end": 4.0, "text": "first"},
        {"start": 4.0, "end": 9.0, "text": "second"},
    ]
    mid = storage.create_meeting(
        title="Planning",
        filename="planning.mp3",
        transcript=_transcript(segments),
        summary={"tldr": "we planned"},
        audio_ext=".mp3",
    )
    got = storage.get_meeting(mid)
    assert got is not None
    assert got["title"] == "Planning"
    assert got["audio_ext"] == ".mp3"
    assert got["duration"] == 9.0
    assert [s["text"] for s in got["segments"]] == ["first", "second"]
    assert got["summary"] == {"tldr": "we planned"}


def test_get_missing_meeting_returns_none():
    assert storage.get_meeting("does-not-exist") is None


def test_blank_title_falls_back_to_a_placeholder():
    mid = storage.create_meeting(title="", filename=None, transcript=_transcript(), summary={})
    stored = storage.get_meeting(mid)
    assert stored is not None
    assert stored["title"] == "Untitled meeting"


def test_delete_meeting_reports_whether_a_row_was_removed():
    mid = _make()
    assert storage.delete_meeting(mid) is True
    assert storage.delete_meeting(mid) is False  # second delete is a no-op


def test_deleting_a_meeting_cascades_to_its_chunks():
    """The FK carries ON DELETE CASCADE and PRAGMA foreign_keys is on, so no orphans."""
    mid = _make()
    storage.replace_chunks(mid, [{"ord": 0, "start": 0.0, "end": 3.0, "text": "chunk", "segs": []}])
    assert storage.count_chunks() == 1

    storage.delete_meeting(mid)
    assert storage.count_chunks() == 0


def test_chunks_cannot_reference_a_missing_meeting():
    """Foreign keys are enforced, not merely declared."""
    with pytest.raises(sqlite3.IntegrityError):
        storage.replace_chunks("ghost-meeting", [{"ord": 0, "text": "orphan", "segs": []}])


# ------------------------------------------------------------------------- pagination
def test_list_meetings_paginates_and_counts():
    ids = [_make(f"Meeting {i}") for i in range(5)]
    assert storage.count_meetings() == 5

    page = storage.list_meetings(limit=2, offset=0)
    assert len(page) == 2
    second = storage.list_meetings(limit=2, offset=2)
    assert len(second) == 2
    assert {m["id"] for m in page}.isdisjoint({m["id"] for m in second})

    everything = storage.list_meetings()
    assert {m["id"] for m in everything} == set(ids)


def test_list_meetings_omits_the_heavy_columns():
    """The sidebar query must not drag full transcripts across the wire."""
    _make()
    row = storage.list_meetings()[0]
    assert set(row) == {"id", "title", "created_at", "duration"}


# ------------------------------------------------------------------------ chunks / RAG
def test_embeddings_survive_the_blob_round_trip():
    mid = _make()
    vector = np.array([0.5, -0.25, 0.125, 2.0], dtype=np.float32)
    storage.replace_chunks(
        mid,
        [{"ord": 0, "start": 1.0, "end": 2.0, "text": "hello", "segs": [], "embedding": vector}],
    )
    stored = storage.get_chunks(mid)[0]
    assert np.array_equal(stored["embedding"], vector)
    assert stored["meeting_title"] == "Meeting"


def test_chunks_without_embeddings_round_trip_as_none():
    """Lexical-only mode stores NULL vectors; retrieval must see None, not zeros."""
    mid = _make()
    storage.replace_chunks(mid, [{"ord": 0, "text": "no vector", "segs": []}])
    assert storage.get_chunks(mid)[0]["embedding"] is None


def test_replace_chunks_is_a_replacement_not_an_append():
    mid = _make()
    storage.replace_chunks(mid, [{"ord": 0, "text": "old", "segs": []}])
    storage.replace_chunks(mid, [{"ord": 0, "text": "new", "segs": []}])
    assert [c["text"] for c in storage.get_chunks(mid)] == ["new"]


def test_get_chunks_scopes_to_one_meeting():
    a, b = _make("A"), _make("B")
    storage.replace_chunks(a, [{"ord": 0, "text": "from a", "segs": []}])
    storage.replace_chunks(b, [{"ord": 0, "text": "from b", "segs": []}])

    assert [c["text"] for c in storage.get_chunks(a)] == ["from a"]
    assert len(storage.get_chunks()) == 2
    assert storage.indexed_meeting_ids() == {a, b}


def test_segments_are_preserved_on_chunks_for_citation_precision():
    mid = _make()
    segs = [{"start": 3.0, "text": "the precise line"}]
    storage.replace_chunks(mid, [{"ord": 0, "text": "chunk text", "segs": segs}])
    assert storage.get_chunks(mid)[0]["segs"] == segs
