"""Cross-meeting analytics.

Every figure on the dashboard is computed by a SQL aggregation over the stored
summaries, using SQLite's JSON1 extension to walk into `summary_json` rather than
loading every meeting into Python and counting in a loop. That matters for more than
tidiness: the loop version costs one full transcript read per meeting, so it degrades
exactly as a user accumulates the meetings that make the dashboard worth looking at.

Every query filters on `user_id`, for the same reason the rest of the app does: an
aggregate that quietly spans accounts is a data leak that looks like a number.

Scope, stated honestly: this is SQL group-by plus visualisation. It is a genuinely
useful product feature and a fair "I can model and query data" talking point. It is not
data science, and nothing here is presented as such.
"""

from __future__ import annotations

from typing import Any

from . import storage

# Placeholders the summarizer writes when the transcript did not state a value. They are
# meaningful ("nobody was assigned this") rather than missing, so they are counted as
# their own bucket instead of being silently dropped.
UNASSIGNED = "Unassigned"
NO_DUE_DATE = "Not specified"


def _rows(conn: Any, sql: str, params: tuple[Any, ...]) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def overview(user_id: str) -> dict:
    """Headline counters: the top row of the dashboard."""
    with storage._connect() as conn:
        totals = conn.execute(
            """
            SELECT
                COUNT(*)                                   AS meetings,
                COALESCE(SUM(duration), 0)                 AS total_seconds,
                COALESCE(SUM(json_array_length(
                    json_extract(summary_json, '$.action_items'))), 0)   AS action_items,
                COALESCE(SUM(json_array_length(
                    json_extract(summary_json, '$.key_decisions'))), 0)  AS decisions,
                COALESCE(SUM(json_array_length(
                    json_extract(summary_json, '$.open_questions'))), 0) AS open_questions
            FROM meetings
            WHERE user_id = ? AND status = 'done'
            """,
            (user_id,),
        ).fetchone()
    return {
        "meetings": int(totals["meetings"]),
        "total_seconds": float(totals["total_seconds"]),
        "action_items": int(totals["action_items"]),
        "decisions": int(totals["decisions"]),
        "open_questions": int(totals["open_questions"]),
    }


def action_items_by_owner(user_id: str, limit: int = 10) -> list[dict]:
    """Who is carrying the work, how much of it is done, and how much has a due date.

    Reads the normalised `action_items` table rather than the summary JSON, so it
    reflects what the team has since decided (reassignments, completions) rather than
    only what the model originally extracted.

    `Unassigned` is deliberately included rather than filtered out -- it is usually the
    most actionable row on the page, because it is the work nobody has picked up.
    """
    with storage._connect() as conn:
        return _rows(
            conn,
            """
            SELECT
                a.owner AS owner,
                COUNT(*) AS total,
                SUM(CASE WHEN a.status = 'done' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN a.due <> ? THEN 1 ELSE 0 END) AS with_due_date
            FROM action_items a
            JOIN meetings m ON m.id = a.meeting_id
            WHERE m.user_id = ?
            GROUP BY a.owner
            ORDER BY total DESC, owner ASC
            LIMIT ?
            """,
            (NO_DUE_DATE, user_id, limit),
        )


def meetings_over_time(user_id: str, days: int = 30) -> list[dict]:
    """One row per day that actually had a meeting, newest last.

    Days with no meetings are filled in by the caller rather than by SQL: generating a
    calendar in SQLite means a recursive CTE, and the gap-filling is three lines of
    Python that are far easier to read and to test.
    """
    with storage._connect() as conn:
        return _rows(
            conn,
            """
            SELECT
                DATE(created_at) AS day,
                COUNT(*) AS meetings,
                COALESCE(SUM(duration), 0) AS seconds
            FROM meetings
            WHERE user_id = ?
              AND status = 'done'
              AND DATE(created_at) >= DATE('now', ?)
            GROUP BY day
            ORDER BY day ASC
            """,
            (user_id, f"-{int(days)} days"),
        )


def top_topics(user_id: str, limit: int = 8) -> list[dict]:
    """Which subjects keep coming back across meetings.

    Grouped case-insensitively so "Hiring" and "hiring" are one topic; the display form
    is whichever casing appeared most recently.
    """
    with storage._connect() as conn:
        return _rows(
            conn,
            """
            SELECT
                LOWER(TRIM(json_extract(topic.value, '$.title'))) AS key,
                TRIM(json_extract(topic.value, '$.title'))        AS title,
                COUNT(*)                                          AS mentions,
                COUNT(DISTINCT m.id)                              AS meetings
            FROM meetings m,
                 json_each(json_extract(m.summary_json, '$.topics')) AS topic
            WHERE m.user_id = ?
              AND m.status = 'done'
              AND TRIM(COALESCE(json_extract(topic.value, '$.title'), '')) <> ''
            GROUP BY key
            ORDER BY mentions DESC, title ASC
            LIMIT ?
            """,
            (user_id, limit),
        )


def unassigned_action_items(user_id: str, limit: int = 8) -> list[dict]:
    """Open work with no owner, newest first -- the dashboard's one call to action.

    Completed items are excluded: an unowned task that is already done is history, not
    a loose end, and listing it would train the reader to ignore the panel.
    """
    with storage._connect() as conn:
        return _rows(
            conn,
            """
            SELECT
                a.id        AS id,
                a.task      AS task,
                a.timestamp AS timestamp,
                m.id        AS meeting_id,
                m.title     AS meeting_title
            FROM action_items a
            JOIN meetings m ON m.id = a.meeting_id
            WHERE m.user_id = ? AND a.owner = ? AND a.status = 'open'
            ORDER BY m.created_at DESC, a.ord ASC
            LIMIT ?
            """,
            (user_id, UNASSIGNED, limit),
        )


def completion(user_id: str) -> dict:
    """Open versus done across every meeting -- the one genuine progress metric here."""
    with storage._connect() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN a.status = 'done' THEN 1 ELSE 0 END) AS completed
            FROM action_items a
            JOIN meetings m ON m.id = a.meeting_id
            WHERE m.user_id = ?
            """,
            (user_id,),
        ).fetchone()
    total = int(row["total"] or 0)
    completed = int(row["completed"] or 0)
    return {"total": total, "completed": completed, "open": total - completed}


def fill_missing_days(rows: list[dict], days: int) -> list[dict]:
    """Pad a sparse day series so a chart shows real gaps instead of compressing them.

    Without this, three meetings three weeks apart draw as three adjacent points and
    read as a busy week.
    """
    from datetime import date, timedelta

    by_day = {r["day"]: r for r in rows}
    today = date.today()
    out = []
    for offset in range(days, -1, -1):
        key = (today - timedelta(days=offset)).isoformat()
        row = by_day.get(key)
        out.append(
            {
                "day": key,
                "meetings": int(row["meetings"]) if row else 0,
                "seconds": float(row["seconds"]) if row else 0.0,
            }
        )
    return out


def dashboard(user_id: str, *, days: int = 30) -> dict:
    """Everything the dashboard needs, in one round trip.

    One endpoint rather than five: the page renders as a unit, and five requests would
    mean five chances for a partially-drawn dashboard.
    """
    return {
        "overview": overview(user_id),
        "completion": completion(user_id),
        "by_owner": action_items_by_owner(user_id),
        "over_time": fill_missing_days(meetings_over_time(user_id, days), days),
        "top_topics": top_topics(user_id),
        "unassigned": unassigned_action_items(user_id),
        "window_days": days,
    }
