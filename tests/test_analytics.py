"""Cross-meeting analytics: the SQL aggregations and the dashboard endpoint.

Each aggregation is tested against a seeded database with hand-computed expected
numbers. That matters more here than elsewhere: an aggregate that is quietly wrong
still renders as a confident chart, so there is nothing in the UI to notice.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app import analytics, storage


def _summary(*, actions=(), decisions=(), questions=(), topics=()) -> dict:
    return {
        "tldr": "x",
        "action_items": [
            {
                "task": a[0],
                "owner": a[1],
                "due": a[2] if len(a) > 2 else "Not specified",
                "timestamp": "00:10",
            }
            for a in actions
        ],
        "key_decisions": [{"decision": d, "timestamp": "00:20"} for d in decisions],
        "open_questions": list(questions),
        "topics": [{"title": t, "summary": "s", "timestamp": "00:00"} for t in topics],
    }


def _meeting(user_id: str, *, title="Meeting", summary=None, duration=600.0, status="done"):
    """Seed a meeting the way the real pipeline does.

    Action items are materialised into their own table as well as staying in the
    summary JSON, because that is exactly what `jobs._run_job` does on completion --
    a fixture that only wrote the JSON would be testing a state the app never produces.
    """
    summary = summary if summary is not None else _summary()
    meeting_id = storage.create_meeting(
        user_id=user_id,
        title=title,
        filename=None,
        transcript={"text": "t", "segments": [], "duration": duration},
        summary=summary,
        status=status,
    )
    storage.replace_action_items(meeting_id, summary.get("action_items", []))
    return meeting_id


def _backdate(meeting_id: str, days_ago: int) -> None:
    """Move a meeting into the past so the time series has something to plot."""
    when = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    with storage._connect() as conn:
        conn.execute("UPDATE meetings SET created_at = ? WHERE id = ?", (when, meeting_id))


# ------------------------------------------------------------------------ overview
def test_overview_of_an_empty_account_is_all_zeros(user_id):
    assert analytics.overview(user_id) == {
        "meetings": 0,
        "total_seconds": 0.0,
        "action_items": 0,
        "decisions": 0,
        "open_questions": 0,
    }


def test_overview_counts_across_meetings(user_id):
    _meeting(
        user_id,
        duration=600.0,
        summary=_summary(
            actions=[("a", "Priya"), ("b", "Rahul")],
            decisions=["ship it"],
            questions=["who reviews?"],
        ),
    )
    _meeting(
        user_id,
        duration=1200.0,
        summary=_summary(actions=[("c", "Priya")], decisions=["cut scope", "hire"]),
    )

    assert analytics.overview(user_id) == {
        "meetings": 2,
        "total_seconds": 1800.0,
        "action_items": 3,
        "decisions": 3,
        "open_questions": 1,
    }


def test_overview_ignores_meetings_that_have_not_finished(user_id):
    """A queued meeting has no summary, so counting it would understate every average."""
    _meeting(user_id, summary=_summary(actions=[("a", "Priya")]))
    storage.create_meeting(user_id=user_id, title="Queued", filename=None, status="queued")

    assert analytics.overview(user_id)["meetings"] == 1


def test_overview_is_scoped_to_one_account(user_id):
    other = storage.create_user(email="other@example.com", password_hash="x")["id"]
    _meeting(user_id, summary=_summary(actions=[("mine", "Priya")]))
    _meeting(other, summary=_summary(actions=[("theirs", "Mallory"), ("more", "Mallory")]))

    assert analytics.overview(user_id)["action_items"] == 1
    assert analytics.overview(other)["action_items"] == 2


# ----------------------------------------------------------------- action items
def test_owner_load_counts_and_ranks(user_id):
    _meeting(
        user_id,
        summary=_summary(
            actions=[("a", "Priya"), ("b", "Priya"), ("c", "Rahul")],
        ),
    )
    rows = analytics.action_items_by_owner(user_id)

    assert [r["owner"] for r in rows] == ["Priya", "Rahul"]
    assert rows[0]["total"] == 2


def test_owner_load_counts_completions(user_id):
    """The table reflects what the team decided, not only what the model extracted."""
    mid = _meeting(user_id, summary=_summary(actions=[("a", "Priya"), ("b", "Priya")]))
    first = storage.list_action_items(mid, user_id)[0]
    storage.update_action_item(first["id"], user_id, {"status": "done"})

    row = analytics.action_items_by_owner(user_id)[0]
    assert row["total"] == 2
    assert row["completed"] == 1


def test_completion_totals_open_versus_done(user_id):
    mid = _meeting(user_id, summary=_summary(actions=[("a", "P"), ("b", "P"), ("c", "R")]))
    items = storage.list_action_items(mid, user_id)
    storage.update_action_item(items[0]["id"], user_id, {"status": "done"})

    assert analytics.completion(user_id) == {"total": 3, "completed": 1, "open": 2}


def test_completion_of_an_empty_account_is_zeros(user_id):
    assert analytics.completion(user_id) == {"total": 0, "completed": 0, "open": 0}


def test_a_completed_unowned_task_is_no_longer_a_loose_end(user_id):
    """History, not a loose end -- listing it would train the reader to ignore the panel."""
    mid = _meeting(user_id, summary=_summary(actions=[("orphan", "")]))
    item = storage.list_action_items(mid, user_id)[0]
    assert len(analytics.unassigned_action_items(user_id)) == 1

    storage.update_action_item(item["id"], user_id, {"status": "done"})
    assert analytics.unassigned_action_items(user_id) == []


def test_reassigning_moves_the_work_on_the_dashboard(user_id):
    """The whole point of normalising: the chart follows the team's decisions."""
    mid = _meeting(user_id, summary=_summary(actions=[("a", "Unassigned")]))
    item = storage.list_action_items(mid, user_id)[0]
    storage.update_action_item(item["id"], user_id, {"owner": "Priya"})

    owners = {r["owner"]: r["total"] for r in analytics.action_items_by_owner(user_id)}
    assert owners == {"Priya": 1}
    assert analytics.unassigned_action_items(user_id) == []


def test_owner_load_counts_how_many_have_a_real_due_date(user_id):
    """`Not specified` is the summarizer's honest placeholder, not a date."""
    _meeting(
        user_id,
        summary=_summary(
            actions=[("a", "Priya", "Friday"), ("b", "Priya", "Not specified"), ("c", "Priya", "")]
        ),
    )
    row = analytics.action_items_by_owner(user_id)[0]

    assert row["total"] == 3
    assert row["with_due_date"] == 1


def test_unassigned_work_is_a_bucket_not_a_gap(user_id):
    """It is usually the most actionable row on the page: work nobody has picked up."""
    _meeting(user_id, summary=_summary(actions=[("a", "Unassigned"), ("b", ""), ("c", "Priya")]))
    rows = {r["owner"]: r["total"] for r in analytics.action_items_by_owner(user_id)}

    assert rows["Unassigned"] == 2
    assert rows["Priya"] == 1


def test_owner_load_is_scoped_to_one_account(user_id):
    other = storage.create_user(email="other@example.com", password_hash="x")["id"]
    _meeting(other, summary=_summary(actions=[("theirs", "Mallory")]))
    assert analytics.action_items_by_owner(user_id) == []


def test_owner_load_respects_its_limit(user_id):
    _meeting(user_id, summary=_summary(actions=[(f"t{i}", f"Owner{i}") for i in range(12)]))
    assert len(analytics.action_items_by_owner(user_id, limit=5)) == 5


# ------------------------------------------------------------------- unassigned
def test_unassigned_items_carry_their_meeting_for_deep_linking(user_id):
    mid = _meeting(user_id, title="Planning", summary=_summary(actions=[("Own the rewrite", "")]))
    rows = analytics.unassigned_action_items(user_id)

    assert len(rows) == 1
    assert rows[0]["task"] == "Own the rewrite"
    assert rows[0]["meeting_id"] == mid
    assert rows[0]["meeting_title"] == "Planning"


def test_assigned_items_are_not_listed_as_loose_ends(user_id):
    _meeting(user_id, summary=_summary(actions=[("done deal", "Priya")]))
    assert analytics.unassigned_action_items(user_id) == []


def test_blank_tasks_are_skipped(user_id):
    """A malformed summary must not put an empty row on the dashboard."""
    _meeting(user_id, summary=_summary(actions=[("", ""), ("real one", "")]))
    assert [r["task"] for r in analytics.unassigned_action_items(user_id)] == ["real one"]


# -------------------------------------------------------------------- over time
def test_time_series_groups_by_day(user_id):
    a = _meeting(user_id, duration=600.0)
    b = _meeting(user_id, duration=300.0)
    c = _meeting(user_id, duration=900.0)
    _backdate(a, 3)
    _backdate(b, 3)
    _backdate(c, 1)

    rows = analytics.meetings_over_time(user_id, days=30)
    assert len(rows) == 2
    assert rows[0]["meetings"] == 2
    assert rows[0]["seconds"] == 900.0
    assert rows[1]["meetings"] == 1


def test_time_series_excludes_anything_older_than_the_window(user_id):
    old = _meeting(user_id)
    _backdate(old, 90)
    assert analytics.meetings_over_time(user_id, days=30) == []


def test_gap_filling_produces_a_continuous_series():
    """Three meetings three weeks apart must not draw as three adjacent points."""
    from datetime import date

    today = date.today().isoformat()
    filled = analytics.fill_missing_days([{"day": today, "meetings": 2, "seconds": 60.0}], days=6)

    assert len(filled) == 7  # today plus the six days before it
    assert filled[-1] == {"day": today, "meetings": 2, "seconds": 60.0}
    assert all(r["meetings"] == 0 for r in filled[:-1])
    assert [r["day"] for r in filled] == sorted(r["day"] for r in filled)


def test_gap_filling_an_empty_series_still_returns_the_window():
    assert len(analytics.fill_missing_days([], days=10)) == 11


# ------------------------------------------------------------------ top topics
def test_topics_are_counted_across_meetings(user_id):
    _meeting(user_id, summary=_summary(topics=["Hiring", "Release scope"]))
    _meeting(user_id, summary=_summary(topics=["Hiring"]))

    rows = analytics.top_topics(user_id)
    assert rows[0]["title"] == "Hiring"
    assert rows[0]["mentions"] == 2
    assert rows[0]["meetings"] == 2


def test_topic_grouping_is_case_insensitive(user_id):
    _meeting(user_id, summary=_summary(topics=["hiring"]))
    _meeting(user_id, summary=_summary(topics=["Hiring"]))

    rows = analytics.top_topics(user_id)
    assert len(rows) == 1
    assert rows[0]["mentions"] == 2


def test_blank_topic_titles_are_skipped(user_id):
    _meeting(user_id, summary=_summary(topics=["", "  ", "Real"]))
    assert [r["title"] for r in analytics.top_topics(user_id)] == ["Real"]


# -------------------------------------------------------------------- endpoint
def test_dashboard_endpoint_returns_every_section(client):
    client.post("/api/v1/meetings/sample")
    body = client.get("/api/v1/analytics").json()

    assert set(body) == {
        "overview",
        "completion",
        "by_owner",
        "over_time",
        "top_topics",
        "unassigned",
        "window_days",
    }
    assert body["overview"]["meetings"] == 1
    assert body["window_days"] == 30


def test_dashboard_endpoint_is_empty_but_valid_for_a_new_account(client):
    body = client.get("/api/v1/analytics").json()
    assert body["overview"]["meetings"] == 0
    assert body["by_owner"] == []
    assert len(body["over_time"]) == 31  # the window is still drawn


def test_dashboard_endpoint_requires_authentication(anon_client):
    assert anon_client.get("/api/v1/analytics").status_code == 401


def test_dashboard_endpoint_never_shows_another_account(client, second_client):
    client.post("/api/v1/meetings/sample")
    assert second_client.get("/api/v1/analytics").json()["overview"]["meetings"] == 0


@pytest.mark.parametrize("days", ["0", "400", "abc"])
def test_dashboard_window_is_validated(client, days):
    assert client.get(f"/api/v1/analytics?days={days}").status_code == 422


def test_dashboard_window_is_honoured(client):
    body = client.get("/api/v1/analytics?days=7").json()
    assert body["window_days"] == 7
    assert len(body["over_time"]) == 8
