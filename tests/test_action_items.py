"""Editable action items: the write path, ownership, and the backfill migration.

This is the app's only mutable user state, so the tests concentrate on the two ways
that goes wrong: one user editing another's task, and a partial update clobbering a
field the caller never sent.
"""

from __future__ import annotations

import sqlite3

import pytest

from app import config, storage


def _summary(actions) -> dict:
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
        "key_decisions": [],
        "open_questions": [],
        "topics": [],
    }


def _meeting(user_id: str, actions=(("Ship the fix", "Priya"),), title="Planning") -> str:
    summary = _summary(actions)
    mid = storage.create_meeting(
        user_id=user_id,
        title=title,
        filename=None,
        transcript={"text": "t", "segments": [], "duration": 60.0},
        summary=summary,
    )
    storage.replace_action_items(mid, summary["action_items"])
    return mid


# ------------------------------------------------------------------------- storage
def test_action_items_are_materialised_from_a_summary(user_id):
    mid = _meeting(user_id, [("Ship the fix", "Priya", "Friday")])
    items = storage.list_action_items(mid, user_id)

    assert len(items) == 1
    assert items[0]["task"] == "Ship the fix"
    assert items[0]["owner"] == "Priya"
    assert items[0]["due"] == "Friday"
    assert items[0]["status"] == "open"
    assert items[0]["edited"] == 0


def test_missing_owner_and_due_become_the_honest_placeholders(user_id):
    mid = _meeting(user_id, [("Do the thing", "")])
    item = storage.list_action_items(mid, user_id)[0]
    assert item["owner"] == "Unassigned"
    assert item["due"] == "Not specified"


def test_blank_tasks_are_never_materialised(user_id):
    """A malformed summary must not create an empty, uneditable row."""
    mid = _meeting(user_id, [("", "Priya"), ("   ", "Priya"), ("real", "Priya")])
    assert [i["task"] for i in storage.list_action_items(mid, user_id)] == ["real"]


def test_items_keep_the_order_the_model_produced(user_id):
    mid = _meeting(user_id, [("first", "A"), ("second", "B"), ("third", "C")])
    assert [i["task"] for i in storage.list_action_items(mid, user_id)] == [
        "first",
        "second",
        "third",
    ]


def test_replacing_is_wholesale_not_a_merge(user_id):
    """Only ever called on completion, before anyone can have edited anything."""
    mid = _meeting(user_id, [("old", "A")])
    storage.replace_action_items(mid, [{"task": "new", "owner": "B"}])
    assert [i["task"] for i in storage.list_action_items(mid, user_id)] == ["new"]


def test_deleting_a_meeting_takes_its_action_items(user_id):
    mid = _meeting(user_id)
    storage.delete_meeting(mid, user_id)
    with storage._connect() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM action_items").fetchone()["n"] == 0


def test_action_items_cannot_reference_a_missing_meeting():
    with pytest.raises(sqlite3.IntegrityError):
        storage.replace_action_items("ghost", [{"task": "orphan", "owner": "X"}])


# -------------------------------------------------------------------------- updates
def test_updating_marks_the_row_as_edited(user_id):
    """`edited` is what distinguishes 'the model said Priya owns this' from 'we
    decided Priya owns this' -- the first question anyone asks of an AI task list."""
    mid = _meeting(user_id)
    item = storage.list_action_items(mid, user_id)[0]

    updated = storage.update_action_item(item["id"], user_id, {"owner": "Rahul"})
    assert updated["owner"] == "Rahul"
    assert updated["edited"] == 1
    assert updated["updated_at"]


def test_a_partial_update_leaves_other_fields_alone(user_id):
    """PATCH semantics: ticking a checkbox must not clobber the owner."""
    mid = _meeting(user_id, [("Ship the fix", "Priya", "Friday")])
    item = storage.list_action_items(mid, user_id)[0]

    updated = storage.update_action_item(item["id"], user_id, {"status": "done"})
    assert updated["status"] == "done"
    assert updated["owner"] == "Priya"
    assert updated["due"] == "Friday"
    assert updated["task"] == "Ship the fix"


def test_an_empty_update_is_a_no_op_that_still_returns_the_row(user_id):
    mid = _meeting(user_id)
    item = storage.list_action_items(mid, user_id)[0]

    unchanged = storage.update_action_item(item["id"], user_id, {})
    assert unchanged["edited"] == 0  # not marked edited by a no-op


def test_unknown_fields_are_ignored_not_written(user_id):
    """The allowlist is the guard: a client cannot set `edited`, `ord` or `meeting_id`."""
    mid = _meeting(user_id)
    item = storage.list_action_items(mid, user_id)[0]

    updated = storage.update_action_item(
        item["id"], user_id, {"owner": "Rahul", "meeting_id": "somewhere-else", "ord": 99}
    )
    assert updated["meeting_id"] == mid
    assert updated["ord"] == item["ord"]


def test_updating_an_unknown_item_returns_none(user_id):
    assert storage.update_action_item("no-such-item", user_id, {"owner": "X"}) is None


def test_counting_by_status(user_id):
    mid = _meeting(user_id, [("a", "A"), ("b", "B"), ("c", "C")])
    items = storage.list_action_items(mid, user_id)
    storage.update_action_item(items[0]["id"], user_id, {"status": "done"})

    assert storage.count_action_items(user_id) == 3
    assert storage.count_action_items(user_id, status="done") == 1
    assert storage.count_action_items(user_id, status="open") == 2


# ---------------------------------------------------------------- ownership isolation
def test_one_user_cannot_read_anothers_action_items(user_id):
    other = storage.create_user(email="mallory@example.com", password_hash="x")["id"]
    mid = _meeting(user_id)

    assert storage.list_action_items(mid, other) == []
    item = storage.list_action_items(mid, user_id)[0]
    assert storage.get_action_item(item["id"], other) is None


def test_one_user_cannot_edit_anothers_action_item(user_id):
    """The ownership check lives in the UPDATE's WHERE clause: a check-then-write would
    be a race, and the race would let one user edit another's task."""
    other = storage.create_user(email="mallory@example.com", password_hash="x")["id"]
    mid = _meeting(user_id)
    item = storage.list_action_items(mid, user_id)[0]

    assert storage.update_action_item(item["id"], other, {"owner": "Mallory"}) is None
    assert storage.get_action_item(item["id"], user_id)["owner"] == "Priya"


# ------------------------------------------------------------------------ migration
def test_the_backfill_makes_pre_existing_action_items_editable():
    """A genuine v5 database -- built by running only migrations 1-5 -- must come out of
    the upgrade with its action items already editable, not stranded in a JSON blob.

    Built for real rather than by re-running a fragment of the migration by hand: the
    point is that the shipped script does this, on a database that really predates it.
    """
    import json
    import sqlite3 as sqlite

    from app import config

    config.ensure_dirs()
    summary = json.dumps(_summary([("Legacy task", "Meera", "Monday"), ("", "ignored")]))
    with sqlite.connect(config.DB_PATH) as conn:
        for version, script in storage._MIGRATIONS[:5]:
            conn.executescript(script)
            conn.execute(f"PRAGMA user_version = {version}")
        conn.execute(
            "INSERT INTO users (id, email, password_hash, role, created_at) "
            "VALUES ('u1', 'legacy@example.com', 'x', 'user', '2024-01-01')"
        )
        conn.execute(
            "INSERT INTO meetings (id, user_id, title, created_at, summary_json, status) "
            "VALUES ('legacy', 'u1', 'Old', '2024-01-01', ?, 'done')",
            (summary,),
        )
    storage.reset_migration_cache()

    storage.init_db()  # runs migration 6, including its backfill

    items = storage.list_action_items("legacy", "u1")
    assert [i["task"] for i in items] == ["Legacy task"]  # the blank one is skipped
    assert items[0]["owner"] == "Meera"
    assert items[0]["due"] == "Monday"
    assert items[0]["status"] == "open"
    assert items[0]["edited"] == 0  # backfilled, not edited by a human


def test_the_summary_json_keeps_its_own_copy(user_id):
    """Provenance versus state: the JSON records what the model extracted, the table
    records what the team decided. Editing one must not rewrite the other."""
    mid = _meeting(user_id, [("Ship the fix", "Priya")])
    item = storage.list_action_items(mid, user_id)[0]
    storage.update_action_item(item["id"], user_id, {"owner": "Rahul"})

    meeting = storage.get_meeting(mid, user_id)
    assert meeting["summary"]["action_items"][0]["owner"] == "Priya"  # unchanged
    assert storage.get_action_item(item["id"], user_id)["owner"] == "Rahul"


# ------------------------------------------------------------------------- endpoints
def test_listing_action_items_over_the_api(client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    rows = client.get(f"/api/v1/meetings/{mid}/action-items").json()

    assert rows
    assert {"id", "task", "owner", "due", "status", "edited"} <= set(rows[0])


def test_patching_an_action_item(client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    item = client.get(f"/api/v1/meetings/{mid}/action-items").json()[0]

    response = client.patch(
        f"/api/v1/action-items/{item['id']}", json={"owner": "Rahul", "status": "done"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["owner"] == "Rahul"
    assert body["status"] == "done"
    assert body["edited"] is True
    assert body["task"] == item["task"]  # untouched by a partial update


def test_patching_an_unknown_item_is_a_404(client):
    assert client.patch("/api/v1/action-items/nope", json={"owner": "X"}).status_code == 404


@pytest.mark.parametrize(
    "payload",
    [{"status": "maybe"}, {"owner": ""}, {"task": ""}, {"due": "x" * 200}],
)
def test_invalid_updates_are_rejected(client, payload):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    item = client.get(f"/api/v1/meetings/{mid}/action-items").json()[0]
    response = client.patch(f"/api/v1/action-items/{item['id']}", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_action_item_routes_require_authentication(anon_client):
    assert anon_client.get("/api/v1/meetings/x/action-items").status_code == 401
    assert anon_client.patch("/api/v1/action-items/x", json={"owner": "Y"}).status_code == 401


def test_one_account_cannot_patch_anothers_action_item(client, second_client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    item = client.get(f"/api/v1/meetings/{mid}/action-items").json()[0]

    response = second_client.patch(f"/api/v1/action-items/{item['id']}", json={"owner": "Mallory"})
    assert response.status_code == 404
    assert client.get(f"/api/v1/meetings/{mid}/action-items").json()[0]["owner"] != "Mallory"


def test_listing_another_users_meeting_items_is_a_404(client, second_client):
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    assert second_client.get(f"/api/v1/meetings/{mid}/action-items").status_code == 404


def test_a_completed_item_shows_up_in_the_dashboard(client):
    """End to end: edit the task, and the aggregate follows."""
    mid = client.post("/api/v1/meetings/sample").json()["id"]
    item = client.get(f"/api/v1/meetings/{mid}/action-items").json()[0]
    client.patch(f"/api/v1/action-items/{item['id']}", json={"status": "done"})

    completion = client.get("/api/v1/analytics").json()["completion"]
    assert completion["completed"] == 1
    assert completion["open"] == completion["total"] - 1


def test_the_status_vocabulary_is_closed(client):
    """Only `open` and `done` exist. A free-text status would make every aggregate a
    guess about what the strings mean."""
    assert set(storage.ACTION_STATUSES) == {"open", "done"}
    assert config.BASE_DIR  # sanity: fixtures wired
