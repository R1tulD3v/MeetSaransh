"""Provider clients: retry/backoff, error mapping, and response normalization.

Every outbound call is mocked with respx. Nothing in this file touches the network, so
the suite runs offline and deterministically -- and a broken retry loop fails the build
instead of quietly costing money in production.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app import llm, summarize, transcription
from tests.conftest import GROQ_ASR_URL, GROQ_CHAT_URL, SUMMARY_JSON, chat_completion


@pytest.fixture(autouse=True)
def instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry logic, drop the wall-clock sleep."""
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)


# ---------------------------------------------------------------------------- llm.chat
@respx.mock
def test_chat_returns_the_assistant_message(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("hello there"))
    )
    assert llm.chat([{"role": "user", "content": "hi"}]) == "hello there"


def test_chat_without_a_key_fails_before_any_network_call(without_api_key):
    with pytest.raises(llm.LLMError, match="No GROQ_API_KEY"):
        llm.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_chat_retries_a_429_then_succeeds(with_api_key):
    route = respx.post(GROQ_CHAT_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": {"message": "slow down"}}),
            httpx.Response(200, json=chat_completion("recovered")),
        ]
    )
    assert llm.chat([{"role": "user", "content": "hi"}]) == "recovered"
    assert route.call_count == 2


@respx.mock
def test_chat_gives_up_after_the_retry_budget(with_api_key):
    route = respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(429, json={}))
    with pytest.raises(llm.LLMError, match="rate limit"):
        llm.chat([{"role": "user", "content": "hi"}], max_retries=2)
    assert route.call_count == 3  # the initial attempt plus two retries


@respx.mock
def test_chat_retries_transient_network_errors(with_api_key):
    route = respx.post(GROQ_CHAT_URL).mock(
        side_effect=[
            httpx.ConnectError("connection reset"),
            httpx.Response(200, json=chat_completion("second try")),
        ]
    )
    assert llm.chat([{"role": "user", "content": "hi"}]) == "second try"
    assert route.call_count == 2


@respx.mock
def test_a_bad_key_is_reported_as_a_key_problem_not_a_generic_failure(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(401, json={}))
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        llm.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_a_401_is_not_retried(with_api_key):
    """Retrying an auth failure just burns time; the key will not fix itself."""
    route = respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(401, json={}))
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert route.call_count == 1


@respx.mock
def test_an_unexpected_response_shape_raises_rather_than_returning_junk(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(llm.LLMError, match="Unexpected LLM response shape"):
        llm.chat([{"role": "user", "content": "hi"}])


@respx.mock
def test_json_mode_sets_the_response_format_flag(with_api_key):
    route = respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("{}"))
    )
    llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert route.calls[0].request.read().decode().count('"response_format"') == 1


@respx.mock
def test_persistent_network_failure_surfaces_the_last_error(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
    with pytest.raises(llm.LLMError, match="network error"):
        llm.chat([{"role": "user", "content": "hi"}], max_retries=1)


# ------------------------------------------------------------------------ transcription
@respx.mock
def test_transcription_normalizes_the_provider_payload(with_api_key, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3fake")
    respx.post(GROQ_ASR_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "text": "Hello world. Second line.",
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": " Hello world. "},
                    {"start": 3.0, "end": 7.5, "text": "Second line."},
                ],
            },
        )
    )
    result = transcription.transcribe(audio)

    assert result["text"] == "Hello world. Second line."
    assert result["segments"][0]["text"] == "Hello world."  # whitespace trimmed
    assert result["duration"] == 7.5
    assert result["timestamped_text"] == "[00:00] Hello world.\n[00:03] Second line."


@respx.mock
def test_transcription_of_audio_with_no_speech_does_not_crash(with_api_key, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3fake")
    respx.post(GROQ_ASR_URL).mock(
        return_value=httpx.Response(200, json={"text": "", "segments": [], "duration": 12.0})
    )
    result = transcription.transcribe(audio)
    assert result["segments"] == []
    assert result["duration"] == 12.0


def test_transcription_without_a_key_points_at_the_sample(without_api_key, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3fake")
    with pytest.raises(transcription.TranscriptionError, match="Load sample meeting"):
        transcription.transcribe(audio)


@respx.mock
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "401"), (429, "rate limit"), (413, "too large"), (500, "500")],
)
def test_asr_http_errors_map_to_actionable_messages(with_api_key, tmp_path, status, expected):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3fake")
    respx.post(GROQ_ASR_URL).mock(return_value=httpx.Response(status, json={}))
    with pytest.raises(transcription.TranscriptionError, match=expected):
        transcription.transcribe(audio)


@respx.mock
def test_asr_network_failure_is_reported_as_unreachable(with_api_key, tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"ID3fake")
    respx.post(GROQ_ASR_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(transcription.TranscriptionError, match="Could not reach"):
        transcription.transcribe(audio)


def test_timestamped_text_skips_empty_segments():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "kept"},
        {"start": 1.0, "end": 2.0, "text": ""},
        {"start": 3725.0, "end": 3730.0, "text": "long meeting"},
    ]
    assert transcription.build_timestamped_text(segments) == (
        "[00:00] kept\n[1:02:05] long meeting"
    )


# -------------------------------------------------------------------------- summarize
@respx.mock
def test_summarize_parses_a_well_formed_response(with_api_key):
    import json

    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion(json.dumps(SUMMARY_JSON)))
    )
    summary = summarize.summarize("Planning", "[00:00] hello")
    assert summary["tldr"] == SUMMARY_JSON["tldr"]
    assert summary["action_items"][0]["owner"] == "Priya"


@respx.mock
def test_summarize_recovers_from_markdown_fenced_json(with_api_key):
    """Models wrap JSON in ``` fences often enough that this cannot be a crash."""
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion('```json\n{"tldr": "fenced"}\n```'))
    )
    assert summarize.summarize("t", "[00:00] hi")["tldr"] == "fenced"


@respx.mock
def test_summarize_recovers_from_prose_wrapped_json(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=chat_completion('Sure! Here you go:\n{"tldr": "extracted"}\nHope that helps.')
        )
    )
    assert summarize.summarize("t", "[00:00] hi")["tldr"] == "extracted"


@respx.mock
def test_summarize_raises_on_unparseable_output(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("I am not JSON at all."))
    )
    with pytest.raises(summarize.SummarizationError, match="valid JSON"):
        summarize.summarize("t", "[00:00] hi")


def test_summarize_rejects_an_empty_transcript_before_calling_the_model(with_api_key):
    with pytest.raises(summarize.SummarizationError, match="empty"):
        summarize.summarize("t", "   ")


def test_summarize_without_a_key_fails_fast(without_api_key):
    with pytest.raises(summarize.SummarizationError, match="No GROQ_API_KEY"):
        summarize.summarize("t", "[00:00] hi")


# -------------------------------------------------------------- summary normalization
def test_normalization_fills_in_every_key_so_the_ui_never_sees_a_hole():
    assert summarize.normalize_summary({}) == {
        "tldr": "",
        "key_decisions": [],
        "action_items": [],
        "open_questions": [],
        "topics": [],
    }


def test_normalization_survives_a_non_dict_payload():
    assert summarize.normalize_summary(["not", "a", "dict"])["tldr"] == ""


def test_action_items_get_honest_placeholders_not_invented_values():
    """An owner the model did not state must read 'Unassigned', never a guess."""
    result = summarize.normalize_summary({"action_items": [{"task": "Do the thing"}]})
    assert result["action_items"][0] == {
        "task": "Do the thing",
        "owner": "Unassigned",
        "due": "Not specified",
        "timestamp": "",
    }


def test_null_owner_and_due_are_replaced_with_placeholders():
    result = summarize.normalize_summary(
        {"action_items": [{"task": "t", "owner": None, "due": None}]}
    )
    assert result["action_items"][0]["owner"] == "Unassigned"
    assert result["action_items"][0]["due"] == "Not specified"


def test_malformed_list_entries_are_dropped_not_rendered():
    result = summarize.normalize_summary(
        {
            "key_decisions": ["a bare string where an object was required", {"decision": "kept"}],
            "action_items": [None, {"task": "kept"}],
            "topics": ["junk", {"title": "kept"}],
        }
    )
    assert [d["decision"] for d in result["key_decisions"]] == ["kept"]
    assert [a["task"] for a in result["action_items"]] == ["kept"]
    assert [t["title"] for t in result["topics"]] == ["kept"]


def test_a_scalar_where_a_list_belongs_becomes_an_empty_list():
    assert summarize.normalize_summary({"open_questions": "not a list"})["open_questions"] == []


def test_non_string_scalars_are_coerced_rather_than_leaking_types():
    result = summarize.normalize_summary({"tldr": 42, "open_questions": [1, 2]})
    assert result["tldr"] == "42"
    assert result["open_questions"] == ["1", "2"]
