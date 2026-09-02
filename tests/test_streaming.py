"""Streaming answers over Server-Sent Events.

The behaviour worth protecting is ordering and failure handling: citations must arrive
before any text, and a provider failure must never become a half-answer spliced from two
attempts.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from app import config, llm, rag
from tests.conftest import GROQ_CHAT_URL, chat_completion


def sse_stream(*chunks: str) -> httpx.Response:
    """A Groq-shaped streaming response body."""
    frames = [
        "data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n\n" for c in chunks
    ]
    frames.append("data: [DONE]\n\n")
    return httpx.Response(200, text="".join(frames))


def read_events(client, question: str = "payment bug", **body) -> list[dict]:
    """Collect the parsed events from one streamed answer."""
    events = []
    with client.stream(
        "POST", "/api/v1/chat/stream", json={"question": question, **body}
    ) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[5:].strip()))
    return events


# ------------------------------------------------------------------- the LLM client
@respx.mock
def test_chat_stream_yields_content_deltas(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(return_value=sse_stream("Priya ", "owns ", "it."))
    assert list(llm.chat_stream([{"role": "user", "content": "hi"}])) == ["Priya ", "owns ", "it."]


@respx.mock
def test_chat_stream_ignores_keepalives_and_role_only_frames(with_api_key):
    """The provider sends frames with no content; aborting on those would kill good
    answers for no reason."""
    body = (
        'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        ": keep-alive\n\n"
        'data: {"choices":[{"delta":{"content":"real"}}]}\n\n'
        "data: not-json\n\n"
        "data: [DONE]\n\n"
    )
    respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, text=body))
    assert list(llm.chat_stream([{"role": "user", "content": "hi"}])) == ["real"]


@respx.mock
def test_chat_stream_maps_http_errors(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(401, json={}))
    with pytest.raises(llm.LLMError, match="GROQ_API_KEY"):
        list(llm.chat_stream([{"role": "user", "content": "hi"}]))


def test_chat_stream_without_a_key_fails_before_the_network(without_api_key):
    with pytest.raises(llm.LLMError, match="No GROQ_API_KEY"):
        list(llm.chat_stream([{"role": "user", "content": "hi"}]))


@respx.mock
def test_chat_stream_is_not_retried(with_api_key, monkeypatch):
    """Unlike `chat`, a stream is never retried: the caller has already shown the user
    part of the previous attempt, so a retry would splice two answers together."""
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    route = respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(429, json={}))
    with pytest.raises(llm.LLMError):
        list(llm.chat_stream([{"role": "user", "content": "hi"}]))
    assert route.call_count == 1


# --------------------------------------------------------------------- the endpoint
def test_citations_arrive_before_any_text(keyed_client, groq_mock):
    """Sources on screen while the answer is still being written, so a reader can start
    checking the evidence instead of watching a spinner."""
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=sse_stream("Priya ", "owns it."))

    events = read_events(keyed_client)
    types = [e["type"] for e in events]

    assert types[0] == "citations"
    assert events[0]["citations"]
    assert "delta" in types
    assert types.index("citations") < types.index("delta")


def test_the_deltas_reassemble_into_the_answer(keyed_client, groq_mock):
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=sse_stream("Priya ", "owns ", "the fix."))

    events = read_events(keyed_client)
    text = "".join(e["text"] for e in events if e["type"] == "delta")

    assert text == "Priya owns the fix."
    assert events[-1] == {"type": "done", "mode": "answer"}


def test_a_refusal_streams_as_a_single_done_event(keyed_client, groq_mock):
    """The honesty guarantee survives streaming: no citations, no tokens, one refusal."""
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")

    events = read_events(keyed_client, question="What is the capital of France?")

    assert [e["type"] for e in events] == ["done"]
    assert events[0]["mode"] == "refused"
    assert events[0]["answer"] == rag.REFUSAL


def test_an_empty_store_streams_the_empty_mode(keyed_client):
    events = read_events(keyed_client, question="anything")
    assert events[-1]["mode"] == "empty"


def test_without_a_key_the_stream_ends_after_citations(client):
    """Excerpts still stream; prose is never invented without the model."""
    client.post("/api/v1/meetings/sample")
    events = read_events(client)

    assert events[0]["type"] == "citations"
    assert not any(e["type"] == "delta" for e in events)
    assert events[-1]["mode"] == "retrieval_only"


def test_a_provider_failure_before_any_token_falls_back_to_the_buffered_call(
    keyed_client, groq_mock
):
    """Nothing has been shown yet, so retrying whole is safe and better than failing."""
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")

    calls = {"n": 0}

    def _responder(request):
        calls["n"] += 1
        if json.loads(request.content).get("stream"):
            return httpx.Response(500, json={})
        return httpx.Response(200, json=chat_completion("Recovered answer."))

    groq_mock.post(GROQ_CHAT_URL).mock(side_effect=_responder)
    events = read_events(keyed_client)

    assert "".join(e["text"] for e in events if e["type"] == "delta") == "Recovered answer."
    assert events[-1]["mode"] == "answer"


def test_a_failure_with_no_fallback_is_reported_as_an_error_event(keyed_client, groq_mock):
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(500, json={}))

    events = read_events(keyed_client)
    assert events[-1]["type"] == "error"
    assert events[-1]["mode"] == "error"


def test_the_stream_sets_the_headers_a_proxy_needs(keyed_client, groq_mock):
    """Without these an intermediate proxy buffers the whole stream and delivers it in
    one lump, which defeats the entire feature."""
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=sse_stream("hi"))
    with keyed_client.stream(
        "POST", "/api/v1/chat/stream", json={"question": "anything"}
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        assert "no-cache" in response.headers["cache-control"]
        response.read()


def test_streaming_requires_authentication(anon_client):
    assert anon_client.post("/api/v1/chat/stream", json={"question": "hi"}).status_code == 401


def test_streaming_rejects_another_users_meeting_scope(client, second_client):
    mine = client.post("/api/v1/meetings/sample").json()["id"]
    response = second_client.post(
        "/api/v1/chat/stream", json={"question": "hi", "meeting_id": mine}
    )
    assert response.status_code == 404


def test_a_malformed_streaming_request_is_rejected_before_the_stream_opens(client):
    """Validation still returns a normal error envelope; once a stream starts it is too
    late to send a status code."""
    response = client.post("/api/v1/chat/stream", json={"question": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_a_crash_mid_stream_becomes_a_final_error_event(keyed_client, groq_mock, monkeypatch):
    """The response has already started, so this cannot become a 500 -- it has to be
    delivered inside the stream or the client hangs."""
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")

    def _explode(*_a, **_kw):
        yield {"type": "citations", "citations": []}
        raise RuntimeError("internal detail: password hunter2")

    monkeypatch.setattr(rag, "answer_stream", _explode)
    events = read_events(keyed_client)

    assert events[-1]["type"] == "error"
    assert "hunter2" not in json.dumps(events)


def test_streamed_answer_modes_are_counted_in_metrics(keyed_client, groq_mock):
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=sse_stream("hi"))
    read_events(keyed_client)

    assert "meetsaransh_rag_answers_total" in keyed_client.get("/metrics").text


def test_the_non_streaming_endpoint_still_works(keyed_client, groq_mock):
    """Streaming is additive: POST /chat remains the documented fallback."""
    groq_mock.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("Buffered answer."))
    )
    keyed_client.post("/api/v1/meetings/sample")
    body = keyed_client.post("/api/v1/chat", json={"question": "payment bug"}).json()

    assert body["mode"] == "answer"
    assert body["answer"] == "Buffered answer."


def test_both_paths_share_one_refusal_decision(keyed_client, groq_mock):
    """Two copies of the refusal gate would eventually disagree, and the one users
    noticed would be whichever refused a question the other answered."""
    groq_mock.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("{}")))
    keyed_client.post("/api/v1/meetings/sample")
    question = "What is the capital of France?"

    buffered = keyed_client.post("/api/v1/chat", json={"question": question}).json()
    streamed = read_events(keyed_client, question=question)

    assert buffered["mode"] == "refused"
    assert streamed[-1]["mode"] == "refused"


def test_the_refusal_gate_is_shared_code_not_a_copy():
    """Guards the above at the source, so a future edit cannot fork the two paths."""
    source = (config.BASE_DIR / "app" / "rag.py").read_text(encoding="utf-8")
    assert source.count("def _passes_refusal_gate") == 1
    assert source.count("_passes_refusal_gate(result)") == 2
