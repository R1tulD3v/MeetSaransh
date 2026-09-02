"""LLM query rewriting.

Two things matter here and get most of the tests: the rewrite must never be able to
break search (every failure path falls back to the original question), and it must never
be able to weaken the refusal gate.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from app import config, rag, storage
from tests.conftest import GROQ_CHAT_URL, chat_completion


@pytest.fixture
def rewriter(monkeypatch: pytest.MonkeyPatch):
    """Deterministic stand-in rewriter, so retrieval assertions stay stable."""
    calls: list[str] = []

    def _rewrite(question: str) -> str:
        calls.append(question)
        return question + " cart serializer latency"

    monkeypatch.setattr(config, "QUERY_REWRITE_ENABLED", True)
    monkeypatch.setattr(rag, "rewrite_query", _rewrite)
    return calls


# ------------------------------------------------------------------- rewrite_query
@respx.mock
def test_a_rewrite_replaces_the_query(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("cart page slow latency"))
    )
    assert rag.rewrite_query("Why was the basket dragging?") == "cart page slow latency"


@respx.mock
def test_surrounding_quotes_and_newlines_are_stripped(with_api_key):
    """Models wrap output in quotes often enough that searching for them would hurt."""
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion('  "cart page\n  slow"  '))
    )
    assert rag.rewrite_query("q") == "cart page slow"


def test_without_a_key_the_question_is_used_unchanged(without_api_key):
    assert rag.rewrite_query("original question") == "original question"


@respx.mock
def test_a_provider_failure_falls_back_to_the_question(with_api_key, monkeypatch):
    """A rewriter that can break search is worse than no rewriter."""
    monkeypatch.setattr(rag.llm.time, "sleep", lambda _: None)
    respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(500, json={}))
    assert rag.rewrite_query("original question") == "original question"


@respx.mock
def test_an_empty_rewrite_falls_back(with_api_key):
    respx.post(GROQ_CHAT_URL).mock(return_value=httpx.Response(200, json=chat_completion("   ")))
    assert rag.rewrite_query("original question") == "original question"


@respx.mock
def test_a_rambling_rewrite_is_discarded(with_api_key):
    """A long reply means the model started explaining or answering instead of
    rewriting -- searching with that would drag its invented words into retrieval."""
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("word " * 500))
    )
    assert rag.rewrite_query("original question") == "original question"


@respx.mock
def test_the_length_bound_is_configurable(with_api_key, monkeypatch):
    monkeypatch.setattr(config, "QUERY_REWRITE_MAX_CHARS", 10)
    respx.post(GROQ_CHAT_URL).mock(
        return_value=httpx.Response(200, json=chat_completion("a much longer rewrite than ten"))
    )
    assert rag.rewrite_query("q") == "q"


# ------------------------------------------------------------- wiring into retrieval
def _corpus(user_id: str) -> str:
    segments: list[dict[str, object]] = [
        {"start": 0.0, "end": 6.0, "text": "Checkout latency regressed after the deploy."},
        {"start": 6.0, "end": 12.0, "text": "It was an N plus one query in the cart serializer."},
        {"start": 12.0, "end": 18.0, "text": "Hiring stays open for two backend engineers."},
    ]
    mid = storage.create_meeting(
        user_id=user_id,
        title="Infra review",
        filename=None,
        transcript={
            "text": " ".join(str(s["text"]) for s in segments),
            "segments": segments,
            "duration": 18.0,
        },
        summary={},
    )
    rag.index_meeting(mid)
    return mid


def test_retrieval_reports_when_it_rewrote(user_id, rewriter):
    _corpus(user_id)
    result = rag.retrieve("Why was the basket page dragging?", user_id)

    assert result["rewritten"] is True
    assert "cart serializer" in result["search_text"]
    assert rewriter == ["Why was the basket page dragging?"]


def test_rewriting_can_be_turned_off_per_call(user_id, rewriter):
    """The evaluation harness A/Bs this without mutating global config."""
    _corpus(user_id)
    result = rag.retrieve("Why was the basket page dragging?", user_id, rewrite=False)

    assert result["rewritten"] is False
    assert result["search_text"] == "Why was the basket page dragging?"
    assert rewriter == []


def test_disabled_by_config_means_no_llm_call(user_id, rewriter, monkeypatch):
    monkeypatch.setattr(config, "QUERY_REWRITE_ENABLED", False)
    _corpus(user_id)
    rag.retrieve("anything", user_id)
    assert rewriter == []


def test_a_rewrite_that_returns_the_question_is_not_reported_as_rewritten(user_id, monkeypatch):
    monkeypatch.setattr(config, "QUERY_REWRITE_ENABLED", True)
    monkeypatch.setattr(rag, "rewrite_query", lambda q: q)
    _corpus(user_id)
    assert rag.retrieve("anything", user_id)["rewritten"] is False


# ------------------------------------------------------- the honesty guarantee holds
def test_the_refusal_gate_reads_the_original_question_not_the_rewrite(user_id, monkeypatch):
    """The critical safety property.

    A rewriter adds synonyms, and synonyms of an off-topic question can collide with the
    corpus. If the gate read the rewrite, a question about France could pick up a
    keyword and stop being refused -- so the rewrite is allowed to improve ranking, and
    is not allowed to change whether we have an answer at all.
    """
    monkeypatch.setattr(config, "QUERY_REWRITE_ENABLED", True)
    # A deliberately hostile rewriter that injects corpus vocabulary into anything.
    monkeypatch.setattr(rag, "rewrite_query", lambda q: q + " checkout latency cart hiring")
    _corpus(user_id)

    result = rag.retrieve("What is the capital of France?", user_id)
    assert result["rewritten"] is True
    assert result["content_match"] is False  # judged on the original question


def test_an_off_topic_question_is_still_refused_with_rewriting_on(user_id, monkeypatch):
    monkeypatch.setattr(config, "QUERY_REWRITE_ENABLED", True)
    monkeypatch.setattr(rag, "rewrite_query", lambda q: q + " checkout latency cart hiring")
    _corpus(user_id)

    assert rag.answer("What is the capital of France?", user_id)["mode"] == "refused"


def test_a_legitimate_paraphrase_still_retrieves(user_id, rewriter):
    """The point of the feature: vocabulary the transcript never used still finds it."""
    _corpus(user_id)
    result = rag.retrieve("Why was the basket page dragging?", user_id)

    top = result["ranked"][0]["text"]
    assert "cart serializer" in top


# ------------------------------------------------------------------------ defaults
def test_rewriting_is_on_by_default():
    """Shipped on because the harness measured recall@1 0.827 -> 0.981 and paraphrase
    recall@1 0.500 -> 1.000, with the refusal metrics unchanged."""
    assert config.QUERY_REWRITE_ENABLED is True


def test_the_prompt_forbids_answering(with_api_key):
    """A rewriter that adds facts would smuggle a hallucination into retrieval, where
    the grounding prompt can no longer catch it."""
    from app import prompts

    text = prompts.QUERY_REWRITE_PROMPT.lower()
    assert "do not answer" in text
    assert "invent" in text
