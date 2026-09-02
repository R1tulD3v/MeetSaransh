"""Cross-encoder reranking.

The model itself is never loaded here (it is a ~90 MB download), so these tests target
the two things that are actually ours: the graceful-degradation contract, and the wiring
into retrieval. Whether reranking *improves* results is not a unit-test question -- it
is measured by `python -m evaluation.run --rerank`, and the answer on the current corpus
was no, which is why it ships disabled.
"""

from __future__ import annotations

import pytest

from app import config, rag, reranker, storage


@pytest.fixture
def fake_reranker(monkeypatch: pytest.MonkeyPatch):
    """A stand-in cross-encoder that scores by a marker, so order is predictable."""

    class _Model:
        def __init__(self) -> None:
            self.calls: list[tuple[str, list[str]]] = []

        def rerank(self, query: str, documents: list[str]) -> list[float]:
            self.calls.append((query, documents))
            # Anything containing the query text scores highest; ties break by position.
            return [10.0 if query.lower() in d.lower() else 0.0 for d in documents]

    model = _Model()
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(reranker, "_model", model)
    monkeypatch.setattr(reranker, "_load_failed", False)
    return model


@pytest.fixture(autouse=True)
def no_reranker_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing in the suite may download or load the real cross-encoder."""
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_load_failed", True)


# ------------------------------------------------------------------- availability
def test_reranking_is_off_by_default():
    """Shipped disabled on evidence -- the harness measured it hurting this corpus."""
    assert config.RERANK_ENABLED is False


def test_unavailable_when_the_model_cannot_load():
    assert reranker.available() is False


def test_unavailable_when_disabled_even_if_the_model_would_load(monkeypatch, fake_reranker):
    """The config switch wins, so turning it off cannot be defeated by a warm model."""
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    assert reranker.available() is False


def test_available_when_enabled_and_loaded(fake_reranker):
    assert reranker.available() is True


def test_a_model_that_fails_to_load_is_only_attempted_once(monkeypatch):
    """Retrying a broken load on every question would make each one slow."""
    attempts = []

    class _Failing:
        def __init__(self, *_a, **_kw):
            attempts.append(1)
            raise RuntimeError("no ONNX runtime here")

    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(reranker, "_model", None)
    monkeypatch.setattr(reranker, "_load_failed", False)
    module = type("m", (), {"TextCrossEncoder": _Failing})
    monkeypatch.setitem(__import__("sys").modules, "fastembed.rerank.cross_encoder", module)

    assert reranker.available() is False
    assert reranker.available() is False
    assert len(attempts) == 1


# ------------------------------------------------------------------------ rerank()
def test_rerank_reorders_by_cross_encoder_score(fake_reranker):
    chunks = [
        {"text": "unrelated filler"},
        {"text": "the payment bug was fixed"},
        {"text": "more filler"},
    ]
    out = reranker.rerank("payment bug", chunks, top_k=3)

    assert out is not None
    assert out[0]["text"] == "the payment bug was fixed"
    assert out[0]["rerank_score"] == 10.0


def test_rerank_truncates_to_top_k(fake_reranker):
    chunks = [{"text": f"chunk {i}"} for i in range(6)]
    out = reranker.rerank("q", chunks, top_k=2)
    assert out is not None
    assert len(out) == 2


def test_rerank_leaves_the_retrieval_score_intact(fake_reranker):
    """Both rankings have to survive, or the evaluation harness cannot compare them."""
    chunks = [{"text": "a", "score": 0.42}]
    out = reranker.rerank("a", chunks, top_k=1)
    assert out[0]["score"] == 0.42
    assert "rerank_score" in out[0]


def test_rerank_returns_none_when_unavailable():
    """None rather than the input list, so a caller cannot mistake a no-op for a rerank."""
    assert reranker.rerank("q", [{"text": "a"}], top_k=1) is None


def test_rerank_of_nothing_is_none(fake_reranker):
    assert reranker.rerank("q", [], top_k=3) is None


def test_a_crashing_reranker_degrades_instead_of_failing_the_question(monkeypatch, fake_reranker):
    """A reranker is an optimisation. It must never turn a working answer into an error."""

    def _explode(*_a, **_kw):
        raise RuntimeError("onnx session died")

    monkeypatch.setattr(fake_reranker, "rerank", _explode)
    assert reranker.rerank("q", [{"text": "a"}], top_k=1) is None


def test_a_reranker_returning_the_wrong_number_of_scores_is_rejected(monkeypatch, fake_reranker):
    monkeypatch.setattr(fake_reranker, "rerank", lambda q, docs: [1.0])
    assert reranker.rerank("q", [{"text": "a"}, {"text": "b"}], top_k=2) is None


# --------------------------------------------------------------- wiring into retrieval
def _index(user_id: str, segments: list[dict], title: str = "Meeting") -> str:
    mid = storage.create_meeting(
        user_id=user_id,
        title=title,
        filename=None,
        transcript={
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
            "duration": segments[-1]["end"],
        },
        summary={},
    )
    rag.index_meeting(mid)
    return mid


@pytest.fixture
def corpus(user_id, monkeypatch):
    """Several small chunks, so there is genuinely something to reorder."""
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 8)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 2)
    _index(
        user_id,
        [
            {"start": 0.0, "end": 5.0, "text": "Hiring is open for two backend engineers."},
            {"start": 5.0, "end": 10.0, "text": "The payment bug came from a retry loop."},
            {"start": 10.0, "end": 15.0, "text": "Checkout latency regressed after the deploy."},
            {"start": 15.0, "end": 20.0, "text": "We agreed to ship on the fifteenth."},
        ],
    )
    return user_id


def test_retrieval_does_not_rerank_by_default(corpus, fake_reranker, monkeypatch):
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    result = rag.retrieve("payment bug", corpus)

    assert result["reranked"] is False
    assert fake_reranker.calls == []


def test_retrieval_reranks_when_enabled(corpus, fake_reranker):
    result = rag.retrieve("payment bug", corpus)

    assert result["reranked"] is True
    assert fake_reranker.calls, "the cross-encoder was never called"


def test_the_caller_can_override_the_config_either_way(corpus, fake_reranker, monkeypatch):
    """The evaluation harness needs to A/B this without mutating global config."""
    assert rag.retrieve("payment bug", corpus, rerank=False)["reranked"] is False

    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    assert rag.retrieve("payment bug", corpus, rerank=True)["reranked"] is True


def test_reranking_never_drops_chunks(corpus, fake_reranker, monkeypatch):
    """Everything past the candidate window keeps its retrieval order behind the
    reranked head, rather than silently disappearing from the result."""
    monkeypatch.setattr(config, "RERANK_CANDIDATES", 2)
    plain = rag.retrieve("payment bug", corpus, rerank=False)
    reranked = rag.retrieve("payment bug", corpus, rerank=True)

    assert reranked["reranked"] is True
    assert len(reranked["ranked"]) == len(plain["ranked"])
    assert {c["id"] for c in reranked["ranked"]} == {c["id"] for c in plain["ranked"]}


def test_a_broken_reranker_leaves_retrieval_working(corpus, fake_reranker, monkeypatch):
    monkeypatch.setattr(
        fake_reranker, "rerank", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    result = rag.retrieve("payment bug", corpus)

    assert result["reranked"] is False
    assert result["ranked"], "retrieval must still return its own ranking"


def test_reranking_does_not_move_the_refusal_gate(corpus, fake_reranker):
    """`dense_best` is captured before reranking on purpose: whether a question is
    answerable is a judgement about the corpus, and must not depend on whether a
    reranker happened to load."""
    plain = rag.retrieve("payment bug", corpus, rerank=False)
    reranked = rag.retrieve("payment bug", corpus, rerank=True)

    assert plain["dense_best"] == reranked["dense_best"]
    assert plain["content_match"] == reranked["content_match"]


def test_an_off_topic_question_is_still_refused_with_reranking_on(corpus, fake_reranker):
    """The honesty guarantee must not be weakened by a second ranking stage."""
    assert rag.answer("What is the capital of France?", corpus, rerank=True)["mode"] == "refused"


def test_rag_status_reports_reranker_availability(client):
    body = client.get("/api/v1/rag/status").json()
    assert body["reranker_available"] is False  # disabled by default and not loaded
