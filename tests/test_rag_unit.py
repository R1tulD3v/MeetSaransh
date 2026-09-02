"""RAG internals: chunking, BM25, hybrid scoring, the refusal gate, and citations.

These are the pieces that carry the product's honesty guarantees, so they get the most
detailed tests in the suite: a regression here means the app confidently answers a
question it has no evidence for.
"""

from __future__ import annotations

import itertools

import pytest

from app import config, rag, storage


# --------------------------------------------------------------------------- timestamps
@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0, "00:00"),
        (9.4, "00:09"),
        (65, "01:05"),
        (599.99, "09:59"),
        (3600, "1:00:00"),
        (3725, "1:02:05"),
        (None, "00:00"),
    ],
)
def test_fmt_ts(seconds, expected):
    assert rag.fmt_ts(seconds) == expected


# ----------------------------------------------------------------------------- chunking
def test_chunking_empty_input_yields_no_chunks():
    assert rag.chunk_segments([]) == []


def test_chunk_ordinals_are_dense_and_sequential(sample_segments):
    chunks = rag.chunk_segments(sample_segments)
    assert [c["ord"] for c in chunks] == list(range(len(chunks)))


def test_a_short_transcript_becomes_a_single_chunk(sample_segments):
    """Well under the word budget, so splitting would only hurt retrieval precision."""
    chunks = rag.chunk_segments(sample_segments)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 31.0


def test_chunking_splits_once_past_the_word_budget(monkeypatch, sample_segments):
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 5)
    chunks = rag.chunk_segments(sample_segments)
    assert len(chunks) > 1


def test_chunking_terminates_on_pathological_input(monkeypatch):
    """A tiny budget with a large overlap must still make forward progress.

    Without the `guarantee forward progress` step-back guard this loops forever, which
    in production is a hung worker rather than a wrong answer.
    """
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 1)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 100)
    segments = [
        {"start": float(i), "end": float(i + 1), "text": f"word{i} " * 30} for i in range(20)
    ]
    chunks = rag.chunk_segments(segments)
    assert 0 < len(chunks) <= len(segments)


def test_chunks_overlap_so_context_survives_a_boundary(monkeypatch, sample_segments):
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 6)
    chunks = rag.chunk_segments(sample_segments)
    shared = [
        bool(set(a["text"].split()) & set(b["text"].split())) for a, b in itertools.pairwise(chunks)
    ]
    assert all(shared), "consecutive chunks must share words"


def test_chunk_timestamps_are_ordered_and_cover_the_transcript(monkeypatch, sample_segments):
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 4)
    chunks = rag.chunk_segments(sample_segments)

    assert chunks[0]["start"] == sample_segments[0]["start"]
    assert chunks[-1]["end"] == sample_segments[-1]["end"]
    for c in chunks:
        assert c["start"] <= c["end"]
    assert [c["start"] for c in chunks] == sorted(c["start"] for c in chunks)


def test_every_segment_appears_in_some_chunk(monkeypatch, sample_segments):
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 10)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 4)
    joined = " ".join(c["text"] for c in rag.chunk_segments(sample_segments))
    for seg in sample_segments:
        assert seg["text"] in joined


def test_chunks_carry_their_segments_for_precise_citations(sample_segments):
    chunk = rag.chunk_segments(sample_segments)[0]
    assert len(chunk["segs"]) == len(sample_segments)
    assert chunk["segs"][2]["start"] == 11.0


def test_blank_segments_are_skipped():
    segments = [{"start": 0.0, "end": 1.0, "text": "   "}, {"start": 1.0, "end": 2.0, "text": ""}]
    assert rag.chunk_segments(segments) == []


# --------------------------------------------------------------------------------- BM25
def test_bm25_ranks_the_document_containing_the_query_term_first():
    docs = [
        "we discussed the payment gateway migration".split(),
        "hiring plans for the next quarter".split(),
        "the coffee machine is broken".split(),
    ]
    scores = rag._bm25_scores("payment gateway", docs)
    assert scores[0] == max(scores)
    assert scores[2] == 0.0


def test_bm25_ignores_terms_absent_from_the_corpus():
    docs = [["alpha", "beta"], ["gamma"]]
    assert rag._bm25_scores("nonexistent term", docs) == [0.0, 0.0]


def test_bm25_on_an_empty_corpus_is_empty_not_a_crash():
    assert rag._bm25_scores("anything", []) == []


def test_bm25_rewards_rare_terms_over_ubiquitous_ones():
    """IDF must do its job: a term in every document carries no discriminating signal."""
    docs = [["release", "common"], ["release", "common"], ["release", "rare"]]
    common = rag._bm25_scores("common", docs)
    rare = rag._bm25_scores("rare", docs)
    assert max(rare) > max(common)


def test_normalize_scales_to_a_unit_maximum():
    assert rag._normalize([2.0, 1.0, 0.0]) == [1.0, 0.5, 0.0]


def test_normalize_handles_all_zero_and_empty_input():
    assert rag._normalize([0.0, 0.0]) == [0.0, 0.0]
    assert rag._normalize([]) == []


# ------------------------------------------------------------------------ content terms
def test_content_terms_drop_stopwords_and_single_characters():
    assert rag._content_terms("What did we decide about the API?") == {"decide", "api"}


def test_a_question_made_only_of_stopwords_has_no_content_terms():
    assert rag._content_terms("What did we do about that?") == set()


# --------------------------------------------------------------------------- retrieval
def _index(segments: list[dict], title: str = "Planning Sync") -> str:
    mid = storage.create_meeting(
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


def test_retrieve_on_an_empty_store_reports_empty(client):
    assert rag.retrieve("anything")["empty"] is True


def test_lexical_only_retrieval_still_ranks_correctly(sample_segments):
    """With no embedding model the app degrades to BM25 rather than breaking."""
    _index(sample_segments)
    result = rag.retrieve("payment gateway migration")
    assert result["dense_used"] is False
    assert result["ranked"][0]["lexical"] > 0
    assert result["content_match"] is True


def test_retrieval_uses_the_dense_signal_when_vectors_exist(fake_embeddings, sample_segments):
    _index(sample_segments)
    result = rag.retrieve("payment gateway migration")
    assert result["dense_used"] is True
    assert result["dense_best"] > 0


def test_off_topic_question_has_no_content_match(sample_segments):
    _index(sample_segments)
    assert rag.retrieve("What is the capital of France?")["content_match"] is False


def test_retrieval_can_be_scoped_to_one_meeting(sample_segments):
    a = _index(sample_segments, title="Meeting A")
    _index(
        [{"start": 0.0, "end": 4.0, "text": "Unrelated design review notes."}], title="Meeting B"
    )

    scoped = rag.retrieve("notes", scope_meeting_id=a)
    assert {c["meeting_id"] for c in scoped["ranked"]} == {a}
    assert len(rag.retrieve("notes")["ranked"]) > len(scoped["ranked"])


# ------------------------------------------------------------------------ refusal gate
def test_refuses_a_clearly_unrelated_question(sample_segments):
    """The headline honesty guarantee: no keyword overlap and no semantic match."""
    _index(sample_segments)
    result = rag.answer("What is the capital of France?")
    assert result["mode"] == "refused"
    assert result["answer"] == rag.REFUSAL
    assert result["citations"] == []


def test_a_keyword_match_is_enough_to_attempt_an_answer(without_api_key, sample_segments):
    """A valid keyword question must not be refused just because phrasing differs."""
    _index(sample_segments)
    result = rag.answer("What did Rahul say?")
    assert result["mode"] == "retrieval_only"
    assert result["citations"]


def test_answering_with_no_meetings_indexed_reports_empty():
    assert rag.answer("anything")["mode"] == "empty"


def test_a_blank_question_is_rejected_before_retrieval():
    assert rag.answer("   ")["mode"] == "error"


def test_without_a_key_the_answer_is_excerpts_not_a_guess(without_api_key, sample_segments):
    _index(sample_segments)
    result = rag.answer("payment gateway")
    assert result["mode"] == "retrieval_only"
    assert result["answer"] is None  # never fabricate prose without the model
    assert result["note"]


def test_top_k_caps_the_number_of_citations(monkeypatch, without_api_key):
    monkeypatch.setattr(config, "RAG_TOP_K", 2)
    monkeypatch.setattr(config, "CHUNK_TARGET_WORDS", 6)
    monkeypatch.setattr(config, "CHUNK_OVERLAP_WORDS", 2)
    segments = [
        {
            "start": float(i * 5),
            "end": float(i * 5 + 5),
            "text": f"release planning topic number {i}",
        }
        for i in range(8)
    ]
    _index(segments)
    assert len(rag.answer("release planning")["citations"]) == 2


# ---------------------------------------------------------------------------- citations
def test_citation_points_at_the_best_matching_line_not_the_chunk_start(sample_segments):
    """Citation precision: the deep link must land on the line that answers the question."""
    chunk = rag.chunk_segments(sample_segments)[0]
    chunk |= {"meeting_id": "m1", "meeting_title": "Planning Sync", "score": 0.9}

    citation = rag._citation(chunk, "What did Rahul say about checkout latency?")
    assert "Rahul" in citation["snippet"]
    assert citation["start"] == 11.0
    assert citation["timestamp"] == "00:11"


def test_citation_falls_back_to_the_first_segment_when_nothing_matches(sample_segments):
    chunk = rag.chunk_segments(sample_segments)[0]
    chunk |= {"meeting_id": "m1", "meeting_title": "Planning Sync", "score": 0.1}
    assert rag._citation(chunk, "zzz qqq")["start"] == 0.0


def test_long_snippets_are_truncated_on_a_word_boundary():
    chunk = {
        "meeting_id": "m1",
        "meeting_title": "T",
        "start": 0.0,
        "text": "word " * 200,
        "segs": [],
        "score": 0.5,
    }
    snippet = rag._citation(chunk, "word")["snippet"]
    assert len(snippet) <= 241
    assert snippet.endswith("…")


# ------------------------------------------------------------------------- index/status
def test_indexing_an_unknown_meeting_is_a_no_op():
    assert rag.index_meeting("missing") == 0


def test_a_meeting_with_no_segments_indexes_to_zero_chunks():
    mid = storage.create_meeting(
        title="Empty",
        filename=None,
        transcript={"text": "", "segments": [], "duration": 0},
        summary={},
    )
    assert rag.index_meeting(mid) == 0
    assert storage.count_chunks() == 0


def test_reindex_only_touches_meetings_that_are_not_indexed_yet(sample_segments):
    _index(sample_segments)
    result = rag.reindex_all()
    assert result["meetings"] == 1
    assert result["newly_indexed"] == 0  # already indexed; no wasted embedding work
    assert result["total_chunks"] > 0


def test_reindex_picks_up_a_meeting_stored_without_indexing(sample_segments):
    storage.create_meeting(
        title="Unindexed",
        filename=None,
        transcript={"text": "x", "segments": sample_segments, "duration": 31.0},
        summary={},
    )
    assert rag.reindex_all()["newly_indexed"] == 1


def test_status_reports_the_store_contents(sample_segments):
    _index(sample_segments)
    status = rag.status()
    assert status["indexed_meetings"] == 1
    assert status["total_chunks"] >= 1
    assert status["embeddings_available"] is False  # forced off in tests
    assert status["embed_model"] == config.EMBED_MODEL
