"""The evaluation harness: metric correctness, dataset integrity, and the runner.

A measurement tool that is itself untested is just a number generator. Two things get
the most attention here:

* the metric functions, because a subtly wrong recall would quietly justify whatever
  retrieval configuration happened to ship; and
* the dataset labels, because a mistyped `expect` marker can never match anything, so
  the harness would report a permanent failure that is actually a typo -- or, worse, a
  marker that matches the wrong meeting would report a success that is actually a bug.
"""

from __future__ import annotations

import json

import pytest

from evaluation import metrics
from evaluation.run import DATASET_PATH, main

DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- primitives
def test_marker_matching_ignores_case_and_whitespace():
    """Chunk text is joined from segments, so whitespace is not stable enough to match on."""
    assert metrics._contains("The  RETRY   loop\nfired twice", "retry loop fired twice")


def test_marker_matching_is_not_fuzzy_beyond_that():
    assert not metrics._contains("the retry loop fired", "retry loop fired twice")


def test_relevant_ranks_are_one_based_and_complete():
    texts = ["nothing here", "the marker is here", "also the marker"]
    assert metrics.relevant_ranks(texts, ["the marker"]) == [2, 3]


def test_relevant_ranks_empty_when_nothing_matches():
    assert metrics.relevant_ranks(["a", "b"], ["z"]) == []


# ------------------------------------------------------------------------ hit rate
def test_hit_rate_is_one_when_any_marker_lands_in_top_k():
    assert metrics.hit_rate_at_k(["no", "yes marker"], ["marker"], k=2) == 1.0


def test_hit_rate_respects_k():
    """The evidence exists but ranked too low, which is a miss for a top-k pipeline."""
    assert metrics.hit_rate_at_k(["no", "yes marker"], ["marker"], k=1) == 0.0


# -------------------------------------------------------------------------- recall
def test_recall_counts_each_marker_separately():
    """Multi-part questions: finding one of two answers is a half-answer, not a win."""
    texts = ["contains alpha", "irrelevant"]
    assert metrics.recall_at_k(texts, ["alpha", "beta"], k=2) == 0.5


def test_recall_is_one_when_every_marker_is_found():
    texts = ["contains alpha", "contains beta"]
    assert metrics.recall_at_k(texts, ["alpha", "beta"], k=2) == 1.0


def test_recall_does_not_double_count_one_chunk_holding_two_markers():
    assert metrics.recall_at_k(["alpha and beta together"], ["alpha", "beta"], k=1) == 1.0


def test_recall_with_no_labels_is_zero_not_a_crash():
    assert metrics.recall_at_k(["anything"], [], k=1) == 0.0


# ----------------------------------------------------------------------- precision
def test_precision_measures_how_much_of_the_context_was_useful():
    texts = ["marker here", "filler", "filler", "filler"]
    assert metrics.precision_at_k(texts, ["marker"], k=4) == 0.25


def test_precision_on_an_empty_result_is_zero():
    assert metrics.precision_at_k([], ["marker"], k=5) == 0.0


# --------------------------------------------------------------------------- MRR
@pytest.mark.parametrize(("rank", "expected"), [(1, 1.0), (2, 0.5), (4, 0.25)])
def test_reciprocal_rank_rewards_ranking_evidence_higher(rank, expected):
    texts = ["filler"] * (rank - 1) + ["marker"]
    assert metrics.reciprocal_rank(texts, ["marker"], k=10) == expected


def test_reciprocal_rank_is_zero_when_evidence_misses_the_cut():
    assert metrics.reciprocal_rank(["filler", "marker"], ["marker"], k=1) == 0.0


def test_mean_of_nothing_is_zero():
    assert metrics.mean([]) == 0.0


# --------------------------------------------------------------------- scorecard
def _result(**kw) -> metrics.QuestionResult:
    base = {
        "question_id": "q",
        "question": "?",
        "hit": 1.0,
        "recall": 1.0,
        "precision": 0.5,
        "reciprocal_rank": 1.0,
        "correct_meeting": True,
        "mode": "retrieval_only",
    }
    return metrics.QuestionResult(**{**base, **kw})


def test_scorecard_averages_across_questions():
    card = metrics.Scorecard(label="test", k=3)
    card.results = [_result(hit=1.0, recall=1.0), _result(hit=0.0, recall=0.0)]
    assert card.hit_rate == 0.5
    assert card.recall == 0.5


def test_scorecard_separates_literal_from_paraphrased():
    """The subset the retrieval strategy is actually chosen on."""
    card = metrics.Scorecard(label="test", k=3)
    card.results = [
        _result(recall=1.0, paraphrase=False),
        _result(recall=0.0, paraphrase=True),
    ]
    assert card.recall_literal == 1.0
    assert card.recall_paraphrase == 0.0
    assert card.n_literal == 1
    assert card.n_paraphrase == 1


def test_meeting_accuracy_catches_answering_from_the_wrong_meeting():
    card = metrics.Scorecard(label="test", k=3)
    card.results = [_result(correct_meeting=True), _result(correct_meeting=False)]
    assert card.meeting_accuracy == 0.5


def test_refusal_accuracy_and_false_refusals_are_reported_together():
    """A system that refuses everything scores a perfect 1.0 on refusals and is useless,
    so the two numbers only mean anything as a pair."""
    card = metrics.Scorecard(label="test", k=3, off_topic_total=4, off_topic_refused=4)
    card.results = [_result(mode="refused"), _result(mode="retrieval_only")]

    assert card.refusal_accuracy == 1.0
    assert card.false_refusal_rate == 0.5  # and the pair exposes it


def test_refusal_accuracy_with_no_off_topic_questions_is_zero():
    assert metrics.Scorecard(label="test", k=3).refusal_accuracy == 0.0


def test_failures_lists_only_the_total_misses():
    card = metrics.Scorecard(label="test", k=3)
    card.results = [_result(question_id="good", hit=1.0), _result(question_id="bad", hit=0.0)]
    assert [f.question_id for f in card.failures()] == ["bad"]


# ------------------------------------------------------------- dataset integrity
def _meeting_text(meeting_id: str) -> str:
    meeting = next(m for m in DATASET["meetings"] if m["id"] == meeting_id)
    return " ".join(s["text"] for s in meeting["segments"])


def test_every_question_points_at_a_meeting_that_exists():
    meeting_ids = {m["id"] for m in DATASET["meetings"]}
    for q in DATASET["questions"]:
        assert q["meeting_id"] in meeting_ids, f"{q['id']} names an unknown meeting"


@pytest.mark.parametrize("question", DATASET["questions"], ids=lambda q: q["id"])
def test_every_label_actually_appears_in_its_meeting(question):
    """The single most important test here.

    A mistyped marker can never match, so the harness would report a permanent
    retrieval failure that is really a typo in the labels -- and a plausible-looking
    number that is quietly wrong is worse than no number at all.
    """
    haystack = _meeting_text(question["meeting_id"])
    for marker in question["expect"]:
        assert metrics._contains(haystack, marker), (
            f"{question['id']}: label {marker!r} does not appear in {question['meeting_id']}"
        )


@pytest.mark.parametrize("question", DATASET["questions"], ids=lambda q: q["id"])
def test_no_label_leaks_into_a_different_meeting(question):
    """A marker that also matches another meeting cannot distinguish between them, so a
    correct-meeting score built on it would be meaningless."""
    others = [m["id"] for m in DATASET["meetings"] if m["id"] != question["meeting_id"]]
    for marker in question["expect"]:
        for other in others:
            assert not metrics._contains(_meeting_text(other), marker), (
                f"{question['id']}: label {marker!r} also matches {other}"
            )


def test_question_and_meeting_ids_are_unique():
    ids = [q["id"] for q in DATASET["questions"]] + [q["id"] for q in DATASET["off_topic"]]
    assert len(ids) == len(set(ids))
    meeting_ids = [m["id"] for m in DATASET["meetings"]]
    assert len(meeting_ids) == len(set(meeting_ids))


def test_the_set_has_enough_paraphrased_questions_to_discriminate():
    """Without them every configuration scores 1.000 and the ablation proves nothing."""
    paraphrased = [q for q in DATASET["questions"] if q.get("paraphrase")]
    assert len(paraphrased) >= 5


def test_off_topic_questions_share_no_content_words_with_the_corpus():
    """If an off-topic question shared vocabulary with a meeting, a refusal failure
    would be the label's fault rather than the system's."""
    from app import rag

    corpus = " ".join(_meeting_text(m["id"]) for m in DATASET["meetings"])
    vocabulary = set(rag._tokenize(corpus))
    for item in DATASET["off_topic"]:
        overlap = rag._content_terms(item["question"]) & vocabulary
        assert not overlap, f"{item['id']} overlaps the corpus on {overlap}"


def test_every_meeting_is_actually_asked_about():
    """An unreferenced meeting is dead weight that only slows the run down."""
    asked = {q["meeting_id"] for q in DATASET["questions"]}
    assert asked == {m["id"] for m in DATASET["meetings"]}


# ------------------------------------------------------------------------ runner
@pytest.mark.slow
def test_the_runner_completes_and_reports(capsys):
    """End-to-end smoke test in lexical mode: no model, no network, deterministic.

    The autouse fixtures force the embedding model off, so the runner takes its
    'dense is unavailable, fall back to lexical' path -- which is also the path a
    machine without the model download takes in real life.
    """
    exit_code = main(["--mode", "lexical", "-k", "1", "3"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "MeetSaransh retrieval evaluation" in output
    assert "refusal accuracy" in output
    assert "lexical" in output


@pytest.mark.slow
def test_the_runner_gate_fails_on_an_impossible_threshold():
    """The regression gate has to be able to fail, or it is decoration."""
    assert main(["--mode", "lexical", "-k", "1", "--min-recall", "1.01"]) == 1


@pytest.mark.slow
def test_the_runner_gate_passes_on_a_reachable_threshold():
    assert main(["--mode", "lexical", "-k", "3", "--min-refusal", "1.0"]) == 0
