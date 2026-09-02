"""Retrieval metrics.

Pure functions over already-retrieved results, so they are unit-testable without a
database, a model, or a network -- which is the point: a metric you cannot test is a
number you cannot trust.

Relevance is defined by substring markers rather than chunk ids. Labelling by id would
mean every change to the chunking strategy silently invalidates the labels, and
chunking is one of the things this harness exists to evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _contains(haystack: str, marker: str) -> bool:
    """Case-insensitive substring match, tolerant of whitespace differences."""
    return " ".join(marker.lower().split()) in " ".join(haystack.lower().split())


def relevant_ranks(retrieved_texts: list[str], markers: list[str]) -> list[int]:
    """1-based ranks of the retrieved chunks that contain at least one marker."""
    return [
        rank
        for rank, text in enumerate(retrieved_texts, start=1)
        if any(_contains(text, m) for m in markers)
    ]


def hit_rate_at_k(retrieved_texts: list[str], markers: list[str], k: int) -> float:
    """1.0 if any relevant chunk made the top k, else 0.0.

    The question "did the LLM get anything useful to work with at all?" -- which for a
    grounded answer is the difference between an answer and a refusal.
    """
    return 1.0 if relevant_ranks(retrieved_texts[:k], markers) else 0.0


def recall_at_k(retrieved_texts: list[str], markers: list[str], k: int) -> float:
    """Fraction of the labelled evidence that made the top k.

    Distinct from hit rate for multi-part questions: "What are Rahul's action items?"
    has two pieces of evidence, and finding one of them is a half-answer, not a win.
    """
    if not markers:
        return 0.0
    top = retrieved_texts[:k]
    found = sum(1 for m in markers if any(_contains(t, m) for t in top))
    return found / len(markers)


def precision_at_k(retrieved_texts: list[str], markers: list[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    Worth watching because every retrieved chunk is tokens in the prompt: padding the
    context with near-misses costs money and dilutes the model's attention.
    """
    top = retrieved_texts[:k]
    if not top:
        return 0.0
    return len(relevant_ranks(top, markers)) / len(top)


def reciprocal_rank(retrieved_texts: list[str], markers: list[str], k: int) -> float:
    """1 / rank of the first relevant chunk, or 0 if none is in the top k.

    Rank matters beyond hit rate: the citation shown first is the one a user actually
    clicks, so evidence at rank 1 is worth more than the same evidence at rank 6.
    """
    ranks = relevant_ranks(retrieved_texts[:k], markers)
    return 1.0 / ranks[0] if ranks else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class QuestionResult:
    """One question's outcome under one retrieval configuration."""

    question_id: str
    question: str
    hit: float
    recall: float
    precision: float
    reciprocal_rank: float
    correct_meeting: bool
    mode: str  # the answer mode the pipeline returned: answer/refused/retrieval_only/...
    paraphrase: bool = False  # worded to avoid the transcript's own vocabulary


@dataclass
class Scorecard:
    """Aggregated results for one retrieval configuration."""

    label: str
    k: int
    results: list[QuestionResult] = field(default_factory=list)
    off_topic_total: int = 0
    off_topic_refused: int = 0
    faithfulness: float | None = None
    judged: int = 0

    @property
    def hit_rate(self) -> float:
        return mean([r.hit for r in self.results])

    @property
    def recall(self) -> float:
        return mean([r.recall for r in self.results])

    @property
    def precision(self) -> float:
        return mean([r.precision for r in self.results])

    @property
    def mrr(self) -> float:
        return mean([r.reciprocal_rank for r in self.results])

    @property
    def recall_literal(self) -> float:
        """Recall on questions that reuse the transcript's wording."""
        return mean([r.recall for r in self.results if not r.paraphrase])

    @property
    def recall_paraphrase(self) -> float:
        """Recall on questions deliberately worded to avoid the transcript's vocabulary.

        This is the subset the retrieval strategy is chosen for. On literal questions
        BM25 alone already scores near-perfectly, so an ablation run only on those
        would show no difference between configurations and would quietly justify
        whichever one happened to ship.
        """
        return mean([r.recall for r in self.results if r.paraphrase])

    @property
    def n_literal(self) -> int:
        return len([r for r in self.results if not r.paraphrase])

    @property
    def n_paraphrase(self) -> int:
        return len([r for r in self.results if r.paraphrase])

    @property
    def meeting_accuracy(self) -> float:
        """How often the top-ranked chunk came from the right meeting.

        The cross-meeting failure this catches is the embarrassing one: a confident,
        well-cited answer sourced from the wrong meeting entirely.
        """
        return mean([1.0 if r.correct_meeting else 0.0 for r in self.results])

    @property
    def refusal_accuracy(self) -> float:
        """Share of off-topic questions correctly refused."""
        if not self.off_topic_total:
            return 0.0
        return self.off_topic_refused / self.off_topic_total

    @property
    def false_refusal_rate(self) -> float:
        """Share of ANSWERABLE questions that were wrongly refused.

        The counterweight to refusal accuracy: a system that refuses everything scores
        a perfect 1.0 on refusals and is useless. Both numbers only mean something
        together.
        """
        return mean([1.0 if r.mode == "refused" else 0.0 for r in self.results])

    def failures(self) -> list[QuestionResult]:
        """Questions where nothing relevant was retrieved -- the list worth reading."""
        return [r for r in self.results if r.hit == 0.0]
