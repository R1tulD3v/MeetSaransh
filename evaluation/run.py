"""Run the retrieval evaluation and print a scorecard.

    python -m evaluation.run                 # every available mode, with an ablation
    python -m evaluation.run --mode lexical  # deterministic: no model, no network
    python -m evaluation.run --judge         # add LLM-as-judge faithfulness (needs a key)
    python -m evaluation.run --min-recall 0.8 --min-refusal 1.0   # regression gate

Everything runs against a throwaway database in a temp directory, so evaluating never
touches real data, and the corpus is identical on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from app.rag import RetrievalMode
else:  # the app is imported lazily inside functions, so this alias is type-only
    RetrievalMode = str

DATASET_PATH = Path(__file__).with_name("dataset.json")


def _load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _build_corpus(dataset: dict) -> tuple[str, dict[str, str]]:
    """Create the eval user and index every meeting. Returns (user_id, id_map)."""
    from app import rag, storage

    user = storage.create_user(email="evaluation@localhost", password_hash="not-a-login")
    id_map: dict[str, str] = {}
    for meeting in dataset["meetings"]:
        segments = meeting["segments"]
        stored_id = storage.create_meeting(
            user_id=user["id"],
            title=meeting["title"],
            filename=None,
            transcript={
                "text": " ".join(s["text"] for s in segments),
                "segments": segments,
                "duration": segments[-1]["end"],
            },
            summary={},
        )
        rag.index_meeting(stored_id)
        id_map[meeting["id"]] = stored_id
    return user["id"], id_map


def _judge_faithfulness(question: str, answer: str, excerpts: str) -> bool | None:
    """Ask the model whether an answer is supported by its excerpts.

    LLM-as-judge is a proxy, not ground truth -- so it is opt-in, reported separately
    from the retrieval metrics, and never gates CI. It answers one narrow, checkable
    question (is this grounded?) rather than a vague "is this good?".
    """
    from app import llm

    prompt = [
        {
            "role": "system",
            "content": (
                "You judge whether an ANSWER is fully supported by the EXCERPTS. "
                "Reply with exactly one word: SUPPORTED or UNSUPPORTED. "
                "An answer that states anything not present in the excerpts is "
                "UNSUPPORTED. An answer that correctly says the excerpts do not "
                "contain the information is SUPPORTED."
            ),
        },
        {
            "role": "user",
            "content": f"QUESTION:\n{question}\n\nEXCERPTS:\n{excerpts}\n\nANSWER:\n{answer}",
        },
    ]
    try:
        verdict = llm.chat(prompt, temperature=0.0)
    except llm.LLMError:
        return None
    return verdict.strip().upper().startswith("SUPPORTED")


def evaluate(
    mode: RetrievalMode,
    k: int,
    user_id: str,
    id_map: dict[str, str],
    *,
    judge: bool = False,
    alpha: float | None = None,
    rerank: bool = False,
    rewrite: bool = False,
):
    """Run every labelled question through one retrieval configuration.

    Unless `judge` is set, the API key is blanked for the duration of the run. Every
    metric here except faithfulness is decided before the LLM is ever called -- the
    refusal gate included -- so generating real answers would add minutes, cost money,
    and trip the provider's rate limit without changing a single number.
    """
    from app import config, rag
    from evaluation import metrics

    dataset = _load_dataset()
    label = mode if alpha is None else f"a={alpha:g}"
    if rerank:
        label += "+rr"
    if rewrite:
        label += "+qr"
    card = metrics.Scorecard(label=label, k=k)

    real_key = config.GROQ_API_KEY
    # Query rewriting IS an LLM call, so unlike every other arm it cannot run with the
    # key blanked. Everything else stays offline and deterministic.
    if not judge and not rewrite:
        config.GROQ_API_KEY = ""

    for item in dataset["questions"]:
        retrieved = rag.retrieve(
            item["question"], user_id, mode=mode, alpha=alpha, rerank=rerank, rewrite=rewrite
        )
        ranked = retrieved["ranked"][:k]
        texts = [c["text"] for c in ranked]
        expected_meeting = id_map[item["meeting_id"]]

        # The full pipeline, not just retrieval: a question whose evidence was found
        # but which the refusal gate then rejected is still a product failure.
        answered = rag.answer(item["question"], user_id, rerank=rerank)

        card.results.append(
            metrics.QuestionResult(
                question_id=item["id"],
                question=item["question"],
                hit=metrics.hit_rate_at_k(texts, item["expect"], k),
                recall=metrics.recall_at_k(texts, item["expect"], k),
                precision=metrics.precision_at_k(texts, item["expect"], k),
                reciprocal_rank=metrics.reciprocal_rank(texts, item["expect"], k),
                correct_meeting=bool(ranked) and ranked[0]["meeting_id"] == expected_meeting,
                mode=answered["mode"],
                paraphrase=bool(item.get("paraphrase")),
            )
        )

        if judge and answered.get("answer"):
            excerpts = "\n\n".join(texts)
            verdict = _judge_faithfulness(item["question"], answered["answer"], excerpts)
            if verdict is not None:
                card.judged += 1
                card.faithfulness = (card.faithfulness or 0.0) + (1.0 if verdict else 0.0)

    for item in dataset["off_topic"]:
        card.off_topic_total += 1
        if rag.answer(item["question"], user_id, rerank=rerank)["mode"] == "refused":
            card.off_topic_refused += 1

    if card.judged:
        card.faithfulness = (card.faithfulness or 0.0) / card.judged

    config.GROQ_API_KEY = real_key
    return card


def _print_table(cards, chunk_count: int, verbose: bool) -> None:
    """One row per (mode, k).

    A sweep rather than a single k, because with a small corpus a generous k retrieves
    everything and every configuration scores a meaningless 1.000. recall@1 is where a
    ranking is actually judged; the larger k values show how fast it recovers.
    """
    print(f"\n  corpus: {chunk_count} chunks")
    if chunk_count <= max(c.k for c in cards):
        print(
            f"  NOTE: the largest k ({max(c.k for c in cards)}) is >= the corpus size, so "
            "its recall is trivially 1.000."
        )
    print()
    header = (
        f"  {'mode':<9} {'k':>2}  {'hit':>6} {'recall':>7} {'prec':>6} {'MRR':>6} "
        f"{'meet@1':>7} {'literal':>8} {'para':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    last_mode = None
    for c in cards:
        if last_mode is not None and c.label != last_mode:
            print()
        last_mode = c.label
        print(
            f"  {c.label:<9} {c.k:>2}  {c.hit_rate:>6.3f} {c.recall:>7.3f} "
            f"{c.precision:>6.3f} {c.mrr:>6.3f} {c.meeting_accuracy:>7.3f} "
            f"{c.recall_literal:>8.3f} {c.recall_paraphrase:>6.3f}"
        )

    print("\n  Grounding (independent of k -- the refusal gate runs before ranking):")
    seen = set()
    for c in cards:
        if c.label in seen:
            continue
        seen.add(c.label)
        print(
            f"    {c.label:<9} refusal accuracy {c.refusal_accuracy:.3f} "
            f"({c.off_topic_refused}/{c.off_topic_total})   "
            f"false refusals {c.false_refusal_rate:.3f}"
        )
        if c.faithfulness is not None:
            print(f"    {'':<9} faithfulness (LLM judge) {c.faithfulness:.3f} (n={c.judged})")

    misses = [c for c in cards if c.failures()]
    if misses and verbose:
        print("\n  Misses:")
        for c in misses:
            for f in c.failures():
                print(f"    {c.label}@{c.k}: {f.question_id} - {f.question}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate MeetSaransh retrieval quality.")
    parser.add_argument(
        "--mode",
        choices=["hybrid", "dense", "lexical", "all"],
        default="all",
        help="Retrieval configuration to score. 'all' runs the ablation.",
    )
    parser.add_argument(
        "-k",
        type=int,
        nargs="*",
        default=None,
        help="Retrieval depths to score (default: 1 3 and RAG_TOP_K).",
    )
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Also score each mode with LLM query rewriting (needs an API key).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Also score each mode with cross-encoder reranking, for an A/B.",
    )
    parser.add_argument(
        "--alpha-sweep",
        action="store_true",
        help="Sweep the hybrid dense/lexical weight instead of comparing modes.",
    )
    parser.add_argument("--judge", action="store_true", help="LLM-as-judge faithfulness.")
    parser.add_argument("--verbose", action="store_true", help="List every miss.")
    parser.add_argument("--min-recall", type=float, help="Fail below this recall@1.")
    parser.add_argument("--min-refusal", type=float, help="Fail below this refusal accuracy.")
    args = parser.parse_args(argv)

    # Point the app at a throwaway database, so evaluating never touches real data.
    # Assigning the config attributes directly rather than setting DATA_DIR in the
    # environment: config reads env vars once at import, so an env var set here would
    # do nothing if config were already imported, and would outlive the temp directory
    # if it were not.
    with tempfile.TemporaryDirectory(prefix="meetsaransh-eval-") as tmp:
        from app import config, embeddings, storage

        config.DATA_DIR = Path(tmp)
        config.AUDIO_DIR = Path(tmp) / "audio"
        config.DB_PATH = Path(tmp) / "eval.db"
        storage.reset_migration_cache()
        storage.init_db()

        ks = sorted(set(args.k or [1, 3, config.RAG_TOP_K]))
        modes: list[RetrievalMode] = (
            ["hybrid", "dense", "lexical"]
            if args.mode == "all"
            else [cast("RetrievalMode", args.mode)]
        )
        if not embeddings.available():
            skipped = [m for m in modes if m != "lexical"]
            if skipped:
                print(
                    f"\n  NOTE: the embedding model is unavailable, so {', '.join(skipped)} "
                    "would score identically to lexical. Running lexical only.",
                    file=sys.stderr,
                )
            modes = ["lexical"]

        print("\n" + "=" * 76)
        print("  MeetSaransh retrieval evaluation")
        print("=" * 76)

        # Built once and shared by every mode: the ablation is only meaningful if the
        # corpus and the index are identical and the retrieval strategy is the single
        # thing that varies.
        user_id, id_map = _build_corpus(_load_dataset())
        chunk_count = storage.count_chunks(user_id)

        cards = []
        if args.alpha_sweep:
            for alpha in (0.0, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0):
                for k in ks:
                    cards.append(
                        evaluate("hybrid", k, user_id, id_map, judge=args.judge, alpha=alpha)
                    )
        else:
            # With --rerank each mode is scored twice, off and on, so the two rows sit
            # next to each other and the comparison is like-for-like.
            rerank_arms = [False, True] if args.rerank else [False]
            rewrite_arms = [False, True] if args.rewrite else [False]
            for mode in modes:
                for rr in rerank_arms:
                    for qr in rewrite_arms:
                        for k in ks:
                            cards.append(
                                evaluate(
                                    mode,
                                    k,
                                    user_id,
                                    id_map,
                                    judge=args.judge,
                                    rerank=rr,
                                    rewrite=qr,
                                )
                            )

        _print_table(cards, chunk_count, args.verbose)

        if len(modes) > 1 or args.alpha_sweep:
            smallest_k = min(ks)
            at_k = [c for c in cards if c.k == smallest_k]
            best = max(at_k, key=lambda c: c.recall)
            best_para = max(at_k, key=lambda c: c.recall_paraphrase)
            print(
                f"\n  Best recall@{smallest_k}: {best.label} ({best.recall:.3f})   "
                f"Best on paraphrased: {best_para.label} ({best_para.recall_paraphrase:.3f})"
            )

        print("\n" + "=" * 76)

        # Gate on the configuration the app actually ships with, at the strictest k --
        # a gate at a k that returns the whole corpus would never fail.
        gate_k = min(ks)
        gated = next(
            (c for c in cards if c.label == "hybrid" and c.k == gate_k),
            next(c for c in cards if c.k == gate_k),
        )
        failures = []
        if args.min_recall is not None and gated.recall < args.min_recall:
            failures.append(f"recall@{gate_k} {gated.recall:.3f} < {args.min_recall}")
        if args.min_refusal is not None and gated.refusal_accuracy < args.min_refusal:
            failures.append(f"refusal accuracy {gated.refusal_accuracy:.3f} < {args.min_refusal}")
        if failures:
            print(f"  FAILED ({gated.label}@{gate_k}): " + "; ".join(failures) + "\n")
            return 1
        if args.min_recall is not None or args.min_refusal is not None:
            print(f"  PASSED ({gated.label}@{gate_k})\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
