"""RAG engine for "Ask your meetings": chunk -> index -> hybrid retrieve -> grounded answer.

Retrieval is HYBRID: a dense semantic score (fastembed cosine) combined with a lexical
BM25 score (pure Python). Research on spoken/transcript content shows hybrid beats
pure-vector search, and the lexical half means the feature still works if the embedding
model can't load. Answering is grounded: excerpts are the only allowed source, and a
low best-score triggers an honest refusal instead of a hallucinated answer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from typing import Literal

from . import config, embeddings, llm, prompts, reranker, storage
from .observability import get_logger

log = get_logger("meetsaransh.rag")


# --------------------------------------------------------------------- timestamps
def fmt_ts(seconds: float | None) -> str:
    s = int(seconds or 0)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


# --------------------------------------------------------------------- chunking
def chunk_segments(segments: list[dict]) -> list[dict]:
    """Group consecutive transcript segments into overlapping, word-budgeted chunks.

    Each chunk preserves its start/end timestamps so answers stay deep-linkable.
    """
    target = config.CHUNK_TARGET_WORDS
    overlap = config.CHUNK_OVERLAP_WORDS
    n = len(segments)
    chunks: list[dict] = []
    i = 0
    ordinal = 0
    while i < n:
        words = 0
        j = i
        while j < n and words < target:
            words += len(segments[j]["text"].split())
            j += 1
        seg_slice = segments[i:j]
        text = " ".join(s["text"].strip() for s in seg_slice).strip()
        if text:
            chunks.append(
                {
                    "ord": ordinal,
                    "start": float(seg_slice[0].get("start", 0.0)),
                    "end": float(seg_slice[-1].get("end", 0.0)),
                    "text": text,
                    # Keep the constituent segments so citations can point at the exact
                    # matching line (precise timestamp + relevant snippet), not just the
                    # chunk's first line.
                    "segs": [
                        {"start": float(s.get("start", 0.0)), "text": s["text"].strip()}
                        for s in seg_slice
                    ],
                }
            )
            ordinal += 1
        if j >= n:
            break
        # Step back a few segments to create ~`overlap` words of context overlap.
        back_words = 0
        k = j
        while k > i + 1 and back_words < overlap:
            k -= 1
            back_words += len(segments[k]["text"].split())
        i = k if k > i else j  # guarantee forward progress
    return chunks


def _context_header(meeting_title: str, text: str) -> str:
    """Contextual-retrieval prefix: cheap, and lifts recall on conversational data."""
    return f"In a meeting titled '{meeting_title}': {text}"


def index_meeting(meeting_id: str) -> int:
    """Chunk a stored meeting, embed the chunks, and persist them. Returns chunk count.

    Reads unscoped because indexing runs on behalf of the system (the background worker
    and the reindex endpoint), not on behalf of a request. Ownership is enforced where
    the chunks are read back, in `storage.get_chunks`.
    """
    meeting = storage.get_meeting_unscoped(meeting_id)
    if meeting is None:
        return 0
    chunks = chunk_segments(meeting.get("segments") or [])
    if not chunks:
        storage.replace_chunks(meeting_id, [])
        return 0

    # Embed with a context header so retrieval knows which meeting a line came from.
    texts = [_context_header(meeting["title"], c["text"]) for c in chunks]
    matrix = embeddings.embed_texts(texts)  # None if the model is unavailable
    for idx, c in enumerate(chunks):
        c["embedding"] = None if matrix is None else matrix[idx]
    storage.replace_chunks(meeting_id, chunks)
    return len(chunks)


def reindex_all(user_id: str) -> dict:
    """Ensure every one of this user's meetings is indexed. Returns a status summary.

    Only meetings that finished processing are indexed: a queued or failed meeting has
    no transcript to chunk, and retrying it on every chat request would be wasted work.
    """
    meetings = [m for m in storage.list_meetings(user_id) if m.get("status") == "done"]
    ids = [m["id"] for m in meetings]
    already = storage.indexed_meeting_ids(user_id)
    newly = 0
    for mid in ids:
        if mid not in already:
            index_meeting(mid)
            newly += 1
    return {
        "meetings": len(ids),
        "newly_indexed": newly,
        "total_chunks": storage.count_chunks(user_id),
    }


# --------------------------------------------------------------------- BM25 (lexical)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function/question words that shouldn't count as "this topic was discussed" evidence.
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "about",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "we",
    "our",
    "us",
    "i",
    "my",
    "me",
    "you",
    "your",
    "they",
    "them",
    "it",
    "its",
    "he",
    "she",
    "him",
    "her",
    "this",
    "that",
    "these",
    "those",
    "what",
    "when",
    "who",
    "whom",
    "why",
    "how",
    "which",
    "where",
    "any",
    "some",
    "there",
    "here",
    "get",
    "got",
    "have",
    "has",
    "had",
    "will",
    "would",
    "can",
    "could",
    "should",
    "tell",
    "say",
    "said",
    "please",
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _content_terms(text: str) -> set[str]:
    """Meaningful query terms: drop stopwords and single characters."""
    return {t for t in _tokenize(text) if len(t) > 1 and t not in _STOPWORDS}


def _bm25_scores(
    query: str, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Okapi BM25 scores of a query against a small in-memory corpus."""
    n = len(docs_tokens)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs_tokens) / n
    # document frequency per term
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for term in set(toks):
            df[term] = df.get(term, 0) + 1
    q_terms = [t for t in set(_tokenize(query)) if t in df]
    scores = [0.0] * n
    for term in q_terms:
        idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
        for i, toks in enumerate(docs_tokens):
            tf = toks.count(term)
            if tf:
                dl = len(toks)
                scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    return scores


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    hi = max(values)
    return [v / hi for v in values] if hi > 0 else [0.0 for _ in values]


# --------------------------------------------------------------------- query rewriting
def rewrite_query(question: str) -> str:
    """Rewrite a question into retrieval vocabulary, or return it unchanged.

    Never raises and never returns nothing: every failure path -- no API key, provider
    error, an empty or suspiciously long reply -- falls back to the original question.
    A rewriter that can break search is worse than no rewriter.
    """
    if not config.has_api_key():
        return question
    try:
        rewritten = llm.chat(prompts.build_rewrite_messages(question), temperature=0.0)
    except llm.LLMError:
        log.warning("query rewrite failed; searching with the original question")
        return question

    rewritten = " ".join(rewritten.strip().strip('"').split())
    if not rewritten or len(rewritten) > config.QUERY_REWRITE_MAX_CHARS:
        # A long reply means the model started explaining or answering instead of
        # rewriting. Searching with that would drag its invented words into retrieval.
        return question
    return rewritten


# --------------------------------------------------------------------- retrieval
# How the dense and lexical scores are weighted. `hybrid` is what the app uses; the
# other two exist so the evaluation harness can ablate one component at a time and show
# that hybrid actually earns its complexity rather than being assumed to.
RetrievalMode = Literal["hybrid", "dense", "lexical"]

# Weight on the dense score when both components are available.
#
# Raised from 0.65 to 0.8 on evidence, not taste. `python -m evaluation.run
# --alpha-sweep` over the 26-question labelled set showed 0.65 scoring 0.875 recall@3
# on paraphrased questions -- identical to lexical-only -- because BM25's confident
# wrong matches were outvoting the dense signal. At 0.8 that goes to 1.000, and MRR@3
# improves from 0.904 to 0.923.
#
# The sweep keeps improving all the way to 1.0 (pure dense), and that is deliberately
# NOT what ships. Two reasons: 26 questions over 5 chunks is far too small a sample to
# justify deleting a retrieval component, and the lexical half does a job the metric
# cannot see -- it is the only thing that still works when the embedding model fails to
# load, and it matches exact tokens (names, ticket ids, error codes) that embeddings
# blur together. 0.8 takes the measured win without betting the feature on it.
HYBRID_ALPHA = 0.8


def retrieve(
    question: str,
    user_id: str,
    scope_meeting_id: str | None = None,
    *,
    mode: RetrievalMode = "hybrid",
    alpha: float | None = None,
    rerank: bool | None = None,
    rewrite: bool | None = None,
) -> dict:
    """Hybrid retrieval over one user's indexed chunks.

    Returns {"ranked": [chunk+score...], "dense_best": float, "dense_used": bool}.
    `mode`, `alpha`, `rerank` and `rewrite` are evaluation levers: production uses the
    defaults.
    """
    chunks = storage.get_chunks(user_id, scope_meeting_id)
    if not chunks:
        return {"ranked": [], "dense_best": 0.0, "dense_used": False, "empty": True}

    use_rewrite = config.QUERY_REWRITE_ENABLED if rewrite is None else rewrite
    search_text = rewrite_query(question) if use_rewrite else question
    rewritten_used = search_text != question

    # Lexical component (always available).
    docs_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = _normalize(_bm25_scores(search_text, docs_tokens))

    # Dense component (when embeddings loaded AND chunks carry vectors).
    dense_raw = [0.0] * len(chunks)
    dense_used = False
    have_vectors = any(c["embedding"] is not None for c in chunks)
    if mode != "lexical" and embeddings.available() and have_vectors:
        qvec = embeddings.embed_one(search_text)
        if qvec is not None:
            import numpy as np

            mat = np.vstack(
                [
                    c["embedding"]
                    if c["embedding"] is not None
                    else np.zeros(embeddings.DIM, dtype="float32")
                    for c in chunks
                ]
            )
            dense_raw = [float(x) for x in embeddings.cosine_scores(qvec, mat)]
            dense_used = True

    dense_norm = _normalize([max(0.0, d) for d in dense_raw])

    # Combine. Favour dense when available; fall back to lexical-only otherwise, which
    # is also what happens whenever the embedding model cannot load.
    if not dense_used:
        weight = 0.0
    elif mode == "dense":
        weight = 1.0
    else:
        weight = HYBRID_ALPHA if alpha is None else alpha
    combined = [weight * dense_norm[i] + (1 - weight) * bm25[i] for i in range(len(chunks))]

    for i, c in enumerate(chunks):
        c["score"] = combined[i]
        c["dense"] = dense_raw[i]
        c["lexical"] = bm25[i]

    ranked = sorted(chunks, key=lambda c: c["score"], reverse=True)
    dense_best = max(dense_raw) if dense_raw else 0.0

    # Second stage: hand the top candidates to the cross-encoder, which reads the
    # question and each chunk together rather than comparing precomputed vectors.
    # `dense_best` is captured above, before this reorders anything, because the
    # refusal gate is a judgement about the corpus and must not shift depending on
    # whether a reranker happened to load.
    use_rerank = config.RERANK_ENABLED if rerank is None else rerank
    reranked_used = False
    if use_rerank and len(ranked) > 1:
        candidates = ranked[: config.RERANK_CANDIDATES]
        reordered = reranker.rerank(search_text, candidates, config.RERANK_CANDIDATES)
        if reordered is not None:
            # Anything past the candidate window keeps its retrieval order behind the
            # reranked head, so nothing is silently dropped from the result.
            ranked = reordered + ranked[config.RERANK_CANDIDATES :]
            reranked_used = True
    # Does any meaningful query term literally appear in the corpus? This lets valid
    # keyword questions ("Rahul's action items") through even when the exact phrasing
    # isn't spoken, while clearly off-topic questions ("capital of France") stay out.
    #
    # Computed from the ORIGINAL question, never the rewrite. A rewriter adds synonyms,
    # and synonyms of an off-topic question can easily collide with the corpus -- so
    # rewriting for the gate would let "what is the capital of France" find a keyword
    # and stop being refused. The rewrite is allowed to improve ranking; it is not
    # allowed to change whether we have an answer at all.
    vocab: set[str] = set().union(*docs_tokens) if docs_tokens else set()
    content_match = bool(_content_terms(question) & vocab)
    return {
        "ranked": ranked,
        "dense_best": dense_best,
        "dense_used": dense_used,
        "reranked": reranked_used,
        "rewritten": rewritten_used,
        "search_text": search_text,
        "content_match": content_match,
        "empty": False,
    }


def _best_segment(chunk: dict, q_terms: set[str]) -> dict | None:
    """Within a chunk, find the segment with the most query-term overlap.

    Makes the citation point at the line that actually answers the question, with its
    own timestamp, rather than the chunk's opening line.
    """
    segs = chunk.get("segs") or []
    if not segs:
        return None
    best, best_score = None, -1
    for s in segs:
        overlap = len(q_terms & set(_tokenize(s["text"])))
        if overlap > best_score:
            best, best_score = s, overlap
    return best if best_score > 0 else segs[0]


def _citation(chunk: dict, question: str) -> dict:
    q_terms = set(_tokenize(question))
    seg = _best_segment(chunk, q_terms)
    start = float((seg or chunk).get("start") or 0.0)
    snippet = (seg["text"] if seg else chunk["text"]).strip()
    if len(snippet) > 240:
        snippet = snippet[:240].rsplit(" ", 1)[0] + "…"
    return {
        "meeting_id": chunk["meeting_id"],
        "meeting_title": chunk["meeting_title"],
        "timestamp": fmt_ts(start),
        "start": start,
        "snippet": snippet,
        "score": round(float(chunk.get("score", 0.0)), 3),
    }


def _format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[Meeting: {c['meeting_title']} | {fmt_ts(c.get('start'))}]\n{c['text']}" for c in chunks
    )


REFUSAL = "I couldn't find anything about that in your meetings."


def _passes_refusal_gate(result: dict) -> bool:
    """Refuse only when the question is neither semantically close (dense) NOR shares a
    meaningful keyword with any meeting. Either signal is enough to attempt an answer;
    the LLM prompt is the second grounding layer.

    Shared by the streaming and non-streaming paths on purpose -- two copies of this
    decision would eventually disagree, and the one users noticed would be whichever
    said "I could not find that" about a question the other answered.
    """
    if result["dense_used"]:
        return bool(result["dense_best"] >= config.RAG_MIN_SCORE or result["content_match"])
    return bool(result["content_match"])


def answer(
    question: str,
    user_id: str,
    scope_meeting_id: str | None = None,
    *,
    rerank: bool | None = None,
) -> dict:
    """Answer a question over one user's meetings, grounded in retrieved excerpts."""
    question = (question or "").strip()
    if not question:
        return {"answer": "Please enter a question.", "citations": [], "mode": "error"}

    result = retrieve(question, user_id, scope_meeting_id, rerank=rerank)
    if result.get("empty"):
        return {
            "answer": "No meetings have been indexed yet. Add a meeting first.",
            "citations": [],
            "mode": "empty",
        }

    ranked = result["ranked"]
    top = ranked[: config.RAG_TOP_K]

    # Grounded-refusal gate: refuse only when the question is neither semantically close
    # (dense) NOR shares a meaningful keyword with any meeting (lexical). Either signal
    # is enough to attempt an answer; the LLM prompt is the second grounding layer.
    if not _passes_refusal_gate(result):
        return {"answer": REFUSAL, "citations": [], "mode": "refused"}

    citations = [_citation(c, question) for c in top]

    # Without an LLM key we can still be useful: return the matching excerpts.
    if not config.has_api_key():
        return {
            "answer": None,
            "citations": citations,
            "mode": "retrieval_only",
            "note": "Add a GROQ_API_KEY to get a written answer. These are the most relevant excerpts.",
        }

    try:
        text = llm.chat(prompts.build_rag_messages(question, _format_context(top)), temperature=0.2)
    except llm.LLMError as exc:
        return {"answer": None, "citations": citations, "mode": "error", "note": str(exc)}

    return {"answer": text.strip(), "citations": citations, "mode": "answer"}


def answer_stream(
    question: str,
    user_id: str,
    scope_meeting_id: str | None = None,
) -> Iterator[dict]:
    """Answer a question, yielding events as the model produces them.

    Yields dicts the transport turns into SSE frames:

        {"type": "citations", "citations": [...]}   once, before any text
        {"type": "delta", "text": "..."}            many
        {"type": "done", "mode": "answer"}          once
        {"type": "error", "message": "..."}         instead of `done`, on failure

    Citations are sent FIRST, before a single token. They are known the moment
    retrieval finishes, and sending them up front means the sources are on screen while
    the answer is still being written -- so a reader can start checking the evidence
    rather than watching a spinner. It also means a mid-stream failure still leaves the
    user with the excerpts.
    """
    question = (question or "").strip()
    if not question:
        yield {"type": "error", "message": "Please enter a question.", "mode": "error"}
        return

    result = retrieve(question, user_id, scope_meeting_id)
    if result.get("empty"):
        yield {
            "type": "done",
            "mode": "empty",
            "answer": "No meetings have been indexed yet. Add a meeting first.",
        }
        return

    top = result["ranked"][: config.RAG_TOP_K]
    if not _passes_refusal_gate(result):
        yield {"type": "done", "mode": "refused", "answer": REFUSAL}
        return

    citations = [_citation(c, question) for c in top]
    yield {"type": "citations", "citations": citations}

    if not config.has_api_key():
        yield {
            "type": "done",
            "mode": "retrieval_only",
            "note": (
                "Add a GROQ_API_KEY to get a written answer. These are the most relevant excerpts."
            ),
        }
        return

    messages = prompts.build_rag_messages(question, _format_context(top))
    streamed = False
    try:
        for delta in llm.chat_stream(messages, temperature=0.2):
            streamed = True
            yield {"type": "delta", "text": delta}
    except llm.LLMError as exc:
        message = str(exc)
        # Falling back is only safe if nothing has been shown yet: retrying mid-answer
        # would splice two different attempts into one paragraph.
        if not streamed:
            try:
                text = llm.chat(messages, temperature=0.2)
                yield {"type": "delta", "text": text.strip()}
                yield {"type": "done", "mode": "answer"}
                return
            except llm.LLMError as retry_exc:
                message = str(retry_exc)
        yield {"type": "error", "message": message, "mode": "error"}
        return

    yield {"type": "done", "mode": "answer"}


def status(user_id: str) -> dict:
    return {
        "embeddings_available": embeddings.available(),
        "reranker_available": reranker.available(),
        "embed_model": config.EMBED_MODEL,
        "indexed_meetings": len(storage.indexed_meeting_ids(user_id)),
        "total_chunks": storage.count_chunks(user_id),
    }
