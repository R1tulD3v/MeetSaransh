"""Cross-encoder reranking of retrieved chunks.

Retrieval and reranking answer different questions. The hybrid retriever scores a query
against a *precomputed* chunk vector, so it never sees the two texts together; a cross
-encoder reads the question and the chunk as one input and scores their actual
relationship. That is far more accurate and far too slow to run over a whole corpus --
hence the standard shape used here: retrieve a wide candidate set cheaply, then rerank
only those candidates.

The model is `Xenova/ms-marco-MiniLM-L-6-v2` via fastembed's ONNX runtime, chosen for
the same reason as the embedding model: no torch, CPU-friendly, no extra API key. It
degrades exactly like the embedder -- if it cannot load, `available()` is False and
retrieval simply returns its own ranking.

Whether this actually improves results is not assumed: `python -m evaluation.run
--rerank` measures it. See the README for the numbers it produced.
"""

from __future__ import annotations

import threading

from . import config
from .observability import get_logger

log = get_logger("meetsaransh.rerank")

_model = None
_load_failed = False
_lock = threading.Lock()


def _get_model():
    """Lazily load the cross-encoder exactly once (thread-safe).

    Loaded on first use rather than at startup so a server that never answers a
    question never pays the download or the memory.
    """
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is None and not _load_failed:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _model = TextCrossEncoder(config.RERANK_MODEL)
                log.info("reranker loaded", extra={"model": config.RERANK_MODEL})
            except Exception:
                # Same contract as embeddings: unavailable is a degraded mode, not an
                # error. Retrieval keeps working with its own ordering.
                _load_failed = True
                _model = None
                log.warning("reranker unavailable; using retrieval order", exc_info=True)
    return _model


def available() -> bool:
    """True if the cross-encoder can score pairs."""
    if not config.RERANK_ENABLED:
        return False
    return _get_model() is not None


def rerank(question: str, chunks: list[dict], top_k: int) -> list[dict] | None:
    """Reorder `chunks` by cross-encoder relevance and return the best `top_k`.

    Returns None when reranking is unavailable, so callers can fall back to the
    retrieval order rather than having to check `available()` separately and race it.

    Each chunk gains a `rerank_score`; its original `score` is left untouched so the
    two rankings can be compared in the evaluation harness.
    """
    model = _get_model()
    if model is None or not chunks:
        return None

    documents = [c["text"] for c in chunks]
    try:
        scores = list(model.rerank(question, documents))
    except Exception:
        log.exception("reranking failed; falling back to retrieval order")
        return None

    if len(scores) != len(chunks):  # pragma: no cover - defensive
        log.warning("reranker returned %d scores for %d chunks", len(scores), len(chunks))
        return None

    for chunk, score in zip(chunks, scores, strict=True):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
