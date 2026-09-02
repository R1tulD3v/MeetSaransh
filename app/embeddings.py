"""Local dense embeddings via fastembed (bge-small-en-v1.5, ONNX runtime).

Chosen over sentence-transformers because it avoids the heavy `torch` dependency and
runs on CPU, and over a hosted embeddings API because it needs no extra API key and
works offline. The model (~90 MB) downloads once on first use and is cached.

The whole module degrades gracefully: if fastembed isn't installed or the model can't
load, `available()` returns False and retrieval falls back to lexical-only search.
"""

from __future__ import annotations

import threading

import numpy as np

from . import config

_model = None
_load_failed = False
_lock = threading.Lock()

DIM = 384  # bge-small-en-v1.5 output dimension


def _get_model():
    """Lazily load the embedding model exactly once (thread-safe)."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _lock:
        if _model is None and not _load_failed:
            try:
                from fastembed import TextEmbedding  # imported lazily; optional dependency

                _model = TextEmbedding(config.EMBED_MODEL)
            except Exception:
                _load_failed = True
                _model = None
    return _model


def available() -> bool:
    """True if dense embeddings can be produced (model loads successfully)."""
    return _get_model() is not None


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Embed a list of texts -> (n, DIM) float32 array, or None if unavailable."""
    model = _get_model()
    if model is None or not texts:
        return None
    vecs = list(model.embed(texts))
    return np.asarray(vecs, dtype=np.float32)


def embed_one(text: str) -> np.ndarray | None:
    """Embed a single text -> (DIM,) float32 array, or None if unavailable."""
    out = embed_texts([text])
    return None if out is None else out[0]


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a (n, DIM) matrix -> (n,)."""
    if matrix.size == 0:
        return np.array([], dtype=np.float32)
    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
    m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return m @ q
