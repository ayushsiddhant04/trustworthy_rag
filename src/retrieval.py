"""TF-IDF retrieval with optional all-MiniLM-L6-v2 hybrid scoring."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .cache import MemoryCache, stable_hash
from .config import SETTINGS

logger = logging.getLogger(__name__)
_RETRIEVAL_CACHE = MemoryCache(max_items=SETTINGS.cache_max_items, ttl_seconds=SETTINGS.cache_ttl_seconds)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    low, high = float(np.min(values)), float(np.max(values))
    if high - low < 1e-12:
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _embedding_scores(query: str, texts: list[str], model_name: str) -> np.ndarray:
    """Encode the corpus once and reuse corpus/query vectors across claims."""
    model = _load_embedding_model(model_name)
    corpus_key = stable_hash("corpus-vectors", model_name, *texts)
    query_key = stable_hash("query-vector", model_name, query)
    corpus_vectors = _RETRIEVAL_CACHE.get(corpus_key)
    if corpus_vectors is None:
        corpus_vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        corpus_vectors = np.asarray(corpus_vectors, dtype=np.float32)
        _RETRIEVAL_CACHE.set(corpus_key, corpus_vectors)
    else:
        corpus_vectors = np.asarray(corpus_vectors, dtype=np.float32)

    query_vector = _RETRIEVAL_CACHE.get(query_key)
    if query_vector is None:
        query_vector = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        query_vector = np.asarray(query_vector[0], dtype=np.float32)
        _RETRIEVAL_CACHE.set(query_key, query_vector)
    else:
        query_vector = np.asarray(query_vector, dtype=np.float32)
    return np.asarray(corpus_vectors @ query_vector, dtype=float)


def retrieve_with_meta(query: str, chunks: Iterable[dict], top_k: int = None,
                       use_embeddings: bool | None = None) -> list[dict]:
    """Return top passages with combined score and original provenance.

    Embeddings are opt-in (ENABLE_EMBEDDINGS=true). If unavailable, the function
    logs a warning and transparently falls back to TF-IDF rather than failing.
    """
    query = (query or "").strip()
    rows = [dict(c) for c in (chunks or []) if (c.get("text") or "").strip()]
    if not query or not rows:
        return []
    top_k = max(1, int(top_k or SETTINGS.retrieval_default_k))
    texts = [row["text"] for row in rows]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    try:
        matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        tfidf_raw = cosine_similarity(query_vector, matrix).ravel()
    except ValueError:  # e.g. text contains only stop words or punctuation
        tfidf_raw = np.zeros(len(rows), dtype=float)
    tfidf_norm = _minmax(tfidf_raw)

    embeddings_enabled = SETTINGS.enable_embeddings if use_embeddings is None else bool(use_embeddings)
    embedding_raw = np.zeros(len(rows), dtype=float)
    embedding_norm = np.zeros(len(rows), dtype=float)
    embeddings_used = False
    if embeddings_enabled:
        try:
            embedding_raw = _embedding_scores(query, texts, SETTINGS.embedding_model)
            embeddings_used = True
        except Exception as exc:
            logger.warning("Embedding retrieval unavailable; using TF-IDF only: %s", exc)
        if embeddings_used:
            # Cosine scores can be negative; shift to a non-negative range before min-max.
            embedding_norm = _minmax(np.clip(embedding_raw, 0.0, 1.0))

    if embeddings_used:
        total_weight = SETTINGS.tfidf_weight + SETTINGS.embedding_weight
        combined = (
            SETTINGS.tfidf_weight * tfidf_norm + SETTINGS.embedding_weight * embedding_norm
        ) / total_weight
        retrieval_method = "hybrid_tfidf_embeddings"
    else:
        combined = tfidf_norm
        retrieval_method = "tfidf"

    order = np.argsort(-combined, kind="stable")[:top_k]
    results = []
    for rank, idx in enumerate(order, start=1):
        item = dict(rows[int(idx)])
        item.update({
            "score": float(combined[idx]),
            "tfidf_score": float(tfidf_raw[idx]),
            "tfidf_normalized": float(tfidf_norm[idx]),
            "embedding_score": float(embedding_raw[idx]) if embeddings_used else None,
            "embedding_normalized": float(embedding_norm[idx]) if embeddings_used else None,
            "retrieval_method": retrieval_method,
            "rank": rank,
        })
        results.append(item)
    return results


def retrieve(query: str, chunks: Iterable[dict], top_k: int = None) -> list[str]:
    """Convenience function returning passage text only."""
    return [item["text"] for item in retrieve_with_meta(query, chunks, top_k=top_k)]


def clear_retrieval_cache() -> None:
    _RETRIEVAL_CACHE.clear()
