"""End-to-end document-to-claim pipeline (Member A replacement implementation)."""
from __future__ import annotations

import logging
from typing import Any

from .cache import MemoryCache, stable_hash
from .chunking import chunk_document
from .claim_extraction import extract_claims
from .config import SETTINGS
from .ingestion import load_document
from .retrieval import retrieve_with_meta
from .answer_generation import generate_answer, generate_answer_with_meta

logger = logging.getLogger(__name__)
_PIPELINE_CACHE = MemoryCache(max_items=SETTINGS.cache_max_items, ttl_seconds=SETTINGS.cache_ttl_seconds)


def _document_info(doc: Any) -> dict:
    if isinstance(doc, dict) and isinstance(doc.get("sections"), list):
        return doc
    return load_document(doc)


def _passage_ref(passage: dict) -> dict:
    return {
        "source": passage.get("source"),
        "page": passage.get("page"),
        "paragraph": passage.get("paragraph"),
        "chunk_id": passage.get("chunk_id"),
        "score": float(passage.get("score", 0.0)),
    }


def run(question: str, doc: Any, answer: str | None = None, mode: str = "generate", top_k: int = None) -> dict:
    """Run ingestion → chunking → retrieval → answer → claim extraction.

    `doc` may be a filesystem path, uploaded file-like object, bytes, or an
    already-ingested document dictionary. The result is deliberately
    verification-engine agnostic and includes a normalized per-claim contract.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    doc_info = _document_info(doc)
    sections = doc_info.get("sections") or []
    if not sections:
        raise ValueError("The supplied document contains no usable text sections")

    cache_key = stable_hash(
        "chunks-v1", SETTINGS.chunk_size_words, SETTINGS.chunk_overlap_words,
        doc_info.get("source", "unknown"),
        *(section.get("text", "") for section in sections),
    )
    chunks = _PIPELINE_CACHE.get_or_compute(
        cache_key,
        lambda: chunk_document(
            sections,
            chunk_size=SETTINGS.chunk_size_words,
            overlap=SETTINGS.chunk_overlap_words,
        ),
    )
    if not chunks:
        raise ValueError("Document text could not be divided into retrieval chunks")

    question_passages = retrieve_with_meta(question, chunks, top_k=top_k)
    answer_meta = generate_answer_with_meta(
        question=question,
        passages=question_passages,
        answer=answer,
        mode=mode,
    )
    raw_answer = answer_meta["answer"]
    has_evidence = answer_meta.get("has_evidence", True)
    answer_status = answer_meta.get("status", "ok")

    if not has_evidence or answer_status == "no_evidence":
        claim_texts = []
    else:
        claim_texts = extract_claims(
            raw_answer,
            question=question,
            passages=question_passages,
        )

    claims = []
    verification_inputs = []
    for index, claim_text in enumerate(claim_texts, start=1):
        claim_id = f"C{index}"
        claim_passages = retrieve_with_meta(claim_text, chunks, top_k=top_k)
        # If the claim itself is too short or retrieval has no lexical signal,
        # preserve the question-level evidence as a conservative fallback.
        if not claim_passages or max((p.get("score", 0.0) for p in claim_passages), default=0.0) <= 0:
            claim_passages = question_passages
        evidence = [
            {
                "chunk_id": passage.get("chunk_id"),
                "text": passage.get("text", ""),
                "page": passage.get("page"),
                "source": passage.get("source", doc_info.get("source")),
                "score": float(passage.get("score", 0.0)),
                "paragraph": passage.get("paragraph"),
            }
            for passage in claim_passages
        ]
        claim_object = {"claim_id": claim_id, "claim": claim_text, "text": claim_text}
        claims.append(claim_object)
        verification_inputs.append({
            "claim_id": claim_id,
            "claim": claim_text,
            "text": claim_text,
            # `evidence` is the canonical Member B input. `passages` and
            # `passage_refs` remain for compatibility with A's documented API.
            "evidence": evidence,
            "passages": [item["text"] for item in evidence],
            "passage_refs": [_passage_ref(item) for item in claim_passages],
        })

    warnings = list(doc_info.get("warnings") or [])
    if not has_evidence or answer_status == "no_evidence":
        warnings.append("Document evidence did not contain information addressing this question. Offline generation produced an explanatory abstention message; no verification claims were submitted.")
    if mode in {"demo_bad", "demo-bad-answer", "demo_bad_answer"}:
        warnings.append("Known-bad answer mode was selected. The supplied answer is deliberately untrusted demo input.")
    if not SETTINGS.enable_embeddings:
        warnings.append("Dense embeddings are disabled; retrieval is using TF-IDF only. Set ENABLE_EMBEDDINGS=true to enable hybrid retrieval.")
    if not SETTINGS.enable_external_llm:
        warnings.append("External LLM calls are disabled; generation and claim extraction use offline fallbacks unless manually supplied.")

    return {
        "question": question.strip(),
        "source": doc_info.get("source", "unknown"),
        "file_type": doc_info.get("file_type"),
        "mode": mode,
        "answer": raw_answer,
        "raw_answer": raw_answer,
        "answer_status": answer_status,
        "has_evidence": has_evidence,
        "claims": claims,
        "retrieved_passages": question_passages,
        "verification_inputs": verification_inputs,
        "verification_results": [],
        "warnings": warnings,
        "metadata": {
            "chunk_count": len(chunks),
            "claim_count": len(claims),
            "top_k": int(top_k or SETTINGS.retrieval_default_k),
            "retrieval_methods": sorted({p.get("retrieval_method", "tfidf") for p in question_passages}),
            "external_llm_enabled": SETTINGS.enable_external_llm,
            "embeddings_enabled": SETTINGS.enable_embeddings,
        },
    }


def clear_pipeline_cache() -> None:
    _PIPELINE_CACHE.clear()
