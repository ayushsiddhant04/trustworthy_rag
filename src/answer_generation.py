"""Grounded answer generation with offline extractive fallback and paste modes."""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache

from .cache import MemoryCache, stable_hash
from .config import SETTINGS

logger = logging.getLogger(__name__)
_LLM_CACHE = MemoryCache(max_items=SETTINGS.cache_max_items, ttl_seconds=SETTINGS.cache_ttl_seconds)
NO_INFO_FALLBACK = "The retrieved document passages do not contain enough information to answer this question."
_STOPWORDS = {
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "to", "of", "in", "on", "for", "with", "what", "when", "where", "who", "how", "does", "do", "did", "required", "must", "should", "can", "from", "by", "at", "as", "it", "this", "that"
}


def _terms(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9]+", text or "") if len(w) > 1 and w.lower() not in _STOPWORDS}


def extractive_answer_with_meta(question: str, passages: list[dict | str]) -> dict:
    """Select relevant source sentences; returns answer text and evidence metadata."""
    query_terms = _terms(question)
    candidates = []
    for passage_index, passage in enumerate(passages or []):
        text = passage.get("text", "") if isinstance(passage, dict) else str(passage)
        page = passage.get("page") if isinstance(passage, dict) else None
        source = passage.get("source") if isinstance(passage, dict) else None
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = sentence.strip(" \t\n•-*\u2022")
            if len(sentence) < 12:
                continue
            terms = _terms(sentence)
            overlap = len(query_terms & terms)
            score = overlap / max(1, len(query_terms))
            if score > 0:
                candidates.append((score, -passage_index, sentence, page, source))
    if not candidates:
        return {
            "answer": NO_INFO_FALLBACK,
            "has_evidence": False,
            "status": "no_evidence",
        }
    candidates.sort(reverse=True)
    selected = []
    seen = set()
    for score, _, sentence, page, source in candidates:
        key = sentence.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) == 3:
            break
    return {
        "answer": " ".join(selected),
        "has_evidence": True,
        "status": "ok",
    }


def extractive_answer(question: str, passages: list[dict | str]) -> str:
    """Select relevant source sentences; never invent facts when offline."""
    return extractive_answer_with_meta(question, passages)["answer"]


@lru_cache(maxsize=1)
def _anthropic_client():
    from anthropic import Anthropic
    return Anthropic(
        api_key=SETTINGS.anthropic_api_key,
        timeout=SETTINGS.llm_timeout_seconds,
        max_retries=SETTINGS.llm_max_retries,
    )


def _call_anthropic(system: str, prompt: str, cache_key: str) -> str:
    cached = _LLM_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if not SETTINGS.enable_external_llm or not SETTINGS.anthropic_api_key:
        raise RuntimeError("External LLM is disabled or ANTHROPIC_API_KEY is not configured")
    message = _anthropic_client().messages.create(
        model=SETTINGS.llm_model,
        max_tokens=900,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(block.text for block in message.content if getattr(block, "type", None) == "text").strip()
    if text:
        _LLM_CACHE.set(cache_key, text)
    return text


def generate_answer_with_meta(question: str, passages: list[dict | str], answer: str | None = None,
                               mode: str = "generate") -> dict:
    """Generate or accept answer, returning text alongside factual evidence status."""
    normalized_mode = (mode or "generate").strip().lower()
    if normalized_mode in {"manual", "paste", "paste_answer"}:
        if not answer or not answer.strip():
            raise ValueError("Paste-answer mode requires a non-empty answer.")
        return {"answer": answer.strip(), "has_evidence": True, "status": "manual"}
    if normalized_mode in {"demo_bad", "demo-bad-answer", "demo_bad_answer"}:
        if not answer or not answer.strip():
            raise ValueError("Known-bad demo mode requires you to paste the deliberately incorrect answer.")
        return {"answer": answer.strip(), "has_evidence": True, "status": "demo_bad"}
    if normalized_mode not in {"generate", "auto"}:
        raise ValueError(f"Unsupported answer mode: {mode!r}")

    source_passages = []
    for p in passages or []:
        source_passages.append(p if isinstance(p, str) else {
            "text": p.get("text", ""), "page": p.get("page"), "source": p.get("source")
        })
    if SETTINGS.enable_external_llm and SETTINGS.anthropic_api_key and source_passages:
        context = "\n\n".join(
            f"[Source: {p.get('source') or 'document'}, page: {p.get('page') or 'not available'}]\n{p['text']}"
            if isinstance(p, dict) else str(p)
            for p in source_passages
        )
        prompt = (
            f"Question: {question}\n\nRetrieved document evidence:\n{context}\n\n"
            "Answer using only the supplied evidence. Do not use prior knowledge. "
            "Do not infer missing requirements or dates. If evidence is insufficient, say so. "
            "Keep the answer concise and preserve important qualifications."
        )
        key = stable_hash("answer", SETTINGS.llm_model, prompt)
        try:
            generated = _call_anthropic(
                "You answer questions strictly from supplied document excerpts. Never invent facts.",
                prompt,
                key,
            )
            if generated:
                return {"answer": generated, "has_evidence": True, "status": "llm_generated"}
        except Exception as exc:
            logger.warning("External answer generation failed; using offline extractive fallback: %s", exc)

    return extractive_answer_with_meta(question, source_passages)


def generate_answer(question: str, passages: list[dict | str], answer: str | None = None,
                    mode: str = "generate") -> str:
    """Generate, accept a pasted answer, or accept an explicit known-bad demo answer."""
    return generate_answer_with_meta(question, passages, answer=answer, mode=mode)["answer"]
