"""Atomic claim extraction with safe JSON parsing and deterministic fallback."""
from __future__ import annotations

import json
import logging
import re

from .answer_generation import _call_anthropic, NO_INFO_FALLBACK
from .cache import stable_hash
from .config import SETTINGS

logger = logging.getLogger(__name__)


def _parse_json_claims(raw: str) -> list[str]:
    """Parse LLM JSON output without evaluating or executing arbitrary content."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    candidate = raw.strip()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Recover the first JSON array/object, but never use eval/exec.
        start_array, end_array = candidate.find("["), candidate.rfind("]")
        start_obj, end_obj = candidate.find("{"), candidate.rfind("}")
        if start_array >= 0 and end_array > start_array:
            try:
                parsed = json.loads(candidate[start_array:end_array + 1])
            except json.JSONDecodeError:
                parsed = None
        elif start_obj >= 0 and end_obj > start_obj:
            try:
                parsed = json.loads(candidate[start_obj:end_obj + 1])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict):
        parsed = parsed.get("claims", parsed.get("items", []))
    if not isinstance(parsed, list):
        return []
    claims = []
    for item in parsed:
        value = item.get("claim") or item.get("text") if isinstance(item, dict) else item
        if isinstance(value, str):
            value = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", value).strip()
            if value and value not in claims:
                claims.append(value)
    return claims


VERB_PAT = re.compile(
    r"\b(?:is|are|was|were|will|shall|must|can|should|has|have|requires?|mandates?|costs?|takes?|falls?|submits?|submitted|specifies?|states?)\b",
    re.IGNORECASE,
)
CLAUSE_SPLIT = re.compile(r"(?:,?\s+and\s+|,?\s+while\s+|,?\s+but\s+)", re.IGNORECASE)


def split_compound_clause(sentence: str) -> list[str]:
    """Split a compound sentence into atomic factual claims if both sides contain verbs."""
    text = sentence.strip(" \t\n-•")
    if not text:
        return []
    matches = list(CLAUSE_SPLIT.finditer(text))
    if not matches:
        return [text]
    for m in matches:
        p1 = text[:m.start()].strip()
        p2 = text[m.end():].strip()
        # Both sides must have subject+verb structure and at least 3 words
        if VERB_PAT.search(p1) and VERB_PAT.search(p2):
            if len(p1.split()) >= 3 and len(p2.split()) >= 3:
                c1 = p1 if p1.endswith((".", "!", "?")) else p1 + "."
                # Ensure second clause starts capitalized
                c2_start = p2[0].upper() if p2 else ""
                c2_rest = p2[1:] if len(p2) > 1 else ""
                c2 = c2_start + c2_rest
                c2 = c2 if c2.endswith((".", "!", "?")) else c2 + "."
                # Recursively check if c2 contains further compound clauses
                return [c1] + split_compound_clause(c2)
    return [text]


def split_sentences(answer: str) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return []
    # Treat numbered/bulleted lists as separate candidate claims.
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if line:
            lines.append(line)
    source = " ".join(lines) if lines else text
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|(?<=;)\s+", source)
    claims = []
    for part in parts:
        part_clean = part.strip(" \t\n-•")
        if not part_clean:
            continue
        atomic_clauses = split_compound_clause(part_clean)
        for clause in atomic_clauses:
            claim = clause.strip(" \t\n-•")
            if len(claim) >= 4 and claim not in claims:
                claims.append(claim)
    return claims


def extract_claims(answer: str, question: str | None = None, passages: list[dict | str] | None = None,
                   use_llm: bool | None = None) -> list[str]:
    """Return a list of factual claims; sentence splitting is the safe fallback."""
    clean = (answer or "").strip()
    if not clean or clean == NO_INFO_FALLBACK:
        return []
    fallback = split_sentences(clean)
    enabled = SETTINGS.enable_external_llm if use_llm is None else bool(use_llm)
    if not enabled or not SETTINGS.anthropic_api_key or not answer.strip():
        return fallback
    context = "\n\n".join(
        p.get("text", "") if isinstance(p, dict) else str(p)
        for p in (passages or [])
    )
    prompt = (
        f"Question: {question or '(not provided)'}\n\nAnswer to decompose:\n{answer}\n\n"
        f"Document context for resolving references (do not add facts):\n{context[:10000]}\n\n"
        "Decompose the answer into the smallest meaningful factual claims. Each claim must express "
        "one proposition. Resolve pronouns using the supplied context where unambiguous. Do not "
        "invent or strengthen the answer. Preserve dates, quantities, names, negation and modality. "
        "Return ONLY a JSON array of strings, with no markdown."
    )
    key = stable_hash("claims", SETTINGS.llm_model, prompt)
    try:
        raw = _call_anthropic(
            "You extract atomic factual claims from an answer. Output valid JSON only.",
            prompt,
            key,
        )
        parsed = _parse_json_claims(raw)
        return parsed or fallback
    except Exception as exc:
        logger.warning("LLM claim extraction failed; using sentence splitting: %s", exc)
        return fallback
