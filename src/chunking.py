"""Word-based overlapping chunks retaining source/page provenance."""
from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Iterable

from .config import SETTINGS


def _words_with_spans(text: str):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text or "")]


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30, *, source: str = "unknown", page=None, paragraph=None) -> list[dict]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = (text or "").strip()
    words = _words_with_spans(text)
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        char_start, char_end = words[start][1], words[end - 1][2]
        content = text[char_start:char_end].strip()
        if not content:
            continue
        identity = f"{source}|{page}|{paragraph}|{start}|{content}"
        chunk_id = "chunk_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        chunks.append({
            "chunk_id": chunk_id,
            "text": content,
            "source": source,
            "page": page,
            "paragraph": paragraph,
            "start_word": start,
            "end_word": end,
            "word_count": end - start,
        })
        if end >= len(words):
            break
    return chunks


def chunk_document(sections, chunk_size: int = None, overlap: int = None) -> list[dict]:
    """Chunk sections, preserving page boundaries for PDFs and paragraph boundaries for TXT/DOCX."""
    chunk_size = int(chunk_size or SETTINGS.chunk_size_words)
    overlap = int(overlap if overlap is not None else SETTINGS.chunk_overlap_words)
    result = []
    for index, section in enumerate(sections or []):
        text = (section.get("text") or "").strip()
        if not text:
            continue
        source = section.get("source", "unknown")
        page = section.get("page")
        paragraph = section.get("paragraph", index + 1 if page is None else None)
        result.extend(chunk_text(text, chunk_size, overlap, source=source, page=page, paragraph=paragraph))
    return result
