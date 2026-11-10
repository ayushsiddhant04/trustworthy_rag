"""PDF, TXT and DOCX ingestion with page/source provenance."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Any

from .config import SETTINGS


class DocumentIngestionError(ValueError):
    """Raised when a document cannot be converted into usable text."""


def clean_text(text: str) -> str:
    text = (text or "").replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"-\s*\n\s*", "", text)  # repair hyphenated line wraps
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_input(document: Any) -> tuple[str, bytes | None, str | None]:
    if isinstance(document, (bytes, bytearray)):
        return "uploaded_document", bytes(document), None
    if hasattr(document, "read"):
        name = getattr(document, "name", None)
        try:
            if hasattr(document, "seek"):
                document.seek(0)
            payload = document.read()
            if hasattr(document, "seek"):
                document.seek(0)
        except Exception as exc:
            raise DocumentIngestionError(f"Could not read uploaded document: {exc}") from exc
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return Path(str(name or "uploaded_document")).name, bytes(payload), Path(str(name or "")).suffix.lower() or None
    path = Path(str(document)).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")
    try:
        return path.name, path.read_bytes(), path.suffix.lower()
    except OSError as exc:
        raise DocumentIngestionError(f"Could not read {path.name}: {exc}") from exc


def load_document(document: Any) -> dict:
    """Load a PDF, TXT or DOCX path/upload into normalized text sections.

    Returns: {source, file_type, sections, warnings}. Page numbers are 1-based
    for PDF and None for formats without reliable page-layout information.
    """
    source, payload, suffix = _read_input(document)
    if payload is None:
        raise DocumentIngestionError("No document content was supplied")

    # File size validation (Task 5)
    if len(payload) > SETTINGS.max_upload_size_bytes:
        limit_mb = SETTINGS.max_upload_size_bytes / (1024 * 1024)
        actual_mb = len(payload) / (1024 * 1024)
        raise DocumentIngestionError(
            f"Document '{source}' ({actual_mb:.1f} MB) exceeds maximum allowed file size of {limit_mb:.0f} MB."
        )

    if suffix is None:
        suffix = Path(source).suffix.lower()
    if not suffix:
        # Try UTF-8 text as a safe default for byte uploads without a filename.
        suffix = ".txt"

    # Supported file type validation (Task 5)
    if suffix not in SETTINGS.supported_file_extensions:
        raise DocumentIngestionError(
            f"Unsupported file type '{suffix or 'unknown'}'. Supported formats are: {', '.join(SETTINGS.supported_file_extensions)}."
        )

    warnings: list[str] = []
    sections: list[dict] = []

    if suffix == ".pdf":
        try:
            import fitz
            with fitz.open(stream=payload, filetype="pdf") as pdf:
                # PDF page count limit validation (Task 5)
                if len(pdf) > SETTINGS.max_pdf_pages:
                    raise DocumentIngestionError(
                        f"PDF '{source}' contains {len(pdf)} pages, which exceeds the maximum allowed limit of {SETTINGS.max_pdf_pages} pages."
                    )
                if pdf.is_encrypted:
                    raise DocumentIngestionError("The PDF is password-protected; provide an unlocked copy.")
                empty_pages = []
                for page_index, page in enumerate(pdf, start=1):
                    text = clean_text(page.get_text("text"))
                    if not text:
                        empty_pages.append(page_index)
                        continue
                    sections.append({"page": page_index, "paragraph": None, "source": source, "text": text})
                if empty_pages:
                    warnings.append(f"Skipped {len(empty_pages)} page(s) with no extractable text: {empty_pages}.")
        except DocumentIngestionError:
            raise
        except Exception as exc:
            raise DocumentIngestionError(f"Could not parse PDF '{source}': {exc}") from exc
        if not sections:
            raise DocumentIngestionError(
                f"'{source}' contains no extractable text. It may be a scanned/image-only PDF. "
                "OCR is not enabled in this prototype; provide a text-based PDF or TXT/DOCX file."
            )
        file_type = "pdf"

    elif suffix == ".txt":
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
            warnings.append("Some invalid UTF-8 bytes were replaced during text decoding.")
        text = clean_text(text)
        if text:
            # Preserve paragraph boundaries for retrieval and citations.
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
            if not paragraphs:
                paragraphs = [text]
            sections = [
                {"page": None, "paragraph": idx + 1, "source": source, "text": paragraph}
                for idx, paragraph in enumerate(paragraphs)
            ]
        if not sections:
            raise DocumentIngestionError(f"'{source}' is empty after text cleaning.")
        file_type = "txt"

    elif suffix == ".docx":
        try:
            from docx import Document
            doc = Document(BytesIO(payload))
            paragraph_no = 0
            for paragraph in doc.paragraphs:
                text = clean_text(paragraph.text)
                if text:
                    paragraph_no += 1
                    sections.append({"page": None, "paragraph": paragraph_no, "source": source, "text": text})
            # Include table contents, which frequently contain policy values.
            for table in doc.tables:
                for row in table.rows:
                    cells = [clean_text(cell.text) for cell in row.cells]
                    row_text = " | ".join(cell for cell in cells if cell)
                    if row_text:
                        paragraph_no += 1
                        sections.append({"page": None, "paragraph": paragraph_no, "source": source, "text": row_text})
        except Exception as exc:
            raise DocumentIngestionError(f"Could not parse DOCX '{source}': {exc}") from exc
        if not sections:
            raise DocumentIngestionError(f"'{source}' contains no readable paragraphs or table text.")
        warnings.append("DOCX page numbers are unavailable from the document structure; paragraph references are used instead.")
        file_type = "docx"

    else:
        raise DocumentIngestionError(
            f"Unsupported file type '{suffix or 'unknown'}'. Supported formats are PDF, TXT and DOCX."
        )

    return {
        "source": source,
        "file_type": file_type,
        "sections": sections,
        "warnings": warnings,
        "text": "\n\n".join(section["text"] for section in sections),
    }


# Backwards-compatible alias used in earlier drafts.
ingest = load_document
