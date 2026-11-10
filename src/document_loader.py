"""Compatibility bridge for older Member A imports."""
from .ingestion import DocumentIngestionError, clean_text, load_document

__all__ = ["DocumentIngestionError", "clean_text", "load_document"]
