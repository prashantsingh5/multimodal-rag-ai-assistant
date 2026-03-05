"""
Ingestion module — document loaders, preprocessors, chunkers.

Supported source types:
  - PDF files
  - Plain-text / Markdown files
  - DOCX files
  - Web URLs (HTML scraping)
  - YouTube videos (transcript extraction)
  - Images (metadata + OCR caption via Gemini Vision)
"""

from ingestion.loaders import (
    load_pdf,
    load_text,
    load_docx,
    load_url,
    load_youtube,
    load_image,
)
from ingestion.chunker import chunk_documents
from ingestion.preprocessor import preprocess_documents

__all__ = [
    "load_pdf", "load_text", "load_docx", "load_url",
    "load_youtube", "load_image",
    "chunk_documents", "preprocess_documents",
]
