"""
Text cleaning and normalisation applied to raw documents
before they are chunked and embedded.
"""

from __future__ import annotations

import re
from typing import List

from langchain_core.documents import Document

from core import get_logger

logger = get_logger(__name__)


def preprocess_documents(docs: List[Document]) -> List[Document]:
    """
    Run a standard cleaning pipeline on each document.

    Steps:
      1. Strip excessive whitespace and null bytes.
      2. Normalise unicode quotation marks and dashes.
      3. Remove boilerplate patterns (page headers/footers).
      4. Drop near-empty documents (< 30 chars after cleaning).

    Args:
        docs: Raw documents from any loader.

    Returns:
        Cleaned documents (short/empty docs discarded).
    """
    cleaned: List[Document] = []
    for doc in docs:
        text = _clean(doc.page_content)
        if len(text) >= 30:
            cleaned.append(Document(page_content=text, metadata=doc.metadata))
        else:
            logger.debug(f"Dropped near-empty chunk from '{doc.metadata.get('source')}'")

    logger.info(f"Preprocessing: {len(docs)} → {len(cleaned)} documents retained")
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    # Remove null bytes
    text = text.replace("\x00", "")

    # Normalise unicode punctuation
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "--")

    # Collapse multiple blank lines to a single one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Strip common PDF page-number footers like "Page 3 of 10"
    text = re.sub(r"(?i)page\s+\d+\s+of\s+\d+", "", text)

    return text.strip()
