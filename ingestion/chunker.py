"""
Chunker — splits cleaned documents into overlapping text windows
suitable for embedding and retrieval.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


def chunk_documents(
    docs: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split documents into fixed-size overlapping chunks.

    Args:
        docs: Pre-processed documents.
        chunk_size: Override the global config value.
        chunk_overlap: Override the global config value.

    Returns:
        List of chunk Documents, each inheriting the parent's metadata
        plus a ``chunk_index`` key.
    """
    size = chunk_size or settings.chunking.chunk_size
    overlap = chunk_overlap or settings.chunking.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    # Tag each chunk with its index within the source document
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(f"Chunking: {len(docs)} document(s) → {len(chunks)} chunk(s) "
                f"(size={size}, overlap={overlap})")
    return chunks
