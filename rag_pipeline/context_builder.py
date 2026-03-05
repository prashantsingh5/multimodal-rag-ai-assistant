"""
Context Builder — assembles retrieved chunks into a clean, numbered
context block for the LLM prompt. Deduplicates near-identical passages.
"""

from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from core import get_logger

logger = get_logger(__name__)

_MAX_CONTEXT_CHARS = 6000  # guard against giant prompts


def build_context(docs: List[Document]) -> str:
    """
    Convert a list of retrieved Document chunks into a single formatted
    context string to be injected into the LLM system prompt.

    - Numbers each chunk for source attribution.
    - Strips chunks that are near-duplicates of earlier ones.
    - Truncates total context if it exceeds ``_MAX_CONTEXT_CHARS``.

    Args:
        docs: Retrieved and ranked document chunks.

    Returns:
        A formatted multi-line string ready for inclusion in a prompt.
    """
    if not docs:
        return "No relevant context found."

    seen: set[str] = set()
    blocks: List[str] = []
    total_chars = 0

    for i, doc in enumerate(docs, start=1):
        # Simple deduplication: compare first 120 chars
        fingerprint = doc.page_content[:120].lower().strip()
        if fingerprint in seen:
            logger.debug("Skipping near-duplicate chunk %d", i)
            continue
        seen.add(fingerprint)

        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "")
        header = f"[{i}] Source: {source} ({source_type})"
        block = f"{header}\n{doc.page_content.strip()}"

        if total_chars + len(block) > _MAX_CONTEXT_CHARS:
            logger.info("Context truncated at chunk %d (char limit reached)", i)
            break

        blocks.append(block)
        total_chars += len(block)

    context = "\n\n---\n\n".join(blocks)
    logger.info("Built context: %d chunk(s), %d chars", len(blocks), total_chars)
    return context
