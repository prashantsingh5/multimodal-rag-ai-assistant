"""
Query Processor — cleans, classifies, and optionally expands user queries
before they hit the retriever. Modelled on the intelligent planner in
project-samarth but generalised to open-domain knowledge tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from core import get_logger

logger = get_logger(__name__)

# Common stop-words that add noise to embedding searches
_STOP_WORDS = {
    "what", "is", "the", "a", "an", "of", "in", "on", "to", "for",
    "and", "or", "how", "why", "when", "where", "who", "which",
    "please", "tell", "give", "me", "explain", "describe",
}


@dataclass
class ProcessedQuery:
    original: str
    cleaned: str
    intent: str        # "factual" | "summary" | "comparison" | "open"
    keywords: List[str]
    expanded_query: str    # used as the actual embedding query


class QueryProcessor:
    """
    Preprocess and classify user queries.

    The intent classification drives how the pipeline formats the LLM prompt:
    - ``factual``    → precision-focused system prompt
    - ``summary``    → summarisation-focused system prompt
    - ``comparison`` → comparison table prompt
    - ``open``       → general QA prompt
    """

    def process(self, query: str) -> ProcessedQuery:
        """
        Analyse and enrich a raw user query.

        Args:
            query: Raw text from the user.

        Returns:
            ``ProcessedQuery`` with cleaned text, detected intent, keywords,
            and an expanded query string optimised for vector search.
        """
        cleaned = self._clean(query)
        intent = self._classify_intent(cleaned)
        keywords = self._extract_keywords(cleaned)
        expanded = self._expand(cleaned, keywords, intent)

        logger.info(
            "Query processed | intent=%s | keywords=%s | expanded='%s...'",
            intent, keywords, expanded[:60],
        )
        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            intent=intent,
            keywords=keywords,
            expanded_query=expanded,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        # Normalise whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove trailing punctuation except "?"
        text = re.sub(r"[.!,;:]+$", "", text).strip()
        return text

    @staticmethod
    def _classify_intent(text: str) -> str:
        lower = text.lower()
        summary_triggers = {"summarize", "summarise", "summary", "overview", "tldr", "brief"}
        comparison_triggers = {"compare", "difference", "versus", "vs", "contrast", "better"}
        factual_triggers = {"define", "what is", "who is", "when did", "where is", "how many"}

        if any(t in lower for t in summary_triggers):
            return "summary"
        if any(t in lower for t in comparison_triggers):
            return "comparison"
        if any(t in lower for t in factual_triggers):
            return "factual"
        return "open"

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return [w for w in words if w not in _STOP_WORDS][:10]

    @staticmethod
    def _expand(text: str, keywords: List[str], intent: str) -> str:
        """
        Light query expansion: append key terms so the embedding captures
        broader context without distorting meaning.
        """
        if intent == "summary":
            return f"Provide a summary. {text}"
        if intent == "comparison":
            return f"Compare and contrast. {text}"
        if keywords:
            # Append top-3 keywords as a hint
            hint = " ".join(keywords[:3])
            return f"{text} [{hint}]"
        return text
