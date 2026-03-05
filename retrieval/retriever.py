"""
Retriever — thin wrapper that adds re-ranking and score filtering
on top of the vector store's similarity search.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

from core import get_logger, RetrievalError
from core.config import settings
from retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class Retriever:
    """
    High-level retriever that can be used independently of LangChain chains.

    Example::

        retriever = Retriever(vector_store)
        docs = retriever.retrieve("What is gradient descent?")
    """

    def __init__(self, vector_store: VectorStore):
        self._vs = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> List[Document]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language question.
            top_k: How many chunks to return.
            score_threshold: Minimum similarity score (0–1). Chunks below
                             this threshold are discarded.

        Returns:
            Ordered list of relevant Document chunks.

        Raises:
            RetrievalError: If the vector store is empty or search fails.
        """
        if self._vs.is_empty():
            raise RetrievalError(
                "Knowledge base is empty. Please ingest documents first."
            )

        k = top_k or settings.vector_store.top_k
        threshold = score_threshold or settings.vector_store.similarity_threshold

        try:
            lc_retriever = self._vs.as_retriever(top_k=k)
            docs = lc_retriever.invoke(query)
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

        logger.info("Retrieved %d chunk(s) for query: '%s...'", len(docs), query[:60])
        return docs

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int | None = None,
    ) -> List[Tuple[Document, float]]:
        """
        Same as ``retrieve`` but also returns similarity scores.
        Useful for debugging and evaluation.
        """
        if self._vs.is_empty():
            raise RetrievalError("Knowledge base is empty.")

        k = top_k or settings.vector_store.top_k

        # Access the underlying store directly for score-aware search
        store = self._vs._store
        if store is None:
            raise RetrievalError("Vector store not initialised.")

        results: List[Tuple[Document, float]] = store.similarity_search_with_score(query, k=k)
        logger.info("Score-aware retrieval: %d result(s)", len(results))
        return results
