"""
Vector store wrapper — abstracts FAISS and ChromaDB behind a single interface.
Handles persistence so the knowledge base survives restarts.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

from core import get_logger, RetrievalError
from core.config import settings
from retrieval.embeddings import get_embedding_function

logger = get_logger(__name__)


class VectorStore:
    """
    Manages document ingestion into and retrieval from a vector store.

    Usage::

        vs = VectorStore()
        vs.add_documents(chunks)          # first time (or to add more)
        retriever = vs.as_retriever()     # pass to RAG pipeline
    """

    def __init__(self, backend: str | None = None):
        self._backend = backend or settings.vector_store.backend
        self._embedding_fn = None  # lazy — initialised on first use
        self._store = None  # lazy-loaded

        # Ensure persistence directory exists
        os.makedirs(settings.vector_store.persist_directory, exist_ok=True)

    def _get_embedding_fn(self):
        """Return (and cache) the embedding function, initialising it on first call."""
        if self._embedding_fn is None:
            self._embedding_fn = get_embedding_function()
        return self._embedding_fn

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> None:
        """
        Embed and persist a list of document chunks.
        If a store already exists it is extended, not replaced.
        """
        if not docs:
            logger.warning("add_documents called with empty list — nothing to do.")
            return

        try:
            if self._backend == "faiss":
                self._add_faiss(docs)
            elif self._backend == "chroma":
                self._add_chroma(docs)
            else:
                raise RetrievalError(f"Unknown vector store backend: '{self._backend}'")

            logger.info("Added %d chunk(s) to %s store", len(docs), self._backend)

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Vector store ingestion failed: {exc}") from exc

    def as_retriever(self, top_k: int | None = None):
        """
        Return a LangChain BaseRetriever for use in chains.

        Args:
            top_k: Number of chunks to fetch per query.
        """
        if self._store is None:
            self._load_existing()

        if self._store is None:
            raise RetrievalError(
                "Vector store is empty. Ingest documents first via add_documents()."
            )

        k = top_k or settings.vector_store.top_k
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def is_empty(self) -> bool:
        """Return True if no documents have been indexed yet."""
        if self._store is not None:
            return False
        self._load_existing()
        return self._store is None

    def clear(self) -> None:
        """Delete all persisted data and reset the in-memory store."""
        import shutil

        persist_dir = settings.vector_store.persist_directory
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            os.makedirs(persist_dir, exist_ok=True)

        self._store = None
        logger.info("Vector store cleared.")

    # ──────────────────────────────────────────────────────────────────────────
    # FAISS backend
    # ──────────────────────────────────────────────────────────────────────────

    def _add_faiss(self, docs: List[Document]) -> None:
        from langchain_community.vectorstores import FAISS

        emb = self._get_embedding_fn()
        persist_path = os.path.join(
            settings.vector_store.persist_directory, "faiss_index"
        )

        if self._store is None and os.path.exists(persist_path):
            self._store = FAISS.load_local(
                persist_path,
                emb,
                allow_dangerous_deserialization=True,
            )

        if self._store is None:
            self._store = FAISS.from_documents(docs, emb)
        else:
            self._store.add_documents(docs)

        self._store.save_local(persist_path)

    def _load_existing(self) -> None:
        """Try to load a previously persisted store silently."""
        try:
            if self._backend == "faiss":
                from langchain_community.vectorstores import FAISS

                emb = self._get_embedding_fn()
                persist_path = os.path.join(
                    settings.vector_store.persist_directory, "faiss_index"
                )
                if os.path.exists(persist_path):
                    self._store = FAISS.load_local(
                        persist_path,
                        emb,
                        allow_dangerous_deserialization=True,
                    )
                    logger.info("Loaded existing FAISS index from disk.")

            elif self._backend == "chroma":
                self._load_chroma_existing()

        except Exception as exc:
            logger.warning("Could not load existing vector store: %s", exc)

    # ──────────────────────────────────────────────────────────────────────────
    # ChromaDB backend
    # ──────────────────────────────────────────────────────────────────────────

    def _add_chroma(self, docs: List[Document]) -> None:
        from langchain_community.vectorstores import Chroma

        if self._store is None:
            self._store = Chroma(
                collection_name=settings.vector_store.collection_name,
                embedding_function=self._get_embedding_fn(),
                persist_directory=settings.vector_store.persist_directory,
            )
        self._store.add_documents(docs)

    def _load_chroma_existing(self) -> None:
        from langchain_community.vectorstores import Chroma

        self._store = Chroma(
            collection_name=settings.vector_store.collection_name,
            embedding_function=self._get_embedding_fn(),
            persist_directory=settings.vector_store.persist_directory,
        )
        logger.info("Loaded existing ChromaDB collection.")
