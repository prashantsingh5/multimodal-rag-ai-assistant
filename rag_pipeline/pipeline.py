"""
RAG Pipeline — the central orchestrator that stitches every module together.

Flow:
  User query
    → QueryProcessor        (clean + classify)
    → Retriever             (vector similarity search)
    → ContextBuilder        (format retrieved chunks)
    → LLMClient             (generate answer)
    → Summarizer (optional) (condense long answers)
    → Humanizer  (optional) (improve readability)
    → Final answer
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from langchain_core.documents import Document

from core import get_logger, PipelineError
from core.config import settings
from rag_pipeline.query_processor import QueryProcessor, ProcessedQuery
from rag_pipeline.context_builder import build_context
from rag_pipeline.llm_client import LLMClient
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

logger = get_logger(__name__)

# ─────────────────────────── prompt templates ────────────────────────────────

_SYSTEM_PROMPTS = {
    "factual": (
        "You are a precise knowledge assistant. Answer the question using ONLY "
        "the context provided below. If the answer is not in the context, say "
        "'I don't have enough information to answer that.' Do not hallucinate.\n\n"
        "Context:\n{context}"
    ),
    "summary": (
        "You are a summarisation expert. Produce a clear, concise summary of "
        "the topic using the context below. Use bullet points where helpful.\n\n"
        "Context:\n{context}"
    ),
    "comparison": (
        "You are an analytical assistant. Compare the subjects mentioned in "
        "the question using the context below. Present differences and "
        "similarities in a structured way.\n\n"
        "Context:\n{context}"
    ),
    "open": (
        "You are a helpful AI assistant. Use the context below to answer the "
        "user's question as thoroughly as possible. If the context is "
        "insufficient, acknowledge it honestly.\n\n"
        "Context:\n{context}"
    ),
}


# ─────────────────────────── result dataclass ────────────────────────────────

@dataclass
class RAGResult:
    query: str
    answer: str
    sources: List[str] = field(default_factory=list)
    intent: str = "open"
    retrieved_chunks: int = 0
    processing_applied: List[str] = field(default_factory=list)


# ─────────────────────────── pipeline ────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Args:
        vector_store: A ``VectorStore`` instance (may be pre-populated or empty).
        enable_summarization: Condense long LLM responses automatically.
        enable_humanization: Post-process responses for natural readability.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        enable_summarization: bool = True,
        enable_humanization: bool = True,
    ):
        self._vs = vector_store or VectorStore()
        self._retriever = Retriever(self._vs)
        self._query_processor = QueryProcessor()
        self._llm = LLMClient()
        self._enable_summarization = enable_summarization
        self._enable_humanization = enable_humanization

        # Lazy imports for heavy processing modules
        self._summarizer = None
        self._humanizer = None

        os.makedirs("assets", exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Ingestion
    # ──────────────────────────────────────────────────────────────────────────

    def ingest(self, docs: List[Document]) -> int:
        """
        Pre-process, chunk, and embed documents into the vector store.

        Args:
            docs: Raw documents from any loader.

        Returns:
            Number of chunks ingested.
        """
        from ingestion.preprocessor import preprocess_documents
        from ingestion.chunker import chunk_documents

        cleaned = preprocess_documents(docs)
        chunks = chunk_documents(cleaned)
        self._vs.add_documents(chunks)
        logger.info("Ingested %d chunk(s) into knowledge base", len(chunks))
        return len(chunks)

    # ──────────────────────────────────────────────────────────────────────────
    # Query → Answer
    # ──────────────────────────────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        top_k: int | None = None,
        apply_summarization: bool | None = None,
        apply_humanization: bool | None = None,
    ) -> RAGResult:
        """
        Process a user query through the full RAG pipeline.

        Args:
            user_query: Any natural language question.
            top_k: Override the number of retrieved chunks.
            apply_summarization: Override instance-level setting.
            apply_humanization: Override instance-level setting.

        Returns:
            ``RAGResult`` with the final answer and metadata.

        Raises:
            PipelineError: On unrecoverable failures.
        """
        try:
            # 1 — Pre-process query
            pq: ProcessedQuery = self._query_processor.process(user_query)

            # 2 — Retrieve relevant chunks
            chunks = self._retriever.retrieve(pq.expanded_query, top_k=top_k)

            # 3 — Build context
            context = build_context(chunks)

            # 4 — Select prompt template and generate
            system_prompt = _SYSTEM_PROMPTS[pq.intent].format(context=context)
            raw_answer = self._llm.generate(system_prompt, pq.cleaned)

            processing_applied: List[str] = []

            # 5 — Optional summarization (only if answer is long)
            do_summarize = apply_summarization if apply_summarization is not None else self._enable_summarization
            if do_summarize and len(raw_answer.split()) > 250:
                raw_answer = self._summarize(raw_answer)
                processing_applied.append("summarization")

            # 6 — Optional humanization
            do_humanize = apply_humanization if apply_humanization is not None else self._enable_humanization
            if do_humanize:
                raw_answer = self._humanize(raw_answer)
                processing_applied.append("humanization")

            sources = list({c.metadata.get("source", "unknown") for c in chunks})

            result = RAGResult(
                query=user_query,
                answer=raw_answer,
                sources=sources,
                intent=pq.intent,
                retrieved_chunks=len(chunks),
                processing_applied=processing_applied,
            )

            self._save_to_history(result)
            return result

        except Exception as exc:
            logger.error("Pipeline error: %s", exc)
            raise PipelineError(str(exc)) from exc

    # ──────────────────────────────────────────────────────────────────────────
    # Processing helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _summarize(self, text: str) -> str:
        if self._summarizer is None:
            from processing.summarizer.summarizer import Summarizer
            self._summarizer = Summarizer()
        return self._summarizer.summarize(text)

    def _humanize(self, text: str) -> str:
        if self._humanizer is None:
            from processing.humanizer.humanizer import Humanizer
            self._humanizer = Humanizer()
        return self._humanizer.humanize(text)

    # ──────────────────────────────────────────────────────────────────────────
    # History
    # ──────────────────────────────────────────────────────────────────────────

    def _save_to_history(self, result: RAGResult) -> None:
        csv_path = settings.history_csv
        write_header = not os.path.exists(csv_path)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["timestamp", "intent", "query", "answer",
                                "sources", "chunks", "processing"],
                )
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "intent": result.intent,
                    "query": result.query,
                    "answer": result.answer,
                    "sources": "; ".join(result.sources),
                    "chunks": result.retrieved_chunks,
                    "processing": ", ".join(result.processing_applied),
                })
        except Exception as exc:
            logger.warning("Could not save conversation history: %s", exc)
