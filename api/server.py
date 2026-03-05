"""
FastAPI application — production-ready REST API for the Multimodal RAG Assistant.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import csv
import os
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    URLIngestRequest,
    YouTubeIngestRequest,
)
from core import get_logger, IngestionError, PipelineError, RetrievalError
from core.config import settings
from rag_pipeline.pipeline import RAGPipeline
from retrieval.vector_store import VectorStore

logger = get_logger(__name__)

# ─────────────────────────── app setup ───────────────────────────────────────

app = FastAPI(
    title="Multimodal RAG AI Assistant",
    description=(
        "Production-style RAG system with multimodal ingestion, "
        "intelligent retrieval, LLM generation, summarization, and humanization."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state — initialised lazily on first request so the server can start
# even before an API key is configured (the error surfaces on the first call).
_vector_store: VectorStore | None = None
_pipeline: RAGPipeline | None = None


def _get_pipeline() -> tuple[VectorStore, RAGPipeline]:
    """Return (vector_store, pipeline), creating them on first call."""
    global _vector_store, _pipeline
    if _vector_store is None:
        _vector_store = VectorStore()
    if _pipeline is None:
        _pipeline = RAGPipeline(vector_store=_vector_store)
    return _vector_store, _pipeline


# ─────────────────────────── helpers ─────────────────────────────────────────

def _detect_loader(filename: str):
    """Return the correct loader function based on file extension."""
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".pdf":
        from ingestion.loaders import load_pdf
        return load_pdf
    elif ext in (".txt", ".md"):
        from ingestion.loaders import load_text
        return load_text
    elif ext == ".docx":
        from ingestion.loaders import load_docx
        return load_docx
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        from ingestion.loaders import load_image
        return load_image
    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{ext}'. Supported: pdf, txt, md, docx, png, jpg.",
        )


# ─────────────────────────── routes ──────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — returns 200 if the server is running."""
    vs, _ = _get_pipeline()
    return HealthResponse(
        status="ok",
        knowledge_base_empty=vs.is_empty(),
    )


@app.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload and ingest a document file into the knowledge base.

    Supported formats: PDF, TXT, MD, DOCX, PNG, JPG.
    """
    try:
        content = await file.read()
        loader_fn = _detect_loader(file.filename)
        docs = loader_fn(content, filename=file.filename)
        _, pipeline = _get_pipeline()
        chunks_indexed = pipeline.ingest(docs)

        return IngestResponse(
            status="success",
            chunks_indexed=chunks_indexed,
            source=file.filename,
        )
    except HTTPException:
        raise
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Unexpected ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during ingestion.")


@app.post("/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(request: URLIngestRequest):
    """Scrape and ingest content from a public web page URL."""
    try:
        from ingestion.loaders import load_url
        docs = load_url(request.url)
        _, pipeline = _get_pipeline()
        chunks_indexed = pipeline.ingest(docs)

        return IngestResponse(
            status="success",
            chunks_indexed=chunks_indexed,
            source=request.url,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("URL ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during URL ingestion.")


@app.post("/ingest/youtube", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_youtube(request: YouTubeIngestRequest):
    """Extract and ingest the transcript from a YouTube video."""
    try:
        from ingestion.loaders import load_youtube
        docs = load_youtube(request.url)
        _, pipeline = _get_pipeline()
        chunks_indexed = pipeline.ingest(docs)

        return IngestResponse(
            status="success",
            chunks_indexed=chunks_indexed,
            source=request.url,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("YouTube ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during YouTube ingestion.")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_knowledge_base(request: QueryRequest):
    """
    Ask a question against the ingested knowledge base.

    Returns the generated answer, source documents, and pipeline metadata.
    """
    vs, pipeline = _get_pipeline()
    if vs.is_empty():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Knowledge base is empty. Please ingest at least one document first.",
        )

    try:
        result = pipeline.query(
            user_query=request.query,
            top_k=request.top_k,
            apply_summarization=request.apply_summarization,
            apply_humanization=request.apply_humanization,
        )
        return QueryResponse(
            query=result.query,
            answer=result.answer,
            intent=result.intent,
            sources=result.sources,
            retrieved_chunks=result.retrieved_chunks,
            processing_applied=result.processing_applied,
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.error("Query error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error during query.")


@app.get("/history", tags=["System"])
async def get_history(limit: int = 50):
    """Return the last N conversation turns from history (default 50)."""
    csv_path = settings.history_csv
    if not os.path.exists(csv_path):
        return JSONResponse(content={"history": []})

    rows: List[dict] = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read history: {exc}")

    return JSONResponse(content={"history": rows[-limit:]})


@app.delete("/knowledge-base", tags=["System"])
async def clear_knowledge_base():
    """Delete all indexed documents from the vector store."""
    vs, _ = _get_pipeline()
    vs.clear()
    return JSONResponse(content={"status": "Knowledge base cleared."})
