"""
Pydantic request / response schemas for the FastAPI endpoints.
Keeping them separate from server.py keeps the API layer clean.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, HttpUrl, Field


class URLIngestRequest(BaseModel):
    url: str = Field(..., description="A publicly accessible web page URL to ingest.")


class YouTubeIngestRequest(BaseModel):
    url: str = Field(..., description="A YouTube video URL.")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The user's natural language question.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")
    apply_summarization: Optional[bool] = Field(None, description="Force-enable or disable summarization.")
    apply_humanization: Optional[bool] = Field(None, description="Force-enable or disable humanization.")


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: int
    source: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    intent: str
    sources: List[str]
    retrieved_chunks: int
    processing_applied: List[str]


class HealthResponse(BaseModel):
    status: str
    knowledge_base_empty: bool
