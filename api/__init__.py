"""
FastAPI server — exposes the RAG pipeline as a REST API.

Endpoints:
  POST /ingest/file      — upload a PDF, DOCX, or text file
  POST /ingest/url       — ingest a web page
  POST /ingest/youtube   — ingest a YouTube video transcript
  POST /query            — ask a question against the knowledge base
  GET  /history          — retrieve conversation history (CSV → JSON)
  DELETE /knowledge-base — clear all indexed documents
  GET  /health           — liveness probe
"""
