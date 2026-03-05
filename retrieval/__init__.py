"""
Retrieval module — embeddings, vector store, retriever.
"""

from retrieval.embeddings import get_embedding_function
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever

__all__ = ["get_embedding_function", "VectorStore", "Retriever"]
