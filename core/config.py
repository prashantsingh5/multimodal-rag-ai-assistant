"""
Central configuration for the Multimodal RAG AI Assistant.
All tuneable parameters live here — no magic strings scattered across modules.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = "gemini"                    # "gemini" | "openai"
    model: str = "gemini-2.5-flash-lite"
    temperature: float = 0.3
    max_output_tokens: int = 2048


@dataclass
class EmbeddingConfig:
    provider: str = "gemini"                    # "gemini" | "sentence-transformers"
    model: str = "gemini-embedding-001"          # or "all-mpnet-base-v2"
    # Used only when provider == "sentence-transformers"
    local_model: str = "all-mpnet-base-v2"


@dataclass
class VectorStoreConfig:
    backend: str = "faiss"                      # "faiss" | "chroma"
    persist_directory: str = "assets/vectorstore"
    collection_name: str = "rag_collection"
    top_k: int = 5
    similarity_threshold: float = 0.4


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class SummarizationConfig:
    model: str = "facebook/bart-large-cnn"
    max_length: int = 200
    min_length: int = 50
    chunk_size: int = 1024


@dataclass
class HumanizerConfig:
    default_style: str = "casual"               # "casual" | "conversational" | "professional"
    preserve_threshold: float = 0.75
    similarity_model: str = "all-mpnet-base-v2"
    ai_score_threshold: float = 0.40            # treat as AI if score > this


@dataclass
class AppConfig:
    # API keys — prefer env vars; fall back to placeholders
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # Sub-configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    humanizer: HumanizerConfig = field(default_factory=HumanizerConfig)

    # Conversation history
    history_csv: str = "assets/conversation_history.csv"

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Logging
    log_level: str = "INFO"


# Singleton — import this everywhere
settings = AppConfig()
