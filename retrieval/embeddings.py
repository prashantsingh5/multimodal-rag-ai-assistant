"""
Embedding factory — returns a callable that converts text → dense vector.

Supports:
  - "gemini"               → Google GenerativeAI Embeddings (API)
  - "sentence-transformers" → local SentenceTransformer model (offline)
"""

from __future__ import annotations

from core import get_logger, ConfigurationError
from core.config import settings

logger = get_logger(__name__)


def get_embedding_function(provider: str | None = None):
    """
    Return a LangChain-compatible embedding object.

    Args:
        provider: "gemini" or "sentence-transformers". Defaults to
                  ``settings.embedding.provider``.

    Returns:
        A LangChain Embeddings instance.

    Raises:
        ConfigurationError: If Gemini is requested but the API key is missing.
    """
    provider = provider or settings.embedding.provider

    if provider == "gemini":
        if settings.gemini_api_key in ("", "YOUR_GEMINI_API_KEY_HERE"):
            raise ConfigurationError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        logger.info("Using Gemini embeddings: %s", settings.embedding.model)
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding.model,
            google_api_key=settings.gemini_api_key,
        )

    elif provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = settings.embedding.local_model
        logger.info("Using local SentenceTransformer embeddings: %s", model_name)
        return HuggingFaceEmbeddings(model_name=model_name)

    else:
        raise ConfigurationError(
            f"Unknown embedding provider '{provider}'. "
            "Choose 'gemini' or 'sentence-transformers'."
        )
