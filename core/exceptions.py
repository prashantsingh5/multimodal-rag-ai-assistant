"""
Project-wide custom exceptions.
Raise these instead of bare Exception so callers can catch specific failure modes.
"""


class IngestionError(Exception):
    """Raised when a document loader or pre-processor fails."""


class RetrievalError(Exception):
    """Raised when vector-store operations fail."""


class PipelineError(Exception):
    """Raised for orchestration-level failures in the RAG pipeline."""


class ProcessingError(Exception):
    """Raised by summarization or humanization modules."""


class ConfigurationError(Exception):
    """Raised when a required config value (e.g. API key) is missing."""
