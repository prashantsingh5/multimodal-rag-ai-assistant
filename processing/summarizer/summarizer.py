"""
Summarizer — condenses long text using facebook/bart-large-cnn.
Adapted and improved from Summarization_project/code_with_gradio.py.
Handles texts longer than the model's context window by chunk-processing.
"""

from __future__ import annotations

from typing import List

from core import get_logger, ProcessingError
from core.config import settings

logger = get_logger(__name__)


class Summarizer:
    """
    High-quality extractive + abstractive summariser.

    Lazy-loads the BART model on first use to keep startup fast
    when summarization is not needed.
    """

    def __init__(self):
        self._pipeline = None  # loaded lazily

    def summarize(self, text: str, max_length: int | None = None,
                  min_length: int | None = None) -> str:
        """
        Produce a concise summary of ``text``.

        Args:
            text: Input text (any length).
            max_length: Max summary token length per chunk.
            min_length: Min summary token length per chunk.

        Returns:
            Summarised string.

        Raises:
            ProcessingError: If the model fails to load or run.
        """
        if not text or len(text.split()) < 80:
            # Short enough — no summarisation needed
            return text

        try:
            pipeline = self._get_pipeline()

            max_len = max_length or settings.summarization.max_length
            min_len = min_length or settings.summarization.min_length
            chunk_size = settings.summarization.chunk_size

            chunks = self._split_for_model(text, chunk_size)
            summaries: List[str] = []

            for chunk in chunks:
                result = pipeline(
                    chunk,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    truncation=True,
                )
                summaries.append(result[0]["summary_text"])

            merged = " ".join(summaries)

            # If the merged summaries are still long, do a second pass
            if len(merged.split()) > max_len * 1.5:
                merged = self.summarize(merged, max_length=max_len, min_length=min_len)

            logger.info(
                "Summarised %d words → %d words", len(text.split()), len(merged.split())
            )
            return merged

        except Exception as exc:
            raise ProcessingError(f"Summarization failed: {exc}") from exc

    def is_available(self) -> bool:
        """Return True if the summarization model can be loaded."""
        try:
            self._get_pipeline()
            return True
        except Exception:
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            import torch
            from transformers import pipeline as hf_pipeline

            device = 0 if _cuda_available() else -1
            logger.info(
                "Loading summarization model '%s' on %s",
                settings.summarization.model,
                "GPU" if device == 0 else "CPU",
            )
            self._pipeline = hf_pipeline(
                "summarization",
                model=settings.summarization.model,
                device=device,
            )
        return self._pipeline

    @staticmethod
    def _split_for_model(text: str, chunk_size: int) -> List[str]:
        """Split text into word-level chunks that fit within the model window."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunks.append(" ".join(words[i: i + chunk_size]))
        return chunks


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
