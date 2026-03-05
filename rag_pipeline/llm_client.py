"""
LLM Client — thin, provider-agnostic wrapper around Google Gemini (primary)
and OpenAI (fallback). Returns plain text so the rest of the pipeline
doesn't depend on any SDK-specific response objects.
"""

from __future__ import annotations

from core import get_logger, ConfigurationError
from core.config import settings

logger = get_logger(__name__)


class LLMClient:
    """
    Abstraction over different LLM backends.

    Instantiate once and call ``generate(prompt)`` throughout the pipeline.
    """

    def __init__(self, provider: str | None = None):
        self._provider = provider or settings.llm.provider
        self._client = None  # lazy — built on first generate() call

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        Send a prompt to the LLM and return the generated text.

        Args:
            system_prompt: Instruction context (injected as system role).
            user_message: The user's question or formatted request.

        Returns:
            Generated response as a plain string.
        """
        if self._client is None:
            self._client = self._build_client()
        try:
            if self._provider == "gemini":
                return self._gemini_generate(system_prompt, user_message)
            elif self._provider == "openai":
                return self._openai_generate(system_prompt, user_message)
            else:
                raise ConfigurationError(f"Unknown LLM provider: '{self._provider}'")
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            return f"[LLM Error] {exc}"

    # ──────────────────────────────────────────────────────────────────────────
    # Gemini
    # ──────────────────────────────────────────────────────────────────────────

    def _build_client(self):
        if self._provider == "gemini":
            if settings.gemini_api_key in ("", "YOUR_GEMINI_API_KEY_HERE"):
                raise ConfigurationError(
                    "GEMINI_API_KEY is not set. Add it to your .env file."
                )
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=settings.llm.model,
                google_api_key=settings.gemini_api_key,
                temperature=settings.llm.temperature,
                max_output_tokens=settings.llm.max_output_tokens,
            )

        elif self._provider == "openai":
            if not settings.openai_api_key:
                raise ConfigurationError("OPENAI_API_KEY is not set.")
            from openai import OpenAI
            return OpenAI(api_key=settings.openai_api_key)

        return None

    def _gemini_generate(self, system_prompt: str, user_message: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage
        response = self._client.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ])
        return response.content.strip()

    # ──────────────────────────────────────────────────────────────────────────
    # OpenAI
    # ──────────────────────────────────────────────────────────────────────────

    def _openai_generate(self, system_prompt: str, user_message: str) -> str:
        response = self._client.chat.completions.create(
            model=settings.llm.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_output_tokens,
        )
        return response.choices[0].message.content.strip()
