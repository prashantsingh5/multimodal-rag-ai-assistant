"""
Humanizer — transforms stiff, AI-sounding text into natural, readable prose.

Ported and heavily refactored from AiHumanizer/src/humanizer.py and
AiHumanizer/src/paraphrasing/paraphraser.py.

Strategy:
  1. Detect how "AI-sounding" the text is (score 0–1).
  2. If score < threshold, return as-is (no unnecessary processing).
  3. Apply layered transformations:
       a. AI-phrase substitution using the pattern library.
       b. Contraction injection.
       c. Sentence length variation.
       d. Gemini-powered light paraphrase (optional, API-based).
"""

from __future__ import annotations

import re
import random
from typing import Dict, List

from core import get_logger
from core.config import settings

logger = get_logger(__name__)


# ─────────────────────── pattern library (inline, no dep) ────────────────────

_CONTRACTIONS: Dict[str, str] = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "will not": "won't", "would not": "wouldn't", "should not": "shouldn't",
    "could not": "couldn't", "cannot": "can't", "is not": "isn't",
    "are not": "aren't", "was not": "wasn't", "were not": "weren't",
    "have not": "haven't", "has not": "hasn't", "had not": "hadn't",
    "I am": "I'm", "you are": "you're", "we are": "we're",
    "they are": "they're", "it is": "it's", "that is": "that's",
    "I have": "I've", "you have": "you've", "we have": "we've",
    "I will": "I'll", "you will": "you'll", "we will": "we'll",
    "let us": "let's", "there is": "there's", "here is": "here's",
}

_REPLACEMENTS: Dict[str, List[str]] = {
    "utilize": ["use", "make use of"],
    "facilitate": ["help", "make easier"],
    "implement": ["do", "carry out", "put into practice"],
    "demonstrate": ["show", "prove"],
    "indicate": ["show", "suggest", "point to"],
    "establish": ["set up", "create", "build"],
    "furthermore": ["also", "plus", "and"],
    "moreover": ["plus", "also", "on top of that"],
    "additionally": ["also", "and", "plus"],
    "consequently": ["so", "as a result", "because of this"],
    "therefore": ["so", "that's why", "which means"],
    "however": ["but", "though", "that said"],
    "comprehensive": ["complete", "full", "thorough"],
    "significant": ["big", "important", "major"],
    "substantial": ["large", "big", "considerable"],
    "numerous": ["many", "lots of", "a number of"],
    "various": ["different", "many", "several"],
    "in order to": ["to"],
    "with regard to": ["about", "on"],
    "in terms of": ["for", "about"],
    "it is important to note": ["note that", "keep in mind"],
    "it should be noted": ["note that"],
    "it is worth noting": ["notably", "worth noting"],
    "in conclusion": ["to wrap up", "finally", "overall"],
    "to summarize": ["in short", "to sum up"],
    "delve into": ["look at", "explore", "cover"],
    "cutting-edge": ["latest", "modern", "advanced"],
    "state-of-the-art": ["top-tier", "modern", "latest"],
}

_AI_PATTERNS: List[str] = [
    r'\b(furthermore|moreover|additionally|consequently|therefore|thus)\b',
    r'\b(comprehensive|utilize|facilitate|implement|demonstrate|indicate)\b',
    r'\b(significant|substantial|numerous|various|endeavor|obtain)\b',
    r'\b(in conclusion|to summarize|in summary|to conclude)\b',
    r'\b(it is important to note|it should be noted|it is worth noting)\b',
    r'\b(in order to|with regard to|in terms of|for the purpose of)\b',
    r'\b(predominantly|approximately|methodology|optimal|paramount)\b',
    r'\b(delve into|dive deep|explore thoroughly|examine closely)\b',
    r'\b(cutting-edge|state-of-the-art|revolutionary|groundbreaking)\b',
]


# ─────────────────────────────── main class ──────────────────────────────────

class Humanizer:
    """
    Rule-based + optional LLM-powered text humanizer.

    The class is intentionally lightweight at import time — heavy ML models
    are NOT loaded. Gemini API is used only when available and when the
    AI score is high, keeping costs minimal.
    """

    def __init__(self):
        self._threshold = settings.humanizer.ai_score_threshold

    def humanize(self, text: str, style: str | None = None) -> str:
        """
        Make AI-generated text sound more natural.

        Args:
            text: Input text to humanize.
            style: One of "casual", "conversational", "professional".
                   Defaults to the global config value.

        Returns:
            Humanized text.
        """
        if not text or len(text.split()) < 20:
            return text

        style = style or settings.humanizer.default_style
        score = self.ai_score(text)
        logger.info("AI score: %.3f (threshold: %.3f)", score, self._threshold)

        if score < self._threshold:
            logger.info("Text already reads naturally — skipping humanization.")
            return text

        # Layer 1: phrase substitution
        result = self._substitute_phrases(text)

        # Layer 2: inject contractions
        result = self._inject_contractions(result)

        # Layer 3: vary sentence lengths to break uniform cadence
        result = self._vary_sentence_lengths(result)

        # Layer 4: light Gemini rewrite if score is high
        if score > 0.65:
            result = self._gemini_rewrite(result, style)

        final_score = self.ai_score(result)
        logger.info(
            "Humanization complete | score: %.3f → %.3f | words: %d → %d",
            score, final_score, len(text.split()), len(result.split()),
        )
        return result

    def ai_score(self, text: str) -> float:
        """
        Estimate how AI-generated a piece of text sounds (0 = human, 1 = AI).

        Uses pattern frequency, sentence uniformity, and contraction density.
        No external models required.
        """
        score = 0.0
        lower = text.lower()

        for pattern in _AI_PATTERNS:
            hits = len(re.findall(pattern, lower))
            score += hits * 0.08

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) > 2:
            lengths = [len(s.split()) for s in sentences]
            avg = sum(lengths) / len(lengths)
            variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
            if variance < 16 and avg > 18:      # very uniform long sentences
                score += 0.20

        words = re.findall(r'\b\w+\b', text)
        contractions = re.findall(r"\b\w+'[a-z]+\b", text)
        if len(words) > 30 and len(contractions) < 2:
            score += 0.15

        return min(score, 1.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _substitute_phrases(text: str) -> str:
        for phrase, alternatives in _REPLACEMENTS.items():
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            replacement = random.choice(alternatives)
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _inject_contractions(text: str) -> str:
        for full, contraction in _CONTRACTIONS.items():
            # Case-sensitive replacement to preserve sentence structure
            text = text.replace(full, contraction)
        return text

    @staticmethod
    def _vary_sentence_lengths(text: str) -> str:
        """
        Break very long sentences by inserting a dash or splitting at conjunctions.
        Merge very short consecutive sentences with a comma.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        varied: List[str] = []
        i = 0
        while i < len(sentences):
            s = sentences[i]
            word_count = len(s.split())

            # Merge two short sentences
            if word_count < 8 and i + 1 < len(sentences):
                next_s = sentences[i + 1]
                if len(next_s.split()) < 8:
                    varied.append(s.rstrip(".!?") + ", " + next_s[0].lower() + next_s[1:])
                    i += 2
                    continue

            # Split overly long sentence at "and" or "which"
            if word_count > 45:
                split_at = re.search(r',\s+(and|which|but|so)\s+', s, re.IGNORECASE)
                if split_at:
                    point = split_at.start()
                    varied.append(s[:point].strip() + ".")
                    varied.append(s[point + 2:].strip())
                    i += 1
                    continue

            varied.append(s)
            i += 1

        return " ".join(varied)

    @staticmethod
    def _gemini_rewrite(text: str, style: str) -> str:
        """Use Gemini for a light conversational rewrite (best-effort)."""
        try:
            import google.generativeai as genai
            from core.config import settings

            if settings.gemini_api_key in ("", "YOUR_GEMINI_API_KEY_HERE"):
                return text

            genai.configure(api_key=settings.gemini_api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = (
                f"Rewrite the following text in a {style} tone to make it sound more "
                f"natural and human. Keep the meaning identical. Do not add new information. "
                f"Return only the rewritten text.\n\nText:\n{text}"
            )
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.6,
                    max_output_tokens=1024,
                ),
            )
            result = response.text.strip()
            # Sanity check: if result is too short, fall back
            if len(result.split()) < len(text.split()) * 0.4:
                return text
            return result

        except Exception as exc:
            logger.warning("Gemini rewrite failed (non-critical): %s", exc)
            return text
