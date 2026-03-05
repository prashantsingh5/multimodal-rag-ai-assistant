"""
Document loaders — each returns a list of LangChain `Document` objects with
rich metadata so downstream modules know the provenance of every chunk.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import List

from langchain_core.documents import Document

from core import get_logger, IngestionError

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────────────────────

def load_pdf(source: str | bytes, filename: str = "document.pdf") -> List[Document]:
    """
    Load a PDF from a file path or raw bytes.

    Args:
        source: Absolute file path OR raw bytes from an upload.
        filename: Human-readable name stored in Document metadata.

    Returns:
        List of Documents, one per page, with ``source``, ``page``,
        and ``source_type`` metadata keys.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader

        if isinstance(source, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            docs = PyPDFLoader(tmp_path).load()
            os.remove(tmp_path)
        else:
            docs = PyPDFLoader(source).load()

        for doc in docs:
            doc.metadata.update({"source": filename, "source_type": "pdf"})

        logger.info(f"Loaded PDF '{filename}' — {len(docs)} page(s)")
        return docs

    except Exception as exc:
        raise IngestionError(f"PDF load failed for '{filename}': {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Plain text / Markdown
# ──────────────────────────────────────────────────────────────────────────────

def load_text(source: str | bytes, filename: str = "document.txt") -> List[Document]:
    """Load a plain-text or Markdown file from path or bytes."""
    try:
        if isinstance(source, (bytes, bytearray)):
            text = source.decode("utf-8", errors="replace")
        else:
            with open(source, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

        doc = Document(
            page_content=text,
            metadata={"source": filename, "source_type": "text"},
        )
        logger.info(f"Loaded text file '{filename}' — {len(text)} chars")
        return [doc]

    except Exception as exc:
        raise IngestionError(f"Text load failed for '{filename}': {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# DOCX
# ──────────────────────────────────────────────────────────────────────────────

def load_docx(source: str | bytes, filename: str = "document.docx") -> List[Document]:
    """Load a DOCX file from path or bytes using docx2txt."""
    try:
        import docx2txt

        if isinstance(source, (bytes, bytearray)):
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(source)
                tmp_path = tmp.name
            text = docx2txt.process(tmp_path)
            os.remove(tmp_path)
        else:
            text = docx2txt.process(source)

        doc = Document(
            page_content=text,
            metadata={"source": filename, "source_type": "docx"},
        )
        logger.info(f"Loaded DOCX '{filename}' — {len(text)} chars")
        return [doc]

    except Exception as exc:
        raise IngestionError(f"DOCX load failed for '{filename}': {exc}") from exc


# ──────────────────────────────────────────────────────────────────────────────
# Web URL
# ──────────────────────────────────────────────────────────────────────────────

def load_url(url: str) -> List[Document]:
    """
    Scrape and load a web page. Uses UnstructuredURLLoader which strips
    most HTML noise and returns readable text.
    """
    try:
        from langchain_community.document_loaders import UnstructuredURLLoader

        docs = UnstructuredURLLoader(urls=[url]).load()
        for doc in docs:
            doc.metadata.update({"source": url, "source_type": "url"})

        logger.info(f"Loaded URL '{url}' via UnstructuredURLLoader — {len(docs)} document(s)")
        return docs

    except Exception as exc:
        logger.warning(
            "Primary URL loader failed for '%s': %s. Falling back to requests+BeautifulSoup.",
            url,
            exc,
        )

        try:
            import requests
            from bs4 import BeautifulSoup

            response = requests.get(
                url,
                timeout=20,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"
                    )
                },
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text).strip()

            if not text:
                raise IngestionError("No readable text content extracted from URL.")

            doc = Document(
                page_content=text,
                metadata={"source": url, "source_type": "url", "loader": "requests_bs4"},
            )
            logger.info("Loaded URL '%s' via requests+BeautifulSoup — %d chars", url, len(text))
            return [doc]

        except Exception as fallback_exc:
            raise IngestionError(
                f"URL load failed for '{url}': {fallback_exc}. "
                "Install optional deps with: pip install unstructured beautifulsoup4"
            ) from fallback_exc


# ──────────────────────────────────────────────────────────────────────────────
# YouTube
# ──────────────────────────────────────────────────────────────────────────────

def load_youtube(url: str) -> List[Document]:
    """
    Extract transcript from a YouTube video and return it as a single Document.
    Also embeds title + description as leading context.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtubesearchpython import Video

        video_id = _extract_video_id(url)
        if not video_id:
            raise IngestionError(f"Could not parse video ID from URL: {url}")

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join(entry["text"] for entry in transcript_list)

        # Best-effort metadata enrichment
        try:
            info = Video.getInfo(video_id)
            title = info.get("title", "Unknown")
            description = info.get("description", "")
        except Exception:
            title, description = "Unknown", ""

        content = f"Title: {title}\n\nDescription: {description}\n\nTranscript:\n{transcript_text}"

        doc = Document(
            page_content=content,
            metadata={"source": url, "source_type": "youtube", "title": title},
        )
        logger.info(f"Loaded YouTube '{title}' — {len(transcript_text)} chars")
        return [doc]

    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"YouTube load failed for '{url}': {exc}") from exc


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:watch\?v=)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if len(url) == 11:
        return url
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Image (OCR / Vision caption via Gemini)
# ──────────────────────────────────────────────────────────────────────────────

def load_image(source: str | bytes, filename: str = "image.png") -> List[Document]:
    """
    Describe an image using Gemini Vision and return the description as a Document.
    Falls back to filename-only metadata if the API call fails.
    """
    try:
        import google.generativeai as genai
        from core.config import settings
        import PIL.Image
        import io

        genai.configure(api_key=settings.gemini_api_key)
        vision_model = genai.GenerativeModel("gemini-1.5-flash")

        if isinstance(source, (bytes, bytearray)):
            image = PIL.Image.open(io.BytesIO(source))
        else:
            image = PIL.Image.open(source)

        response = vision_model.generate_content(
            ["Describe this image in detail for use in a knowledge retrieval system:", image]
        )
        description = response.text.strip()

        doc = Document(
            page_content=description,
            metadata={"source": filename, "source_type": "image"},
        )
        logger.info(f"Captioned image '{filename}' via Gemini Vision")
        return [doc]

    except Exception as exc:
        logger.warning(f"Image captioning failed for '{filename}': {exc}. Using placeholder.")
        doc = Document(
            page_content=f"[Image: {filename}] — visual content not extracted.",
            metadata={"source": filename, "source_type": "image"},
        )
        return [doc]
