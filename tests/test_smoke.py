"""
Smoke-test suite for the Multimodal RAG AI Assistant.

Tests everything that does NOT require an API key or GPU:
  - Config loading
  - Logger
  - Custom exceptions
  - QueryProcessor
  - Preprocessor
  - Chunker
  - ContextBuilder
  - Humanizer (rule-based path, no Gemini)
  - VectorStore / Retriever (skipped if no API key set)
  - FastAPI routes (via TestClient)

Run from the project root (with venv active):
    python tests/test_smoke.py
"""

import os
import sys
import tempfile
import traceback
from typing import Callable

# ── ensure project root is importable ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── colour helpers ─────────────────────────────────────────────────────────

GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW= "\033[93m"
RESET = "\033[0m"
BOLD  = "\033[1m"

_results = []

def run(label: str, fn: Callable):
    try:
        fn()
        _results.append((True, label, None))
        print(f"  {GREEN}✓{RESET}  {label}")
    except Exception as exc:
        _results.append((False, label, exc))
        print(f"  {RED}✗{RESET}  {label}")
        print(f"     {RED}{type(exc).__name__}: {exc}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_config():
    from core.config import AppConfig, settings
    assert isinstance(settings, AppConfig)
    assert settings.api_port == 8000
    assert settings.chunking.chunk_size == 1000
    assert settings.humanizer.default_style == "casual"


def test_logger():
    from core.logger import get_logger
    log = get_logger("test")
    log.info("Logger OK")
    assert log is not None


def test_exceptions():
    from core.exceptions import (
        IngestionError, RetrievalError, PipelineError,
        ProcessingError, ConfigurationError,
    )
    for cls in (IngestionError, RetrievalError, PipelineError,
                ProcessingError, ConfigurationError):
        try:
            raise cls("test")
        except cls:
            pass


def test_query_processor_basic():
    from rag_pipeline.query_processor import QueryProcessor
    qp = QueryProcessor()

    r = qp.process("What is the capital of France?")
    assert r.intent == "factual"
    assert "capital" in r.keywords or "france" in r.keywords
    assert r.cleaned == "What is the capital of France?"

    r2 = qp.process("Summarize the document for me")
    assert r2.intent == "summary"

    r3 = qp.process("Compare Python and JavaScript")
    assert r3.intent == "comparison"


def test_query_processor_expansion():
    from rag_pipeline.query_processor import QueryProcessor
    qp = QueryProcessor()
    r = qp.process("What are the benefits of exercise?")
    assert r.expanded_query != ""
    assert len(r.keywords) > 0


def test_preprocessor():
    from langchain_core.documents import Document
    from ingestion.preprocessor import preprocess_documents

    raw = [
        Document(page_content="Hello\x00 World\n\n\n\n   Multiple  blanks.", metadata={"source": "test"}),
        Document(page_content="  ", metadata={"source": "empty"}),   # should be dropped
        Document(page_content="Page 3 of 10\nActual content here that is long enough.", metadata={"source": "test2"}),
    ]
    cleaned = preprocess_documents(raw)
    assert len(cleaned) == 2, f"Expected 2, got {len(cleaned)}"
    assert "\x00" not in cleaned[0].page_content
    assert "Page 3 of 10" not in cleaned[1].page_content


def test_chunker():
    from langchain_core.documents import Document
    from ingestion.chunker import chunk_documents

    text = " ".join(["word"] * 500)   # 500-word doc
    docs = [Document(page_content=text, metadata={"source": "test"})]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1, "Expected multiple chunks"
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["source"] == "test"


def test_context_builder():
    from langchain_core.documents import Document
    from rag_pipeline.context_builder import build_context

    docs = [
        Document(page_content="Apples are red fruits.", metadata={"source": "fruit.pdf", "source_type": "pdf"}),
        Document(page_content="Bananas are yellow.", metadata={"source": "fruit.pdf", "source_type": "pdf"}),
        # near-duplicate of first — should be deduplicated
        Document(page_content="Apples are red fruits.", metadata={"source": "dup.pdf", "source_type": "pdf"}),
    ]
    ctx = build_context(docs)
    assert "Apples" in ctx
    assert "Bananas" in ctx
    assert ctx.count("Apples are red fruits.") == 1, "Deduplication failed"
    assert "[1]" in ctx and "[2]" in ctx


def test_context_builder_empty():
    from rag_pipeline.context_builder import build_context
    result = build_context([])
    assert result == "No relevant context found."


def test_humanizer_ai_score():
    from processing.humanizer.humanizer import Humanizer
    h = Humanizer()

    ai_text = (
        "Furthermore, it is important to note that the comprehensive utilization "
        "of state-of-the-art methodologies facilitates the establishment of "
        "significant improvements. Consequently, numerous endeavors have been "
        "undertaken to demonstrate the substantial benefits."
    )
    human_text = "I just tried the new method and it works really well. Give it a go!"

    ai_score = h.ai_score(ai_text)
    human_score = h.ai_score(human_text)
    assert ai_score > human_score, f"Expected ai_score({ai_score:.2f}) > human_score({human_score:.2f})"
    assert ai_score > 0.3, f"AI text should score high, got {ai_score:.2f}"


def test_humanizer_phrase_substitution():
    from processing.humanizer.humanizer import Humanizer
    h = Humanizer()
    h._threshold = 0.0   # always humanize
    # Use 20+ words so the short-text guard is bypassed
    long_text = (
        "Furthermore we must utilize this technology to facilitate better outcomes. "
        "It is important to note that comprehensive methods are utilized across various domains."
    )
    result = h.humanize(long_text)
    assert "Furthermore" not in result or "utilize" not in result, \
        "Expected at least one AI phrase to be replaced"


def test_humanizer_contractions():
    from processing.humanizer.humanizer import Humanizer
    h = Humanizer()
    h._threshold = 0.0
    # 20+ words so short-text guard is bypassed
    text = (
        "We are going to do this experiment carefully. "
        "I am not sure it will not work as planned. "
        "They are confident the results are not going to disappoint us at all."
    )
    result = h.humanize(text)
    # Contractions should appear
    assert "we're" in result.lower() or "i'm" in result.lower() or "won't" in result.lower() or "aren't" in result.lower()


def settings_threshold_restore(h):
    from core.config import settings
    return settings.humanizer.ai_score_threshold


def test_pdf_loader_bytes():
    """Test PDF loader with a minimal synthetic PDF."""
    try:
        import pypdf
    except ImportError:
        print("     (skipped — pypdf not installed)")
        return

    # Create a minimal PDF in memory with pypdf
    from pypdf import PdfWriter
    import io
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    from ingestion.loaders import load_pdf
    docs = load_pdf(pdf_bytes, filename="test.pdf")
    assert isinstance(docs, list)
    for doc in docs:
        assert doc.metadata["source"] == "test.pdf"
        assert doc.metadata["source_type"] == "pdf"


def test_text_loader_bytes():
    from ingestion.loaders import load_text
    content = b"Hello from a text file. This is a test."
    docs = load_text(content, filename="hello.txt")
    assert len(docs) == 1
    assert "Hello" in docs[0].page_content
    assert docs[0].metadata["source_type"] == "text"


def test_text_loader_file():
    from ingestion.loaders import load_text
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("Test file content for the loader.")
        path = f.name
    try:
        docs = load_text(path, filename="tmp.txt")
        assert "Test file content" in docs[0].page_content
    finally:
        os.remove(path)


def test_fastapi_health():
    from fastapi.testclient import TestClient
    from api.server import app
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "knowledge_base_empty" in body


def test_fastapi_query_empty_kb():
    """Querying an empty knowledge base should return 400."""
    from fastapi.testclient import TestClient
    import api.server as srv
    from api.server import app
    # Reset pipeline so the test starts with a clean slate
    srv._vector_store = None
    srv._pipeline = None
    client = TestClient(app)
    r = client.post("/query", json={"query": "What is RAG?"})
    assert r.status_code == 400


def test_fastapi_ingest_unsupported_type():
    """Uploading an unsupported file type should return 415."""
    from fastapi.testclient import TestClient
    from api.server import app
    client = TestClient(app)
    r = client.post(
        "/ingest/file",
        files={"file": ("malware.exe", b"\x00\x01\x02", "application/octet-stream")},
    )
    assert r.status_code == 415


def test_fastapi_ingest_text_and_query_no_api():
    """
    Ingest a text file and attempt a query.
    The query will fail at the LLM step (no real API key),
    but ingestion itself should succeed (status 200, chunks > 0).
    """
    from fastapi.testclient import TestClient
    from api.server import app, _vector_store

    # Skip this test if no embedding API is available
    from core.config import settings
    if settings.gemini_api_key in ("", "YOUR_GEMINI_API_KEY_HERE", "your_gemini_api_key_here"):
        print("     (skipped — no GEMINI_API_KEY set)")
        return

    _vector_store.clear()
    client = TestClient(app)

    payload = b"The Eiffel Tower is located in Paris, France. It was built in 1889."
    r = client.post(
        "/ingest/file",
        files={"file": ("facts.txt", payload, "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunks_indexed"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

TESTS = [
    ("Config loads correctly",              test_config),
    ("Logger creates without error",        test_logger),
    ("Custom exceptions raise/catch",       test_exceptions),
    ("QueryProcessor — intent: factual",    test_query_processor_basic),
    ("QueryProcessor — expansion",          test_query_processor_expansion),
    ("Preprocessor cleans text",            test_preprocessor),
    ("Chunker splits documents",            test_chunker),
    ("ContextBuilder formats chunks",       test_context_builder),
    ("ContextBuilder — empty input",        test_context_builder_empty),
    ("Humanizer — AI score detection",      test_humanizer_ai_score),
    ("Humanizer — phrase substitution",     test_humanizer_phrase_substitution),
    ("Humanizer — contraction injection",   test_humanizer_contractions),
    ("Loader — PDF bytes",                  test_pdf_loader_bytes),
    ("Loader — text bytes",                 test_text_loader_bytes),
    ("Loader — text file path",             test_text_loader_file),
    ("FastAPI /health",                     test_fastapi_health),
    ("FastAPI /query on empty KB → 400",    test_fastapi_query_empty_kb),
    ("FastAPI unsupported file → 415",      test_fastapi_ingest_unsupported_type),
    ("FastAPI ingest text (needs key)",     test_fastapi_ingest_text_and_query_no_api),
]


if __name__ == "__main__":
    print(f"\n{BOLD}=== Multimodal RAG AI Assistant — Smoke Tests ==={RESET}\n")

    for label, fn in TESTS:
        run(label, fn)

    passed = sum(1 for ok, _, _ in _results if ok)
    failed = sum(1 for ok, _, _ in _results if not ok)
    total  = len(_results)

    print(f"\n{BOLD}Results: {GREEN}{passed} passed{RESET}{BOLD}, "
          f"{RED if failed else ''}{failed} failed{RESET}{BOLD} / {total} total{RESET}")

    if failed:
        print(f"\n{YELLOW}Failed tests:{RESET}")
        for ok, label, exc in _results:
            if not ok:
                print(f"  • {label}")
                traceback.print_exception(type(exc), exc, exc.__traceback__, limit=3)
        sys.exit(1)
    else:
        print(f"\n{GREEN}{BOLD}All tests passed!{RESET}")
