"""
Gradio frontend for the Multimodal RAG AI Assistant.

Tabs:
  1. Ingest — upload files, paste a URL, or drop a YouTube link
  2. Ask    — query the knowledge base
  3. History — view conversation history

Run:
    python frontend/app.py
    # or via venv:
    .\\venv\\Scripts\\python.exe frontend/app.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from core import get_logger
from ingestion.loaders import load_pdf, load_text, load_docx, load_url, load_youtube, load_image
from rag_pipeline.pipeline import RAGPipeline
from retrieval.vector_store import VectorStore

logger = get_logger("frontend")

# ─────────────────────────── shared state ────────────────────────────────────

_vs = VectorStore()
_pipeline = RAGPipeline(vector_store=_vs, enable_summarization=True, enable_humanization=True)

# ─────────────────────────── ingest handlers ─────────────────────────────────

def ingest_file(files) -> str:
    if not files:
        return "No files selected."
    messages = []
    for file in (files if isinstance(files, list) else [files]):
        try:
            path = file.name
            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[-1].lower()

            with open(path, "rb") as f:
                content = f.read()

            if ext == ".pdf":
                docs = load_pdf(content, filename=filename)
            elif ext in (".txt", ".md"):
                docs = load_text(content, filename=filename)
            elif ext == ".docx":
                docs = load_docx(content, filename=filename)
            elif ext in (".png", ".jpg", ".jpeg", ".webp"):
                docs = load_image(content, filename=filename)
            else:
                messages.append(f"⚠ Skipped '{filename}' — unsupported format.")
                continue

            n = _pipeline.ingest(docs)
            messages.append(f"✅ '{filename}' ingested — {n} chunks indexed.")
        except Exception as exc:
            messages.append(f"❌ Error ingesting '{filename}': {exc}")

    return "\n".join(messages)


def ingest_url(url: str) -> str:
    if not url.strip():
        return "Please enter a URL."
    try:
        docs = load_url(url.strip())
        n = _pipeline.ingest(docs)
        return f"✅ URL ingested — {n} chunks indexed.\nSource: {url}"
    except Exception as exc:
        return f"❌ Failed to ingest URL: {exc}"


def ingest_youtube(url: str) -> str:
    if not url.strip():
        return "Please enter a YouTube URL."
    try:
        docs = load_youtube(url.strip())
        n = _pipeline.ingest(docs)
        return f"✅ YouTube video ingested — {n} chunks indexed.\nSource: {url}"
    except Exception as exc:
        return f"❌ Failed to ingest YouTube video: {exc}"


def clear_kb() -> str:
    _vs.clear()
    return "🗑 Knowledge base cleared."


# ─────────────────────────── query handler ───────────────────────────────────

def answer_query(
    query: str,
    top_k: int,
    apply_summarization: bool,
    apply_humanization: bool,
    history: list,
) -> tuple[list, str]:
    """Return (updated chat history, sources string)."""
    if not query.strip():
        return history, ""

    if _vs.is_empty():
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "⚠ Knowledge base is empty. Please ingest documents first."},
        ]
        return history, ""

    try:
        result = _pipeline.query(
            user_query=query,
            top_k=top_k,
            apply_summarization=apply_summarization,
            apply_humanization=apply_humanization,
        )
        sources_text = "\n".join(f"• {s}" for s in result.sources) or "None"
        meta = (
            f"**Intent:** {result.intent} | "
            f"**Chunks retrieved:** {result.retrieved_chunks} | "
            f"**Post-processing:** {', '.join(result.processing_applied) or 'none'}"
        )
        answer = f"{result.answer}\n\n---\n{meta}"
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]
        return history, sources_text
    except Exception as exc:
        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": f"❌ Error: {exc}"},
        ]
        return history, ""


# ─────────────────────────── history loader ──────────────────────────────────

def load_history() -> str:
    from core.config import settings
    import csv

    csv_path = settings.history_csv
    if not os.path.exists(csv_path):
        return "No conversation history yet."

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return "History is empty."

    lines = []
    for r in rows[-30:]:   # show last 30
        lines.append(
            f"**[{r.get('timestamp','')}]** ({r.get('intent','')})\n"
            f"Q: {r.get('query','')}\n"
            f"A: {r.get('answer','')[:300]}...\n"
        )
    return "\n---\n".join(lines)


# ─────────────────────────── UI layout ───────────────────────────────────────

_CSS = """
#title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
#subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; }
"""

with gr.Blocks(title="Multimodal RAG AI Assistant") as demo:

    gr.Markdown("# 🤖 Multimodal RAG AI Assistant", elem_id="title")
    gr.Markdown(
        "Ingest documents, web pages, YouTube videos, and images — then ask anything.",
        elem_id="subtitle",
    )

    with gr.Tabs():

        # ── Tab 1: Ingest ──────────────────────────────────────────────────
        with gr.TabItem("📥 Ingest"):
            gr.Markdown("### Add knowledge to the assistant")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Upload files** (PDF, TXT, DOCX, PNG, JPG)")
                    file_input = gr.File(
                        file_count="multiple",
                        file_types=[".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"],
                        label="Drop files here",
                    )
                    file_btn = gr.Button("Ingest Files", variant="primary")
                    file_status = gr.Textbox(label="Status", lines=4, interactive=False)
                    file_btn.click(ingest_file, inputs=[file_input], outputs=[file_status])

                with gr.Column():
                    gr.Markdown("**Web URL**")
                    url_input = gr.Textbox(label="Enter a URL", placeholder="https://example.com/article")
                    url_btn = gr.Button("Ingest URL", variant="primary")
                    url_status = gr.Textbox(label="Status", interactive=False)
                    url_btn.click(ingest_url, inputs=[url_input], outputs=[url_status])

                    gr.Markdown("**YouTube Video**")
                    yt_input = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
                    yt_btn = gr.Button("Ingest YouTube", variant="primary")
                    yt_status = gr.Textbox(label="Status", interactive=False)
                    yt_btn.click(ingest_youtube, inputs=[yt_input], outputs=[yt_status])

            with gr.Row():
                clear_btn = gr.Button("🗑 Clear Knowledge Base", variant="stop")
                clear_status = gr.Textbox(label="", interactive=False)
                clear_btn.click(clear_kb, outputs=[clear_status])

        # ── Tab 2: Ask ─────────────────────────────────────────────────────
        with gr.TabItem("💬 Ask"):
            gr.Markdown("### Query the knowledge base")

            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Conversation", height=480)
                    query_box = gr.Textbox(
                        placeholder="Ask anything about your ingested documents…",
                        label="Your question",
                        lines=2,
                    )
                    with gr.Row():
                        ask_btn = gr.Button("Ask", variant="primary")
                        clear_chat_btn = gr.Button("Clear Chat")

                with gr.Column(scale=1):
                    gr.Markdown("**Pipeline options**")
                    top_k_slider = gr.Slider(1, 15, value=5, step=1, label="Chunks to retrieve (top_k)")
                    summarize_cb = gr.Checkbox(value=True, label="Auto-summarize long answers")
                    humanize_cb = gr.Checkbox(value=True, label="Humanize answer")
                    sources_box = gr.Textbox(label="Sources used", lines=5, interactive=False)

            chat_state = gr.State([])
            ask_btn.click(
                answer_query,
                inputs=[query_box, top_k_slider, summarize_cb, humanize_cb, chat_state],
                outputs=[chatbot, sources_box],
            ).then(lambda h: h, inputs=[chatbot], outputs=[chat_state])
            query_box.submit(
                answer_query,
                inputs=[query_box, top_k_slider, summarize_cb, humanize_cb, chat_state],
                outputs=[chatbot, sources_box],
            ).then(lambda h: h, inputs=[chatbot], outputs=[chat_state])
            clear_chat_btn.click(lambda: ([], []), outputs=[chatbot, chat_state])

        # ── Tab 3: History ─────────────────────────────────────────────────
        with gr.TabItem("📜 History"):
            gr.Markdown("### Conversation history (last 30 turns)")
            refresh_btn = gr.Button("Refresh")
            history_box = gr.Markdown()
            refresh_btn.click(load_history, outputs=[history_box])
            demo.load(load_history, outputs=[history_box])


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(),
        css=_CSS,
    )
