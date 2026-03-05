# Multimodal RAG AI Assistant

Portfolio-grade, production-style AI assistant that combines multimodal ingestion, retrieval-augmented generation (RAG), intent-aware query planning, answer post-processing, and dual interfaces (FastAPI + Gradio).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Live Demo Video

- Full walkthrough: [Watch the demo](https://github.com/user-attachments/assets/0d10551e-10e9-469e-ab5c-6645642b81c0)
- Includes: PDF + URL + YouTube ingestion, cross-source Q&A, summarization and humanization comparison

## Why This Project Stands Out

- Multimodal knowledge ingestion: PDF, DOCX, TXT/MD, URL, YouTube transcripts, images
- Intent-aware retrieval pipeline: factual, summary, comparison, open-ended
- Pluggable architecture: FAISS/Chroma, Gemini/OpenAI, local/offline fallback paths
- Post-processing layer: optional summarization and answer humanization
- Engineering maturity: modular codebase, lazy initialization, typed exceptions, smoke tests

## System Architecture

```text
Ingestion Layer
    -> Loaders (PDF, DOCX, TXT, URL, YouTube, Image)
    -> Preprocessor
    -> Chunker
    -> Vector Store (FAISS/Chroma)

Query Layer
    -> QueryProcessor (clean + classify intent + expand)
    -> Retriever
    -> ContextBuilder
    -> LLMClient (Gemini/OpenAI)

Post Processing
    -> Summarizer (optional)
    -> Humanizer (optional)

Delivery Layer
    -> FastAPI REST endpoints
    -> Gradio interactive UI
```

## Ingestion Sources

| Source | Handler |
|---|---|
| PDF | `load_pdf()` |
| DOCX | `load_docx()` |
| TXT/MD | `load_text()` |
| URL | `load_url()` |
| YouTube | `load_youtube()` |
| Image | `load_image()` |

## Repository Layout

```text
multimodal-rag-assistant/
|-- core/                  # config, logger, exceptions
|-- ingestion/             # loaders, preprocessor, chunker
|-- retrieval/             # embeddings, vector store, retriever
|-- rag_pipeline/          # query processing and orchestration
|-- processing/
|   |-- summarizer/        # BART summarization
|   `-- humanizer/         # rule + LLM humanization
|-- api/                   # FastAPI server and schemas
|-- frontend/              # Gradio app
|-- assets/                # persisted vectors and history
|-- tests/                 # smoke tests
|-- setup.ps1              # Windows setup
|-- setup.sh               # Linux/macOS setup
`-- README.md
```

## Quick Start

### 1. Environment Setup

Windows (PowerShell):

```powershell
cd multimodal-rag-assistant
.\setup.ps1
```

Linux/macOS:

```bash
cd multimodal-rag-assistant
bash setup.sh
```

### 2. Configure Keys

Edit `.env`:

```env
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=
```

### 3. Run API and UI

Terminal 1 (API):

```powershell
.\venv\Scripts\python.exe -m uvicorn api.server:app --reload --port 8000
```

Terminal 2 (UI):

```powershell
.\venv\Scripts\python.exe frontend\app.py
```

Endpoints:

- API docs: `http://localhost:8000/docs`
- Gradio UI: `http://localhost:7860`

## API Endpoints

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Service health + KB status |
| `POST` | `/ingest/file` | Ingest uploaded files |
| `POST` | `/ingest/url` | Ingest web content |
| `POST` | `/ingest/youtube` | Ingest YouTube transcript |
| `POST` | `/query` | Ask grounded questions |
| `GET` | `/history` | Read conversation history |
| `DELETE` | `/knowledge-base` | Reset vector store |

Example:

```bash
curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{
        "query": "Summarize the key implementation decisions",
        "top_k": 5,
        "apply_summarization": true,
        "apply_humanization": true
    }'
```

## Configurable Components

Core options live in `core/config.py`.

- LLM model/provider
- embedding provider/model
- vector backend (`faiss`/`chroma`)
- chunk size/overlap
- summarization and humanization toggles

## Test Status

Smoke tests:

```powershell
.\venv\Scripts\python.exe tests\test_smoke.py
```

Current baseline: `19/19` passing.

## Demo Video Playbook

Use this script to record a high-impact demo in under 4 minutes.

### Step 1: Show ingest breadth

1. Upload one PDF
2. Ingest one URL
3. Ingest one YouTube link

Expected talking point:

"This assistant can unify heterogeneous knowledge sources into a single retrievable memory."

### Step 2: Ask strong, evaluation-grade questions

Ask these in order:

1. Document-only grounding
     - "From the uploaded PDF, what are the top 5 core topics and where are they discussed?"
2. URL-only extraction
     - "From the webpage I ingested, summarize the main argument, constraints, and practical recommendations."
3. YouTube-only extraction
     - "From the YouTube transcript, what are the speaker's key claims, and what actionable steps are suggested?"
4. Cross-source synthesis
     - "Compare the PDF and YouTube perspectives on this topic. Where do they agree, and where do they differ?"
5. Evidence-backed answer
     - "Give me a final recommendation using all ingested sources and clearly cite the evidence used."

### Step 3: Show quality controls

Ask the same question twice:

- once with summarization on
- once with humanization on

Then highlight the difference in readability and concision.

## Roadmap

- Add reranker layer for retrieval quality
- Add citation spans with page/time offsets
- Add async ingestion pipeline and queueing
- Add benchmark suite (faithfulness, context precision, latency)

## Acknowledgements

This project is a merged and upgraded evolution of:

- `RAG_multi_media_QA_system`
- `RAG-based_Chatbot`
- `Summarization_project`
- `AiHumanizer`
- `project-samarth`

## License

MIT
