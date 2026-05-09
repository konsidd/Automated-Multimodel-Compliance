# 🛡️ Automated Compliance Orchestrator

**Azure Multi-Modal Compliance Ingestion Engine** — rebuilt with **fully local tools**:
Whisper · Tesseract OCR · FAISS · Ollama · LangGraph · LangSmith

---

## Architecture

```
Entry Points          Orchestration (LangGraph)          Local Infra
─────────────         ──────────────────────────         ────────────────
main.py (CLI)  ───►  RAG Workflow (graph.py)    ───►    FAISS Vector Index
FastAPI Server ───►  Video Processor            ───►    Whisper (transcribe)
               ───►  Retrieval Engine           ───►    Tesseract (OCR)
               ───►  Compliance Auditor         ───►    Ollama LLM
                                                         sentence-transformers

External Intelligence + Observability
──────────────────────────────────────
Ollama (LLM + Embeddings) · LangSmith Tracing · structlog/rich
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.10+
pip install -e ".[dev]"

# Tesseract OCR
sudo apt install tesseract-ocr          # Ubuntu/Debian
brew install tesseract                  # macOS

# Ollama (local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3                      # or any model you prefer

# yt-dlp (video download)
pip install yt-dlp
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env – add LANGCHAIN_API_KEY if you want LangSmith tracing
# Edit configs/settings.yaml to change LLM model, Whisper size, etc.
```

### 3. Index your policies

```bash
# Place .txt policy documents in data/policies/
python scripts/index_policies.py
```

### 4. Run a compliance audit

```bash
# Audit a YouTube video
python scripts/run_pipeline.py --source "https://www.youtube.com/watch?v=XXXX"

# Audit a local video file
python scripts/run_pipeline.py --source ./data/raw_videos/demo.mp4

# Audit plain text
python scripts/run_pipeline.py --type text \
  --text "We store personal data without user consent."

# Save JSON report
python scripts/run_pipeline.py --type text --text "..." --output report.json
```

### 5. Start the API server

```bash
uvicorn app.api.server:app --host 127.0.0.1 --port 8000 --reload
```

Or on Windows, run:

```powershell
run_server.bat
```

API docs available at `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness check |
| `POST` | `/audit/video` | Audit a YouTube URL or local video |
| `POST` | `/audit/text` | Audit raw text |
| `POST` | `/index/document` | Index a policy `.txt` file |

### Example – Audit text

```bash
curl -X POST http://localhost:8000/audit/text \
  -H "Content-Type: application/json" \
  -d '{"text": "We collect your credit card number in plain text.", "source_label": "test"}'
```

---

## Project Structure

```
automated-compliance-orchestrator/
├── app/
│   ├── orchestration/       # LangGraph pipeline
│   │   ├── state.py         # PipelineState (Pydantic)
│   │   ├── nodes.py         # ingest / retrieve / audit / output nodes
│   │   ├── graph.py         # graph.compile() + run_pipeline()
│   │   └── router.py        # conditional edge functions
│   ├── compliance/
│   │   ├── schemas.py       # ComplianceRule, RuleResult, AuditReport
│   │   ├── rules_engine.py  # YAML rules loader + keyword filter
│   │   └── evaluator.py     # LLM-based compliance audit
│   ├── retrieval/
│   │   ├── chunking.py      # Overlap chunker
│   │   ├── embeddings.py    # sentence-transformers (local)
│   │   ├── vector_store.py  # FAISS index + JSON metadata
│   │   └── retriever.py     # query rewrite + RAG search
│   ├── ingestion/
│   │   ├── video_transcriber.py  # yt-dlp + Whisper
│   │   ├── ocr_processor.py      # Tesseract on video frames
│   │   └── normalizer.py         # merge transcript + OCR
│   ├── llm/
│   │   ├── client.py        # Ollama / OpenAI abstraction
│   │   ├── prompts.py       # all prompt templates
│   │   └── output_parser.py # JSON response parser
│   ├── api/
│   │   └── server.py        # FastAPI endpoints
│   └── telemetry/
│       ├── logging_config.py  # structlog + rich
│       └── tracing.py         # LangSmith tracing
├── configs/
│   ├── settings.yaml          # all configuration
│   └── rules.yaml             # GDPR / HIPAA / SOC2 / PCI rules
├── data/
│   ├── policies/              # .txt policy documents to index
│   ├── transcripts/           # Whisper output cache
│   ├── raw_videos/            # yt-dlp download cache
│   └── vector_index/          # FAISS .index + metadata.json
├── scripts/
│   ├── run_pipeline.py        # CLI entry point
│   └── index_policies.py      # policy indexer
└── tests/
    ├── test_retrieval.py
    └── test_compliance.py
```

---

## Configuration (configs/settings.yaml)

| Key | Default | Description |
|-----|---------|-------------|
| `llm.provider` | `ollama` | `ollama` or `openai` |
| `llm.model` | `llama3` | Any model pulled in Ollama |
| `embeddings.model` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `ingestion.whisper_model` | `base` | `tiny/base/small/medium/large` |
| `compliance.score_threshold` | `0.75` | Minimum passing compliance score |
| `vector_store.top_k` | `5` | Retrieved context chunks |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Tool | Replaces Azure |
|-----------|------|----------------|
| Speech-to-Text | OpenAI Whisper (local) | Azure Speech / Video Indexer |
| OCR | Tesseract (local) | Azure Computer Vision |
| Vector DB | FAISS (local) | Azure AI Search |
| LLM | Ollama / llama3 (local) | Azure OpenAI |
| Embeddings | sentence-transformers | Azure OpenAI Embeddings |
| Orchestration | LangGraph | – |
| Tracing | LangSmith | Azure Application Insights |
| Video Download | yt-dlp | Azure Blob Storage temp |
