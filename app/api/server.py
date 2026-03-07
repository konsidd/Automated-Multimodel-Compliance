"""
app/api/server.py
──────────────────
FastAPI REST API for the compliance orchestration engine.

Endpoints:
  POST /audit/video    – submit a YouTube URL or local path
  POST /audit/text     – submit raw text for compliance audit
  GET  /health         – liveness check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel

from app.compliance.schemas import AuditReport
from app.orchestration.graph import run_pipeline
from app.retrieval.chunking import chunk_document
from app.retrieval.retriever import get_vector_store
from app.telemetry.logging_config import get_logger, setup_logging
from app.utils.config import get_config

cfg = get_config()
setup_logging(cfg.telemetry.log_level)
logger = get_logger(__name__)

app = FastAPI(
    title="Compliance Orchestration Engine",
    version=cfg.version,
    description="Local multimodal compliance auditing via LangGraph + Whisper + FAISS + Ollama",
)


# ── Request / Response models ──────────────────────────────────────────────────

class VideoAuditRequest(BaseModel):
    source: str                   # YouTube URL or local path
    input_type: str = "video"


class TextAuditRequest(BaseModel):
    text: str
    source_label: str = "manual_input"


class AuditResponse(BaseModel):
    source: str
    compliance_score: float
    is_compliant: bool
    summary: str
    results: list[dict]
    error: Optional[str] = None


class IndexRequest(BaseModel):
    file_path: str                # path to a .txt policy document


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": cfg.version}


@app.post("/audit/video", response_model=AuditResponse)
def audit_video(req: VideoAuditRequest):
    """Download + transcribe a video then audit for compliance."""
    logger.info("api_audit_video", source=req.source)
    state = run_pipeline(source=req.source, input_type=req.input_type)

    if state.get("error"):
        raise HTTPException(status_code=500, detail=state["error"])

    report_data = state.get("audit_report")
    if not report_data:
        raise HTTPException(status_code=500, detail="No audit report generated")

    report = AuditReport(**report_data) if isinstance(report_data, dict) else report_data
    return AuditResponse(
        source=report.source,
        compliance_score=report.compliance_score,
        is_compliant=report.is_compliant,
        summary=report.summary,
        results=[r.model_dump() for r in report.results],
    )


@app.post("/audit/text", response_model=AuditResponse)
def audit_text(req: TextAuditRequest):
    """Audit a plain-text snippet for compliance."""
    logger.info("api_audit_text", source=req.source_label)
    state = run_pipeline(source=req.source_label, input_type="text", raw_text=req.text)

    if state.get("error"):
        raise HTTPException(status_code=500, detail=state["error"])

    report_data = state.get("audit_report")
    if not report_data:
        raise HTTPException(status_code=500, detail="No audit report generated")

    report = AuditReport(**report_data) if isinstance(report_data, dict) else report_data
    return AuditResponse(
        source=report.source,
        compliance_score=report.compliance_score,
        is_compliant=report.is_compliant,
        summary=report.summary,
        results=[r.model_dump() for r in report.results],
    )


@app.post("/index/document")
def index_document(req: IndexRequest):
    """Index a plain-text policy document into the FAISS vector store."""
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")

    text = path.read_text(encoding="utf-8")
    chunks = chunk_document(text, source=str(path))
    store = get_vector_store()
    store.add_chunks(chunks)
    store.save()

    logger.info("document_indexed", path=str(path), chunks=len(chunks))
    return {"indexed_chunks": len(chunks), "total_vectors": store.total_vectors}
