"""
app/orchestration/state.py
───────────────────────────
Typed state object passed between all LangGraph nodes.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.compliance.schemas import AuditReport


class PipelineState(BaseModel):
    """
    Mutable state that flows through the LangGraph pipeline.
    Each node reads from and writes to this object.
    """

    # ── Input ────────────────────────────────────────────────────────────────
    source: str = ""                   # YouTube URL or local file path
    input_type: str = "video"          # "video" | "text" | "image"
    raw_text: str = ""                 # pre-provided text (skip ingestion)

    # ── Ingestion outputs ────────────────────────────────────────────────────
    transcript: str = ""
    ocr_text: str = ""
    combined_text: str = ""

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieved_context: str = ""

    # ── Compliance audit ─────────────────────────────────────────────────────
    audit_report: Optional[AuditReport] = None

    # ── Control flow ─────────────────────────────────────────────────────────
    error: Optional[str] = None
    next_node: Optional[str] = None    # used by router

    class Config:
        arbitrary_types_allowed = True
