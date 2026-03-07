"""
app/orchestration/router.py
────────────────────────────
Conditional edge routing functions for LangGraph.
"""

from __future__ import annotations

from app.orchestration.state import PipelineState


def route_after_ingestion(state: dict) -> str:
    """
    After ingestion:
      - If error → end
      - If combined_text is empty → end
      - Otherwise → retrieve context
    """
    s = PipelineState(**state)
    if s.error or not s.combined_text.strip():
        return "output"
    return "retrieve"


def route_after_retrieval(state: dict) -> str:
    """After retrieval always proceed to compliance audit."""
    s = PipelineState(**state)
    if s.error:
        return "output"
    return "audit"
