"""
app/orchestration/graph.py
───────────────────────────
Builds and compiles the LangGraph compliance pipeline.

Graph topology:

  [START]
     │
     ▼
  ingest_video
     │
     ├─(error / empty)──► output ──► [END]
     │
     ▼
  retrieve_context
     │
     ├─(error)──────────► output ──► [END]
     │
     ▼
  compliance_audit
     │
     ▼
  output
     │
     ▼
  [END]
"""

from __future__ import annotations

from functools import lru_cache
from langgraph.graph import StateGraph, START, END

from app.orchestration.state import PipelineState
from app.orchestration.router import route_after_ingestion, route_after_retrieval

from app.orchestration.nodes import (
    node_ingest_video,
    node_retrieve_context,
    node_compliance_audit,
    node_output
)

from app.telemetry.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def build_graph():

    builder = StateGraph(PipelineState)

    # Nodes
    builder.add_node("ingest", node_ingest_video)
    builder.add_node("retrieve", node_retrieve_context)
    builder.add_node("audit", node_compliance_audit)
    builder.add_node("output", node_output)

    # Edges
    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "retrieve")
    builder.add_edge("retrieve", "audit")
    builder.add_edge("audit", "output")
    builder.add_edge("output", END)

    return builder.compile()


def run_pipeline(
    source: str,
    input_type: str = "video",
    raw_text: str = "",
) -> dict:
    """
    Execute the compliance pipeline.

    Args:
        source:     YouTube URL, local file path, or label string for raw text.
        input_type: "video" | "text" | "image"
        raw_text:   Pre-extracted text (skips ingestion if provided).

    Returns:
        Final pipeline state dict (includes audit_report).
    """
    initial_state = PipelineState(
        source=source,
        input_type=input_type,
        raw_text=raw_text,
    )

    graph = build_graph()
    # Invoke the compiled graph.  The type checker currently infers the
    # return value as a `dict` even though we know it will be a
    # `PipelineState`, so cast to keep Pylance happy.
    from typing import cast

    final_state = cast(PipelineState, graph.invoke(initial_state))
    # the caller still wants a plain dict, so dump the model before returning
    return final_state.model_dump()
