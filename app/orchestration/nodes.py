"""
app/orchestration/nodes.py
───────────────────────────
All LangGraph node functions.
Each node receives and returns a PipelineState dict.
"""

from __future__ import annotations

from app.compliance.evaluator import audit_content
from app.ingestion.normalizer import normalise_document
from app.ingestion.ocr_processor import ocr_video
from app.ingestion.video_transcriber import transcribe_local_file, transcribe_youtube_video
from app.orchestration.state import PipelineState
from app.retrieval.retriever import retrieve_context
from app.telemetry.logging_config import get_logger

logger = get_logger(__name__)


# ── Helper ────────────────────────────────────────────────────────────────────

def _is_youtube_url(source: str) -> bool:
    return "youtube.com" in source or "youtu.be" in source


# ── Node 1: Video Processor ───────────────────────────────────────────────────

def node_ingest_video(state: PipelineState) -> PipelineState:
    """
    Download (if YouTube URL) and transcribe the video with Whisper.
    Also runs OCR on video frames.

    The graph infrastructure expects a `PipelineState` object rather than a
    plain dictionary.  Previously we reconstructed the model from a dict and
    then returned a dict, which confused the type checker when the function was
    passed to `StateGraph.add_node`.  Using `PipelineState` directly yields
    correct signatures.
    """
    s = state
    logger.info("node_ingest_video", source=s.source)

    try:
        if s.input_type == "text" and s.raw_text:
            # No ingestion needed – use provided text
            s.combined_text = s.raw_text
            return s

        source = s.source

        # Transcribe
        if _is_youtube_url(source):
            transcript = transcribe_youtube_video(source)
        else:
            transcript = transcribe_local_file(source)

        # OCR (video frames)
        ocr_text = ""
        if not _is_youtube_url(source):  # OCR only on local video files
            try:
                ocr_text = ocr_video(source)
            except Exception as e:
                logger.warning("ocr_skipped", reason=str(e))

        # Normalise
        doc = normalise_document(source=source, transcript=transcript, ocr_text=ocr_text)
        s.transcript = doc.transcript
        s.ocr_text = doc.ocr_text
        s.combined_text = doc.combined_text

    except Exception as e:
        logger.error("node_ingest_video_error", error=str(e))
        s.error = f"Ingestion failed: {e}"

    return s


# ── Node 2: Retrieval Engine ──────────────────────────────────────────────────

def node_retrieve_context(state: PipelineState) -> PipelineState:
    """Retrieve relevant policy context from the FAISS vector store."""
    s = state
    if s.error:
        return s

    logger.info("node_retrieve_context")
    try:
        s.retrieved_context = retrieve_context(s.combined_text, rewrite=True)
    except Exception as e:
        logger.error("node_retrieve_error", error=str(e))
        s.retrieved_context = ""   # non-fatal – continue without context

    return s


# ── Node 3: Compliance Auditor ────────────────────────────────────────────────

def node_compliance_audit(state: PipelineState) -> PipelineState:
    """Run the LLM-based compliance audit."""
    s = state
    if s.error:
        return s

    logger.info("node_compliance_audit", source=s.source)
    try:
        s.audit_report = audit_content(s.combined_text, source=s.source)
    except Exception as e:
        logger.error("node_compliance_error", error=str(e))
        s.error = f"Compliance audit failed: {e}"

    return s


# ── Node 4: Terminal / output formatter ───────────────────────────────────────

def node_output(state: PipelineState) -> PipelineState:
    """Log and return the final report."""
    s = state
    if s.audit_report:
        logger.info(
            "pipeline_complete",
            score=s.audit_report.compliance_score,
            compliant=s.audit_report.is_compliant,
        )
    elif s.error:
        logger.error("pipeline_failed", error=s.error)
    return s
