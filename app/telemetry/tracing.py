"""
app/telemetry/tracing.py
────────────────────────
Optional LangSmith tracing initialisation.
Set LANGCHAIN_API_KEY in .env to enable.
"""

from __future__ import annotations

import os

from app.telemetry.logging_config import get_logger

logger = get_logger(__name__)


def setup_tracing(project_name: str, enable: bool = True) -> None:
    """Configure LangSmith tracing if API key is present."""
    if not enable:
        logger.info("tracing_disabled")
        return

    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        logger.warning("langsmith_api_key_missing", hint="Set LANGCHAIN_API_KEY in .env to enable tracing")
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name
    logger.info("langsmith_tracing_enabled", project=project_name)
