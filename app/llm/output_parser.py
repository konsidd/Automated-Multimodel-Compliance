"""
app/llm/output_parser.py
────────────────────────
Parse structured JSON output from the LLM.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.telemetry.logging_config import get_logger

logger = get_logger(__name__)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    return re.sub(r"```(?:json)?\s*([\s\S]*?)```", r"\1", text).strip()


def parse_json_response(raw: str) -> Any:
    """
    Safely parse a JSON response from the LLM.
    Handles markdown fences and trailing commas.
    Raises ValueError on failure.
    """
    cleaned = _strip_markdown_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("json_parse_failed", error=str(e), raw_snippet=cleaned[:200])
        # Last-ditch: find first JSON array or object
        match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", cleaned)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse LLM JSON output: {cleaned[:300]}") from e


def parse_compliance_results(raw: str) -> list[dict[str, str]]:
    """Parse the compliance auditor JSON array."""
    data = parse_json_response(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of compliance results, got: {type(data)}")
    return data
