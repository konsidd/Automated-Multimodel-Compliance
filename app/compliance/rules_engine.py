"""
app/compliance/rules_engine.py
────────────────────────────────
Load compliance rules from YAML and filter those relevant
to a piece of content based on keyword matching.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from app.compliance.schemas import ComplianceRule, Severity
from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_all_rules() -> list[ComplianceRule]:
    """Load and cache all rules from configs/rules.yaml."""
    cfg = get_config()
    rules_path = cfg.root_dir / cfg.compliance.rules_file

    with open(rules_path) as f:
        raw = yaml.safe_load(f)

    rules = [
        ComplianceRule(
            id=r["id"],
            category=r["category"],
            description=r["description"],
            keywords=r.get("keywords", []),
            severity=Severity(r.get("severity", "medium")),
        )
        for r in raw.get("rules", [])
    ]
    logger.info("rules_loaded", count=len(rules))
    return rules


def filter_relevant_rules(content: str, rules: list[ComplianceRule] | None = None) -> list[ComplianceRule]:
    """
    Return rules whose keywords appear in the content.
    If no rules match via keywords, return ALL rules (safe fallback).
    """
    all_rules = rules or load_all_rules()
    content_lower = content.lower()

    matched = [
        rule for rule in all_rules
        if any(kw.lower() in content_lower for kw in rule.keywords)
    ]

    if not matched:
        logger.info("no_keyword_match_returning_all_rules")
        return all_rules

    logger.info("rules_filtered", matched=len(matched), total=len(all_rules))
    return matched


def format_rules_for_prompt(rules: list[ComplianceRule]) -> str:
    """Render rules as a numbered list for the LLM prompt."""
    lines = []
    for r in rules:
        lines.append(
            f"- Rule {r.id} [{r.severity.upper()}] ({r.category}): {r.description}"
        )
    return "\n".join(lines)
