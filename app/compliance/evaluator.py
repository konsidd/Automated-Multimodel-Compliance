"""
app/compliance/evaluator.py
────────────────────────────
Orchestrates a compliance audit:
  1. Filter relevant rules
  2. Retrieve policy context via RAG
  3. Call LLM to evaluate content
  4. Parse and return AuditReport
"""

from __future__ import annotations

from app.compliance.rules_engine import filter_relevant_rules, format_rules_for_prompt
from app.compliance.schemas import AuditReport, ComplianceStatus, RuleResult, Severity
from app.llm.client import invoke_llm
from app.llm.output_parser import parse_compliance_results
from app.llm.prompts import COMPLIANCE_AUDITOR_SYSTEM, COMPLIANCE_AUDITOR_USER, SUMMARY_SYSTEM, SUMMARY_USER
from app.retrieval.retriever import retrieve_context
from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


def audit_content(content: str, source: str = "unknown") -> AuditReport:
    """
    Run a full compliance audit on the given content.

    Steps:
      1. Keyword-filter relevant rules
      2. Retrieve matching policy context (RAG)
      3. Ask LLM to evaluate each rule
      4. Build and return AuditReport
    """
    cfg = get_config()

    # 1. Rules
    rules = filter_relevant_rules(content)
    rules_text = format_rules_for_prompt(rules)
    logger.info("audit_start", source=source, rules=len(rules))

    # 2. RAG context
    context = retrieve_context(content, rewrite=True)

    # 3. LLM evaluation
    user_prompt = COMPLIANCE_AUDITOR_USER.format(
        content=content[:4000],   # truncate very long content
        rules=rules_text,
        context=context[:2000],
    )

    raw_response = invoke_llm(
        system_prompt=COMPLIANCE_AUDITOR_SYSTEM,
        user_prompt=user_prompt,
    )

    # 4. Parse results
    try:
        parsed = parse_compliance_results(raw_response)
    except ValueError as e:
        logger.error("parse_error", error=str(e))
        parsed = []

    rule_results: list[RuleResult] = []
    for item in parsed:
        try:
            rule_results.append(RuleResult(
                rule_id=item.get("rule_id", "UNKNOWN"),
                status=ComplianceStatus(item.get("status", "UNCLEAR")),
                reason=item.get("reason", ""),
                severity=Severity(item.get("severity", "medium")),
            ))
        except Exception as e:
            logger.warning("result_parse_skip", item=item, error=str(e))

    # 5. Build report
    report = AuditReport(source=source, results=rule_results)
    report.compute_score()
    report.is_compliant = report.compliance_score >= cfg.compliance.score_threshold

    # 6. Generate executive summary
    import json
    try:
        summary_raw = invoke_llm(
            system_prompt=SUMMARY_SYSTEM,
            user_prompt=SUMMARY_USER.format(audit_results=json.dumps([r.model_dump() for r in rule_results], indent=2)),
        )
        report.summary = summary_raw
    except Exception as e:
        logger.error("summary_generation_failed", error=str(e))
        report.summary = "Summary generation failed due to LLM error."

    logger.info(
        "audit_complete",
        source=source,
        score=report.compliance_score,
        compliant=report.is_compliant,
        fails=[r.rule_id for r in rule_results if r.status == ComplianceStatus.FAILS],
    )
    return report
