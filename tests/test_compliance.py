"""
tests/test_compliance.py
─────────────────────────
Unit tests for compliance rules engine and output parsing.
"""

from __future__ import annotations

import pytest

from app.compliance.rules_engine import filter_relevant_rules, load_all_rules, format_rules_for_prompt
from app.compliance.schemas import AuditReport, ComplianceStatus, RuleResult, Severity
from app.llm.output_parser import parse_compliance_results, parse_json_response


# ── Rules Engine ─────────────────────────────────────────────────────────────

def test_load_all_rules():
    rules = load_all_rules()
    assert len(rules) > 0
    for r in rules:
        assert r.id
        assert r.description


def test_filter_relevant_rules_gdpr():
    rules = load_all_rules()
    content = "We collect personal data from users without their consent."
    matched = filter_relevant_rules(content, rules)
    ids = [r.id for r in matched]
    assert any("GDPR" in rid for rid in ids)


def test_filter_relevant_rules_no_match_returns_all():
    rules = load_all_rules()
    content = "The sky is blue today."
    matched = filter_relevant_rules(content, rules)
    assert len(matched) == len(rules)


def test_format_rules_for_prompt():
    rules = load_all_rules()[:2]
    prompt_text = format_rules_for_prompt(rules)
    assert "Rule" in prompt_text
    assert rules[0].id in prompt_text


# ── Output Parser ─────────────────────────────────────────────────────────────

def test_parse_json_response_clean():
    raw = '[{"rule_id": "GDPR-001", "status": "FAILS", "reason": "No consent", "severity": "critical"}]'
    result = parse_json_response(raw)
    assert isinstance(result, list)
    assert result[0]["rule_id"] == "GDPR-001"


def test_parse_json_response_fenced():
    raw = '```json\n[{"rule_id": "SOC2-001", "status": "PASSES", "reason": "Good", "severity": "high"}]\n```'
    result = parse_compliance_results(raw)
    assert result[0]["status"] == "PASSES"


def test_parse_json_response_invalid_raises():
    with pytest.raises(ValueError):
        parse_json_response("this is not json at all!!!")


# ── AuditReport scoring ───────────────────────────────────────────────────────

def test_audit_report_score_all_pass():
    report = AuditReport(
        source="test",
        results=[
            RuleResult(rule_id="R1", status=ComplianceStatus.PASSES, reason="ok", severity=Severity.HIGH),
            RuleResult(rule_id="R2", status=ComplianceStatus.PASSES, reason="ok", severity=Severity.MEDIUM),
        ],
    )
    report.compute_score()
    assert report.compliance_score == 1.0


def test_audit_report_score_all_fail():
    report = AuditReport(
        source="test",
        results=[
            RuleResult(rule_id="R1", status=ComplianceStatus.FAILS, reason="bad", severity=Severity.CRITICAL),
        ],
    )
    report.compute_score()
    assert report.compliance_score == 0.0


def test_audit_report_score_mixed():
    report = AuditReport(
        source="test",
        results=[
            RuleResult(rule_id="R1", status=ComplianceStatus.PASSES, reason="ok", severity=Severity.HIGH),   # weight 3
            RuleResult(rule_id="R2", status=ComplianceStatus.FAILS, reason="bad", severity=Severity.LOW),    # weight 1
        ],
    )
    report.compute_score()
    # pass_weight = 3, total = 4 → 0.75
    assert report.compliance_score == 0.75
