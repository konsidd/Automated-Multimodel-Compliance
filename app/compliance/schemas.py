"""
app/compliance/schemas.py
──────────────────────────
Pydantic models for compliance rules and audit results.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplianceStatus(str, Enum):
    PASSES = "PASSES"
    FAILS = "FAILS"
    UNCLEAR = "UNCLEAR"


class ComplianceRule(BaseModel):
    id: str
    category: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    severity: Severity = Severity.MEDIUM


class RuleResult(BaseModel):
    rule_id: str
    status: ComplianceStatus
    reason: str
    severity: Severity = Severity.MEDIUM


class AuditReport(BaseModel):
    source: str
    results: list[RuleResult] = Field(default_factory=list)
    summary: str = ""
    compliance_score: float = 0.0        # 0.0 – 1.0
    is_compliant: bool = False

    def compute_score(self) -> None:
        """Calculate compliance score and set is_compliant flag."""
        if not self.results:
            self.compliance_score = 1.0
            self.is_compliant = True
            return

        weights = {Severity.CRITICAL: 4, Severity.HIGH: 3, Severity.MEDIUM: 2, Severity.LOW: 1}
        total_weight = sum(weights[r.severity] for r in self.results)
        pass_weight = sum(
            weights[r.severity] for r in self.results if r.status == ComplianceStatus.PASSES
        )
        self.compliance_score = round(pass_weight / total_weight, 4) if total_weight else 1.0
        threshold = 0.75  # default; overridden from config in evaluator
        self.is_compliant = self.compliance_score >= threshold
