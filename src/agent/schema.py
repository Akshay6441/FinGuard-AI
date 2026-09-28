"""
Data models and Pydantic schemas for the FinGuard AI Autonomous Investigator Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

RecommendedAction = Literal[
    "APPROVE",
    "FLAG_FOR_MANUAL_REVIEW",
    "REQUEST_ADDITIONAL_KYC",
    "BLOCK_TRANSACTION",
    "FREEZE_ACCOUNT",
]


class PolicyCitation(BaseModel):
    policy_name: str = Field(..., description="Name of policy or regulatory document")
    section: str = Field(..., description="Section number and title")
    relevant_clause: str = Field(..., description="Exact rule or standard applied")
    citation_id: Optional[str] = None


class InvestigationEvidence(BaseModel):
    category: str = Field(..., description="Category of evidence (e.g. ML_SCORE, VELOCITY, INVOICE, GEO)")
    finding: str = Field(..., description="Specific finding or quantitative measurement")
    severity: Literal["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"] = "INFO"


class AgentStepTrace(BaseModel):
    step_number: int
    thought: str
    tool_called: str
    tool_input: Dict[str, Any]
    tool_output_summary: str


class CaseInvestigationReport(BaseModel):
    case_id: str
    transaction_id: str
    user_id: str
    timestamp: str
    risk_level: RiskLevel
    recommended_action: RecommendedAction
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    investigation_summary: str
    evidence_list: List[InvestigationEvidence]
    policy_citations: List[PolicyCitation]
    tool_traces: List[AgentStepTrace] = []
    requires_human_escalation: bool = False
    sla_minutes: int = 45
