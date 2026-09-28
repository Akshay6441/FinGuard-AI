"""
Unit test suite for the autonomous fraud investigator agent.
"""

from __future__ import annotations

from src.agent.investigator import FraudInvestigatorAgent
from src.agent.schema import CaseInvestigationReport


def test_agent_investigation_routine_approval():
    agent = FraudInvestigatorAgent()
    # TXN_0015301 is a normal $45 grocery spend
    report = agent.investigate("TXN_0015301")
    assert isinstance(report, CaseInvestigationReport)
    assert report.risk_level in ["LOW", "MEDIUM"]
    assert report.recommended_action in ["APPROVE", "REQUEST_ADDITIONAL_KYC"]
    assert report.confidence_score >= 0.85
    assert len(report.tool_traces) > 0


def test_agent_investigation_high_risk():
    agent = FraudInvestigatorAgent()
    # TXN_0000001 is a high risk account takeover with foreign IP
    report = agent.investigate("TXN_0000001")
    assert isinstance(report, CaseInvestigationReport)
    assert report.risk_level in ["HIGH", "CRITICAL"]
    assert report.recommended_action in ["BLOCK_TRANSACTION", "FREEZE_ACCOUNT", "FLAG_FOR_MANUAL_REVIEW"]
    assert len(report.evidence_list) > 0
    assert len(report.policy_citations) > 0


def test_agent_not_found_handling():
    agent = FraudInvestigatorAgent()
    report = agent.investigate("TXN_NON_EXISTENT_9999")
    assert report.risk_level == "HIGH"
    assert report.requires_human_escalation is True
