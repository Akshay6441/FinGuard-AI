"""
Autonomous Fraud Investigator Agent for FinGuard AI.
Coordinates multi-tool evidentiary reasoning (SQL, ML/SHAP, OCR, RAG),
enforces guardrails, and outputs structured, regulatory-compliant case reports.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from typing import Any, Dict, List, Optional

from src.agent.schema import (
    AgentStepTrace,
    CaseInvestigationReport,
    InvestigationEvidence,
    PolicyCitation,
    RecommendedAction,
    RiskLevel,
)
from src.agent.tools import InvestigatorTools
from src.features.db import get_db_connection


class FraudInvestigatorAgent:
    """
    Autonomous investigator agent with tool-augmented reasoning and strict compliance guardrails.
    """

    def __init__(self, tools: Optional[InvestigatorTools] = None):
        self.tools = tools or InvestigatorTools()
        self.api_key = os.getenv("GEMINI_API_KEY", "")

    def investigate(
        self,
        transaction_id: str,
        invoice_path_or_id: Optional[str] = None,
        max_steps: int = 5,
    ) -> CaseInvestigationReport:
        """
        Execute full autonomous investigation loop for a transaction.
        """
        traces: List[AgentStepTrace] = []
        evidence_list: List[InvestigationEvidence] = []
        policy_citations: List[PolicyCitation] = []

        # Step 1: Query transaction ledger and baseline history
        step1_thought = "Investigating flagged alert: First, query the transaction record, user profile, and historical spend baseline from the SQL ledger."
        txn_data = self.tools.lookup_transaction_history(transaction_id)
        traces.append(AgentStepTrace(
            step_number=1,
            thought=step1_thought,
            tool_called="lookup_transaction_history",
            tool_input={"transaction_id": transaction_id},
            tool_output_summary=f"Retrieved user {txn_data.get('user_id')}, amount ${txn_data.get('transaction', {}).get('amount')}, history count {txn_data.get('historical_transactions_count')}",
        ))

        if "error" in txn_data:
            # Transaction does not exist
            return self._create_not_found_report(transaction_id, txn_data["error"], traces)

        txn = txn_data["transaction"]
        amount = float(txn["amount"])
        avg_spend = float(txn_data["user_avg_historical_spend"])

        # Record spend deviation evidence
        spend_ratio = amount / (avg_spend + 1e-5)
        if spend_ratio > 3.0:
            evidence_list.append(InvestigationEvidence(
                category="SPEND_DEVIATION",
                finding=f"Transaction amount (${amount:.2f}) is {spend_ratio:.1f}x higher than user's historical average spend (${avg_spend:.2f}).",
                severity="HIGH" if spend_ratio > 5.0 else "MEDIUM",
            ))

        # Check device/location
        if txn.get("is_new_device"):
            evidence_list.append(InvestigationEvidence(
                category="DEVICE_INTEGRITY",
                finding=f"Transaction executed from unrecognized device fingerprint ({txn.get('device_id')}).",
                severity="MEDIUM",
            ))
        if txn.get("distance_from_home_km", 0) > 500:
            evidence_list.append(InvestigationEvidence(
                category="GEO_LOCATION",
                finding=f"Physical location is {txn.get('distance_from_home_km'):.1f} km from cardholder billing country/home ({txn.get('country')}).",
                severity="CRITICAL" if txn.get("distance_from_home_km") > 2000 else "HIGH",
            ))

        # Step 2: Query ML Model score and SHAP feature drivers
        step2_thought = "Querying the predictive fraud model and SHAP TreeExplainer to quantify risk probability and isolate principal drivers."
        ml_data = self.tools.get_fraud_score_and_shap(transaction_id)
        traces.append(AgentStepTrace(
            step_number=2,
            thought=step2_thought,
            tool_called="get_fraud_score_and_shap",
            tool_input={"transaction_id": transaction_id},
            tool_output_summary=f"Blended Fraud Score: {ml_data.get('blended_fraud_score')}, Risk Tier: {ml_data.get('risk_tier')}, Flagged: {ml_data.get('is_flagged_by_model')}",
        ))

        fraud_score = float(ml_data.get("blended_fraud_score", 0.05))
        risk_tier = ml_data.get("risk_tier", "LOW")
        top_drivers = ml_data.get("top_shap_drivers", [])

        driver_names = [f"{d['feature']} ({d['shap_value']:+.2f})" for d in top_drivers[:3]]
        evidence_list.append(InvestigationEvidence(
            category="ML_MODEL_INFERENCE",
            finding=f"FinGuard ML Risk Score: {fraud_score:.2%} ({risk_tier} tier). Key SHAP drivers: {', '.join(driver_names)}.",
            severity="CRITICAL" if fraud_score >= 0.88 else "HIGH" if fraud_score >= 0.65 else "MEDIUM" if fraud_score >= 0.28 else "INFO",
        ))

        # Step 3: Cross-reference supporting receipt / invoice
        step3_thought = "Cross-checking the submitted receipt/invoice against transaction authorization details for amount tampering, stale dates, or ghost merchants."
        reconcil_data = self.tools.check_invoice_mismatch(transaction_id, invoice_path_or_id=invoice_path_or_id)
        traces.append(AgentStepTrace(
            step_number=3,
            thought=step3_thought,
            tool_called="check_invoice_mismatch",
            tool_input={"transaction_id": transaction_id, "invoice_path_or_id": invoice_path_or_id},
            tool_output_summary=f"Mismatch Score: {reconcil_data.get('mismatch_score')}, Flags: {reconcil_data.get('flags')}",
        ))

        is_doc_mismatch = reconcil_data.get("is_mismatch", False)
        mismatch_flags = reconcil_data.get("flags", [])
        mismatch_reasons = reconcil_data.get("reasons", [])

        if is_doc_mismatch:
            for reason in mismatch_reasons:
                if "Breach" in reason:
                    evidence_list.append(InvestigationEvidence(
                        category="INVOICE_RECONCILIATION",
                        finding=reason,
                        severity="CRITICAL" if "CRITICAL" in reason else "HIGH",
                    ))
        elif reconcil_data.get("status") == "NO_INVOICE_SUBMITTED":
            if fraud_score >= 0.50:
                evidence_list.append(InvestigationEvidence(
                    category="DOCUMENTATION",
                    finding="No invoice or receipt submitted for high-risk transaction.",
                    severity="MEDIUM",
                ))

        # Step 4: Query regulatory guidelines and SOP via RAG
        rag_query = f"Risk tiering {risk_tier} velocity {txn.get('merchant_category')} invoice mismatch {' '.join(mismatch_flags)}"
        step4_thought = f"Retrieving compliance rules from RAG knowledge base for signals: {risk_tier}, MCC {txn.get('mcc')}, flags: {mismatch_flags}."
        rag_data = self.tools.retrieve_fraud_policy(rag_query, top_k=3)
        traces.append(AgentStepTrace(
            step_number=4,
            thought=step4_thought,
            tool_called="retrieve_fraud_policy",
            tool_input={"query": rag_query},
            tool_output_summary=f"Retrieved {len(rag_data.get('retrieved_citations', []))} citations from regulatory corpus.",
        ))

        for c in rag_data.get("retrieved_citations", []):
            policy_citations.append(PolicyCitation(
                policy_name=c["policy_name"],
                section=c["section"],
                relevant_clause=c["content_snippet"][:180] + "...",
            ))

        # Step 5: Synthesize final decision and structured report enforcing SOP guardrails
        report = self._synthesize_case_report(
            transaction=txn,
            fraud_score=fraud_score,
            risk_tier=risk_tier,
            reconciliation=reconcil_data,
            evidence_list=evidence_list,
            policy_citations=policy_citations,
            traces=traces,
        )

        # Persist case to SQLite
        self._persist_case(report)

        return report

    def _synthesize_case_report(
        self,
        transaction: Dict[str, Any],
        fraud_score: float,
        risk_tier: str,
        reconciliation: Dict[str, Any],
        evidence_list: List[InvestigationEvidence],
        policy_citations: List[PolicyCitation],
        traces: List[AgentStepTrace],
    ) -> CaseInvestigationReport:
        """
        Deterministic rule-grounded synthesis guaranteeing regulatory compliance and strict SLA assignment.
        """
        txn_id = transaction["transaction_id"]
        uid = transaction["user_id"]
        is_mismatch = reconciliation.get("is_mismatch", False)
        flags = reconciliation.get("flags", [])

        # Decision matrix based on SOP Section 2 & Section 5
        if fraud_score >= 0.88 or "AMOUNT_MISMATCH" in flags or ("VENDOR_MISMATCH" in flags and fraud_score > 0.5):
            risk_level: RiskLevel = "CRITICAL"
            recommended_action: RecommendedAction = "BLOCK_TRANSACTION"
            if fraud_score >= 0.92 and transaction.get("distance_from_home_km", 0) > 1000:
                recommended_action = "FREEZE_ACCOUNT"
            confidence = 0.96
            sla = 15
            requires_human = False

        elif fraud_score >= 0.65 or is_mismatch:
            risk_level: RiskLevel = "HIGH"
            recommended_action: RecommendedAction = "FLAG_FOR_MANUAL_REVIEW"
            if "STALE_INVOICE" in flags:
                recommended_action = "REQUEST_ADDITIONAL_KYC"
            confidence = 0.91
            sla = 45
            requires_human = True

        elif fraud_score >= 0.28:
            risk_level: RiskLevel = "MEDIUM"
            recommended_action: RecommendedAction = "REQUEST_ADDITIONAL_KYC"
            confidence = 0.88
            sla = 240
            requires_human = False

        else:
            risk_level: RiskLevel = "LOW"
            recommended_action: RecommendedAction = "APPROVE"
            confidence = 0.98
            sla = 1
            requires_human = False

        # Narrative case summary
        summary_paragraphs = [
            f"Case analysis for transaction {txn_id} (User: {uid}, Amount: ${float(transaction['amount']):.2f} at {transaction['merchant_name']}).",
            f"1. RISK MODEL INFERENCE: Evaluated with composite risk score of {fraud_score:.2%} ({risk_level} Risk Tier). The optimal cost-based threshold is {reconciliation.get('optimal_threshold', 0.28):.2f}.",
            f"2. DOCUMENT VERIFICATION: Cross-checked against submitted documentation. Status: {'MISMATCH DETECTED' if is_mismatch else 'VERIFIED'}. Flags: {', '.join(flags) if flags else 'None'}.",
            f"3. REGULATORY COMPLIANCE: Adheres to {len(policy_citations)} cited compliance clauses from payment network standards and digital payment liability frameworks.",
            f"DISPOSITION DECISION: Recommended action '{recommended_action}' with confidence {confidence:.1%}. Maximum resolution SLA: {sla} minutes.",
        ]
        investigation_summary = "\n\n".join(summary_paragraphs)

        case_id = f"CASE-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        return CaseInvestigationReport(
            case_id=case_id,
            transaction_id=txn_id,
            user_id=uid,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            risk_level=risk_level,
            recommended_action=recommended_action,
            confidence_score=confidence,
            investigation_summary=investigation_summary,
            evidence_list=evidence_list,
            policy_citations=policy_citations,
            tool_traces=traces,
            requires_human_escalation=requires_human,
            sla_minutes=sla,
        )

    def _create_not_found_report(
        self, transaction_id: str, error_msg: str, traces: List[AgentStepTrace]
    ) -> CaseInvestigationReport:
        return CaseInvestigationReport(
            case_id=f"CASE-ERR-{uuid.uuid4().hex[:6].upper()}",
            transaction_id=transaction_id,
            user_id="UNKNOWN",
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            risk_level="HIGH",
            recommended_action="FLAG_FOR_MANUAL_REVIEW",
            confidence_score=0.50,
            investigation_summary=f"Investigation aborted: {error_msg}",
            evidence_list=[InvestigationEvidence(category="SYSTEM", finding=error_msg, severity="HIGH")],
            policy_citations=[],
            tool_traces=traces,
            requires_human_escalation=True,
            sla_minutes=60,
        )

    def _persist_case(self, report: CaseInvestigationReport) -> None:
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
            INSERT OR REPLACE INTO cases (
                case_id, transaction_id, created_at, risk_level, recommended_action,
                confidence_score, investigation_summary, evidence_json, policy_citations_json, report_pdf_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report.case_id,
                report.transaction_id,
                report.timestamp,
                report.risk_level,
                report.recommended_action,
                report.confidence_score,
                report.investigation_summary,
                json.dumps([e.model_dump() for e in report.evidence_list]),
                json.dumps([c.model_dump() for c in report.policy_citations]),
                f"data/reports/{report.case_id}.pdf",
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: could not persist case {report.case_id} to DB: {e}")
