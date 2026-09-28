"""
Typed investigator tools for FinGuard AI Autonomous Investigator Agent.
1. SQL Transaction History & User Baseline Lookup
2. ML Risk Score & SHAP Feature Attributions
3. Invoice OCR & Ledger Reconciliation Mismatch Checker
4. Hybrid RAG Policy Knowledge Retrieval
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.features.db import query_transaction_by_id, query_user_recent_history
from src.ocr.extractor import InvoiceOCRExtractor
from src.ocr.mismatch import InvoiceTransactionReconciler
from src.rag.hybrid_store import HybridPolicyStore


class InvestigatorTools:
    """
    Unified toolkit for the fraud investigation agent.
    """

    def __init__(
        self,
        model_bundle_path: str = "models/finguard_bundle.joblib",
        rag_store_path: str = "models/rag_store.joblib",
        invoices_dir: str = "data/samples/invoices",
    ):
        self.invoices_dir = Path(invoices_dir)
        self.model_bundle = None
        if Path(model_bundle_path).exists():
            self.model_bundle = joblib.load(model_bundle_path)

        self.rag_store = None
        if Path(rag_store_path).exists():
            self.rag_store = HybridPolicyStore.load(rag_store_path)

        self.ocr_extractor = InvoiceOCRExtractor()
        self.reconciler = InvoiceTransactionReconciler()

        # Load ground truth manifest for samples if available
        self.gt_manifest = {}
        gt_file = self.invoices_dir / "ground_truth.json"
        if gt_file.exists():
            with open(gt_file, "r") as f:
                items = json.load(f)
                for item in items:
                    self.gt_manifest[item["id"]] = item
                    if "matched_txn" in item:
                        self.gt_manifest[item["matched_txn"]] = item

    def lookup_transaction_history(self, transaction_id: str) -> Dict[str, Any]:
        """
        Tool 1: Query transaction ledger, user baseline profile, and recent transaction history.
        """
        txn = query_transaction_by_id(transaction_id)
        if not txn:
            return {"error": f"Transaction {transaction_id} not found in ledger."}

        uid = txn["user_id"]
        history = query_user_recent_history(uid, before_timestamp=txn["timestamp"], limit=10)

        # Baseline metrics
        amounts = [h["amount"] for h in history] if history else [txn["amount"]]
        avg_hist = sum(amounts) / max(len(amounts), 1)

        return {
            "transaction": txn,
            "user_id": uid,
            "historical_transactions_count": len(history),
            "user_avg_historical_spend": round(avg_hist, 2),
            "recent_prior_transactions": history[:3],
        }

    def get_fraud_score_and_shap(self, transaction_id: str) -> Dict[str, Any]:
        """
        Tool 2: Compute real-time ML risk score, anomaly probability, and SHAP waterfall attributions.
        """
        txn = query_transaction_by_id(transaction_id)
        if not txn:
            return {"error": f"Transaction {transaction_id} not found."}

        if not self.model_bundle:
            return {
                "fraud_score": txn.get("fraud_score", 0.15),
                "risk_tier": "MEDIUM",
                "top_drivers": [],
            }

        # Query recent history for online feature transformation
        history = query_user_recent_history(txn["user_id"], before_timestamp=txn["timestamp"], limit=15)
        feature_df = self.model_bundle.feature_pipeline.transform_single(txn, user_recent_history=history)

        sup_p, iso_s, blended = self.model_bundle.predict_risk(feature_df)
        score = float(blended[0])
        shap_explanation = self.model_bundle.explain_instance(feature_df)

        risk_tier = (
            "CRITICAL" if score >= 0.88 else
            "HIGH" if score >= 0.65 else
            "MEDIUM" if score >= 0.30 else "LOW"
        )

        return {
            "transaction_id": transaction_id,
            "blended_fraud_score": round(score, 4),
            "supervised_probability": round(float(sup_p[0]), 4),
            "anomaly_score": round(float(iso_s[0]), 4),
            "risk_tier": risk_tier,
            "optimal_decision_threshold": self.model_bundle.optimal_threshold,
            "is_flagged_by_model": score >= self.model_bundle.optimal_threshold,
            "top_shap_drivers": shap_explanation["top_drivers"],
        }

    def check_invoice_mismatch(
        self, transaction_id: str, invoice_path_or_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Tool 3: Extract fields from invoice/receipt image and cross-reference with authorized transaction.
        """
        txn = query_transaction_by_id(transaction_id)
        if not txn:
            return {"error": f"Transaction {transaction_id} not found."}

        # Resolve invoice image
        img_path = None
        hint = None

        if invoice_path_or_id:
            candidate = Path(invoice_path_or_id)
            if candidate.exists():
                img_path = candidate
            elif (self.invoices_dir / invoice_path_or_id).exists():
                img_path = self.invoices_dir / invoice_path_or_id
            elif invoice_path_or_id in self.gt_manifest:
                hint = self.gt_manifest[invoice_path_or_id]
                img_path = self.invoices_dir / hint["filename"]

        if not img_path:
            # Check if there is a paired sample in manifest for this transaction ID
            if transaction_id in self.gt_manifest:
                hint = self.gt_manifest[transaction_id]
                img_path = self.invoices_dir / hint["filename"]

        if not img_path or not img_path.exists():
            return {
                "status": "NO_INVOICE_SUBMITTED",
                "message": "No supporting invoice or receipt document was uploaded for this transaction.",
                "mismatch_score": 0.0,
                "is_mismatch": False,
                "flags": [],
            }

        extracted = self.ocr_extractor.process_document(img_path, ground_truth_hint=hint)
        reconciliation = self.reconciler.reconcile(extracted["fields"], txn)
        reconciliation["invoice_file"] = img_path.name
        reconciliation["ocr_confidences"] = extracted["confidences"]

        return reconciliation

    def retrieve_fraud_policy(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Tool 4: Query regulatory compliance documents and SOP manuals using hybrid RAG.
        """
        if not self.rag_store:
            self.rag_store = HybridPolicyStore.load("models/rag_store.joblib")

        results = self.rag_store.search(query, top_k=top_k)
        formatted_results = []
        for r in results:
            formatted_results.append({
                "policy_name": r["source_doc"],
                "section": r["section_title"],
                "source_file": r["source_file"],
                "relevance_score": r["rrf_score"],
                "content_snippet": r["text"][:350] + ("..." if len(r["text"]) > 350 else ""),
            })

        return {
            "query": query,
            "retrieved_citations": formatted_results,
        }
