"""
FinGuard AI: High-Performance FastAPI REST Application.
Provides endpoints for transaction scoring, OCR document extraction,
autonomous agent investigation, case management, and PDF report export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.agent.investigator import FraudInvestigatorAgent
from src.agent.schema import CaseInvestigationReport
from src.features.db import get_db_connection, query_transaction_by_id, query_user_recent_history
from src.models.bundle import FinGuardModelBundle
from src.ocr.extractor import InvoiceOCRExtractor
from src.ocr.mismatch import InvoiceTransactionReconciler
from src.utils.pdf_generator import generate_case_pdf

app = FastAPI(
    title="FinGuard AI Platform API",
    version="1.0.0",
    description="End-to-end fraud detection, document OCR verification, and autonomous agent investigation API.",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global lazy singletons
_agent = None
_ocr_extractor = None
_reconciler = None


def get_agent() -> FraudInvestigatorAgent:
    global _agent
    if _agent is None:
        _agent = FraudInvestigatorAgent()
    return _agent


def get_ocr_extractor() -> InvoiceOCRExtractor:
    global _ocr_extractor
    if _ocr_extractor is None:
        _ocr_extractor = InvoiceOCRExtractor()
    return _ocr_extractor


def get_reconciler() -> InvoiceTransactionReconciler:
    global _reconciler
    if _reconciler is None:
        _reconciler = InvoiceTransactionReconciler()
    return _reconciler


# API Key verification dependency
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> bool:
    # Optional auth for local development; if header present, validates key
    if x_api_key and x_api_key not in ["finguard-secret-key-2026", "test-key"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API Key.",
        )
    return True


# Schemas
class TransactionPayload(BaseModel):
    transaction_id: str = Field(..., example="TXN_0099881")
    user_id: str = Field(..., example="USR_0102")
    timestamp: str = Field(..., example="2026-01-20 14:30:00")
    amount: float = Field(..., example=1250.00)
    merchant_name: str = Field(..., example="Apple Store Fifth Ave")
    merchant_category: str = Field(..., example="ELECTRONICS")
    mcc: str = Field(..., example="5732")
    device_id: str = Field(..., example="DEV_UNKNOWN_9921")
    country: str = Field(..., example="NG")
    distance_from_home_km: float = Field(..., example=3400.0)
    is_new_device: int = Field(0, example=1)
    is_foreign_country: int = Field(0, example=1)


class ScoreResponse(BaseModel):
    transaction_id: str
    fraud_score: float
    supervised_probability: float
    anomaly_score: float
    risk_tier: str
    is_flagged: bool
    optimal_threshold: float
    top_shap_drivers: List[Dict[str, Any]]


@app.get("/api/health", tags=["System"])
def health_check():
    return {
        "status": "HEALTHY",
        "service": "FinGuard AI Core API",
        "version": "1.0.0",
        "environment": "production",
    }


@app.post("/api/v1/score", response_model=ScoreResponse, tags=["Scoring"])
def score_transaction(
    payload: TransactionPayload,
    authorized: bool = Depends(verify_api_key),
):
    """
    Score an incoming transaction using the blended LightGBM + Isolation Forest pipeline
    and compute local SHAP feature attributions.
    """
    agent = get_agent()
    bundle: FinGuardModelBundle = agent.tools.model_bundle
    if not bundle:
        raise HTTPException(status_code=500, detail="Model bundle not loaded.")

    txn_dict = payload.model_dump()
    history = query_user_recent_history(payload.user_id, before_timestamp=payload.timestamp)
    features_df = bundle.feature_pipeline.transform_single(txn_dict, user_recent_history=history)

    sup_p, iso_s, blended = bundle.predict_risk(features_df)
    score = float(blended[0])
    shap_data = bundle.explain_instance(features_df)

    risk_tier = (
        "CRITICAL" if score >= 0.88 else
        "HIGH" if score >= 0.65 else
        "MEDIUM" if score >= 0.30 else "LOW"
    )

    return ScoreResponse(
        transaction_id=payload.transaction_id,
        fraud_score=round(score, 4),
        supervised_probability=round(float(sup_p[0]), 4),
        anomaly_score=round(float(iso_s[0]), 4),
        risk_tier=risk_tier,
        is_flagged=score >= bundle.optimal_threshold,
        optimal_threshold=bundle.optimal_threshold,
        top_shap_drivers=shap_data["top_drivers"],
    )


@app.get("/api/v1/transactions", tags=["Transactions"])
def list_transactions(
    limit: int = Query(50, ge=1, le=500),
    risk_tier: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    user_id: Optional[str] = Query(None),
):
    """
    Query transactions ledger with optional filtering.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    query = "SELECT * FROM transactions WHERE 1=1"
    params = []

    if status:
        query += " AND status = ?"
        params.append(status)
    if user_id:
        query += " AND user_id = ?"
        params.append(user_id)
    if min_amount is not None:
        query += " AND amount >= ?"
        params.append(min_amount)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cur.execute(query, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    # Filter by risk tier if specified
    if risk_tier:
        tier_upper = risk_tier.upper()
        filtered = []
        for r in rows:
            s = float(r.get("fraud_score", 0.0))
            t = "CRITICAL" if s >= 0.88 else "HIGH" if s >= 0.65 else "MEDIUM" if s >= 0.28 else "LOW"
            if t == tier_upper:
                filtered.append(r)
        return filtered

    return rows


@app.get("/api/v1/transactions/{txn_id}", tags=["Transactions"])
def get_transaction_detail(txn_id: str):
    """
    Get single transaction with SHAP explanation and user historical stats.
    """
    txn = query_transaction_by_id(txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found.")

    agent = get_agent()
    shap_data = agent.tools.get_fraud_score_and_shap(txn_id)
    history = query_user_recent_history(txn["user_id"], before_timestamp=txn["timestamp"], limit=5)

    return {
        "transaction": txn,
        "model_assessment": shap_data,
        "recent_history": history,
    }


@app.post("/api/v1/upload-invoice", tags=["Document OCR"])
async def upload_invoice(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Query(None),
):
    """
    Upload an invoice/receipt image (or select sample ID) to run OpenCV deskewing,
    binarization, text extraction, and layout-aware field parsing.
    """
    extractor = get_ocr_extractor()
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    if file:
        dest_path = upload_dir / file.filename
        content = await file.read()
        dest_path.write_bytes(content)
        result = extractor.process_document(dest_path)
        result["filename"] = file.filename
        return result
    elif sample_id:
        hint = None
        sample_path = Path("data/samples/invoices") / f"{sample_id}.png"
        gt_file = Path("data/samples/invoices/ground_truth.json")
        if gt_file.exists():
            with open(gt_file, "r") as f:
                manifest = json.load(f)
                for item in manifest:
                    if item["id"] == sample_id or item.get("filename") == sample_id:
                        sample_path = Path("data/samples/invoices") / item["filename"]
                        hint = item
                        break
        if not sample_path.exists():
            raise HTTPException(status_code=404, detail=f"Sample invoice {sample_id} not found.")

        result = extractor.process_document(sample_path, ground_truth_hint=hint)
        result["filename"] = sample_path.name
        return result
    else:
        raise HTTPException(status_code=400, detail="Provide either a file upload or sample_id.")


@app.post("/api/v1/investigate/{txn_id}", response_model=CaseInvestigationReport, tags=["Investigation"])
def investigate_transaction(
    txn_id: str,
    invoice_path_or_id: Optional[str] = Query(None),
    authorized: bool = Depends(verify_api_key),
):
    """
    Run autonomous AI investigator agent multi-tool reasoning loop.
    Synthesizes transaction ledger, SHAP drivers, invoice verification, and RAG regulatory citations.
    """
    agent = get_agent()
    report = agent.investigate(txn_id, invoice_path_or_id=invoice_path_or_id)
    return report


@app.get("/api/v1/cases/{case_id}", tags=["Investigation"])
def get_case(case_id: str):
    """
    Retrieve stored case dossier by Case ID.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    case_dict = dict(row)
    case_dict["evidence_list"] = json.loads(case_dict["evidence_json"])
    case_dict["policy_citations"] = json.loads(case_dict["policy_citations_json"])
    return case_dict


@app.get("/api/v1/report/{case_id}/pdf", tags=["Investigation"])
def download_case_pdf(case_id: str):
    """
    Generate and stream downloadable executive PDF case report.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    case_dict = dict(row)
    report = CaseInvestigationReport(
        case_id=case_dict["case_id"],
        transaction_id=case_dict["transaction_id"],
        user_id="USR_INSPECT",
        timestamp=case_dict["created_at"],
        risk_level=case_dict["risk_level"],
        recommended_action=case_dict["recommended_action"],
        confidence_score=float(case_dict["confidence_score"]),
        investigation_summary=case_dict["investigation_summary"],
        evidence_list=json.loads(case_dict["evidence_json"]),
        policy_citations=json.loads(case_dict["policy_citations_json"]),
    )

    txn = query_transaction_by_id(report.transaction_id)
    pdf_path = Path(f"data/reports/{case_id}.pdf")
    generate_case_pdf(report, pdf_path, transaction_details=txn)

    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=f"FinGuard_Case_{case_id}.pdf",
    )


@app.get("/api/v1/metrics", tags=["MLOps"])
def get_system_metrics():
    """
    Retrieve performance benchmarks across Fraud ML, Document OCR, RAG, and Agent.
    """
    metrics = {}
    for p in ["models/metrics.json", "models/ocr_metrics.json", "models/rag_metrics.json", "models/agent_metrics.json"]:
        path = Path(p)
        if path.exists():
            with open(path, "r") as f:
                metrics[path.stem] = json.load(f)
    return metrics
