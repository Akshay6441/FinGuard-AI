"""
Integration test suite for FinGuard AI FastAPI application.
Validates all endpoints using FastAPI TestClient.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "version" in data


def test_score_transaction():
    payload = {
        "transaction_id": "TXN_TEST_001",
        "user_id": "USR_0001",
        "timestamp": "2026-01-15 12:00:00",
        "amount": 2400.0,
        "merchant_name": "Apple Store Fifth Ave",
        "merchant_category": "ELECTRONICS",
        "mcc": "5732",
        "device_id": "DEV_UNKNOWN_99",
        "country": "NG",
        "distance_from_home_km": 4200.0,
        "is_new_device": 1,
        "is_foreign_country": 1,
    }
    response = client.post("/api/v1/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_score" in data
    assert "risk_tier" in data
    assert "top_shap_drivers" in data
    assert len(data["top_shap_drivers"]) > 0


def test_list_transactions():
    response = client.get("/api/v1/transactions?limit=10")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) > 0
    assert "transaction_id" in items[0]


def test_get_transaction_detail():
    # Fetch first transaction from list
    list_res = client.get("/api/v1/transactions?limit=1")
    assert list_res.status_code == 200
    first_id = list_res.json()[0]["transaction_id"]

    res = client.get(f"/api/v1/transactions/{first_id}")
    assert res.status_code == 200
    detail = res.json()
    assert "transaction" in detail
    assert "model_assessment" in detail


def test_investigate_and_download_pdf():
    # Pick a transaction to investigate
    list_res = client.get("/api/v1/transactions?limit=1")
    first_id = list_res.json()[0]["transaction_id"]

    inv_res = client.post(f"/api/v1/investigate/{first_id}")
    assert inv_res.status_code == 200
    case_report = inv_res.json()
    assert "case_id" in case_report
    assert "risk_level" in case_report
    assert "recommended_action" in case_report
    assert len(case_report["evidence_list"]) > 0

    case_id = case_report["case_id"]

    # Verify retrieval
    get_res = client.get(f"/api/v1/cases/{case_id}")
    assert get_res.status_code == 200

    # Verify PDF generation and streaming
    pdf_res = client.get(f"/api/v1/report/{case_id}/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF")


def test_upload_invoice_sample():
    response = client.post("/api/v1/upload-invoice?sample_id=INV_DEMO_01_MATCH")
    assert response.status_code == 200
    data = response.json()
    assert "fields" in data
    assert data["fields"]["vendor"] is not None
    assert "total" in data["fields"]


def test_metrics_endpoint():
    response = client.get("/api/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "ocr_metrics" in data
    assert "rag_metrics" in data
    assert "agent_metrics" in data
