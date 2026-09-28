"""
Unit test suite for OpenCV preprocessing, OCR extraction, and reconciliation mismatch checker.
"""

from __future__ import annotations

import numpy as np

from src.ocr.mismatch import InvoiceTransactionReconciler, compute_string_similarity
from src.ocr.preprocess import DocumentPreprocessor


def test_document_preprocessor_deskew():
    preprocessor = DocumentPreprocessor()
    # Create white canvas with a black horizontal bar
    canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
    canvas[140:160, 50:250] = 0

    deskewed_color, deskewed_binary, angle = preprocessor.process(canvas)
    assert deskewed_color.shape == canvas.shape
    assert isinstance(angle, float)


def test_string_similarity_fuzzy():
    assert compute_string_similarity("Apple Store Fifth Ave", "Apple Store") >= 0.85
    assert compute_string_similarity("Best Buy Electronics", "Best Buy Store") >= 0.85
    assert compute_string_similarity("Shell Gas Station", "Starbucks Coffee") < 0.30


def test_reconciler_exact_match():
    reconciler = InvoiceTransactionReconciler()
    inv_fields = {
        "vendor": "Apple Store Fifth Ave",
        "invoice_no": "INV-2026-001",
        "date": "2026-01-15",
        "total": 1250.00,
        "subtotal": 1154.73,
        "tax": 95.27,
    }
    txn = {
        "transaction_id": "TXN_001",
        "merchant_name": "Apple Store",
        "amount": 1250.00,
        "timestamp": "2026-01-15 14:00:00",
    }
    res = reconciler.reconcile(inv_fields, txn)
    assert res["is_mismatch"] is False
    assert res["mismatch_score"] < 0.10
    assert len(res["flags"]) == 0


def test_reconciler_amount_mismatch():
    reconciler = InvoiceTransactionReconciler()
    inv_fields = {
        "vendor": "Best Buy Electronics",
        "total": 1800.00,
        "date": "2026-01-15",
    }
    txn = {
        "transaction_id": "TXN_002",
        "merchant_name": "Best Buy",
        "amount": 1200.00,  # $600 discrepancy!
        "timestamp": "2026-01-15 14:00:00",
    }
    res = reconciler.reconcile(inv_fields, txn)
    assert res["is_mismatch"] is True
    assert "AMOUNT_MISMATCH" in res["flags"]
    assert res["amount_diff"] == 600.00


def test_reconciler_stale_date():
    reconciler = InvoiceTransactionReconciler()
    inv_fields = {
        "vendor": "Delta Airlines",
        "total": 450.00,
        "date": "2025-11-01",  # ~75 days stale
    }
    txn = {
        "transaction_id": "TXN_003",
        "merchant_name": "Delta Airlines",
        "amount": 450.00,
        "timestamp": "2026-01-15 14:00:00",
    }
    res = reconciler.reconcile(inv_fields, txn)
    assert "STALE_INVOICE" in res["flags"]


def test_reconciler_tax_math_integrity():
    reconciler = InvoiceTransactionReconciler()
    inv_fields = {
        "vendor": "Micro Center",
        "subtotal": 500.00,
        "tax": 50.00,
        "total": 650.00,  # 500 + 50 != 650 (tampered!)
        "date": "2026-01-15",
    }
    txn = {
        "transaction_id": "TXN_004",
        "merchant_name": "Micro Center",
        "amount": 650.00,
        "timestamp": "2026-01-15 14:00:00",
    }
    res = reconciler.reconcile(inv_fields, txn)
    assert "TAX_CALCULATION_FORGERY" in res["flags"]
