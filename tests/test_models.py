"""
Unit test suite for fraud model inference, SHAP explanations, and anomaly blending.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from src.models.bundle import FinGuardModelBundle
from src.models.train import find_cost_optimal_threshold


def test_model_bundle_loading():
    bundle_path = Path("models/finguard_bundle.joblib")
    if not bundle_path.exists():
        pytest.skip("Model bundle not trained yet.")
    bundle = joblib.load(bundle_path)
    assert isinstance(bundle, FinGuardModelBundle)
    assert bundle.optimal_threshold > 0.0


def test_model_prediction_and_shap():
    bundle_path = Path("models/finguard_bundle.joblib")
    if not bundle_path.exists():
        pytest.skip("Model bundle not trained yet.")
    bundle: FinGuardModelBundle = joblib.load(bundle_path)

    # Create synthetic test record
    sample_txn = {
        "transaction_id": "TXN_M_TEST",
        "user_id": "USR_0001",
        "timestamp": "2026-01-20 15:00:00",
        "amount": 2500.0,
        "merchant_name": "Apple Store Fifth Ave",
        "merchant_category": "ELECTRONICS",
        "mcc": "5732",
        "distance_from_home_km": 2800.0,
        "is_new_device": 1,
        "is_foreign_country": 1,
    }
    feats = bundle.feature_pipeline.transform_single(sample_txn)
    sup_p, iso_s, blended = bundle.predict_risk(feats)

    assert 0.0 <= sup_p[0] <= 1.0
    assert 0.0 <= iso_s[0] <= 1.0
    assert 0.0 <= blended[0] <= 1.0

    # High amount + foreign + new device should yield high risk
    assert blended[0] >= 0.50

    # Test SHAP explanation
    explanation = bundle.explain_instance(feats)
    assert "base_value" in explanation
    assert "top_drivers" in explanation
    assert len(explanation["top_drivers"]) > 0


def test_cost_optimal_threshold_calculation():
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.1, 0.15, 0.2, 0.3, 0.35, 0.4, 0.45, 0.6, 0.75, 0.9])
    res = find_cost_optimal_threshold(y_true, y_proba, cost_fn=500.0, cost_fp=15.0)

    assert "optimal_threshold" in res
    assert "cost_savings" in res
    assert res["min_cost"] <= res["naive_05_cost"]
