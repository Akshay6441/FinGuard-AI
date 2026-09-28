"""
Unit test suite for feature engineering pipeline and dataset generation.
"""

from __future__ import annotations

from src.features.generate_dataset import generate_synthetic_transactions
from src.features.pipeline import FEATURE_COLUMNS, TransactionFeaturePipeline
from src.models.train import time_based_split


def test_synthetic_data_generation():
    df = generate_synthetic_transactions(num_users=20, num_transactions=150, fraud_rate=0.05)
    assert len(df) == 150
    assert "transaction_id" in df.columns
    assert "amount" in df.columns
    assert "is_fraud" in df.columns
    assert df["is_fraud"].sum() > 0


def test_time_based_split():
    df = generate_synthetic_transactions(num_users=20, num_transactions=200, fraud_rate=0.05)
    train, val, test = time_based_split(df, 0.70, 0.15)
    assert len(train) == 140
    assert len(val) == 30
    assert len(test) == 30

    # Ensure strictly chronological ordering
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_feature_pipeline_fit_transform():
    df = generate_synthetic_transactions(num_users=20, num_transactions=200, fraud_rate=0.05)
    pipe = TransactionFeaturePipeline()
    pipe.fit(df, df["is_fraud"])

    X = pipe.transform(df)
    assert len(X) == len(df)
    for col in FEATURE_COLUMNS:
        assert col in X.columns
        assert not X[col].isna().any(), f"NaN found in feature column {col}"


def test_feature_pipeline_single_transform():
    df = generate_synthetic_transactions(num_users=20, num_transactions=100, fraud_rate=0.05)
    pipe = TransactionFeaturePipeline()
    pipe.fit(df, df["is_fraud"])

    sample_txn = {
        "transaction_id": "TXN_SINGLE_TEST",
        "user_id": "USR_0001",
        "timestamp": "2026-01-20 15:00:00",
        "amount": 150.0,
        "merchant_name": "Target Store #104",
        "merchant_category": "RETAIL",
        "mcc": "5311",
        "distance_from_home_km": 12.0,
        "is_new_device": 0,
        "is_foreign_country": 0,
    }
    history = [
        {"timestamp": "2026-01-20 14:30:00", "amount": 45.0},
        {"timestamp": "2026-01-19 16:00:00", "amount": 80.0},
    ]

    single_df = pipe.transform_single(sample_txn, user_recent_history=history)
    assert len(single_df) == 1
    assert single_df["velocity_1h"].iloc[0] == 1
    assert single_df["velocity_24h"].iloc[0] == 2
    assert single_df["amount_sum_24h"].iloc[0] == 125.0
