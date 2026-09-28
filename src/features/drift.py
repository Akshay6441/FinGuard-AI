"""
Feature and Population Stability Index (PSI) drift monitoring for FinGuard AI.
Monitors production inference data against training baselines to detect data decay.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between expected (training)
    and actual (production inference) feature distributions.

    PSI Interpretation:
    - PSI < 0.10: No significant distribution change (Healthy)
    - 0.10 <= PSI < 0.25: Moderate change (Monitor)
    - PSI >= 0.25: Significant change (Action required / Retrain)
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected baseline
    quantiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0

    # Calculate frequencies in each bucket
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division with smoothing
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)

    # Re-normalize
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    # PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))


def run_drift_analysis(
    baseline_path: str = "data/raw/transactions.csv",
    inference_path: str = "models/test_evaluated_transactions.csv",
    output_path: str = "models/drift_report.json",
) -> Dict[str, Any]:
    """
    Run complete feature drift analysis across numeric features.
    """
    base_file = Path(baseline_path)
    inf_file = Path(inference_path)

    if not base_file.exists() or not inf_file.exists():
        # Fallback summary
        return {"status": "SKIPPED", "message": "Files not available for drift report."}

    df_base = pd.read_csv(base_file)
    df_inf = pd.read_csv(inf_file)

    numeric_cols = ["amount", "distance_from_home_km", "fraud_score"]
    drift_metrics = {}
    overall_drift_flag = False

    for col in numeric_cols:
        if col in df_base.columns and col in df_inf.columns:
            psi = calculate_psi(df_base[col].values, df_inf[col].values)
            status = "HEALTHY" if psi < 0.10 else "MODERATE" if psi < 0.25 else "CRITICAL_DRIFT"
            if psi >= 0.25:
                overall_drift_flag = True

            drift_metrics[col] = {
                "psi": psi,
                "status": status,
                "baseline_mean": round(float(df_base[col].mean()), 2),
                "inference_mean": round(float(df_inf[col].mean()), 2),
                "baseline_std": round(float(df_base[col].std()), 2),
                "inference_std": round(float(df_inf[col].std()), 2),
            }

    report = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_status": "CRITICAL_DRIFT" if overall_drift_flag else "HEALTHY",
        "requires_retraining": overall_drift_flag,
        "features": drift_metrics,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_drift_analysis()
    print(f"Drift Analysis Completed: Status = {report.get('overall_status')}")
