"""
Unit test suite for feature drift and Population Stability Index (PSI).
"""

from __future__ import annotations

import numpy as np

from src.features.drift import calculate_psi, run_drift_analysis


def test_psi_identical_distributions():
    data = np.random.normal(loc=100.0, scale=15.0, size=1000)
    psi = calculate_psi(data, data)
    assert psi < 0.05  # Should be near zero for identical distributions


def test_psi_shifted_distributions():
    baseline = np.random.normal(loc=50.0, scale=10.0, size=1000)
    shifted = np.random.normal(loc=120.0, scale=25.0, size=1000)
    psi = calculate_psi(baseline, shifted)
    assert psi > 0.25  # Should flag significant distribution drift


def test_drift_report_execution():
    report = run_drift_analysis()
    assert "overall_status" in report
    assert "features" in report
