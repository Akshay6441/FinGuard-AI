"""
FinGuardModelBundle definition module.
Ensures stable pickle unpickling across scripts, API, Streamlit, and tests.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from src.features.pipeline import FEATURE_COLUMNS, TransactionFeaturePipeline


class FinGuardModelBundle:
    """
    Complete production inference bundle containing:
    1. Feature engineering pipeline
    2. Scaler (for baseline models)
    3. Supervised LightGBM classifier
    4. Unsupervised Isolation Forest anomaly layer
    5. SHAP TreeExplainer
    6. Calibrated cost-optimal threshold
    """

    def __init__(
        self,
        feature_pipeline: TransactionFeaturePipeline,
        classifier: Any,
        anomaly_detector: IsolationForest,
        optimal_threshold: float = 0.28,
        blend_weight_supervised: float = 0.85,
    ):
        self.feature_pipeline = feature_pipeline
        self.classifier = classifier
        self.anomaly_detector = anomaly_detector
        self.optimal_threshold = optimal_threshold
        self.blend_weight_supervised = blend_weight_supervised
        self.feature_names = FEATURE_COLUMNS
        self.explainer = shap.TreeExplainer(self.classifier)

    def predict_risk(
        self, features_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (supervised_proba, anomaly_score, blended_risk_score)
        """
        sup_proba = self.classifier.predict_proba(features_df[self.feature_names])[:, 1]

        # Isolation forest scores: lower score = more anomalous. Normalize to [0, 1]
        iso_raw = self.anomaly_detector.decision_function(
            features_df[self.feature_names]
        )
        iso_score = 1.0 / (1.0 + np.exp(iso_raw * 5.0))

        blended = (self.blend_weight_supervised * sup_proba) + (
            (1.0 - self.blend_weight_supervised) * iso_score
        )
        return sup_proba, iso_score, blended

    def explain_instance(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Produce SHAP waterfall breakdown for a single transaction.
        """
        feat_vals = features_df[self.feature_names].iloc[0].to_dict()
        shap_vals = self.explainer.shap_values(features_df[self.feature_names])

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        elif shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1]
        else:
            vals = shap_vals[0]

        base_val = float(
            self.explainer.expected_value[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray))
            else self.explainer.expected_value
        )

        contributions = []
        for name, val, contrib in zip(self.feature_names, feat_vals.values(), vals, strict=False):
            contributions.append({
                "feature": name,
                "value": round(float(val), 4),
                "shap_value": round(float(contrib), 4),
                "impact": "INCREASES_RISK" if contrib > 0 else "DECREASES_RISK",
            })

        # Sort by absolute SHAP contribution
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value": round(base_val, 4),
            "top_drivers": contributions[:6],
            "all_contributions": contributions,
        }
