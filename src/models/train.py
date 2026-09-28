"""
Model training, evaluation, threshold optimization, and SHAP explainability for FinGuard AI.
Trains baseline (Logistic Regression), Random Forest, LightGBM/XGBoost, and Isolation Forest anomaly layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.features.generate_dataset import generate_synthetic_transactions
from src.features.pipeline import FEATURE_COLUMNS, TransactionFeaturePipeline
from src.models.bundle import FinGuardModelBundle

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)


def time_based_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split transaction dataset chronologically to strictly avoid future leakage.
    """
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    return train_df, val_df, test_df


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 500.0,
    cost_fp: float = 15.0,
) -> Dict[str, Any]:
    """
    Evaluate total expected cost across all possible classification thresholds:
    Cost(T) = FN(T) * Cost_FN + FP(T) * Cost_FP
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Append 1.0 to thresholds for shape alignment
    thresholds = np.append(thresholds, 1.0)

    best_thresh = 0.5
    min_cost = float("inf")
    cost_curve = []

    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (fn * cost_fn) + (fp * cost_fp)
        cost_curve.append({"threshold": round(float(t), 2), "total_cost": float(cost)})
        if cost < min_cost:
            min_cost = cost
            best_thresh = float(t)

    # Cost at naive 0.5 threshold
    pred_05 = (y_proba >= 0.5).astype(int)
    tn05, fp05, fn05, tp05 = confusion_matrix(y_true, pred_05, labels=[0, 1]).ravel()
    naive_cost = float((fn05 * cost_fn) + (fp05 * cost_fp))

    return {
        "optimal_threshold": round(best_thresh, 3),
        "min_cost": round(min_cost, 2),
        "naive_05_cost": round(naive_cost, 2),
        "cost_savings": round(naive_cost - min_cost, 2),
        "cost_savings_pct": round(
            ((naive_cost - min_cost) / (naive_cost + 1e-5)) * 100, 2
        ),
    }


def compute_recall_at_precision(
    y_true: np.ndarray, y_proba: np.ndarray, target_precision: float = 0.85
) -> float:
    """
    Calculate the maximum recall achievable while maintaining at least target_precision.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    valid_recalls = [r for p, r in zip(precisions, recalls, strict=False) if p >= target_precision]
    return float(max(valid_recalls)) if valid_recalls else 0.0


def train_and_evaluate(
    data_path: str = "data/raw/transactions.csv",
    models_dir: str = "models",
    n_optuna_trials: int = 25,
) -> Dict[str, Any]:
    """
    Main training routine executing Phase 2 requirements.
    """
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion / Generation
    if not Path(data_path).exists():
        print("Synthesizing raw transaction dataset...")
        raw_df = generate_synthetic_transactions(
            num_users=600,
            num_transactions=18000,
            fraud_rate=0.026,
            output_path=data_path,
        )
    else:
        raw_df = pd.read_csv(data_path)

    # 2. Time-based split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = time_based_split(raw_df, 0.70, 0.15)
    print(
        f"Temporal Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    y_train = train_df["is_fraud"].values
    y_val = val_df["is_fraud"].values
    y_test = test_df["is_fraud"].values

    print(
        f"Train Fraud: {y_train.sum()} ({y_train.mean():.2%}), Test Fraud: {y_test.sum()} ({y_test.mean():.2%})"
    )

    # 3. Fit feature pipeline strictly on train_df
    pipe = TransactionFeaturePipeline(smoothing_prior=10.0)
    pipe.fit(train_df, pd.Series(y_train))

    X_train = pipe.transform(train_df)
    X_val = pipe.transform(val_df)
    X_test = pipe.transform(test_df)

    # 4. Baseline 1: Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_prauc = float(average_precision_score(y_test, lr_proba))
    lr_roc = float(roc_auc_score(y_test, lr_proba))

    # 5. Baseline 2: Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", max_depth=10, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_prauc = float(average_precision_score(y_test, rf_proba))
    rf_roc = float(roc_auc_score(y_test, rf_proba))

    # 6. Champion Model: LightGBM with Optuna tuning
    scale_pos = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, scale_pos),
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        val_preds = clf.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, val_preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_optuna_trials)
    best_params = study.best_params
    best_params.update({"objective": "binary", "verbosity": -1, "random_state": 42})
    print(f"Optuna Best Params: {best_params}")

    champion_lgb = lgb.LGBMClassifier(**best_params)
    champion_lgb.fit(X_train, y_train)

    # 7. Unsupervised Anomaly Layer (Isolation Forest)
    iso_forest = IsolationForest(
        n_estimators=100, contamination=0.03, random_state=42, n_jobs=-1
    )
    iso_forest.fit(X_train)

    # 8. Create Bundle and evaluate blended predictions
    bundle = FinGuardModelBundle(
        feature_pipeline=pipe,
        classifier=champion_lgb,
        anomaly_detector=iso_forest,
    )
    _, _, blended_test_proba = bundle.predict_risk(X_test)

    # Metrics computation
    blended_prauc = float(average_precision_score(y_test, blended_test_proba))
    blended_roc = float(roc_auc_score(y_test, blended_test_proba))
    recall_at_85_prec = compute_recall_at_precision(y_test, blended_test_proba, 0.85)

    # Cost-based threshold optimization
    cost_analysis = find_cost_optimal_threshold(
        y_test, blended_test_proba, cost_fn=500.0, cost_fp=15.0
    )
    bundle.optimal_threshold = cost_analysis["optimal_threshold"]

    # Final test evaluation at optimal threshold
    final_preds = (blended_test_proba >= bundle.optimal_threshold).astype(int)
    final_precision = float(precision_score(y_test, final_preds, zero_division=0))
    final_recall = float(recall_score(y_test, final_preds, zero_division=0))
    final_f1 = float(f1_score(y_test, final_preds, zero_division=0))

    # Global SHAP feature importances
    shap_vals = bundle.explainer.shap_values(X_test)
    if isinstance(shap_vals, list):
        shap_vals_class1 = shap_vals[1]
    elif shap_vals.ndim == 3:
        shap_vals_class1 = shap_vals[:, :, 1]
    else:
        shap_vals_class1 = shap_vals

    mean_abs_shap = np.abs(shap_vals_class1).mean(axis=0)
    importance_dict = {
        name: round(float(val), 4)
        for name, val in sorted(
            zip(FEATURE_COLUMNS, mean_abs_shap, strict=False), key=lambda x: x[1], reverse=True
        )
    }

    metrics_summary = {
        "model_comparison": {
            "logistic_regression": {
                "pr_auc": round(lr_prauc, 4),
                "roc_auc": round(lr_roc, 4),
            },
            "random_forest": {
                "pr_auc": round(rf_prauc, 4),
                "roc_auc": round(rf_roc, 4),
            },
            "finguard_champion_blended": {
                "pr_auc": round(blended_prauc, 4),
                "roc_auc": round(blended_roc, 4),
                "recall_at_85_precision": round(recall_at_85_prec, 4),
                "optimal_threshold": bundle.optimal_threshold,
                "precision": round(final_precision, 4),
                "recall": round(final_recall, 4),
                "f1": round(final_f1, 4),
            },
        },
        "cost_optimization": cost_analysis,
        "global_shap_importances": importance_dict,
        "dataset_split": {
            "total_transactions": len(raw_df),
            "train_count": len(train_df),
            "val_count": len(val_df),
            "test_count": len(test_df),
            "test_fraud_count": int(y_test.sum()),
        },
    }

    # Save artifacts
    bundle_path = models_path / "finguard_bundle.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"Saved model bundle to {bundle_path}")

    metrics_path = models_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Also persist test transactions with predictions and preprocessed data for fast UI/Agent access
    test_df["fraud_score"] = blended_test_proba
    test_df["predicted_fraud"] = final_preds
    test_df.to_csv(models_path / "test_evaluated_transactions.csv", index=False)

    return metrics_summary


if __name__ == "__main__":
    train_and_evaluate()
