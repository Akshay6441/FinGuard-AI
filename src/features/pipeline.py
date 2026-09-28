"""
Feature engineering and transformation pipeline for FinGuard AI.
Handles velocity calculation, user spend deviation z-scores, temporal cyclical features,
merchant category risk encoding, and single-transaction inference.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

HIGH_RISK_MCCS = {"7995", "6051", "5944", "5732"}

FEATURE_COLUMNS = [
    "amount",
    "velocity_1h",
    "velocity_24h",
    "amount_sum_24h",
    "amount_to_mean_ratio",
    "amount_z_score",
    "hour_of_day",
    "is_night",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "category_risk_score",
    "high_risk_mcc",
    "distance_from_home_km",
    "is_new_device",
    "is_foreign_country",
]


class TransactionFeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Stateful feature pipeline that computes historical user baselines,
    time-windowed velocity, cyclical time features, and risk indicators.
    """

    def __init__(self, smoothing_prior: float = 10.0, global_fraud_rate: float = 0.025):
        self.smoothing_prior = smoothing_prior
        self.global_fraud_rate = global_fraud_rate
        self.user_stats_: Dict[str, Dict[str, float]] = {}
        self.category_risk_map_: Dict[str, float] = {}
        self.feature_names_: List[str] = FEATURE_COLUMNS
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> TransactionFeaturePipeline:
        """
        Fit pipeline on training transactions:
        1. Learn historical mean and std spend per user.
        2. Learn smoothed empirical Bayes fraud rate per merchant category.
        """
        df = X.copy()
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 1. User historical stats
        user_grouped = df.groupby("user_id")["amount"].agg(["mean", "std", "count"]).reset_index()
        overall_mean = float(df["amount"].mean())
        overall_std = float(df["amount"].std()) if float(df["amount"].std()) > 0 else 1.0

        self.user_stats_ = {}
        for _, row in user_grouped.iterrows():
            uid = row["user_id"]
            u_mean = float(row["mean"]) if pd.notna(row["mean"]) else overall_mean
            u_std = float(row["std"]) if pd.notna(row["std"]) and float(row["std"]) > 0 else overall_std * 0.4
            self.user_stats_[uid] = {
                "mean": u_mean,
                "std": max(u_std, 1.0),
                "count": int(row["count"]),
            }
        self.overall_mean_ = overall_mean
        self.overall_std_ = max(overall_std, 1.0)

        # 2. Category risk score (Empirical Bayes target encoding)
        if y is not None:
            df["target"] = y.values
            cat_grouped = df.groupby("merchant_category")["target"].agg(["sum", "count"]).reset_index()
            global_prior = float(y.mean()) if len(y) > 0 else self.global_fraud_rate
            self.global_fraud_rate = global_prior

            self.category_risk_map_ = {}
            for _, row in cat_grouped.iterrows():
                cat = row["merchant_category"]
                k = row["sum"]
                n = row["count"]
                smoothed_rate = (k + self.smoothing_prior * global_prior) / (n + self.smoothing_prior)
                self.category_risk_map_[cat] = float(smoothed_rate)
        else:
            # Fallback default risk map if y not provided
            self.category_risk_map_ = {
                "GROCERY": 0.015,
                "DINING": 0.020,
                "RETAIL": 0.035,
                "UTILITIES": 0.010,
                "TRAVEL": 0.080,
                "ELECTRONICS": 0.120,
                "CRYPTO": 0.280,
                "GAMBLING": 0.350,
            }

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all feature columns for a dataframe of transactions.
        Preserves temporal order and respects user windows.
        """
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before calling transform().")

        df = X.copy()
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Preserve original index so output rows match input rows exactly
        orig_index = df.index
        df = df.sort_values(["user_id", "timestamp"])

        # Velocity calculations per user
        v_1h = []
        v_24h = []
        sum_24h = []

        # Group by user and compute rolling windows
        for _uid, group in df.groupby("user_id", sort=False):
            times = group["timestamp"].values
            amounts = group["amount"].values
            n = len(times)

            # Rolling search
            for i in range(n):
                current_t = times[i]
                # 1 hour window (preceding)
                t_1h_ago = current_t - np.timedelta64(1, "h")
                t_24h_ago = current_t - np.timedelta64(24, "h")

                # Preceding transactions within window (exclusive of current)
                mask_1h = (times[:i] >= t_1h_ago)
                mask_24h = (times[:i] >= t_24h_ago)

                v_1h.append(int(np.sum(mask_1h)))
                v_24h.append(int(np.sum(mask_24h)))
                sum_24h.append(float(np.sum(amounts[:i][mask_24h])) if np.any(mask_24h) else 0.0)

        df["velocity_1h"] = v_1h
        df["velocity_24h"] = v_24h
        df["amount_sum_24h"] = sum_24h

        # User spend deviations
        u_means = []
        u_stds = []
        for uid in df["user_id"]:
            stats = self.user_stats_.get(uid, {"mean": self.overall_mean_, "std": self.overall_std_})
            u_means.append(stats["mean"])
            u_stds.append(stats["std"])

        u_means = np.array(u_means)
        u_stds = np.array(u_stds)
        amounts = df["amount"].values

        df["amount_to_mean_ratio"] = amounts / (u_means + 1e-5)
        df["amount_z_score"] = (amounts - u_means) / (u_stds + 1e-5)

        # Temporal features
        if "timestamp" in df.columns:
            hours = df["timestamp"].dt.hour.values
            dayofweek = df["timestamp"].dt.dayofweek.values
        else:
            hours = np.full(len(df), 12)
            dayofweek = np.full(len(df), 2)

        df["hour_of_day"] = hours
        df["is_night"] = ((hours >= 0) & (hours <= 5)).astype(int)
        df["is_weekend"] = (dayofweek >= 5).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        # Categorical risk mapping
        default_risk = self.global_fraud_rate
        df["category_risk_score"] = df["merchant_category"].map(
            lambda c: self.category_risk_map_.get(c, default_risk)
        )
        mcc_vals = df["mcc"].astype(str) if "mcc" in df.columns else pd.Series(["0000"] * len(df))
        df["high_risk_mcc"] = mcc_vals.apply(lambda m: 1 if m in HIGH_RISK_MCCS else 0)

        # Device & geography defaults
        if "distance_from_home_km" not in df.columns:
            df["distance_from_home_km"] = 15.0
        if "is_new_device" not in df.columns:
            df["is_new_device"] = 0
        if "is_foreign_country" not in df.columns:
            df["is_foreign_country"] = 0

        # Restore original index order so output aligns with input
        df = df.loc[orig_index]
        return df[self.feature_names_]

    def transform_single(self, txn_dict: Dict, user_recent_history: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Fast online feature extraction for a single incoming transaction.
        Utilizes user's recent transactions (e.g. from DB) to compute exact velocity.
        """
        t_str = txn_dict.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        current_t = pd.to_datetime(t_str)
        uid = txn_dict.get("user_id", "UNKNOWN")
        amount = float(txn_dict.get("amount", 0.0))

        # Velocity from recent history
        v_1h = 0
        v_24h = 0
        sum_24h = 0.0
        if user_recent_history:
            t_1h_ago = current_t - pd.Timedelta(hours=1)
            t_24h_ago = current_t - pd.Timedelta(hours=24)
            for prev in user_recent_history:
                pt = pd.to_datetime(prev["timestamp"])
                pa = float(prev["amount"])
                if pt >= t_1h_ago and pt < current_t:
                    v_1h += 1
                if pt >= t_24h_ago and pt < current_t:
                    v_24h += 1
                    sum_24h += pa

        # User baseline stats
        stats = self.user_stats_.get(uid, {"mean": self.overall_mean_, "std": self.overall_std_})
        u_mean = stats["mean"]
        u_std = stats["std"]
        amount_ratio = amount / (u_mean + 1e-5)
        amount_z = (amount - u_mean) / (u_std + 1e-5)

        # Temporal
        hour = current_t.hour
        is_night = 1 if 0 <= hour <= 5 else 0
        is_weekend = 1 if current_t.dayofweek >= 5 else 0
        hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2 * np.pi * hour / 24.0))

        # Category
        cat = txn_dict.get("merchant_category", "RETAIL")
        cat_risk = self.category_risk_map_.get(cat, self.global_fraud_rate)
        mcc = str(txn_dict.get("mcc", "0000"))
        high_mcc = 1 if mcc in HIGH_RISK_MCCS else 0

        row = {
            "amount": amount,
            "velocity_1h": v_1h,
            "velocity_24h": v_24h,
            "amount_sum_24h": sum_24h,
            "amount_to_mean_ratio": amount_ratio,
            "amount_z_score": amount_z,
            "hour_of_day": hour,
            "is_night": is_night,
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "category_risk_score": cat_risk,
            "high_risk_mcc": high_mcc,
            "distance_from_home_km": float(txn_dict.get("distance_from_home_km", 10.0)),
            "is_new_device": int(txn_dict.get("is_new_device", 0)),
            "is_foreign_country": int(txn_dict.get("is_foreign_country", 0)),
        }
        return pd.DataFrame([row])[self.feature_names_]
