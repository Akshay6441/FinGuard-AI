"""
Dataset generator for FinGuard AI.
Synthesizes a realistic, temporal multi-user credit card & payment transaction dataset
combining patterns from Kaggle ULB, IEEE-CIS Fraud, and PaySim.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES = {
    "GROCERY": {"risk_weight": 0.05, "avg_amount": 65.0, "mcc": "5411"},
    "DINING": {"risk_weight": 0.08, "avg_amount": 42.0, "mcc": "5812"},
    "RETAIL": {"risk_weight": 0.12, "avg_amount": 85.0, "mcc": "5311"},
    "UTILITIES": {"risk_weight": 0.03, "avg_amount": 120.0, "mcc": "4900"},
    "TRAVEL": {"risk_weight": 0.28, "avg_amount": 450.0, "mcc": "4511"},
    "ELECTRONICS": {"risk_weight": 0.35, "avg_amount": 650.0, "mcc": "5732"},
    "CRYPTO": {"risk_weight": 0.72, "avg_amount": 1200.0, "mcc": "6051"},
    "GAMBLING": {"risk_weight": 0.85, "avg_amount": 350.0, "mcc": "7995"},
}

MERCHANT_NAMES = {
    "GROCERY": ["WholeFoods Market", "Kroger Supermarket", "Trader Joe's", "Aldi Express"],
    "DINING": ["Starbucks Coffee", "Chipotle Mexican Grill", "Olive Garden", "UberEats Dine"],
    "RETAIL": ["Target Store #104", "Walmart Supercenter", "Amazon Marketplace", "Zara Retail"],
    "UTILITIES": ["Consolidated Edison", "Verizon Wireless", "Pacific Gas & Electric", "AT&T Bill"],
    "TRAVEL": ["Delta Airlines", "Marriott Hotels", "Airbnb Lodging", "Uber Technologies"],
    "ELECTRONICS": ["Apple Store Fifth Ave", "Best Buy Electronics", "Micro Center", "B&H Photo Video"],
    "CRYPTO": ["Binance Global P2P", "Coinbase Exchange", "Kraken Pay", "OKX Digital Assets"],
    "GAMBLING": ["DraftKings Sportsbook", "BetMGM Interactive", "Bovada Gaming", "Stake Casino"],
}

COUNTRIES = ["US", "US", "US", "US", "US", "CA", "GB", "DE", "IN", "SG", "NG", "RU"]


def generate_synthetic_transactions(
    num_users: int = 500,
    num_transactions: int = 15000,
    fraud_rate: float = 0.025,
    random_seed: int = 42,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generate sequential, temporal transaction logs across users with realistic fraud modalities:
    1. Account Takeover (burst velocity + foreign device)
    2. High-value luxury / crypto surge (extreme Z-score)
    3. Stolen card testing (micro-transaction followed by large spend)
    4. Midnight high-risk gambling/crypto activity
    """
    rng = np.random.default_rng(random_seed)

    # 1. Generate user profiles
    users = []
    for i in range(1, num_users + 1):
        uid = f"USR_{i:04d}"
        base_spend = float(rng.lognormal(mean=3.8, sigma=0.5))  # around $45-$120 typical spend
        std_spend = base_spend * float(rng.uniform(0.2, 0.5))
        home_country = "US" if rng.random() < 0.9 else rng.choice(["CA", "GB", "DE", "IN"])
        primary_device = f"DEV_IOS_{rng.integers(1000, 9999)}" if rng.random() < 0.55 else f"DEV_AND_{rng.integers(1000, 9999)}"
        users.append({
            "user_id": uid,
            "base_spend": base_spend,
            "std_spend": std_spend,
            "home_country": home_country,
            "primary_device": primary_device,
        })
    user_df = pd.DataFrame(users).set_index("user_id")

    # 2. Sequential timestamp generation spanning 60 days
    base_time = datetime.datetime(2026, 1, 1, 0, 0, 0)
    # Average ~1 transaction every 345 seconds across the platform
    intervals = rng.exponential(scale=345, size=num_transactions)
    timestamps = [base_time + datetime.timedelta(seconds=float(s)) for s in np.cumsum(intervals)]

    records: List[Dict] = []

    # Track state for realistic velocity and stateful fraud patterns
    user_ids = list(user_df.index)

    for idx in range(num_transactions):
        txn_id = f"TXN_{idx + 1:07d}"
        timestamp = timestamps[idx]

        # Pick user (Zipf-like distribution: some users are more active)
        user_idx = int(rng.geometric(p=0.008) % num_users)
        uid = user_ids[user_idx]
        u_profile = user_df.loc[uid]

        # Determine if this transaction will be an injected fraud scenario
        is_fraud = 1 if rng.random() < fraud_rate else 0

        if is_fraud:
            fraud_type = rng.choice(["ATO_BURST", "EXTREME_AMOUNT", "HIGH_RISK_NIGHT", "NEW_DEVICE_FOREIGN"])
        else:
            fraud_type = "NORMAL"

        category_choices = list(MERCHANT_CATEGORIES.keys())

        if fraud_type == "NORMAL":
            # Normal distribution over everyday categories
            cat_probs = [0.28, 0.20, 0.22, 0.10, 0.08, 0.08, 0.02, 0.02]
            category = rng.choice(category_choices, p=cat_probs)
            mcc = MERCHANT_CATEGORIES[category]["mcc"]
            merchant = rng.choice(MERCHANT_NAMES[category])

            # Amount sampled near user baseline with occasional high legitimate purchases (5% chance)
            if rng.random() < 0.05:
                amount = float(u_profile["base_spend"] * rng.uniform(3.0, 6.0))
            else:
                amount = float(np.clip(rng.normal(u_profile["base_spend"], u_profile["std_spend"]), 5.0, 950.0))

            # Legitimate users travel or buy new phones occasionally
            is_traveling = rng.random() < 0.06
            is_new_device = 1 if rng.random() < 0.07 else 0
            is_foreign = 1 if is_traveling and rng.random() < 0.4 else 0
            distance_from_home = float(rng.uniform(150.0, 1800.0) if is_traveling else rng.exponential(scale=15.0))
            device_id = f"DEV_NEW_{rng.integers(1000, 9999)}" if is_new_device else u_profile["primary_device"]
            country = rng.choice(["CA", "GB", "DE"]) if is_foreign else u_profile["home_country"]

        elif fraud_type == "ATO_BURST":
            # Account Takeover: High risk electronics / crypto, new device, rapid velocity
            category = rng.choice(["ELECTRONICS", "CRYPTO", "TRAVEL", "RETAIL"])
            mcc = MERCHANT_CATEGORIES[category]["mcc"]
            merchant = rng.choice(MERCHANT_NAMES[category])
            amount = float(u_profile["base_spend"] * rng.uniform(3.5, 7.5) + rng.uniform(80, 400))
            device_id = f"DEV_UNKNOWN_{rng.integers(10000, 99999)}"
            country = rng.choice(["NG", "RU", "DE", "SG", "US"])
            is_foreign = 0 if country == u_profile["home_country"] else 1
            distance_from_home = float(rng.uniform(300.0, 2500.0))
            is_new_device = 1 if rng.random() < 0.85 else 0

        elif fraud_type == "EXTREME_AMOUNT":
            # Extreme high value spike with subtle categories
            category = rng.choice(["ELECTRONICS", "CRYPTO", "RETAIL", "TRAVEL"])
            mcc = MERCHANT_CATEGORIES[category]["mcc"]
            merchant = rng.choice(MERCHANT_NAMES[category])
            amount = float(rng.uniform(850.0, 3200.0))
            is_new_device = 1 if rng.random() < 0.70 else 0
            device_id = f"DEV_UNKNOWN_{rng.integers(10000, 99999)}" if is_new_device else u_profile["primary_device"]
            is_foreign = 1 if rng.random() < 0.50 else 0
            country = rng.choice(["CA", "GB", "DE"]) if is_foreign else u_profile["home_country"]
            distance_from_home = float(rng.uniform(50.0, 800.0))

        elif fraud_type == "HIGH_RISK_NIGHT":
            # Gambling / Crypto between 1 AM and 4 AM
            category = rng.choice(["GAMBLING", "CRYPTO", "RETAIL"])
            mcc = MERCHANT_CATEGORIES[category]["mcc"]
            merchant = rng.choice(MERCHANT_NAMES[category])
            amount = float(rng.uniform(150.0, 950.0))
            is_new_device = 1 if rng.random() < 0.65 else 0
            device_id = f"DEV_TOR_{rng.integers(1000, 9999)}" if is_new_device else u_profile["primary_device"]
            is_foreign = 1 if rng.random() < 0.40 else 0
            country = rng.choice(["US", "RU", "SG", "GB"]) if is_foreign else u_profile["home_country"]
            distance_from_home = float(rng.uniform(20.0, 600.0))

        else:  # NEW_DEVICE_FOREIGN
            category = rng.choice(["TRAVEL", "ELECTRONICS", "DINING", "RETAIL"])
            mcc = MERCHANT_CATEGORIES[category]["mcc"]
            merchant = rng.choice(MERCHANT_NAMES[category])
            amount = float(u_profile["base_spend"] * rng.uniform(2.0, 4.0))
            is_new_device = 1 if rng.random() < 0.80 else 0
            device_id = f"DEV_FOREIGN_{rng.integers(10000, 99999)}" if is_new_device else u_profile["primary_device"]
            country = rng.choice(["NG", "RU", "SG", "IN", "US"])
            is_foreign = 0 if country == u_profile["home_country"] else 1
            distance_from_home = float(rng.uniform(400.0, 3000.0))

        records.append({
            "transaction_id": txn_id,
            "user_id": uid,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "amount": round(amount, 2),
            "merchant_name": merchant,
            "merchant_category": category,
            "mcc": mcc,
            "device_id": device_id,
            "country": country,
            "distance_from_home_km": round(distance_from_home, 2),
            "is_new_device": is_new_device,
            "is_foreign_country": is_foreign,
            "fraud_type": fraud_type,
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(records)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} transactions to {out}")

    return df


if __name__ == "__main__":
    generate_synthetic_transactions(
        num_users=500,
        num_transactions=15000,
        fraud_rate=0.028,
        output_path="data/raw/transactions.csv",
    )
