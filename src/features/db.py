"""
Database interface and SQLite persistent storage for FinGuard AI.
Provides tables for transactions, user profiles, and investigation case dossiers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DB_PATH = Path("data/finguard.db")


def get_db_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path | str = DB_PATH, populate_from_csv: bool = True) -> None:
    """
    Initialize SQLite schema with tables: transactions, users, and cases.
    """
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        amount REAL NOT NULL,
        merchant_name TEXT NOT NULL,
        merchant_category TEXT NOT NULL,
        mcc TEXT NOT NULL,
        device_id TEXT NOT NULL,
        country TEXT NOT NULL,
        distance_from_home_km REAL NOT NULL,
        is_new_device INTEGER NOT NULL,
        is_foreign_country INTEGER NOT NULL,
        fraud_type TEXT,
        is_fraud INTEGER NOT NULL,
        fraud_score REAL,
        status TEXT DEFAULT 'PENDING'
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        home_country TEXT NOT NULL,
        primary_device TEXT NOT NULL,
        risk_segment TEXT NOT NULL,
        avg_spend REAL,
        total_transactions INTEGER DEFAULT 0
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY,
        transaction_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        recommended_action TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        investigation_summary TEXT NOT NULL,
        evidence_json TEXT NOT NULL,
        policy_citations_json TEXT NOT NULL,
        report_pdf_path TEXT,
        FOREIGN KEY (transaction_id) REFERENCES transactions (transaction_id)
    )
    """)

    conn.commit()

    if populate_from_csv:
        csv_path = Path("data/raw/transactions.csv")
        eval_csv_path = Path("models/test_evaluated_transactions.csv")
        source_csv = csv_path if csv_path.exists() else eval_csv_path

        if source_csv.exists():
            df = pd.read_csv(source_csv)
            # Check if all transactions loaded
            count = cur.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            if count < len(df):
                print(f"Populating DB from {source_csv} ({len(df)} records)...")
                # Insert or ignore transactions
                for _, row in df.iterrows():
                    score = float(row["fraud_score"]) if "fraud_score" in row and pd.notna(row["fraud_score"]) else 0.05
                    status = "FLAGGED" if score >= 0.28 else "APPROVED"
                    cur.execute("""
                    INSERT OR REPLACE INTO transactions (
                        transaction_id, user_id, timestamp, amount, merchant_name,
                        merchant_category, mcc, device_id, country, distance_from_home_km,
                        is_new_device, is_foreign_country, fraud_type, is_fraud, fraud_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(row["transaction_id"]),
                        str(row["user_id"]),
                        str(row["timestamp"]),
                        float(row["amount"]),
                        str(row["merchant_name"]),
                        str(row["merchant_category"]),
                        str(row["mcc"]),
                        str(row["device_id"]),
                        str(row["country"]),
                        float(row["distance_from_home_km"]),
                        int(row["is_new_device"]),
                        int(row["is_foreign_country"]),
                        str(row.get("fraud_type", "NORMAL")),
                        int(row["is_fraud"]),
                        score,
                        status
                    ))

                # Populate users table
                user_stats = df.groupby("user_id").agg({
                    "amount": "mean",
                    "transaction_id": "count",
                    "country": "first",
                    "device_id": "first",
                }).reset_index()

                for _, u_row in user_stats.iterrows():
                    cur.execute("""
                    INSERT OR REPLACE INTO users (
                        user_id, home_country, primary_device, risk_segment, avg_spend, total_transactions
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        str(u_row["user_id"]),
                        str(u_row["country"]),
                        str(u_row["device_id"]),
                        "STANDARD",
                        round(float(u_row["amount"]), 2),
                        int(u_row["transaction_id"]),
                    ))

                conn.commit()
                print("Database populated successfully.")

    conn.close()


def query_transaction_by_id(txn_id: str, db_path: Path | str = DB_PATH) -> Optional[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM transactions WHERE transaction_id = ?", (txn_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def query_user_recent_history(
    user_id: str,
    before_timestamp: Optional[str] = None,
    limit: int = 20,
    db_path: Path | str = DB_PATH,
) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cur = conn.cursor()
    if before_timestamp:
        cur.execute("""
        SELECT * FROM transactions
        WHERE user_id = ? AND timestamp < ?
        ORDER BY timestamp DESC LIMIT ?
        """, (user_id, before_timestamp, limit))
    else:
        cur.execute("""
        SELECT * FROM transactions
        WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT ?
        """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
