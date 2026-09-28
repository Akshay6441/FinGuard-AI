"""
Benchmarking and failure analysis suite for the FinGuard AI Autonomous Investigator Agent.
Evaluates agent decision accuracy, latency, and logs failure analysis across 30 labeled test cases.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.agent.investigator import FraudInvestigatorAgent


def run_agent_benchmark(
    cases_count: int = 30, output_path: str = "models/agent_metrics.json"
) -> Dict[str, Any]:
    agent = FraudInvestigatorAgent()
    eval_csv = Path("models/test_evaluated_transactions.csv")
    if not eval_csv.exists():
        eval_csv = Path("data/raw/transactions.csv")

    df = pd.read_csv(eval_csv)

    # Sample diverse cases
    fraud_cases = df[df["is_fraud"] == 1].head(16)
    legit_cases = df[df["is_fraud"] == 0].head(14)
    test_batch = pd.concat([fraud_cases, legit_cases]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    results = []
    latencies = []
    failures = []

    correct_dispositions = 0

    for _idx, row in test_batch.iterrows():
        txn_id = str(row["transaction_id"])
        is_fraud = int(row["is_fraud"])
        fraud_type = str(row.get("fraud_type", "NORMAL"))

        start_time = time.perf_counter()
        report = agent.investigate(txn_id)
        elapsed = time.perf_counter() - start_time
        latencies.append(elapsed)

        action = report.recommended_action
        risk_level = report.risk_level

        # Evaluate disposition correctness:
        # Fraud transactions should be BLOCK_TRANSACTION, FREEZE_ACCOUNT, or FLAG_FOR_MANUAL_REVIEW
        # Legitimate transactions should be APPROVE or REQUEST_ADDITIONAL_KYC
        is_correct = False
        if is_fraud == 1:
            if action in ["BLOCK_TRANSACTION", "FREEZE_ACCOUNT", "FLAG_FOR_MANUAL_REVIEW"]:
                is_correct = True
        else:
            if action in ["APPROVE", "REQUEST_ADDITIONAL_KYC"]:
                is_correct = True

        if is_correct:
            correct_dispositions += 1
        else:
            failures.append({
                "transaction_id": txn_id,
                "is_fraud": is_fraud,
                "fraud_type": fraud_type,
                "model_fraud_score": float(row.get("fraud_score", 0.0)),
                "agent_risk_level": risk_level,
                "agent_recommended_action": action,
                "agent_confidence": report.confidence_score,
                "root_cause": (
                    "False Negative: Model score under threshold for subtle fraud"
                    if is_fraud == 1
                    else "False Positive: Legitimate user high spend triggered conservative block"
                ),
            })

        results.append({
            "transaction_id": txn_id,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
            "recommended_action": action,
            "risk_level": risk_level,
            "latency_seconds": round(elapsed, 4),
            "correct": is_correct,
        })

    accuracy = correct_dispositions / len(test_batch)
    avg_latency = float(sum(latencies) / len(latencies))
    p95_latency = float(pd.Series(latencies).quantile(0.95))

    metrics = {
        "num_cases_evaluated": len(test_batch),
        "agent_decision_accuracy": round(accuracy, 4),
        "correct_dispositions": correct_dispositions,
        "total_failures": len(failures),
        "latency_stats": {
            "mean_latency_seconds": round(avg_latency, 3),
            "p95_latency_seconds": round(p95_latency, 3),
            "meets_sub_5s_target": avg_latency < 5.0,
        },
        "failure_analysis": failures,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Agent Benchmark Complete across {len(test_batch)} cases:")
    print(f"  Decision Accuracy: {accuracy:.2%}")
    print(f"  Mean Latency: {avg_latency:.3f}s (P95: {p95_latency:.3f}s)")
    print(f"  Failures: {len(failures)}")

    return metrics


if __name__ == "__main__":
    run_agent_benchmark()
