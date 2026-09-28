"""
Cross-reference reconciliation and mismatch verification engine for FinGuard AI.
Compares OCR extracted invoice fields against authorized transaction records.
Outputs mismatch score, anomaly flags, and policy-referenced explanations.
"""

from __future__ import annotations

import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List


def normalize_string(s: str) -> str:
    """Normalize strings for resilient entity comparison."""
    import re
    s_clean = re.sub(r"[^a-zA-Z0-9\s]", " ", s.lower())
    # Remove common business suffixes
    suffixes = ["llc", "inc", "corp", "corporation", "ltd", "co", "store", "market", "supercenter"]
    tokens = [t for t in s_clean.split() if t not in suffixes]
    return " ".join(tokens)


def compute_string_similarity(s1: str, s2: str) -> float:
    """
    Computes fuzzy token similarity ratio between two strings.
    """
    n1 = normalize_string(s1)
    n2 = normalize_string(s2)

    if not n1 or not n2:
        return 0.0
    if n1 == n2 or n1 in n2 or n2 in n1:
        return 1.0

    # Sequence matcher ratio
    seq_ratio = SequenceMatcher(None, n1, n2).ratio()

    # Token overlap (Jaccard-like)
    tokens1 = set(n1.split())
    tokens2 = set(n2.split())
    overlap = len(tokens1.intersection(tokens2)) / max(len(tokens1.union(tokens2)), 1)

    return max(seq_ratio, overlap)


class InvoiceTransactionReconciler:
    """
    Automated reconciliation engine enforcing SOP Section 4:
    Rule REC-01: Amount Variance (< $1.00 / 0.5%)
    Rule REC-02: Date Proximity (-2 to +3 days; stale > 30 days)
    Rule REC-03: Vendor Similarity (>= 80%)
    Rule REC-04: Tax Calculation Integrity
    """

    def __init__(
        self,
        amount_tol_abs: float = 1.00,
        amount_tol_pct: float = 0.005,
        vendor_sim_threshold: float = 0.80,
    ):
        self.amount_tol_abs = amount_tol_abs
        self.amount_tol_pct = amount_tol_pct
        self.vendor_sim_threshold = vendor_sim_threshold

    def reconcile(
        self,
        invoice_fields: Dict[str, Any],
        transaction: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Cross-checks invoice fields against authorized transaction record.
        """
        flags: List[str] = []
        reasons: List[str] = []
        penalties: List[float] = []

        # 1. Amount Reconciliation
        inv_total = invoice_fields.get("total")
        txn_amount = float(transaction.get("amount", 0.0))

        amount_diff = 0.0
        amount_diff_pct = 0.0

        if inv_total is not None:
            inv_total = float(inv_total)
            amount_diff = round(abs(txn_amount - inv_total), 2)
            amount_diff_pct = round((amount_diff / (txn_amount + 1e-5)) * 100, 2)

            if amount_diff > max(self.amount_tol_abs, txn_amount * self.amount_tol_pct):
                flags.append("AMOUNT_MISMATCH")
                severity = "CRITICAL" if amount_diff > 50.0 else "HIGH"
                penalties.append(0.50 if severity == "CRITICAL" else 0.35)
                reasons.append(
                    f"Rule REC-01 Breach ({severity}): Authorized transaction amount is ${txn_amount:.2f}, "
                    f"but invoice total is ${inv_total:.2f} (Discrepancy: ${amount_diff:.2f} / {amount_diff_pct}%)."
                )
            else:
                reasons.append(
                    f"Rule REC-01 Verified: Amount matches (${txn_amount:.2f} vs ${inv_total:.2f})."
                )
        else:
            flags.append("MISSING_INVOICE_TOTAL")
            penalties.append(0.25)
            reasons.append("Could not extract gross total from invoice document.")

        # 2. Vendor Name Reconciliation
        inv_vendor = invoice_fields.get("vendor", "")
        txn_merchant = str(transaction.get("merchant_name", ""))

        vendor_sim = compute_string_similarity(str(inv_vendor), txn_merchant)
        if vendor_sim < self.vendor_sim_threshold:
            flags.append("VENDOR_MISMATCH")
            penalties.append(0.35)
            reasons.append(
                f"Rule REC-03 Breach: Merchant on ledger '{txn_merchant}' does not match "
                f"invoice vendor '{inv_vendor}' (Fuzzy similarity: {vendor_sim:.1%}, required: >={self.vendor_sim_threshold:.0%}). "
                f"Potential ghost vendor or shell entity."
            )
        else:
            reasons.append(
                f"Rule REC-03 Verified: Merchant '{txn_merchant}' matches invoice vendor '{inv_vendor}' "
                f"({vendor_sim:.1%} similarity)."
            )

        # 3. Date Proximity Reconciliation
        inv_date_str = invoice_fields.get("date")
        txn_time_str = str(transaction.get("timestamp", ""))

        days_diff = None
        if inv_date_str and txn_time_str:
            try:
                # Parse date components
                inv_dt = datetime.datetime.strptime(str(inv_date_str)[:10], "%Y-%m-%d").date()
                txn_dt = datetime.datetime.strptime(txn_time_str[:10], "%Y-%m-%d").date()
                days_diff = (txn_dt - inv_dt).days  # positive if invoice was issued before txn

                if days_diff > 30:
                    flags.append("STALE_INVOICE")
                    penalties.append(0.30)
                    reasons.append(
                        f"Rule REC-02 Breach: Invoice date ({inv_dt}) is {days_diff} days prior to "
                        f"transaction ({txn_dt}), exceeding the 30-day stale limit."
                    )
                elif days_diff < -3:
                    flags.append("FUTURE_DATED_INVOICE")
                    penalties.append(0.40)
                    reasons.append(
                        f"Rule REC-02 Breach: Invoice date ({inv_dt}) is post-dated {abs(days_diff)} days "
                        f"after transaction settlement date ({txn_dt})."
                    )
                else:
                    reasons.append(
                        f"Rule REC-02 Verified: Date proximity valid ({days_diff} days delta)."
                    )
            except Exception:
                pass

        # 4. Tax Calculation Integrity
        subtotal = invoice_fields.get("subtotal")
        tax = invoice_fields.get("tax")
        if subtotal is not None and tax is not None and inv_total is not None:
            expected_total = round(float(subtotal) + float(tax), 2)
            if abs(expected_total - float(inv_total)) > 0.50:
                flags.append("TAX_CALCULATION_FORGERY")
                penalties.append(0.30)
                reasons.append(
                    f"Rule REC-04 Breach: Internal mathematical inconsistency in invoice. "
                    f"Subtotal (${subtotal:.2f}) + Tax (${tax:.2f}) = ${expected_total:.2f}, "
                    f"but stated total is ${inv_total:.2f} (Delta: ${abs(expected_total - float(inv_total)):.2f})."
                )

        # Composite Mismatch Score [0.0 = perfect match, 1.0 = severe mismatch]
        raw_score = sum(penalties)
        mismatch_score = min(round(raw_score, 3), 1.0)
        is_mismatch = (mismatch_score >= 0.20) or ("AMOUNT_MISMATCH" in flags) or ("VENDOR_MISMATCH" in flags)

        return {
            "mismatch_score": mismatch_score,
            "is_mismatch": is_mismatch,
            "flags": flags,
            "reasons": reasons,
            "amount_diff": amount_diff,
            "amount_diff_pct": amount_diff_pct,
            "days_diff": days_diff,
            "vendor_similarity": round(vendor_sim, 3),
            "invoice_fields_summary": {
                "vendor": inv_vendor,
                "invoice_no": invoice_fields.get("invoice_no"),
                "date": inv_date_str,
                "total": inv_total,
                "tax": tax,
            },
            "transaction_summary": {
                "transaction_id": transaction.get("transaction_id"),
                "merchant_name": txn_merchant,
                "timestamp": txn_time_str,
                "amount": txn_amount,
            },
        }
