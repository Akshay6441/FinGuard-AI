# FinGuard Standard Operating Procedure: Fraud Investigation & Case Disposition Manual
**Document ID:** SOP-OPS-2026-V3
**Classification:** Confidential - Internal Compliance & Fraud Operations
**Effective Date:** January 15, 2026
**Review Cycle:** Annual

---

### Section 1: Objective & Scope
This Standard Operating Procedure (SOP) governs the triage, evidentiary assessment, escalation, and final disposition of flagged credit card, wire, and digital payment transactions within the FinGuard risk monitoring platform. This protocol applies to all Tier-1 fraud analysts, automated AI investigator agents, and senior risk escalation supervisors.

---

### Section 2: Automated Risk Tiering & SLA Matrix

The FinGuard predictive fraud model produces a composite risk probability score \(P \in [0.0, 1.0]\) combining supervised gradient boosting with unsupervised anomaly detection. Transactions are classified according to the following thresholds:

| Risk Tier | Score Range | Primary Action | Max Investigation SLA | Auto-Action Allowed |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Low Risk** | \(0.00 \le P < 0.30\) | Auto-Approve / Clear Alert | Instant (< 100 ms) | Yes |
| **Tier 2: Medium Risk** | \(0.30 \le P < 0.65\) | Soft Challenge / Step-Up Auth | 4 Hours | Yes (SMS OTP / Biometric) |
| **Tier 3: High Risk** | \(0.65 \le P < 0.88\) | Hold Settlement & Agent Triage | 45 Minutes | No (Investigator Review Required) |
| **Tier 4: Critical Risk** | \(0.88 \le P \le 1.00\) | Immediate Block & Account Freeze | 15 Minutes | Yes (Immediate Decline) |

---

### Section 3: Velocity & Behavioral Anomaly Rules

1. **Hourly Velocity Cap (Rule VEL-01):**
   - Any cardholder account executing more than **3 distinct transactions** within a rolling 60-minute window exceeding a cumulative sum of \$1,000 must be immediately placed on temporary authorization hold pending device verification.
2. **Historical Spend Deviation (Rule DEV-02):**
   - If a single transaction amount exceeds **4.5 standard deviations** (\(Z > 4.5\)) from the cardholder's 90-day moving average, automated risk weighting must elevate the base score by a minimum factor of \(+0.25\).
3. **Geo-Velocity Impossibility (Rule GEO-03):**
   - Physical or IP transaction locations occurring within 2 hours of each other with a geographical distance exceeding 500 kilometers ("impossible travel") must be flagged as `CRITICAL_GEO_ANOMALY` and blocked immediately.
4. **New Device & High-Risk MCC Concurrency (Rule MCC-04):**
   - Transactions originating from an unverified device fingerprint targeting Merchant Category Codes 7995 (Gambling/Betting), 6051 (Cryptocurrency/Quasi-Cash), or 5944 (Precious Gems/Jewelry) require mandatory step-up 2FA and manual receipt inspection.

---

### Section 4: Invoice & Receipt Cross-Reference Reconciliation

When an invoice or point-of-sale receipt is submitted for an expense claim, commercial payment, or disputed transaction, the automated optical character recognition (OCR) pipeline extracts:
1. Vendor / Merchant Name
2. Invoice Date & Timestamp
3. Itemized Subtotal, Applicable Tax, and Gross Total
4. Unique Invoice Number

#### Reconciliation Rules & Mismatch Thresholds:
- **Total Amount Variance (Rule REC-01):** The OCR-extracted gross invoice total must match the authorized transaction ledger amount within an absolute tolerance of **\$1.00 USD** or **0.5%** (whichever is lower) to account for currency conversion rounding. A discrepancy exceeding \$5.00 or 1.0% triggers an automatic `AMOUNT_MISMATCH` flag.
- **Date Proximity (Rule REC-02):** The invoice timestamp must be within **-2 days to +3 days** of the transaction authorization date. Invoices dated older than 30 days prior to the transaction date must be marked as `STALE_INVOICE_SUBMISSION`.
- **Vendor Name Similarity (Rule REC-03):** Normalized fuzzy string similarity (Levenshtein distance / Token Sort Ratio) between the invoice header and merchant statement descriptor must score \(\ge 80\%\). Scores below 80% require investigation for ghost company creation or split-merchant billing.
- **Tax Calculation Integrity (Rule REC-04):** If stated tax + subtotal \(\ne\) total by more than \$0.50, flag the invoice as potentially forged or mathematically altered.

---

### Section 5: Standard Action Dispositions & Enforcement

Upon conclusion of an investigation, the investigator (human or AI agent) must issue one of the following five standardized dispositions:

1. **`APPROVE`**: Transaction verified as legitimate. Low risk score, verified invoice reconciliation, normal velocity.
2. **`FLAG_FOR_MANUAL_REVIEW`**: Ambiguous signals present (e.g. minor name discrepancy, moderate score 0.50-0.70). Release funds only after secondary supervisor confirmation.
3. **`REQUEST_ADDITIONAL_KYC`**: Identity or device mismatch. Prompt user for government photo ID and live selfie match before clearing.
4. **`BLOCK_TRANSACTION`**: High risk score (\(P \ge 0.75\)), significant invoice mismatch (\(>\$50\)), or confirmed policy breach. Decline the authorization and reverse pre-auth holds.
5. **`FREEZE_ACCOUNT`**: Confirmed takeover or organized syndicate attack (\(P \ge 0.90\), rapid velocity, impossible travel). Immediately disable all cards, online banking credentials, and submit an internal Suspicious Activity Report (SAR).
