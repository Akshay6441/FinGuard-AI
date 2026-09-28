# Global Card Network Operating Regulations: Fraud & Dispute Resolution Standards
**Document Reference:** GCN-DISPUTE-SPEC-2025
**Scope:** Visa, Mastercard, and Major Payment Rail Dispute Management

---

### Section 1: Overview of Dispute Categories
Payment networks establish uniform dispute rules to allocate liability fairly between issuers (cardholder banks) and acquirers (merchant processing banks). All automated dispute evaluations within FinGuard must cite appropriate Network Reason Codes when proposing chargeback initiation or defense.

---

### Section 2: Reason Code 10.4 — Card-Not-Present (CNP) Fraud
**Definition:** The transaction was processed in an e-commerce, mail-order, or telephone environment where physical card presence and PIN were absent, and the legitimate cardholder denies authorizing or participating in the transaction.

#### Key Conditions & Liability Shifts:
1. **EMV 3-D Secure (3DS / 2.2+ Protocol):**
   - If the merchant successfully verified the cardholder using full 3DS challenge or frictionless biometric verification with a valid Authentication Verification Value (CAVV/AAV), **liability shifts to the card issuer**.
   - If the merchant did not attempt 3DS authentication on a high-risk CNP transaction, the **merchant bears 100% financial liability** under Reason Code 10.4.
2. **Compelling Evidence Standard (CE 3.0):**
   - To defend against a 10.4 dispute, the merchant or investigator must supply:
     - IP address and geographical device location matching previous undisputed purchases.
     - Device fingerprint ID and operating system version verified on prior valid sessions.
     - Delivery address matching the cardholder's verified billing address on file with proof of physical delivery signature.
     - Documented customer correspondence acknowledging receipt of goods.

---

### Section 3: Reason Code 10.5 — Fraudulent Processing & Ghost Merchants
**Definition:** Unauthorized transactions generated through merchant collusion, merchant terminal manipulation, account cycling, or fictitious invoices.

#### Trigger Indicators:
- Repeated round-number transactions (e.g. \$500.00, \$1,000.00) processed within seconds across multiple card numbers.
- Invoices where the legal merchant corporate entity does not match the merchant category or business registration database.
- Absence of verifiable supply chain or shipment tracking numbers for tangible physical goods.

---

### Section 4: High-Risk Merchant Category Codes (MCC)

Transactions within the following MCCs require heightened due diligence and lower alert thresholds:

| MCC Code | Description | Special Requirements |
| :--- | :--- | :--- |
| **7995** | Betting, Lottery & Casino Gaming | Mandatory 3DS 2.0; immediate decline if foreign card or VPN detected. |
| **6051** | Quasi-Cash, Cryptocurrency Exchanges | KYC level 3 verification; daily volume cap enforced per cardholder. |
| **5944** | Jewelry, Watches, Precious Stones | Physical delivery signature required; signature must match invoice recipient. |
| **5732** | Electronics & Computer Hardware | Resale fraud target; invoice must list individual serial numbers for items > \$500. |

---

### Section 5: Time Limits & Arbitration Standards
1. **First Chargeback Filing Window:** The card issuer must submit the chargeback within **120 calendar days** from the transaction settlement date.
2. **Representment / Dispute Response:** The merchant/acquirer has exactly **30 calendar days** to submit rebuttal documentation, including invoice verification reports, OCR logs, and IP match certificates.
3. **Pre-Arbitration:** Unresolved disputes exceeding \$250 may proceed to network arbitration. The losing party incurs a \$500 administrative fee plus total claim indemnification.
