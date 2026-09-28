# Regulatory Framework for Digital Payments & Customer Liability Protection
**Regulatory Authority:** Central Banking & Payment Systems Regulatory Framework (RBI/NPCI & Basel FinTech Standards)
**Reference:** REG-DP-COMPLIANCE-2025/26
**Subject:** Limiting Customer Liability in Unauthorized Electronic Payment Transactions and Mandatory Security Controls

---

### Section 1: Customer Liability Determination Framework

Customer financial liability in unauthorized electronic banking and payment transactions is determined strictly based on breach origin and reporting timeline:

#### A. Zero Liability of Customer (Full Institutional Indemnification)
A customer shall have ZERO liability in the following circumstances:
1. **Contributory Fraud / Negligence by Bank / Payment System:** Contributory fraud, technical system error, or systemic security failure occurring on the payment processing platform or issuing bank infrastructure, regardless of whether reported by customer.
2. **Third-Party Breach with Prompt Reporting:** Third-party compromise (e.g. merchant database leak, cyber breach, malware skimming) where neither the payment provider nor the customer is at fault, **provided the customer notifies the payment provider within three (3) working days** of receiving the transaction notification alert.

#### B. Limited Liability of Customer (Tiered Liability Matrix)
If the customer reports the unauthorized transaction after three working days:

| Notification Timeframe | Customer Liability Extent | Maximum Liability Cap |
| :--- | :--- | :--- |
| **Within 3 working days** | Zero liability | \$0.00 / Zero |
| **4 to 7 working days** | Transaction value or statutory cap (whichever is lower) | Max \$100 (Basic Accounts) / Max \$250 (Credit Cards) |
| **Beyond 7 working days** | As per Board-approved institutional risk policy | Full loss subject to judicial review |

#### C. Full Customer Liability
The customer shall bear the entire loss in cases where the loss is solely due to negligence by the customer, such as:
1. Voluntarily sharing secret authentication credentials (PIN, SMS OTP, biometric bypass, CVV) with unauthorized third parties.
2. Compromised device jailbreaking / root manipulation with explicit credential export.
*(Note: Even in cases of customer negligence, any unauthorized transactions occurring AFTER the customer has reported the credential compromise or blocked the card must be 100% indemnified by the financial institution).*

---

### Section 2: Mandatory Two-Factor Authentication (2FA / AFA)
1. **Additional Factor of Authentication (AFA):**
   - All card-not-present (CNP) e-commerce, digital wallet top-up, and recurring mandate modifications must mandate dynamic out-of-band two-factor authentication (dynamic SMS OTP, TOTP authenticator app, or device hardware crypto key).
   - Relaxations apply only to contactless tap-to-pay under regulatory micro-transaction limits (\$50 / INR 5,000 equivalent) or whitelisted recurring subscriptions with established historical mandates.
2. **Dynamic Frictionless Step-Up:**
   - Any transaction exhibiting anomalous risk attributes (abnormal velocity, foreign IP, new device fingerprint, or ML fraud score \(P \ge 0.65\)) MUST trigger a dynamic step-up challenge even if the transaction amount is below the micro-payment threshold.

---

### Section 3: Suspicious Activity Reporting (SAR / STR) & FIU Obligations
1. **Threshold for Mandatory Filing:**
   - Any transaction or series of connected transactions totaling \$10,000 or greater exhibiting patterns consistent with money laundering, synthetic identity fraud, or structured smurfing must be logged for a Suspicious Transaction Report (STR) to the Financial Intelligence Unit (FIU) within **7 calendar days**.
2. **Tipping-Off Prohibition:**
   - Under no circumstances shall the platform or investigator inform the transaction originator that an STR or AML inquiry has been registered.
