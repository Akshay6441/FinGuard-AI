# FinGuard AI: Problem Statement & Product Requirements Document (PRD)

## 1. Executive Summary
Modern financial institutions, payment gateways, and enterprise procurement platforms process millions of transactions and supporting documentation (invoices, receipts, shipping manifests) daily. Fraudulent activities cost global commerce upwards of $40 billion annually. Current fraud prevention systems operate in silos:
1. **Rule-based & tabular ML systems** flag anomalies without unstructured document validation.
2. **Accounts payable and claims teams** manually inspect scanned invoices to detect billing inflation, ghost vendors, or altered totals.
3. **Fraud investigators** spend an average of 45–75 minutes per flagged case querying database histories, calculating feature deviations, consulting regulatory manuals (e.g., RBI guidelines, card network dispute chargeback rules), and manually writing case disposition reports.

**FinGuard AI** unifies these three disciplines into an end-to-end, auditable fraud intelligence platform:
- **Classical Tabular Machine Learning:** Real-time risk scoring, velocity computation, and SHAP explainability.
- **Computer Vision & OCR:** Automated document deskewing, binarization, bounding box text extraction, and cross-reference reconciliation against transaction ledgers.
- **RAG-Powered Autonomous Investigator Agent:** Multi-tool reasoning loop that queries transaction databases, inspects SHAP risk drivers, verifies invoice alignment, retrieves authoritative compliance clauses, and outputs structured, audit-ready case summaries with automated PDF export.

---

## 2. Target Personas & User Journeys

| Persona | Role | Primary Needs | FinGuard AI Impact |
| :--- | :--- | :--- | :--- |
| **Fraud Operations Analyst** | Tier-1 & Tier-2 triage | Rapidly inspect flagged alerts, understand model drivers, verify receipt authenticity | Reduces case triage time from 45 min to < 10 seconds via SHAP waterfalls and automated mismatch scores. |
| **AML & Compliance Officer** | Regulatory oversight & audit | Ensure decisions adhere to digital payment rules (RBI/NPCI/Network rules), document citations | Provides grounded regulatory citations for every suggested action (Hold, Block, Approve). |
| **Risk Operations Lead** | Platform health & monitoring | Track false positive rates, dollar loss prevented, feature drift, model decay | Real-time monitoring dashboard with drift tracking (PSI/KS), PR-AUC evaluation, and case volume analytics. |

---

## 3. High-Level Architecture & Information Flow

```
[ Incoming Transaction ]               [ Scanned Invoice/Receipt ]
          │                                        │
          ▼                                        ▼
[ Feature Engineering Pipeline ]         [ CV Preprocessing & OCR ]
 (Velocity, Z-Score, Merchant Risk)       (Deskew, Binarize, Layout Parser)
          │                                        │
          ▼                                        ▼
[ Supervised & Anomaly Models ]          [ Extracted Document Schema ]
 (LightGBM/XGBoost + Isolation Forest)    (Vendor, Amount, Date, Tax, ID)
          │                                        │
          └─────────────────┬──────────────────────┘
                            ▼
              [ Cross-Check Mismatch Engine ]
               (Amount variance, Date mismatch,
                Fuzzy vendor similarity)
                            │
                            ▼
           [ Autonomous Investigator Agent ]
              ├─ Tool 1: SQL Transaction Ledger History
              ├─ Tool 2: SHAP Feature Attributions
              ├─ Tool 3: Document Verification Output
              └─ Tool 4: Hybrid RAG over Fraud Regulations
                            │
                            ▼
           [ Structured Case Disposition ]
              ├─ Risk Level (LOW, MEDIUM, HIGH, CRITICAL)
              ├─ Key Risk Drivers & Evidence Log
              ├─ Regulatory & SOP Citations
              ├─ Recommended Action & Confidence
              └─ Exportable Case Dossier (PDF)
```

---

## 4. Key Success Metrics & Target KPIs

| Metric Category | Target KPI | Benchmark / Baseline | Business Impact |
| :--- | :--- | :--- | :--- |
| **Model Detection** | **PR-AUC > 0.85**, Recall@90% Precision > 0.75 | Logistic Regression baseline (PR-AUC ~0.45) | Captures 80%+ fraud volume while keeping investigator alert fatigue low. |
| **Cost Optimization** | Optimal Cost Threshold ($C_{FN} = \$500$, $C_{FP} = \$15$) | Fixed 0.5 threshold | Maximizes net dollar savings by minimizing missed fraud costs. |
| **OCR Extraction** | Field-level F1 > 0.88 (Vendor, Date, Total, Tax) | Raw regex on noisy OCR (F1 ~0.60) | High confidence automated reconciliation between receipt and payment. |
| **RAG Retrieval** | Top-3 Retrieval Hit Rate > 90%, Faithfulness > 95% | Naive keyword search (Hit Rate ~65%) | Prevents agent hallucinations; guarantees regulatory auditability. |
| **Investigation Velocity** | End-to-end investigation < 3.5 seconds | Manual investigator review: 45–75 mins | 99% reduction in investigation latency per high-risk alert. |

---

## 5. Security, Compliance & Ethical Guardrails
1. **PII Redaction & Masking:** Cardholder numbers, PANs, and sensitive contact details are masked (`**** **** **** 1234`) across logs, agent prompts, and PDF exports.
2. **Anti-Hallucination Guardrails:** The investigator agent is strictly constrained to verify evidence from tool outputs and cite retrieved policy sections. Speculative conclusions are flagged for mandatory human review.
3. **Audit Trail Immutability:** Every case file maintains a cryptographic hash of the input transaction, OCR evidence, SHAP values, and prompt traces.
