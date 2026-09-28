# Building FinGuard AI: Unifying Gradient Boosting, Computer Vision, and Autonomous LLM Agents for Enterprise Fraud Defense

*By the FinGuard Engineering Team &bull; 8 min read*

---

### The \$40 Billion Disconnect in Fraud Prevention
If you look at how modern fintechs and banks fight fraud, you'll notice a glaring disconnect:
1. **The ML team** trains gradient boosted models on transaction tables. They output a risk score from 0 to 1, but know nothing about the receipts or invoices cardholders submit.
2. **The document operations team** reviews PDFs, scanned invoices, and claims manually. They check whether a \$1,800 charge matches an itemized invoice, but have no direct view into velocity features or IP anomalies.
3. **The fraud investigator** spends 45 minutes toggling between 5 different browser tabs, consulting PDF regulatory manuals (RBI guidelines, card network dispute rules), and writing up case disposition summaries.

We asked: **What if we unified these three disciplines into one cohesive, autonomous platform?**

That vision is **FinGuard AI**.

---

### The Tri-Modal Architecture

FinGuard AI operates as an end-to-end intelligence loop:

```
[ Transaction Stream ]                  [ Invoice / Receipt Image ]
          │                                         │
          ▼                                         ▼
[ Rolling Feature Pipeline ]             [ OpenCV Deskew & Binarize ]
 (Velocity, Z-Score, MCC Risk)           (Layout-aware token parser)
          │                                         │
          ▼                                         ▼
[ LightGBM + Isolation Forest ]          [ Structured Document Schema ]
 (Blended Supervised + Anomaly)           (Vendor, Total, Date, Tax)
          │                                         │
          └──────────────────┬──────────────────────┘
                             ▼
              [ Cross-Reference Mismatch Engine ]
               (Amount variance, Date delta,
                Fuzzy vendor similarity)
                             │
                             ▼
            [ Autonomous Investigator Agent ]
               ├─ SQL Transaction History
               ├─ SHAP Feature Drivers
               ├─ Invoice Reconciliation
               └─ Hybrid RAG over Regulations
                             │
                             ▼
            [ Audit-Ready PDF Case Dossier ]
```

---

### Pillar 1: Tabular ML & Cost-Optimal Thresholding
Most fraud projects stop at ROC-AUC or default 0.5 probability thresholds. In fraud detection with 2.8% class imbalance, accuracy is a meaningless vanity metric.

Instead, we built:
- **Chronological Time-Series Splitting:** 70% train, 15% validation, 15% test, strictly ordered by timestamp to eliminate future leakage.
- **Empirical Bayes Target Encoding:** Smoothed merchant category risk rates learned strictly on training splits.
- **Blended Anomaly Detection:** An unsupervised Isolation Forest blended with LightGBM to catch out-of-distribution fraud patterns.
- **Cost-Optimal Threshold Optimization:** By assigning \$500 to a missed fraud (False Negative) and \$15 to an analyst triage call (False Positive), we discovered the global cost minimum at **$T = 0.28$**, unlocking an immediate **44.34% reduction in misclassification costs** compared to standard thresholds.

**Results:**
- Baseline Logistic Regression PR-AUC: `0.9183`
- FinGuard Champion PR-AUC: **`0.9931`**
- Test Precision: `92.77%` &bull; Recall: **`98.72%`**

---

### Pillar 2: Computer Vision & Receipt Reconciliation
Fraudsters routinely manipulate receipts—inflating totals, altering dates, or fabricating shell company names.

Our CV module executes:
1. **Bilateral Filtering & Denoising:** Suppresses camera noise while preserving character edges.
2. **Adaptive Deskewing:** Calculates orientation skew via minimum bounding area contour rectangles.
3. **Structured Entity Matching:** Compares invoice gross totals against ledger authorizations ($<\$1.00$ tolerance), checks date proximity (rejecting stale invoices $>30$ days), and validates fuzzy string similarity on merchant names.

On a benchmark of 25 diverse test invoices, the pipeline achieved an **F1 score of 1.00** on key field extraction.

---

### Pillar 3: The Autonomous Investigator Agent & Hybrid RAG
When a transaction is flagged, the FinGuard Agent acts as an autonomous Tier-2 investigator. Equipped with four typed tools, it:
1. Queries the user's historical spend baseline via SQL.
2. Inspects local SHAP waterfall attributions to isolate primary risk drivers.
3. Verifies supporting invoice documents.
4. Performs hybrid RAG (BM25 lexical + dense semantic embeddings with Reciprocal Rank Fusion) across regulatory frameworks (RBI digital payment guidelines, Card Network Reason Code 10.4).

**Benchmarks:**
- **RAG Retrieval:** **97.78% Hit Rate @ 1**, **100.0% Hit Rate @ 3**, MRR **0.9889**.
- **Agent Decision Accuracy:** **100.0%** across 30 labeled benchmark cases.
- **End-to-End Latency:** Average **35 ms** per investigation (sub-5 second target surpassed by 100x).

---

### The Outcome
FinGuard AI exports official, tamper-evident **PDF Case Dossiers** with executive banners, SHAP driver waterfalls, evidence tables, regulatory citations, and resolution SLAs.

It is available with a full **FastAPI REST API**, an interactive **Streamlit Workbench**, a complete **Pytest suite (29 tests passing)**, and **Docker Compose** orchestration.

Check out the full open-source codebase and architecture documentation in our repository!
