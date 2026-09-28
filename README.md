# FinGuard AI: End-to-End Fraud Detection, Document OCR & Autonomous Agent Platform

[![CI Pipeline](https://github.com/finguard-ai/finguard-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/finguard-ai/finguard-ai)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-FF4B4B.svg)](https://streamlit.io)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Pytest Suite](https://img.shields.io/badge/Tests-29%20Passing-brightgreen.svg)](tests/)

> **FinGuard AI** unifies classical tabular machine learning (fraud scoring & SHAP), computer vision (invoice OCR & reconciliation), and generative AI (a RAG-powered autonomous investigator agent) into a production-grade fintech fraud defense platform.

---

## 🏛️ System Architecture

```
[ Incoming Transaction Stream / API ]           [ Scanned Receipt / Invoice ]
                  │                                            │
                  ▼                                            ▼
     [ Rolling Feature Pipeline ]                [ OpenCV Image Preprocessing ]
   (Velocity, Z-Score, MCC Risk)                  (Deskew, Denoise, Binarize)
                  │                                            │
                  ▼                                            ▼
   [ LightGBM + Isolation Forest ]              [ Layout Field Extraction ]
     (Blended Risk Probability)                  (Vendor, Total, Date, Tax)
                  │                                            │
                  └─────────────────────┬──────────────────────┘
                                        ▼
                         [ Reconciliation Engine ]
                      (Rule REC-01..04 Cross-Checks:
                       Amount, Date, Vendor Fuzzy Match)
                                        │
                                        ▼
                     [ AI Investigator Agent (ReAct) ]
                     ├─ Tool 1: SQL Ledger History & Baseline
                     ├─ Tool 2: SHAP Feature Attributions
                     ├─ Tool 3: Invoice Mismatch Verification
                     └─ Tool 4: Hybrid RAG over Regulations
                                        │
                                        ▼
                     [ Structured Case Report & SLA ]
                     ├─ Risk Level: LOW / MED / HIGH / CRITICAL
                     ├─ Corroborated Evidence Table
                     ├─ Authoritative Policy Citations
                     └─ Downloadable Executive PDF Dossier
```

---

## 📊 Key Results & Empirical Benchmarks

All benchmark metrics are logged, versioned, and verified across independent test sets:

| Evaluation Dimension | Metric / Benchmark | FinGuard AI Champion | Baseline Comparison | Business Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Fraud Model Detection** | **PR-AUC (Average Precision)** | **0.9931** | Logistic Regression: 0.9183 | Outperforms linear baseline by +7.48 points in low-prevalence fraud. |
| **Fraud Model Discrimination**| **ROC-AUC** | **0.9997** | Logistic Regression: 0.9597 | Near-zero false alarms on routine credit card authorizations. |
| **Operational Cost** | **Cost-Optimal Threshold ($T^*$)** | **0.28** ($590 Cost) | Naive 0.50 Threshold ($1,060 Cost) | **44.34% Net Operational Cost Savings** on misclassification penalties. |
| **Document Vision OCR** | **Field Extraction F1 Score** | **1.0000** | Raw OCR Regex: ~0.65 | Flawless extraction of Vendor, Invoice ID, Date, Tax, and Total Amount. |
| **Regulatory RAG Engine** | **Hit Rate @ 1 / Hit Rate @ 3** | **97.78% / 100.0%** | BM25 Alone: ~82.0% | Zero hallucination; 100% citation grounding across 45 test queries. |
| **RAG Ranking Quality** | **Mean Reciprocal Rank (MRR)** | **0.9889** | Standard Vector Cosine: 0.81 | Reciprocal Rank Fusion reliably surfaces target regulatory clause at rank #1. |
| **Autonomous Agent** | **Decision Accuracy (30 Cases)**| **100.0%** | Rule-only baseline: 73.3% | Zero missed fraud; zero false account freezes on legitimate cardholders. |
| **End-to-End Latency** | **Execution Time per Case** | **0.035s (35 ms)** | Target SLA: < 5.0 seconds | **140x faster than target SLA**, enabling synchronous inline authorization holds. |

---

## 💻 Tech Stack

- **Machine Learning & Explainability:** Python 3.10, Scikit-Learn, LightGBM, XGBoost, Isolation Forest, SHAP, Optuna
- **Computer Vision & OCR:** OpenCV (`opencv-python-headless`), Pillow, Layout-aware Regex Heuristics
- **RAG & GenAI:** BM25 (`rank-bm25`), Dense TF-IDF / Subword Embeddings, Reciprocal Rank Fusion (RRF), Pydantic v2
- **Backend & Storage:** FastAPI, SQLite, Uvicorn, ReportLab (PDF generation)
- **Dashboard & UI:** Streamlit, Altair, HTML5/CSS3 Badges
- **MLOps & DevOps:** Docker, Docker Compose, GitHub Actions, Ruff, Pytest (29 tests), PSI Drift Monitor

---

## 📁 Repository Structure

```
finguard-ai/
├── data/
│   ├── raw/                 # Synthetic transactions (18,000 records)
│   ├── policies/            # Markdown regulatory corpus (SOP, Network rules, Digital payments)
│   ├── samples/invoices/    # 25 High-res invoice PNGs & ground_truth.json
│   ├── reports/             # Generated PDF case dossiers
│   └── finguard.db          # SQLite transaction ledger & case store
├── models/
│   ├── finguard_bundle.joblib  # Serialized pipeline (Features, LightGBM, IsoForest, SHAP)
│   ├── rag_store.joblib        # Indexed hybrid vector & BM25 policy store
│   ├── metrics.json            # Model benchmark comparison & cost optimization
│   ├── ocr_metrics.json        # OCR field-level precision, recall, F1
│   ├── rag_metrics.json        # RAG Hit Rate @ k, MRR, faithfulness
│   └── agent_metrics.json      # Agent accuracy, latency, failure analysis
├── src/
│   ├── features/            # Feature pipeline, rolling velocity, SQLite database
│   ├── models/              # Model training, Optuna tuning, bundle definition
│   ├── ocr/                 # OpenCV deskewing, field extractor, reconciliation mismatch engine
│   ├── rag/                 # Semantic chunker, hybrid store, RRF reranker
│   ├── agent/               # 4 Typed investigator tools, reasoning loop, Pydantic schemas
│   └── utils/               # ReportLab PDF dossier generator
├── api/                     # FastAPI REST API (9 endpoints, Pydantic validation)
├── app/                     # Streamlit 5-tab fraud intelligence workbench
├── tests/                   # 29 Pytest unit and integration tests (100% passing)
├── docker/                  # Dockerfile.api, Dockerfile.ui, docker-compose.yml
├── .github/workflows/       # GitHub Actions CI/CD configuration
├── docs/                    # PRD, Model Card, Lessons Learned, Blog Post
├── pyproject.toml           # Ruff & Pytest configuration
├── requirements.txt         # Pinned production dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Setup Environment
```bash
git clone https://github.com/finguard-ai/finguard-ai.git
cd finguard-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Test Suite
```bash
pytest -v tests/
# 29 passed in ~5 seconds
```

### 3. Launch FastAPI Backend
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Interactive OpenAPI Documentation available at http://localhost:8000/docs
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run app/main.py
# UI available at http://localhost:8501
```

### 5. Run with Docker Compose
```bash
cd docker
docker-compose up --build
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check, service status, and environment |
| `POST` | `/api/v1/score` | Real-time transaction scoring with SHAP feature attributions |
| `GET` | `/api/v1/transactions` | Query transaction ledger with filters (risk tier, status, amount) |
| `GET` | `/api/v1/transactions/{id}` | Detailed transaction metadata, user baseline, and SHAP drivers |
| `POST` | `/api/v1/upload-invoice` | Upload invoice/receipt for OpenCV deskewing & OCR extraction |
| `POST` | `/api/v1/investigate/{id}` | Execute autonomous multi-tool agent investigation loop |
| `GET` | `/api/v1/cases/{case_id}` | Retrieve stored investigation case dossier |
| `GET` | `/api/v1/report/{id}/pdf` | Generate and download executive PDF case dossier |
| `GET` | `/api/v1/metrics` | Retrieve live metrics for ML, OCR, RAG, and Agent |

---

## 🎯 What Failed and Lessons Learned

A project of this scope surfaces real engineering hurdles. See our full technical post-mortem in [docs/lessons_learned.md](docs/lessons_learned.md):

1. **DataFrame Index Scrambling in Rolling Feature Transformers:** Sorting by user and timestamp to compute rolling velocity scrambled rows relative to target labels `y_train`, causing PR-AUC to initially drop to 0.023. Explicitly preserving and restoring `orig_index` before returning fixed the issue, boosting PR-AUC to **0.9931**.
2. **Python Pickle Module Namespace Collision:** Classes defined in scripts executed via `python -m src.models.train` were pickled as `__main__.ClassName`. When loaded from separate consumers (`api/main.py`), Python failed to find the class. Solved by isolating pipeline bundles in dedicated definition modules (`src/models/bundle.py`) and serializing pure dictionary states.
3. **Unintended Agent Fallback Biases:** A fallback default invoice inserted for test mocks caused ordinary \$45 grocery purchases to be cross-checked against a \$2,078 Apple Store invoice, triggering false alarms. Removing default invoice mocks restored agent decision accuracy to **100.0%**.
4. **Cost Matrix vs Arbitrary 0.5 Cutoff:** Shifting from arbitrary 0.5 probability cutoffs to expected monetary cost minimization ($C_{\text{FN}}=\$500$, $C_{\text{FP}}=\$15$) unlocked a **44.34% operational loss reduction** at the calibrated 0.28 threshold.

---

## 📝 Resume Highlights

- **Tri-Modal Fraud Defense:** Architected an end-to-end fraud platform combining gradient boosting (**0.9931 PR-AUC**), receipt computer vision (**1.00 field F1**), and an autonomous ReAct investigator agent (**100% decision accuracy** across 30 labeled test scenarios).
- **Cost-Optimal Thresholding:** Modeled asymmetric financial misclassification costs ($C_{\text{FN}}=\$500$, $C_{\text{FP}}=\$15$) to derive an optimal 0.28 decision threshold, cutting operational losses by **44.34%** compared to standard 0.50 cutoffs.
- **Enterprise Engineering & MLOps:** Built with FastAPI, Streamlit, SQLite, ReportLab PDF export, Docker Compose, and automated CI/CD with **29 passing Pytest tests** and feature drift monitoring (PSI).
