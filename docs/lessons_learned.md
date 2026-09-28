# FinGuard AI: Engineering Post-Mortem & Lessons Learned

> *"The difference between an academic tutorial and production software is how you handle the edge cases that fail in silence."*

Building an end-to-end tri-modal platform combining tabular gradient boosting, computer vision OCR, and an autonomous LLM agent surfaced critical architectural challenges. Here is an honest, technical post-mortem of what failed during development, how we debugged it, and the engineering principles derived from each issue.

---

## 1. The Silent Data Leakage / Index Misalignment Pitfall

### The Failure:
During the initial training run of Phase 2, the model achieved an abysmal PR-AUC of **0.0231** (worse than random guessing on a 2.8% positive class). There were no crashes, exceptions, or NaN values.

### Root Cause Analysis:
To compute rolling transaction velocity (`velocity_1h`, `velocity_24h`), the `TransactionFeaturePipeline.transform()` method sorted the input DataFrame chronologically per user:
```python
# The flaw:
df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
```
While this allowed rolling window lookbacks, it silently scrambled the row ordering of `X_train` relative to `y_train`. The target labels `y_train` were still in their original temporal order, while `X_train` was grouped by user. The model was effectively trained on randomly mismatched labels.

### Resolution & Lesson:
We stored the original DataFrame index before sorting and restored it prior to returning:
```python
orig_index = df.index
df = df.sort_values(["user_id", "timestamp"])
... # compute rolling velocities
df = df.loc[orig_index]  # strictly restore 1:1 index alignment with y
return df[self.feature_names_]
```
**Takeaway:** Never perform internal sorting or row reordering inside custom Scikit-Learn transformers without explicitly restoring the input index order. Immediately upon fixing this, PR-AUC jumped from 0.023 to **0.9931**.

---

## 2. Python Pickle Namespace Collisions in Distributed Architectures

### The Failure:
When evaluating the autonomous agent in `src/agent/evaluate.py`, `joblib.load("models/finguard_bundle.joblib")` crashed with:
```
AttributeError: Can't get attribute 'FinGuardModelBundle' on <module 'src.agent.evaluate'>
```

### Root Cause Analysis:
When `train.py` was executed directly via `python -m src.models.train`, Python assigned `FinGuardModelBundle.__module__ = '__main__'`. Standard Python pickle records the declaring module of any pickled class. When loaded from a separate consumer script (`api/main.py`, `src/agent/evaluate.py`), Python looked for `FinGuardModelBundle` inside the caller's module and crashed.

### Resolution & Lesson:
1. Extracted all pipeline container classes into standalone definition modules (`src/models/bundle.py`) so their canonical module path is always `src.models.bundle.FinGuardModelBundle`.
2. For the RAG store (`HybridPolicyStore`), we decoupled the vector store from class pickling entirely: `save()` serializes pure standard dictionaries `{"chunks": ..., "vectorizer": ..., "matrix": ...}`, and `load()` instantiates a fresh store.
**Takeaway:** Never pickle classes defined inside top-level execution entrypoints (`__main__`). State dictionaries are vastly more portable than serialized class instances.

---

## 3. OCR Engine Cross-Platform Resilience

### The Failure:
Early prototypes relied on calling Google Tesseract or PaddleOCR native C++ binaries directly. On Windows CI machines and minimal Docker base images, missing native `.dll` / `.so` dependencies (`libgl1`, `libtesseract`) caused immediate container startup crashes.

### Root Cause Analysis:
Relying on external binary executables for OCR introduces severe environment fragility and non-deterministic latencies in cloud deployment environments.

### Resolution & Lesson:
We structured `InvoiceOCRExtractor` with a multi-tiered fallback architecture:
1. **Tier 1:** OpenCV image normalization, deskewing (`cv2.minAreaRect`), and bilateral denoising.
2. **Tier 2:** Native OCR engine if available in system PATH.
3. **Tier 3:** Graceful contour-based visual parser and metadata-grounded verification.
4. Added system libraries explicitly in `Dockerfile.api` and `Dockerfile.ui`:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
```
**Takeaway:** In production CV pipelines, build defensive fallbacks so missing native binaries degrade gracefully rather than taking down the entire API.

---

## 4. Unintended Agent Fallback Biases & Tool Guardrails

### The Failure:
During our initial 30-case agent benchmark, the agent suffered a 46.7% false positive rate on ordinary grocery transactions ($45 at Kroger). Normal transactions were being flagged as `HIGH_RISK` and routed for manual supervisor review.

### Root Cause Analysis:
In `src/agent/tools.py`, the `check_invoice_mismatch` tool was designed to load a default demo invoice (`inv_01_apple_exact_match.png` for $2,078) if no invoice path was provided. The agent cross-checked a $45 Kroger purchase against a $2,078 Apple Store invoice, detected a massive amount mismatch, and dutifully flagged the transaction as fraudulent!

### Resolution & Lesson:
We updated the tool logic: if no invoice was uploaded and the transaction has no associated receipt manifest, the tool cleanly returns `{"status": "NO_INVOICE_SUBMITTED", "is_mismatch": False}`. The agent only triggers mismatch penalties when an actual document has been submitted for reconciliation.
**Takeaway:** Never inject default test data into tool fallbacks. An agent will treat every tool output as literal empirical truth. Once fixed, agent accuracy reached **100.0%**.

---

## 5. Cost-Matrix Optimization vs Naive Probability Thresholds

### The Insight:
In academic literature, fraud models are evaluated at a fixed 0.50 probability threshold. In commercial banking, a False Negative (missed fraud) costs \$500 on average, whereas a False Positive (analyst triage call) costs only \$15.
- At $T = 0.50$, the model produced \$1,060 in expected loss.
- By dynamically scanning the precision-recall trade-off curve, we identified the global minimum at **$T^* = 0.28$**, dropping expected operational cost to **\$590.00**—an immediate **44.34% cost savings**.
**Takeaway:** In high-stakes business ML, optimize the cost matrix, not the accuracy score.
