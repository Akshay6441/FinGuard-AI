# FinGuard AI: Model Card

## Model Overview
- **Model Name:** FinGuard Blended Fraud Classifier (GBDT + Isolation Forest)
- **Model Version:** v1.0.0 (Production Release)
- **Model Type:** Hybrid Supervised Gradient Boosted Decision Tree (LightGBM) blended with Unsupervised Isolation Forest Anomaly Detection.
- **Explainability Engine:** SHAP (SHapley Additive exPlanations) TreeExplainer with local per-transaction waterfall attributions.
- **License:** Apache 2.0
- **Maintainers:** FinGuard AI Risk Operations Engineering

---

## Intended Use & Target Users
- **Primary Use Case:** Real-time scoring, triage, and autonomous investigation of authorized credit card, digital wallet, and commercial procurement transactions.
- **Primary Users:** Tier-1 and Tier-2 Fraud Analysts, Compliance Officers, Risk Operations Managers, and Autonomous AI Investigator Agents.
- **Out-of-Scope Uses:** Credit scoring, automated loan rejection without adverse action notices, or automated blocking of essential public utility payments without step-up authentication.

---

## Training Data & Split Methodology
- **Dataset Composition:** 18,000 multi-user sequential transactions spanning 60 continuous days, modeling real-world patterns from Kaggle ULB Credit Card Fraud, IEEE-CIS Fraud Detection, and PaySim.
- **Class Imbalance:** Base fraud prevalence of 2.79% (Train: 352 fraud cases / 12,600 total).
- **Split Strategy:** Strict **chronological time-based split** (70% Train, 15% Validation, 15% Test). Random k-fold cross-validation was explicitly avoided to prevent look-ahead bias and temporal leakage.
- **Feature Set (16 Features):**
  - **Velocity:** `velocity_1h`, `velocity_24h`, `amount_sum_24h`
  - **User Baselines:** `amount`, `amount_to_mean_ratio`, `amount_z_score`
  - **Temporal & Cyclical:** `hour_of_day`, `hour_sin`, `hour_cos`, `is_night`, `is_weekend`
  - **Risk Weightings:** `category_risk_score` (smoothed empirical Bayes), `high_risk_mcc`
  - **Spatial & Device Integrity:** `distance_from_home_km`, `is_new_device`, `is_foreign_country`

---

## Benchmark Performance & Comparative Evaluation

Evaluated on an independent, strictly chronologically partitioned test set of 2,700 transactions (78 fraud cases, 2.89% prevalence):

| Metric | Logistic Regression (Baseline) | Random Forest (Baseline) | FinGuard Champion (Blended LightGBM + IsoForest) |
| :--- | :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | 0.9183 | 0.9912 | **0.9931** |
| **ROC-AUC** | 0.9597 | 0.9997 | **0.9997** |
| **Recall @ 85% Precision** | 0.8846 | 0.9744 | **0.9872** |
| **Test Precision** | 0.8298 | 0.9383 | **0.9277** |
| **Test Recall** | 0.9487 | 0.9744 | **0.9872** |
| **Test F1 Score** | 0.8852 | 0.9559 | **0.9565** |

---

## Cost-Optimal Threshold Optimization
Standard models deploy an arbitrary 0.50 probability cutoff. FinGuard optimizes the expected financial cost matrix:
$$\text{Cost}(T) = \text{FN}(T) \times C_{\text{FN}} + \text{FP}(T) \times C_{\text{FP}}$$
where $C_{\text{FN}} = \$500$ (average loss per missed fraud) and $C_{\text{FP}} = \$15$ (analyst manual triage cost).

- **Default Naive Threshold (0.50) Cost:** \$1,060.00
- **FinGuard Optimal Threshold (0.28) Cost:** \$590.00
- **Net Cost Savings:** **\$470.00 (44.34% operational loss reduction)**

---

## Ethical Considerations & Bias Guardrails
1. **Protected Attributes:** Demographic identifiers (race, gender, age, marital status, nationality) are strictly excluded from the feature space.
2. **Geographical Distance vs Home Country:** Spatial anomalies are measured via physical kilometers from cardholder billing center rather than static country blacklisting.
3. **Adverse Action Transparency:** For every transaction classified above threshold, local SHAP values explicitly itemize the exact statistical drivers responsible for the disposition.
