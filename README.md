# Create a comprehensive README for the project and save it in two locations:
# - /mnt/data/README.md (download link available in this chat)
# - /content/eda_artifacts/README.md (convenient for Colab users)

from pathlib import Path

readme = r"""# Predicting Equipment Failure in Mining Operations

End‑to‑end, reproducible workflow for predicting **Machine Failure** from daily machine telemetry in mining operations.  
The project includes **EDA**, **feature engineering**, **multi‑model development** (with an imbalanced target), **LightGBM hyperparameter tuning**, **thresholding**, and **explainability with SHAP** — all **Colab‑ready**.

> Dataset: 10,000 daily records. Target is the binary `machine_failure` (or equivalent).  
> Failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are **not used as features** (to avoid leakage) and are discussed for **future modelling**.

---

## Table of Contents
- [Project Goals](#project-goals)
- [Data & Assumptions](#data--assumptions)
- [Environment & Setup](#environment--setup)
- [Quickstart (Colab)](#quickstart-colab)
- [Repository / Artifact Structure](#repository--artifact-structure)
- [Workflow Overview](#workflow-overview)
- [Model Development](#model-development)
- [Explainability (SHAP)](#explainability-shap)
- [Key Results Summary](#key-results-summary)
- [Using the Model in Production](#using-the-model-in-production)
- [Integrating Failure Modes (TWF/HDF/PWF/OSF/RNF)](#integrating-failure-modes-twfhdfpwfosfrnf)
- [Monitoring, Bias & Drift](#monitoring-bias--drift)
- [License & Acknowledgements](#license--acknowledgements)

---

## Project Goals
1. **Predict** whether a machine will **fail** on a given day (rare event ~3–4%).  
2. **Explain** drivers of risk via SHAP (load–age–heat story).  
3. **Operate** at a business‑aligned **precision–recall** point (threshold tuning).  
4. **Package** a deployable pipeline (imputation + model) with feature list and params.

---

## Data & Assumptions
- Each row = **one machine‑day** of telemetry. Typical numeric columns:
  - `air_temperature_k`, `process_temperature_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`
  - Categorical: `type` (e.g., L/M/H). High‑cardinality IDs like `product_id` may exist.
- **Target**: `machine_failure` (or column containing “fail”).  
- **Failure modes**: `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are **outcomes** and **excluded** as features.  
- Sanity checks pass: `process_temperature_k ≥ air_temperature_k`, non‑negative rpm/torque/wear.

---

## Environment & Setup
Use the provided `requirements.txt`:

pip install -r requirements.txt

Contents include: `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `lightgbm`, `imbalanced-learn`, `joblib`, `shap`.

**File locations**
- **Colab raw dataset**: `/content/drive/MyDrive/Dataset.csv`
- **Artifacts (saved here)**: `/content/eda_artifacts/`

---

## Quickstart (Colab)

1) **Mount Drive (if using raw CSV in Drive)**
```python
from google.colab import drive
drive.mount('/content/drive')

Run the provided notebooks/cells in order

EDA → EDA report + quality checks

Feature Engineering → saves Dataset_engineered.csv + data dictionary

Multi‑model pipeline (LogReg, RF, SVM, KNN, GBDT, AdaBoost, LightGBM, DecisionTree; optional SMOTE‑NC)

LightGBM tuning → baseline CV → randomized search → refit best → threshold tuning

SHAP → importance table + plots

Export → pickle model + best params + feature names

Artifacts are saved to /content/eda_artifacts/ (see structure below).

/content/eda_artifacts/
├── Dataset_engineered.csv
├── Feature_Engineering_Data_Dictionary.csv
├── EDA_Report.md                     # optional if you ran the EDA report cell
├── LGBM_CV_Top10.csv                 # hyperparameter leaderboard
├── LGBM_BestParams.json              # chosen hyperparameters
├── LGBM_TestMetrics.csv              # AP, ROC-AUC, Precision/Recall/F1, confusion
├── LGBM_TestPredictions.csv          # test set predictions (sorted by risk)
├── LGBM_FeatureNames.json            # features used by the pipeline
├── lgbm_best_model.pkl               # deployable (imputer + LightGBM) pipeline
├── SHAP_Importance.csv               # global mean(|SHAP|) importance
└── README.md                         # this file

Workflow Overview
1) EDA

Numeric/categorical identification, descriptive stats, histograms, boxplots by class.

Imbalance: ~3.39% failures → use PR‑AUC (Average Precision).

Missingness: tiny (≤0.32%) → median imputation in pipeline.

Sanity/physics: process ≥ air; no negative rpm/torque/wear.

Outliers: right‑tail rpm/torque are plausible operating extremes (keep them).

2) Feature Engineering (physics‑informed)

Thermal load: temp_diff_k = process − air; temp_ratio

Mechanical power (kW): power_kw = torque_nm × (2π × rpm / 60) / 1000

Load × age: torque_per_wear, power_per_wear, temp_diff_k × torque

Curvature: torque_sq, rpm_sq, torque_to_rpm

Robust anomalies: median/MAD z‑scores

Type handling: one‑hot type; optional per‑type mean‑centering

Drop/avoid leakage: drop udi, avoid raw product_id (or frequency‑encode)

3) Model Selection (with class imbalance)

Compared: Logistic Regression, Random Forest, SVM, K‑NN, Gradient Boosting, AdaBoost, LightGBM, Decision Tree.

Evaluated with stratified CV, primary metric Average Precision, secondary ROC‑AUC.

Optional SMOTE‑NC variant applied inside CV folds for non‑tree baselines.

4) Best Model: LightGBM

Baseline CV: median imputer + class weighting (scale_pos_weight).

Hyperparameter tuning: randomized search over tree depth/leaves, child samples, subsample/colsample, L1/L2, learning rate, estimators, and pos‑class weight.

Refit best params on Train+Val; threshold tuning on a small val split to max F1 (adjust later per capacity/costs).

5) Explainability

SHAP TreeExplainer on the tuned LightGBM: global importance, beeswarm, top‑3 dependence plots.

Confirms the load–age–heat story: power/torque (↑), wear (↑), temp gap (↑), and low‑rpm/high‑torque pockets drive risk; type_* are smaller baseline shifts.

Model Development

Primary metric: Average Precision (PR‑AUC) (rare target).
Secondary: ROC‑AUC.
Thresholding: tuned to maximise F1; choose production threshold from the PR curve to match inspection capacity and cost of false alarms vs misses.

Key engineered drivers

power_kw, torque_nm, tool_wear_min, temp_diff_k, torque_per_wear / power_per_wear, rotational_speed_rpm (negative effect at high torque), and temp_diff_k × torque interactions.

Global SHAP importance table saved to SHAP_Importance.csv.

Explainability (SHAP)

Use the robust SHAP cell (works with Pipeline or bare estimator). It:

resolves the final estimator (e.g., "lgbm" step),

applies the pipeline preprocessor to align feature spaces,

uses TreeExplainer for LightGBM (falls back to KernelExplainer if needed),

produces: global importance bar, beeswarm, and top‑3 dependence plots,

saves SHAP_Importance.csv.

If you see a column‑name error, ensure you pass a DataFrame with feature names into .predict_proba in any kernel fallback; the provided cell handles this.

Key Results Summary

Best model: LightGBM (class‑weighted), tuned via randomized search.

Performance: see LGBM_TestMetrics.csv for test AP, ROC‑AUC, Precision/Recall/F1, and confusion matrix at the tuned threshold.

Top drivers (SHAP): power_kw (load), torque_nm, tool_wear_min, temp_diff_k, rotational_speed_rpm (lower is riskier at given torque), and their interactions; smaller shifts from type_*.

Using the Model in Production
Load the pickle & score new data
Always show details
import json, joblib, pandas as pd

ART = "/content/eda_artifacts"  # or your deployment path
model = joblib.load(f"{ART}/lgbm_best_model.pkl")
features = json.loads(open(f"{ART}/LGBM_FeatureNames.json").read())

# Load new daily telemetry
new_df = pd.read_csv("new_daily_telemetry.csv")

# Select the exact features in the right order (model has its own imputer)
X = new_df[features]

# Probabilities and binary predictions at your chosen threshold
p_fail = model.predict_proba(X)[:, 1]
thresh = 0.5  # replace with your threshold from validation
y_pred = (p_fail >= thresh).astype(int)

out = new_df.assign(p_fail=p_fail, y_pred=y_pred)
out.to_csv("alerts_scored.csv", index=False)


Tips

Keep the same safe_col standardisation you used in training (lowercase, snake_case).

Monitor score distribution and alert rate; adjust the threshold to respect inspection bandwidth.

If you need calibrated probabilities, add isotonic calibration on a validation hold‑out.

Integrating Failure Modes (TWF/HDF/PWF/OSF/RNF)

Treat them as targets, not features, to avoid leakage. Options:

Two‑stage (recommended): stage‑1 binary failure, then conditional mode classifier on failed rows. Serve joint probs:
P(mode_k & fail) = P(fail) × P(mode_k | fail)

Flat 6‑class: No‑failure vs five modes (class weights needed).

Multi‑task: shared backbone with heads for failure + mode(s).

Evaluate the mode model on failed rows (macro‑F1, per‑class AP). Use SHAP per mode to drive targeted maintenance.

Monitoring, Bias & Drift

Imbalance & threshold: re‑tune thresholds as costs/volumes change.

Slice metrics: by type, wear bins, and operating regimes (rpm/torque ranges).

Data drift: track feature distributions & PSI; set retraining triggers (e.g., quarterly or when AP drops by N%).

Quality gates: alert on physics violations (e.g., process < air, negative rpm/torque/wear).
