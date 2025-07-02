# FairLoans: Auditing and Debiasing Loan Approval Systems for Responsible AI

*A trustworthy, explainable, and bias‑aware pipeline for automated loan approvals.*

![Project Status](https://img.shields.io/badge/status-complete-brightgreen)
![Hackathon](https://img.shields.io/badge/HacktheFest-AI_Bias_Bounty-critical)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Business & Ethical Impact](#business--ethical-impact)
3. [Dataset](#dataset)
4. [Solution Overview](#solution-overview)
5. [Technical Stack](#technical-stack)
6. [Project Roadmap & Timeline](#project-roadmap--timeline)
7. [Repository Structure](#repository-structure)
8. [Quick Start](#quick-start)
9. [Usage Guide](#usage-guide)
10. [Results & Metrics](#results--metrics)
11. [Fairness Audit & Mitigation](#fairness-audit--mitigation)
12. [Explainability & Transparency](#explainability--transparency)
13. [Demo Video](#demo-video)
14. [Lessons Learned & Future Work](#lessons-learned--future-work)
15. [Team](#team)
16. [License](#license)

---

## Problem Statement

Financial institutions face increasing regulatory and reputational risk from machine‑learning models that unintentionally discriminate against protected groups. Our goal is to **identify, quantify, and mitigate bias** in a loan‑approval dataset while maintaining strong predictive performance.

## Business & Ethical Impact

* **Legal Compliance**: Satisfy fairness mandates (e.g., ECOA, FHA) to avoid fines.
* **Customer Trust**: Transparent AI decisions strengthen brand reputation.
* **Revenue Growth**: Fair models expand the pool of qualified borrowers, boosting loan volume.

## Dataset

> **Release Date**: 30 June 2025 (provided by HacktheFest)

Key fields (tentative): `loan_status`, `applicant_income`, `gender`, `race`, `loan_amount`, `credit_score`, `region`, etc.

## Solution Overview

1. **Data Exploration & Cleaning**
   Identify missing values, outliers, and sensitive attributes.
2. **Baseline Modeling**
   Train an initial classifier (Logistic Regression, XGBoost) to predict loan approval.
3. **Bias Detection**
   Measure fairness metrics (Statistical Parity, Equal Opportunity, Disparate Impact) using **Fairlearn** & **AIF360**.
4. **Bias Mitigation**
   Apply in‑processing (Exponentiated Gradient with Demographic Parity) strategies.
5. **Explainability**
   Use **SHAP** to visualize feature influence for different demographic groups.
6. **Business Impact Analysis**
   Compare financial and ethical trade‑offs before vs. after mitigation.

## Technical Stack

| Purpose             | Library / Tool                   |
| ------------------- | -------------------------------- |
| Data handling       | `pandas`, `numpy`                |
| Modeling            | `scikit‑learn`, `xgboost`        |
| Fairness analysis   | `fairlearn`, `aif360`            |
| Explainability      | `shap`, `matplotlib`, `seaborn`  |
| UI Dashboard        | `streamlit`                      |
| IDE / Notebook      | `jupyterlab`, `notebook`         |

## Project Roadmap & Timeline

| Date (2025)            | Milestone                           | Deliverable                     |
| ---------------------- | ----------------------------------- | ------------------------------- |
| **Jun 30**             | Dataset drop & repo setup           | Branch `dev`, push skeleton     |
| **Jul 01 – AM**        | EDA + Baseline model                | Notebook `01_explore.ipynb`     |
| **Jul 01 – PM**        | Fairness metrics baseline           | Notebook `02_bias_detect.ipynb` |
| **Jul 02 – AM**        | Mitigation experiments              | Notebook `03_mitigate.ipynb`    |
| **Jul 02 – PM**        | Dashboard & Submission module       | `dashboard.py`, submission CSV  |
| **Jul 03 – AM**        | Demo video & final polish           | `demo.mp4`, README              |
| **Jul 03 11:59 PM PT** | **Submission**                      | Zip and submit                  |

## Repository Structure

```
FairLoans/
├── data/
│   ├── loan_dataset.csv
│   └── test.csv
├── results/
│   ├── model_xgb.pkl
│   ├── model_debiased_xgb.pkl
│   ├── label_encoders.pkl
│   ├── shap_explainer.pkl
│   ├── baseline_predictions.csv
│   ├── debiased_predictions.csv
│   ├── debiased_model_features.pkl
│   └── fairness_report.json
├── notebooks/
│   ├── 01_explore.ipynb 
│   ├── 02_bias_detect.ipynb 
│   └── 03_mitigate.ipynb 
├── scripts/
│   ├── generate_debiased_predictions.py
│   ├── generate_shap_explainer.py
│   └── train_debiased_model.py
├── src/
│   ├── data_prep.py
│   ├── fairness_utils.py
│   ├── train_model.py
│   └── generate_assets.py
├── dashboard.py 
├── run_pipeline.py 
├── generate_submission.py 
├── requirements.txt 
└── README.md 
```

## Quick Start

```bash
git clone https://github.com/RajatShinde3/FairLoans.git
cd FairLoans

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

streamlit run dashboard.py
```

## Usage Guide

1. Upload `loan_dataset.csv` and run EDA notebooks (`01_explore.ipynb`)
2. Use `run_pipeline.py` to train and save the baseline model
3. Run `generate_debiased_predictions.py` to generate debiased output
4. Open the dashboard with Streamlit and explore insights

## Results & Metrics

| Model    | Accuracy | AUC   | Demographic Parity Diff | Equal Opportunity Diff |
| -------- | -------- | ----- | ------------------------| ---------------------- |
| Baseline | 0.86     | 0.92  | 0.23                     | 0.18                  |
| Debiased | 0.83     | 0.90  | 0.06                     | 0.04                  |


### 🔍 Interpretation

| Metric | Baseline | Debiased | Change | Comment |
|--------|----------|----------|--------|---------|
| **Accuracy** | 0.86 | 0.83 | ↓ 0.03 | Slight and acceptable drop |
| **AUC** | 0.92 | 0.90 | ↓ 0.02 | Still strong model discrimination |
| **Demographic Parity Diff** | 0.23 | 0.06 | ↓ 0.17 | Great fairness gain |
| **Equal Opportunity Diff** | 0.18 | 0.04 | ↓ 0.14 | Significant improvement |

### ✅ Summary

- **Fairness improved** significantly across key metrics.
- **Accuracy remained high** (≥ 83%), showing strong predictive performance.
- **Meets industry standards** for bias mitigation and model reliability.
- This result is **submission-ready and impactful**.


## Fairness Audit & Mitigation

The project uses Fairlearn's `MetricFrame` and `ExponentiatedGradient` algorithm to quantify and mitigate bias with respect to gender, race, and region.

## Explainability & Transparency

SHAP summary plots explain how features like `income`, `credit_score`, and `loan_amount` influence predictions.

## Demo Video

👉 Coming soon

## Lessons Learned & Future Work

* Tradeoff between fairness and performance is measurable and manageable.
* Further improvements could use post-processing or adversarial debiasing.
* Consider deployment as a real-time API with dashboards for internal audit.

## Team

| Name         | Role                    | Contributions                                                  |
| ------------ | ----------------------- | ---------------------------------------------------------------|
| Rajat Shinde | Lead ML Developer       | Modeling, fairness analysis, SHAP explainability, dashboard UI,|
|                                        | Code support, documentation, debug support, UX suggestions     |

## License

MIT License — see `LICENSE`