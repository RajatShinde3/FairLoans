"""
generate_predictions.py
-----------------------

Usage
-----

Baseline:      python generate_predictions.py
Debiased:      python generate_predictions.py --model results/model_debiased_xgb.pkl \
                                              --output results/deb_predictions.csv
With submit:   python generate_predictions.py --submit

Arguments
---------

--model   Path to a *.pkl* model.   [default: results/model_xgb.pkl]
--data    Dataset CSV (must include loan_approved). [default: data/loan_dataset.csv]
--output  Where to save prediction CSV. [default: results/baseline_predictions.csv]
--submit  Also write Devpost‑style `submission.csv` (ID + LoanApproved)
"""

import pandas as pd
import joblib
import argparse
import os
from pathlib import Path

def main(model_path, data_path, out_path, write_submit):

    # ── 1. Load model & data ───────────────────────────────────────────
    print(f"📥  Model: {model_path}")
    model = joblib.load(model_path)

    print(f"📥  Data : {data_path}")
    df = pd.read_csv(data_path).dropna()
    df.columns = df.columns.str.strip().str.lower()

    # ── 2. Expected columns ────────────────────────────────────────────
    features = ['age', 'income', 'loan_amount', 'credit_score',
                'gender', 'race', 'zip_code_group']
    target = 'loan_approved'

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing columns in dataset: {missing}")

    X = pd.get_dummies(df[features])
    y = df[target]

    # align feature order with model
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in X:
            X[col] = 0
    X = X[model_features]

    # ── 3. Predict ─────────────────────────────────────────────────────
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({"y_true": y, "y_pred": y_pred, "y_prob": y_prob})
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    results.to_csv(out_path, index=False)
    print(f"✅ Prediction file saved → {out_path}")

    # ── 4. Optional Devpost submission file ────────────────────────────
    if write_submit:
        submission = pd.DataFrame({
            "ID": df.index + 1,
            "LoanApproved": y_pred.astype(int)
        })
        submission.to_csv("submission.csv", index=False)
        print("✅ Devpost submission file → submission.csv\n"
              "   (columns: ID, LoanApproved)")

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prediction CSVs.")
    parser.add_argument("--model",  default="results/model_xgb.pkl")
    parser.add_argument("--data",   default="data/loan_dataset.csv")
    parser.add_argument("--output", default="results/baseline_predictions.csv")
    parser.add_argument("--submit", action="store_true",
                        help="Also create submission.csv (ID, LoanApproved)")
    args = parser.parse_args()

    main(args.model, args.data, args.output, args.submit)
