# scripts/generate_debiased_predictions.py
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ───────────── Paths ─────────────
DATA_PATH = Path("data/loan_dataset.csv")
MODEL_PATH = Path("results/model_debiased_xgb.pkl")
OUT_PATH = Path("results/debiased_predictions.csv")

# ───────────── Load dataset ─────
print("📥  Loading dataset...")
if not DATA_PATH.exists():
    raise FileNotFoundError("Dataset not found at data/loan_dataset.csv")

df = pd.read_csv(DATA_PATH)

# ───────────── Preprocessing ────
print("🔤  Encoding categorical variables...")
df_enc = pd.get_dummies(df.drop(columns=["id"]), drop_first=True)

# Target variable
y = df["loan_approved"]
X = df_enc.drop(columns=["loan_approved"], errors="ignore")

# ───────────── Load model ───────
print("📦  Loading debiased model...")
if not MODEL_PATH.exists():
    raise FileNotFoundError("Debiased model not found. Train it first.")

model = joblib.load(MODEL_PATH)

# ───────────── Align features ───
if hasattr(model, "get_booster"):  # XGBoost-style model
    model_feats = model.get_booster().feature_names
    for col in model_feats:
        if col not in X.columns:
            X[col] = 0
    X = X[model_feats]
elif hasattr(model, "feature_names_in_"):  # sklearn wrapper
    model_feats = model.feature_names_in_
    for col in model_feats:
        if col not in X.columns:
            X[col] = 0
    X = X[model_feats]
else:
    raise ValueError("Unsupported model format")

# ───────────── Predict ──────────
print("🧠  Making predictions...")
y_pred = model.predict(X)

# Try to get y_prob
try:
    y_prob = model.predict_proba(X)[:, 1]
    print("✅  Model supports predict_proba.")
except (AttributeError, NotImplementedError):
    y_prob = None
    print("⚠️  Model does NOT support predict_proba.")

# ───────────── Save results ─────
print("💾  Saving predictions...")
result_dict = {
    "y_true": y,
    "y_pred": y_pred
}
if y_prob is not None:
    result_dict["y_prob"] = y_prob

df_out = pd.DataFrame(result_dict)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_PATH, index=False)
print(f"✅  Saved to {OUT_PATH}")
