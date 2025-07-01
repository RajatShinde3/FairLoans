# scripts/generate_debiased_predictions.py

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# ──────────────────────────────
# Paths
# ──────────────────────────────
DATA_PATH = Path("data/loan_dataset.csv")
MODEL_PATH = Path("results/model_debiased_xgb.pkl")
FEATS_PATH = Path("results/debiased_model_features.pkl")
OUT_PATH = Path("results/debiased_predictions.csv")

# ──────────────────────────────
# Load Dataset
# ──────────────────────────────
print("📥  Loading dataset...")
if not DATA_PATH.exists():
    raise FileNotFoundError("❌ Dataset not found at data/loan_dataset.csv")

df = pd.read_csv(DATA_PATH)

if "loan_approved" not in df.columns:
    raise ValueError("❌ 'loan_approved' column missing from dataset.")

# Drop unused columns
df = df.drop(columns=["id"], errors="ignore")

# Save ground truth before encoding
y = df["loan_approved"].map({"Denied": 0, "Approved": 1})


# ──────────────────────────────
# Encode Features
# ──────────────────────────────
print("🔤  Encoding categorical variables...")
df_enc = pd.get_dummies(df, drop_first=True)
X = df_enc.drop(columns=["loan_approved"], errors="ignore")

# ──────────────────────────────
# Load Model + Features
# ──────────────────────────────
print("📦  Loading debiased model...")
model = joblib.load(MODEL_PATH)

print("📄  Loading feature list used during training...")
if not FEATS_PATH.exists():
    raise FileNotFoundError("❌ Feature list not found at results/debiased_model_features.pkl")

model_feats = joblib.load(FEATS_PATH)

# ──────────────────────────────
# Align Features
# ──────────────────────────────
print("📐  Aligning features...")
for col in model_feats:
    if col not in X.columns:
        X[col] = 0
X = X[model_feats]

# ──────────────────────────────
# Predict
# ──────────────────────────────
print("🧠  Making predictions...")
try:
    y_pred = model.predict(X)
except Exception as e:
    raise RuntimeError(f"❌ Error during prediction: {e}")

try:
    y_prob = model.predict_proba(X)[:, 1]
    print("✅  Model supports `predict_proba`.")
except:
    y_prob = None
    print("⚠️  Model does NOT support `predict_proba`.")

# ──────────────────────────────
# Save Results
# ──────────────────────────────
print("💾  Saving predictions...")
result_dict = {
    "y_true": y.values,
    "y_pred": y_pred
}
if y_prob is not None:
    result_dict["y_prob"] = y_prob

df_out = pd.DataFrame(result_dict)
df_out.dropna(inplace=True)  # prevent dashboard crash

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT_PATH, index=False)
print(f"✅  Saved debiased predictions to → {OUT_PATH}")
