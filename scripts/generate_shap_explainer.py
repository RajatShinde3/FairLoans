# scripts/generate_shap_explainer.py

import pandas as pd
import joblib
import shap
import os

print("🔍 Generating SHAP explainer...")

# Load trained baseline model
model = joblib.load("results/model_xgb.pkl")

# Load and clean dataset
df = pd.read_csv("data/loan_dataset.csv").dropna()
df.columns = df.columns.str.strip().str.lower()

# Convert 'loan_approved' if it's in string format
if df['loan_approved'].dtype == object:
    df['loan_approved'] = df['loan_approved'].str.strip().map({
        "Approved": 1,
        "Denied": 0
    })

# Features used in training
features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'zip_code_group']
if not all(f in df.columns for f in features):
    missing = [f for f in features if f not in df.columns]
    raise ValueError(f"Missing required columns: {missing}")

# Prepare feature set
X = df[features]
X_encoded = pd.get_dummies(X)

# Align with model feature names
model_features = model.get_booster().feature_names
for col in model_features:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[model_features]

# SHAP Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X_encoded)

# Save SHAP explainer
os.makedirs("results", exist_ok=True)
joblib.dump(explainer, "results/shap_explainer.pkl")
print("✅ SHAP explainer saved to results/shap_explainer.pkl")
