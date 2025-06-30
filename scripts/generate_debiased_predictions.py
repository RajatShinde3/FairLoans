# scripts/generate_debiased_predictions.py

import pandas as pd
import joblib
import os

print("📥  Loading debiased model & dataset...")

# Load debiased model and feature list
model = joblib.load("results/model_debiased_xgb.pkl")
model_features = joblib.load("results/debiased_model_features.pkl")

# Load and clean dataset
df = pd.read_csv("data/loan_dataset.csv").dropna()
df.columns = df.columns.str.strip().str.lower()

# Map 'loan_approved' to 0/1 if it's still in string form
if df['loan_approved'].dtype == object:
    df['loan_approved'] = df['loan_approved'].str.strip().map({
        "Approved": 1,
        "Denied": 0
    })

# Feature columns
features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'zip_code_group']
X = df[features]
y = df['loan_approved']

# One-hot encoding
X_encoded = pd.get_dummies(X)

# Align with training features
for col in model_features:
    if col not in X_encoded.columns:
        X_encoded[col] = 0
X_encoded = X_encoded[model_features]

# Predict using fair model
y_pred = model.predict(X_encoded)

# Save predictions (no y_prob because Fairlearn doesn’t provide probabilities)
results = pd.DataFrame({
    'y_true': y,
    'y_pred': y_pred,
    'y_prob': [None] * len(y)  # Placeholder
})

os.makedirs("results", exist_ok=True)
results.to_csv("results/debiased_predictions.csv", index=False)
print("✅ Saved to results/debiased_predictions.csv")
