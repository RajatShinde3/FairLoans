import pandas as pd
import joblib
import os

print("📥  Loading model & data …")

# Load trained model
model = joblib.load("results/model_xgb.pkl")

# Load dataset
df = pd.read_csv("data/loan_dataset.csv").dropna()

# Select features and target
features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'region']
df = df[features + ['loan_approved']]
X = df[features]
y = df['loan_approved']

# ✅ One-hot encode categorical features
X_encoded = pd.get_dummies(X)

# ✅ Align columns with training model
model_features = model.get_booster().feature_names
missing_cols = set(model_features) - set(X_encoded.columns)
for col in missing_cols:
    X_encoded[col] = 0  # Add missing columns
X_encoded = X_encoded[model_features]  # Reorder to match model

# Predict
y_pred = model.predict(X_encoded)
y_prob = model.predict_proba(X_encoded)[:, 1]

# Save predictions
results = pd.DataFrame({
    'y_true': y,
    'y_pred': y_pred,
    'y_prob': y_prob
})

os.makedirs("results", exist_ok=True)
results.to_csv("results/baseline_predictions.csv", index=False)
print("✅ Saved to results/baseline_predictions.csv")
