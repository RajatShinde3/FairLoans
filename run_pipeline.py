# run_pipeline.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Config
DATA_PATH = "data/loan_dataset.csv"
MODEL_PATH = "results/model_xgb.pkl"
ENCODER_PATH = "results/label_encoders.pkl"
PREDICTIONS_PATH = "results/preds.csv"

# Load data
print("[📥] Loading data...")
df = pd.read_csv(DATA_PATH)
df.dropna(inplace=True)

# Encode categorical columns
print("[🔤] Encoding categorical variables...")
label_encoders = {}
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Split features and target
X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("[🧠] Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
print("[✅] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[📊] Accuracy: {acc:.4f}")

# Save model & encoders
os.makedirs("results", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(label_encoders, ENCODER_PATH)
print(f"[💾] Model saved to {MODEL_PATH}")
print(f"[💾] Label encoders saved to {ENCODER_PATH}")

# Save predictions for fairness analysis
print("[📁] Saving predictions for fairness audit...")
preds_df = X_test.copy()
preds_df["y_true"] = y_test
preds_df["y_pred"] = y_pred
preds_df.to_csv(PREDICTIONS_PATH, index=False)
print(f"[📊] Predictions saved to {PREDICTIONS_PATH}")
