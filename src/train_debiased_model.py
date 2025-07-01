# src/train_debiased_model.py

import os
import sys
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.metrics import accuracy_score, roc_auc_score

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_prep import load_and_prepare_data

def train_fair_model(data_path, model_path="results/model_debiased_xgb.pkl", features_path="results/debiased_model_features.pkl"):
    print("📥 Loading and preparing dataset...")
    (X_train, X_test, y_train, y_test), _ = load_and_prepare_data(data_path)

    # ───── Select sensitive feature ─────
    sensitive_feature = None
    if 'gender_Male' in X_train.columns:
        sensitive_feature = X_train['gender_Male']
        print("👤 Using 'gender_Male' as sensitive feature.")
    elif 'gender' in X_train.columns:
        sensitive_feature = X_train['gender']
        print("👤 Using 'gender' as sensitive feature.")
    else:
        raise ValueError("❌ Sensitive feature 'gender' or 'gender_Male' not found in dataset.")

    # ───── Fair model training ─────
    print("⚖️ Training debiased model with Demographic Parity constraint...")
    base_model = LogisticRegression(solver="liblinear")
    mitigator = ExponentiatedGradient(base_model, DemographicParity())
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_feature)

    # ───── Evaluate model ─────
    y_pred = mitigator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Debiased Accuracy: {acc:.4f}")

    try:
        y_prob = mitigator._pmf_predict(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"✅ Debiased AUC: {auc:.4f}")
    except Exception:
        print("⚠️ Could not calculate AUC — probabilities not available.")

    # ───── Save model and features ─────
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(mitigator, model_path)
    joblib.dump(X_train.columns.tolist(), features_path)
    print(f"💾 Debiased model saved to: {model_path}")
    print(f"💾 Features saved to: {features_path}")

if __name__ == "__main__":
    train_fair_model("data/loan_dataset.csv")
