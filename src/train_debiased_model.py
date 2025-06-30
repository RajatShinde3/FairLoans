# src/train_debiased_model.py

import os
import sys
import joblib
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_prep import load_and_prepare_data

def train_fair_model(data_path, model_path="results/model_debiased_xgb.pkl", features_path="results/debiased_model_features.pkl"):
    (X_train, X_test, y_train, y_test), _ = load_and_prepare_data(data_path)

    # Use gender as sensitive attribute
    sensitive = X_train['gender_Male'] if 'gender_Male' in X_train.columns else X_train.iloc[:, 0]

    # Fair model training using Demographic Parity
    base_model = LogisticRegression(solver="liblinear")
    constraint = DemographicParity()
    mitigator = ExponentiatedGradient(base_model, constraint)
    mitigator.fit(X_train, y_train, sensitive_features=sensitive)

    # Save model & feature list
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(mitigator, model_path)
    joblib.dump(X_train.columns.tolist(), features_path)

    print(f"✅ Debiased model saved to: {model_path}")
    print(f"✅ Features saved to: {features_path}")

if __name__ == "__main__":
    train_fair_model("data/loan_dataset.csv")
