"""
generate_assets.py
──────────────────
Re‑trains the production model and creates perfectly matched
model_xgb.pkl  +  shap_explainer.pkl   inside the results/ folder.
"""

import os
import joblib
import shap
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = "data/loan_dataset.csv"
RESULT_DIR = "results"
MODEL_PATH = os.path.join(RESULT_DIR, "model_xgb.pkl")
EXPLAINER_PATH = os.path.join(RESULT_DIR, "shap_explainer.pkl")

# ───────────────────────────────────────────────────────────
print("📥  Loading dataset …")
df = pd.read_csv(DATA_PATH).dropna()

# Encode object columns
encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

X = df.drop("loan_approved", axis=1)
y = df["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ───────────────────────────────────────────────────────────
print("🧠  Training XGBoost …")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
preds = (proba >= 0.5).astype(int)
acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, proba)

print(f"✅  Accuracy: {acc:.4f} | AUC: {auc:.4f}")

# ───────────────────────────────────────────────────────────
print("🔍  Building SHAP TreeExplainer …")
explainer = shap.TreeExplainer(model, X_train, feature_perturbation="tree_path_dependent")

# ───────────────────────────────────────────────────────────
os.makedirs(RESULT_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(explainer, EXPLAINER_PATH)
print(f"💾  Saved model  →  {MODEL_PATH}")
print(f"💾  Saved explainer →  {EXPLAINER_PATH}")

print("🎉  Assets regenerated and production‑ready!")
