# ──────────────────────────────────────────────────────────────
#  dashboard.py · FairLoans Interactive Fairness Dashboard
#  (robust to any user‑supplied prediction CSV)
# ──────────────────────────────────────────────────────────────
import os
from pathlib import Path

import joblib
import shap
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
)
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    false_positive_rate, false_negative_rate
)

# ───────────── Page setup ─────────────
st.set_page_config(page_title="FairLoans Dashboard", layout="wide")
st.title("📊 FairLoans – Bias & Performance Dashboard")

# ───────────── Sidebar ────────────────
st.sidebar.header("📂 Upload prediction CSVs")
base_csv = st.sidebar.file_uploader("Baseline predictions (csv)", type="csv")
deb_csv  = st.sidebar.file_uploader("Debiased predictions (csv)", type="csv")
st.sidebar.markdown("---")
enable_shap = st.sidebar.checkbox("🔍 Show SHAP explainability", value=False)
enable_sim  = st.sidebar.checkbox("🧪 Enable input simulator",   value=True)

# ───────────── Required columns ───────
REQ_COLS = {"y_true", "y_pred", "y_prob"}

# ───────────── Load raw feature data ──
DATA_PATH = Path("data/loan_dataset.csv")
if not DATA_PATH.exists():
    st.error("`data/loan_dataset.csv` not found.")
    st.stop()
raw_df = pd.read_csv(DATA_PATH)
raw_df.columns = raw_df.columns.str.strip().str.lower()   # standardize

# ───────────── CSV loader ─────────────
def load_preds(upload, raw):
    """Validate and merge sensitive columns; drop NaNs in essential cols."""
    df = pd.read_csv(upload)
    missing = REQ_COLS - set(df.columns)
    if missing:
        st.error(f"❌ CSV missing column(s): {missing}")
        st.stop()

    # Merge sensitive columns only if not already present
    for col in ["gender", "race", "region"]:
        if col in raw.columns and col not in df.columns:
            df[col] = raw[col]

    # Clean NaNs in key columns
    n0 = len(df)
    df = df.dropna(subset=list(REQ_COLS))
    if len(df) < n0:
        st.warning(f"{n0 - len(df)} rows with NaNs in `y_true`/`y_pred`/`y_prob` were dropped.")

    # Normalise y_true if textual
    if df["y_true"].dtype == object:
        df["y_true"] = df["y_true"].str.strip().map({"Approved": 1, "Denied": 0})

    return df

# ───────────── Tabs ─────────────
tab_over, tab_bias, tab_shap, tab_sim = st.tabs(
    ["🏠 Overview", "⚖️ Fairness", "🔍 SHAP", "🧪 Simulator"]
)

# ═════════════  🏠 OVERVIEW  ═════════════
with tab_over:
    st.subheader("How to use this dashboard")
    st.markdown("""
1. **Upload** baseline and (optionally) debiased prediction CSVs in the sidebar  
2. **Pick** a sensitive feature (gender / race / region) to inspect fairness  
3. **Enable** SHAP to inspect global feature impacts  
4. **Try** the simulator for a what‑if analysis of a single applicant  
""")
    st.info("Upload files in the sidebar to begin!")

# ═════════════  ⚖️ FAIRNESS  ═════════════
with tab_bias:
    st.header("Group‑wise Fairness Metrics")

    if base_csv is None:
        st.warning("Please upload at least a *Baseline* CSV.")
    else:
        df_base = load_preds(base_csv, raw_df)
        df_deb  = load_preds(deb_csv , raw_df) if deb_csv else None

        sens_avail = [c for c in ["gender", "race", "region"] if c in df_base.columns]
        if not sens_avail:
            st.error("Uploaded CSV has no sensitive feature columns (gender / race / region).")
            st.stop()

        sens = st.selectbox("Sensitive feature", sens_avail)

        def metric_frame(df):
            df = df.dropna(subset=[sens, "y_true", "y_pred"])
            return MetricFrame(
                metrics={
                    "Accuracy": accuracy_score,
                    "Selection Rate": selection_rate,
                    "FPR": false_positive_rate,
                    "FNR": false_negative_rate,
                },
                y_true=df["y_true"],
                y_pred=df["y_pred"],
                sensitive_features=df[sens],
            )

        col1, col2 = st.columns(2)

        # ── Baseline
        with col1:
            st.markdown("### Baseline")
            mf_b = metric_frame(df_base)
            st.dataframe(mf_b.by_group)
            fig, ax = plt.subplots(figsize=(5,3))
            mf_b.by_group.T.plot(kind="bar", ax=ax)
            ax.set_title(f"Baseline · {sens}")
            st.pyplot(fig)

        # ── Debiased
        with col2:
            if df_deb is not None:
                st.markdown("### Debiased")
                mf_d = metric_frame(df_deb)
                st.dataframe(mf_d.by_group)
                fig, ax = plt.subplots(figsize=(5,3))
                mf_d.by_group.T.plot(kind="bar", color="salmon", ax=ax)
                ax.set_title(f"Debiased · {sens}")
                st.pyplot(fig)
            else:
                st.info("Upload a Debiased CSV to compare.")

        # ── Performance curves
        st.markdown("---")
        st.subheader("Performance Curves")
        perf_cols = st.columns(3)

        for name, df_show, slot in [("Baseline", df_base, 0), ("Debiased", df_deb, 1)]:
            if df_show is None:
                continue
            with perf_cols[slot]:
                # Confusion Matrix
                cm = confusion_matrix(df_show["y_true"], df_show["y_pred"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"{name} Confusion Matrix")
                st.pyplot(fig)

                # ROC / PR only if y_prob numeric & finite
                if pd.api.types.is_numeric_dtype(df_show["y_prob"]):
                    y_prob_clean = pd.to_numeric(df_show["y_prob"], errors="coerce")
                    if y_prob_clean.notna().all():
                        # ROC
                        fpr, tpr, _ = roc_curve(df_show["y_true"], y_prob_clean)
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
                        ax.set_title(f"{name} ROC"); st.pyplot(fig)

                        # PR
                        prec, rec, _ = precision_recall_curve(df_show["y_true"], y_prob_clean)
                        fig, ax = plt.subplots()
                        ax.plot(rec, prec); ax.set_title(f"{name} PR")
                        st.pyplot(fig)
                    else:
                        st.warning(f"{name}: `y_prob` contains NaNs — ROC/PR skipped.")
                else:
                    st.warning(f"{name}: `y_prob` non‑numeric — ROC/PR skipped.")

# ═════════════  🔍 SHAP  ═════════════
with tab_shap:
    st.header("Global Feature Importance (SHAP)")
    if not enable_shap:
        st.info("Enable SHAP in sidebar.")
    else:
        try:
            model = joblib.load("results/model_xgb.pkl")
            model_feats = model.get_booster().feature_names
            df_enc = pd.get_dummies(raw_df, drop_first=True).reindex(columns=model_feats, fill_value=0)

            explainer = shap.Explainer(model)
            shap_vals = explainer(df_enc)

            fig = plt.figure()
            shap.summary_plot(shap_vals, df_enc, show=False)
            st.pyplot(fig); plt.clf()
        except Exception as e:
            st.error(f"SHAP failed: {e}")

# ═════════════  🧪 SIMULATOR  ═════════════
with tab_sim:
    st.header("Loan‑Approval Simulator")
    if not enable_sim:
        st.info("Enable simulator in sidebar.")
    else:
        left, right = st.columns(2)
        with left:
            age    = st.slider("Age", 18, 70, 30)
            income = st.number_input("Annual Income", 10000, 200000, 50000, 1000)
            credit = st.slider("Credit Score", 300, 850, 650)
        with right:
            loan_amt = st.number_input("Loan Amount", 1000, 100000, 15000, 500)
            gender   = st.radio("Gender", ["Male","Female"])
            race     = st.selectbox("Race", ["White","Black","Asian","Hispanic","Other"])
            region   = st.selectbox("Region", ["Urban","Rural","Suburban"])

        # Build one‑hot dict
        user = {
            "age": age, "income": income, "loan_amount": loan_amt,
            "credit_score": credit,
            f"gender_{gender}": 1,
            f"race_{race}": 1,
            f"region_{region}": 1
        }
        X_user = pd.DataFrame([user])

        # --- Helper to align columns
        def align_cols(X_in, model_cols):
            for col in model_cols:
                if col not in X_in:
                    X_in[col] = 0
            return X_in[model_cols]

        # Baseline model
        try:
            m_base = joblib.load("results/model_xgb.pkl")
            Xb = align_cols(X_user.copy(), m_base.get_booster().feature_names)
            prob_b = m_base.predict_proba(Xb)[0,1]
            st.success(f"Baseline → {'✅ Approved' if prob_b >= 0.5 else '❌ Rejected'}  (p={prob_b:.2f})")
        except Exception as e:
            st.error(f"Baseline model error: {e}")

        # Debiased model (may not provide probas)
        deb_path = Path("results/model_debiased_xgb.pkl")
        if deb_path.exists():
            try:
                m_deb = joblib.load(deb_path)

                # Fairlearn ExponentiatedGradient may not expose get_booster()
                if hasattr(m_deb, "get_booster"):
                    Xd = align_cols(X_user.copy(), m_deb.get_booster().feature_names)
                    prob_d = m_deb.predict_proba(Xd)[0,1]
                else:
                    # Fall back to class prediction only
                    Xd = align_cols(X_user.copy(), m_deb._input_columns)
                    prob_d = None

                pred_d = m_deb.predict(Xd)[0]
                txt = "✅ Approved" if pred_d else "❌ Rejected"
                suffix = f" (p={prob_d:.2f})" if prob_d is not None else ""
                st.info(f"Debiased → {txt}{suffix}")
            except Exception as e:
                st.error(f"Debiased model error: {e}")
        else:
            st.info("Debiased model not found — upload debiased CSV to compare.")

# ───────────── Footer ─────────────
st.markdown("---")
st.caption("© FairLoans · HacktheFest AI Bias Bounty 2025")
