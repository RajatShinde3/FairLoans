# ──────────────────────────────────────────────────────────────
#  dashboard.py · FairLoans Interactive Fairness Dashboard
#  ----------------------------------------------------------
#  Upload baseline / debiased CSVs ➜ explore bias, metrics,
#  SHAP explainability & live prediction simulator.
#  ----------------------------------------------------------
#  Required columns in CSV: y_true, y_pred, y_prob
#  Sensitive columns must exist in data/loan_dataset.csv
# ──────────────────────────────────────────────────────────────
import os, json, joblib, shap
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, accuracy_score
)
from fairlearn.metrics import (
    MetricFrame, selection_rate,
    false_positive_rate, false_negative_rate,
)

# ───── Streamlit page config ─────────────────────────────────
st.set_page_config(page_title="FairLoans Dashboard", layout="wide")
st.title("📊 FairLoans – Bias & Performance Dashboard")

# ───── Sidebar – upload files & toggles ──────────────────────
st.sidebar.header("📂  Upload prediction CSVs")
base_csv = st.sidebar.file_uploader("Baseline predictions", type="csv")
deb_csv  = st.sidebar.file_uploader("Debiased predictions", type="csv")
st.sidebar.markdown("---")
enable_shap = st.sidebar.checkbox("🔍  Show SHAP explainability", value=False)
enable_sim  = st.sidebar.checkbox("🧪  Enable input simulator",   value=True)

# ───── Helper: load + validate prediction files ──────────────
REQ = {"y_true","y_pred","y_prob"}
def load_preds(file):
    df = pd.read_csv(file)
    if not REQ.issubset(df.columns):
        missing = REQ - set(df.columns)
        st.error(f"Missing columns in CSV: {missing}")
        st.stop()
    return df

# ───── Load raw feature data (for sensitive cols & SHAP) ─────
DATA_PATH = Path("data/loan_dataset.csv")
if not DATA_PATH.exists():
    st.error("`data/loan_dataset.csv` not found in project.")
    st.stop()
raw_df = pd.read_csv(DATA_PATH)

# ══════════════ TAB LAYOUT ═══════════════════════════════════
tab_over, tab_bias, tab_shap, tab_sim = st.tabs(
    ["🏠 Overview", "⚖️ Fairness", "🔍 SHAP", "🧪 Simulator"]
)

# ────────────────── 🏠 OVERVIEW ──────────────────────────────
with tab_over:
    st.subheader("How to use this dashboard")
    st.markdown("""
1. **Upload** baseline and (optionally) debiased prediction CSVs using the sidebar.  
2. ⬅️ Choose a sensitive feature (gender / race / region) to inspect bias.  
3. 🔍 Toggle SHAP to see feature‑level impacts.  
4. 🧪 Try the simulator to compare predictions for a custom applicant.  
""")
    st.info("Ready when your CSVs are!")

# ────────────────── ⚖️ FAIRNESS ─────────────────────────────
with tab_bias:
    st.header("Group‑wise Fairness Metrics")
    if base_csv is None:
        st.warning("Upload at least a *baseline* CSV to begin.")
    else:
        df_base = pd.concat([load_preds(base_csv), raw_df[['gender','race','region']]], axis=1)

        # If debiased provided, load similarly
        df_deb = None
        if deb_csv:
            df_deb = pd.concat([load_preds(deb_csv), raw_df[['gender','race','region']]], axis=1)

        sens = st.selectbox("Sensitive feature", ["gender","race","region"])

        def metric_frame(df):
            return MetricFrame(
                metrics={
                    "Accuracy": accuracy_score,
                    "Selection Rate": selection_rate,
                    "FPR": false_positive_rate,
                    "FNR": false_negative_rate,
                },
                y_true=df["y_true"],
                y_pred=df["y_pred"],
                sensitive_features=df[sens],
            )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Baseline")
            mf_b = metric_frame(df_base)
            st.dataframe(mf_b.by_group)
            mf_b.by_group.T.plot(kind="bar", figsize=(4,3))
            plt.title(f"Baseline by {sens}")
            st.pyplot(plt.gcf()); plt.clf()

        with col2:
            if df_deb is not None:
                st.markdown("### Debiased")
                mf_d = metric_frame(df_deb)
                st.dataframe(mf_d.by_group)
                mf_d.by_group.T.plot(kind="bar", color="salmon", figsize=(4,3))
                plt.title(f"Debiased by {sens}")
                st.pyplot(plt.gcf()); plt.clf()
            else:
                st.info("Upload debiased CSV to compare.")

        # --- Performance curves & CM (baseline) ---
        st.markdown("---")
        st.subheader("Performance Curves")
        perf_cols = st.columns(3)
        for name, df_show, c in [("Baseline", df_base, 0), ("Debiased", df_deb, 1)]:
            if df_show is None:
                continue
            with perf_cols[c]:
                # Confusion Matrix
                cm = confusion_matrix(df_show["y_true"], df_show["y_pred"])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.title(f"{name} CM"); st.pyplot(plt.gcf()); plt.clf()

                # ROC
                fpr, tpr, _ = roc_curve(df_show["y_true"], df_show["y_prob"])
                plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--'); plt.title(f"{name} ROC")
                st.pyplot(plt.gcf()); plt.clf()

                # PR
                prec, rec, _ = precision_recall_curve(df_show["y_true"], df_show["y_prob"])
                plt.plot(rec, prec); plt.title(f"{name} PR"); st.pyplot(plt.gcf()); plt.clf()

# ────────────────── 🔍 SHAP TAB ─────────────────────────────
with tab_shap:
    st.header("Global Feature Importance (SHAP)")
    if not enable_shap:
        st.info("Enable SHAP checkbox in sidebar.")
    else:
        try:
            model = joblib.load("results/model_xgb.pkl")
            explainer = shap.Explainer(model)
            data_enc  = pd.get_dummies(raw_df[model.get_booster().feature_names], drop_first=True)
            data_enc  = data_enc.reindex(columns=model.get_booster().feature_names, fill_value=0)
            shap_values = explainer(data_enc)

            st.set_option("deprecation.showPyplotGlobalUse", False)
            shap.summary_plot(shap_values, data_enc, show=False)
            st.pyplot(bbox_inches="tight", dpi=300); plt.clf()
        except Exception as e:
            st.error(f"Unable to generate SHAP: {e}")

# ────────────────── 🧪 SIMULATOR ────────────────────────────
with tab_sim:
    st.header("Loan‑Approval Simulator")
    if not enable_sim:
        st.info("Enable simulator via sidebar.")
    else:
        base_cols = st.columns(2)
        with base_cols[0]:
            age = st.slider("Age", 18, 70, 30)
            income = st.number_input("Annual Income", 10000, 200000, 50000, 1000)
            credit = st.slider("Credit Score", 300, 850, 650)
        with base_cols[1]:
            loan_amt = st.number_input("Loan Amount", 1000, 100000, 15000, 500)
            gender = st.radio("Gender", ["Male","Female"])
            race = st.selectbox("Race", ["White","Black","Asian","Hispanic","Other"])
            region = st.selectbox("Region", ["Urban","Rural","Suburban"])

        sample = {
            "age": age, "income": income, "loan_amount": loan_amt,
            "credit_score": credit, f"gender_{gender}":1,
            f"race_{race}":1, f"region_{region}":1
        }
        sample_df = pd.DataFrame([sample])
        model_b = joblib.load("results/model_xgb.pkl")
        for col in model_b.get_booster().feature_names:
            if col not in sample_df: sample_df[col]=0
        sample_df = sample_df[model_b.get_booster().feature_names]
        prob_b = model_b.predict_proba(sample_df)[0,1]
        pred_b = int(prob_b >= 0.5)
        st.write(f"**Baseline Prediction:** {'✅ Approved' if pred_b else '❌ Rejected'} (p={prob_b:.2f})")

        if Path("results/model_debiased_xgb.pkl").exists():
            model_d = joblib.load("results/model_debiased_xgb.pkl")
            prob_d = model_d.predict_proba(sample_df)[0,1]
            pred_d = int(prob_d >= 0.5)
            st.write(f"**Debiased Prediction:** {'✅ Approved' if pred_d else '❌ Rejected'} (p={prob_d:.2f})")
        else:
            st.info("Debiased model not found (upload debiased CSV to compare).")

st.markdown("---")
st.caption("© FairLoans · Built for HacktheFest AI Bias Bounty 2025")
