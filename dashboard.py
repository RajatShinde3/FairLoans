# dashboard.py · FairLoans Interactive Fairness Dashboard

import os, joblib, shap, pandas as pd, numpy as np, streamlit as st
import seaborn as sns, matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
)
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate, false_negative_rate
)

# ───────────── Setup ─────────────
st.set_page_config(page_title="FairLoans Dashboard", layout="wide")
st.title("📊 FairLoans – Bias & Performance Dashboard")

# ───────────── Sidebar ─────────────
st.sidebar.header("📂 Upload prediction CSVs")
base_csv = st.sidebar.file_uploader("Baseline predictions", type="csv")
deb_csv  = st.sidebar.file_uploader("Debiased predictions", type="csv")
st.sidebar.markdown("---")
enable_shap = st.sidebar.checkbox("🔍  Show SHAP explainability", value=False)
enable_sim  = st.sidebar.checkbox("🧪  Enable input simulator",   value=True)

# ───────────── Constants ─────────────
REQ = {"y_true", "y_pred"}

# ───────────── Load raw feature data ─────────────
DATA_PATH = Path("data/loan_dataset.csv")
if not DATA_PATH.exists():
    st.error("`data/loan_dataset.csv` not found.")
    st.stop()
raw_df = pd.read_csv(DATA_PATH)

# ───────────── CSV loader ─────────────
def load_preds(file, raw_df):
    df = pd.read_csv(file)

    # Lowercase column names
    df.columns = df.columns.str.lower()
    raw_df.columns = raw_df.columns.str.lower()

    # Ensure prediction columns
    missing = REQ - set(df.columns)
    if missing:
        st.error(f"❌ CSV missing column(s): {missing}")
        st.stop()

    # Ensure numeric
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    if "y_prob" in df.columns:
        df["y_prob"] = pd.to_numeric(df["y_prob"], errors="coerce")

    df.dropna(subset=list(REQ), inplace=True)

    # Merge sensitive features from raw data if missing
    for col in ["gender", "race", "region"]:
        if col in raw_df.columns and col not in df.columns:
            df[col] = raw_df[col]

    return df

# ───────────── Tabs ─────────────
tab_over, tab_bias, tab_shap, tab_sim = st.tabs(
    ["🏠 Overview", "⚖️ Fairness", "🔍 SHAP", "🧪 Simulator"]
)

# ═════════════ 🏠 OVERVIEW ═════════════
with tab_over:
    st.subheader("How to use this dashboard")
    st.markdown("""
1. Upload **baseline** and (optionally) **debiased** prediction CSVs ⬅️  
2. Pick a sensitive feature to inspect fairness metrics  
3. Toggle SHAP for feature‑level insights  
4. Use the Simulator to test a custom applicant
""")
    st.info("Upload files to begin!")

# ═════════════ ⚖️ FAIRNESS ═════════════
with tab_bias:
    st.header("Group‑wise Fairness Metrics")

    if base_csv is None:
        st.warning("Upload at least a *Baseline* CSV to begin.")
    else:
        df_base = load_preds(base_csv, raw_df)
        df_deb  = load_preds(deb_csv, raw_df) if deb_csv else None

        sens_avail = [c for c in ["gender", "race", "region"] if c in df_base.columns]
        if not sens_avail:
            st.error("No sensitive features (`gender`, `race`, `region`) found.")
            st.stop()
        sens = st.selectbox("Sensitive feature", sens_avail)

        def metric_frame(df):
            df = df.dropna(subset=[sens, "y_true", "y_pred"])
            if df.empty:
                return None
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

        # Baseline
        with col1:
            st.markdown("### Baseline")
            mf_b = metric_frame(df_base)
            if mf_b is not None:
                st.dataframe(mf_b.by_group)
                fig, ax = plt.subplots(figsize=(5,3))
                mf_b.by_group.T.plot(kind="bar", ax=ax)
                ax.set_title(f"Baseline by {sens}")
                st.pyplot(fig)
            else:
                st.warning("No valid baseline rows.")

        # Debiased
        with col2:
            st.markdown("### Debiased")
            if df_deb is not None:
                mf_d = metric_frame(df_deb)
                if mf_d is not None:
                    st.dataframe(mf_d.by_group)
                    fig, ax = plt.subplots(figsize=(5,3))
                    mf_d.by_group.T.plot(kind="bar", color="salmon", ax=ax)
                    ax.set_title(f"Debiased by {sens}")
                    st.pyplot(fig)
                else:
                    st.warning("No valid debiased rows.")
            else:
                st.info("Upload debiased CSV to compare.")

        # ─── Performance Curves ───
        st.markdown("---")
        st.subheader("Performance Curves")
        perf_cols = st.columns(2)

        for name, df_show, slot in [("Baseline", df_base, 0), ("Debiased", df_deb, 1)]:
            if df_show is None:
                continue

            with perf_cols[slot]:
                st.markdown(f"### {name}")

                cm = confusion_matrix(df_show["y_true"], df_show["y_pred"])
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                ax_cm.set_title(f"{name} · Confusion Matrix")
                ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
                st.pyplot(fig_cm)

                if "y_prob" in df_show.columns and df_show["y_prob"].notna().all():
                    fpr, tpr, _ = roc_curve(df_show["y_true"], df_show["y_prob"])
                    prec, rec, _ = precision_recall_curve(df_show["y_true"], df_show["y_prob"])

                    fig_roc, ax_roc = plt.subplots()
                    ax_roc.plot(fpr, tpr, label="ROC")
                    ax_roc.plot([0,1],[0,1],"--")
                    ax_roc.set_title(f"{name} · ROC")
                    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")
                    st.pyplot(fig_roc)

                    fig_pr, ax_pr = plt.subplots()
                    ax_pr.plot(rec, prec, label="PR")
                    ax_pr.set_title(f"{name} · Precision-Recall")
                    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
                    st.pyplot(fig_pr)
                else:
                    st.warning(f"{name}: `y_prob` missing — ROC/PR skipped.")

# ═════════════ 🔍 SHAP ═════════════
with tab_shap:
    st.header("Global Feature Importance (SHAP)")
    if not enable_shap:
        st.info("Enable SHAP in sidebar.")
    else:
        try:
            model = joblib.load("results/model_xgb.pkl")
            feats = model.get_booster().feature_names
            df_enc = pd.get_dummies(raw_df.drop(columns=["id", "loan_approved"]), drop_first=True)
            df_enc = df_enc.reindex(columns=feats, fill_value=0)

            explainer = shap.Explainer(model)
            shap_vals = explainer(df_enc)

            fig = plt.figure()
            shap.summary_plot(shap_vals, df_enc, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP failed: {e}")

# ═════════════ 🧪 SIMULATOR ═════════════
with tab_sim:
    st.header("Loan‑Approval Simulator")
    if not enable_sim:
        st.info("Enable simulator in sidebar.")
    else:
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", 18, 70, 30)
            income = st.number_input("Income", 10000, 200000, 50000, 1000)
            credit = st.slider("Credit Score", 300, 850, 650)
        with right:
            loan_amt = st.number_input("Loan Amount", 1000, 100000, 15000, 500)
            gender = st.radio("Gender", ["Male","Female"])
            race = st.selectbox("Race", ["White","Black","Asian","Hispanic","Other"])
            region = st.selectbox("Region", ["Urban","Rural","Suburban"])

        user = {
            "age": age, "income": income, "loan_amount": loan_amt,
            "credit_score": credit, f"gender_{gender}": 1,
            f"race_{race}": 1, f"region_{region}": 1
        }
        X_user = pd.DataFrame([user])

        try:
            model_base = joblib.load("results/model_xgb.pkl")
            for col in model_base.get_booster().feature_names:
                if col not in X_user: X_user[col] = 0
            X_user = X_user[model_base.get_booster().feature_names]
            prob = model_base.predict_proba(X_user)[0, 1]
            st.success(f"Baseline → {'✅ Approved' if prob>=0.5 else '❌ Rejected'}  (p={prob:.2f})")
        except Exception as e:
            st.error(f"Baseline model error: {e}")

        deb_path = Path("results/model_debiased_xgb.pkl")
        if deb_path.exists():
            try:
                model_deb = joblib.load(deb_path)
                if hasattr(model_deb, "get_booster"):
                    for col in model_deb.get_booster().feature_names:
                        if col not in X_user: X_user[col] = 0
                    X_user_deb = X_user[model_deb.get_booster().feature_names]
                else:
                    for col in model_deb.feature_names_in_:
                        if col not in X_user: X_user[col] = 0
                    X_user_deb = X_user[model_deb.feature_names_in_]

                prob_deb = model_deb.predict_proba(X_user_deb)[0, 1]
                st.info(f"Debiased → {'✅ Approved' if prob_deb>=0.5 else '❌ Rejected'}  (p={prob_deb:.2f})")
            except Exception as e:
                st.error(f"Debiased model error: {e}")
        else:
            st.info("Debiased model not found.")

# ───────────── Footer ─────────────
st.markdown("---")
st.caption("© FairLoans · HacktheFest AI Bias Bounty 2025")
