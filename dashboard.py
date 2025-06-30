# ──────────────────────────────────────────────────────────────
#  dashboard.py · FairLoans Interactive Fairness Dashboard
#  (robust to any user‑supplied prediction CSV)
# ──────────────────────────────────────────────────────────────
import os, joblib, shap, pandas as pd, numpy as np, streamlit as st
import seaborn as sns, matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
)
from fairlearn.metrics import (
    MetricFrame, selection_rate, false_positive_rate, false_negative_rate
)

# ───────────── Page setup ─────────────
st.set_page_config(page_title="FairLoans Dashboard", layout="wide")
st.title("📊 FairLoans – Bias & Performance Dashboard")

# ───────────── Sidebar ────────────────
st.sidebar.header("📂 Upload prediction CSVs")
base_csv = st.sidebar.file_uploader("Baseline predictions", type="csv")
deb_csv  = st.sidebar.file_uploader("Debiased predictions", type="csv")
st.sidebar.markdown("---")
enable_shap = st.sidebar.checkbox("🔍  Show SHAP explainability", value=False)
enable_sim  = st.sidebar.checkbox("🧪  Enable input simulator",   value=True)

# ───────────── Required columns ───────
REQ = {"y_true", "y_pred", "y_prob"}

# ───────────── Load raw feature data ──
DATA_PATH = Path("data/loan_dataset.csv")
if not DATA_PATH.exists():
    st.error("`data/loan_dataset.csv` not found.")
    st.stop()
raw_df = pd.read_csv(DATA_PATH)

# ───────────── Robust CSV loader ──────
def load_preds(file, raw_df):
    """Read CSV, ensure required columns, merge sensitive cols, drop NaNs."""
    df = pd.read_csv(file)

    # Check mandatory preds
    missing = REQ - set(df.columns)
    if missing:
        st.error(f"❌ Missing column(s) {missing} in uploaded CSV.")
        st.stop()

    # Bring in sensitive cols if present in raw data
    for col in ["gender", "race", "region"]:
        if col in raw_df.columns and col not in df.columns:
            df[col] = raw_df[col]

    # Drop rows with NaN in key columns
    n0 = len(df)
    df = df.dropna(subset=list(REQ))
    dropped = n0 - len(df)
    if dropped:
        st.warning(f"{dropped} rows with NaNs in `y_true`/`y_pred` were removed.")

    return df

# ───────────── Tabs ───────────────────
tab_over, tab_bias, tab_shap, tab_sim = st.tabs(
    ["🏠 Overview", "⚖️ Fairness", "🔍 SHAP", "🧪 Simulator"]
)

# ═════════════  🏠  OVERVIEW  ═════════════
with tab_over:
    st.subheader("How to use this dashboard")
    st.markdown("""
1. Upload **baseline** and (optionally) **debiased** prediction CSVs in the sidebar  
2. Pick a sensitive feature (only those present are shown)  
3. Enable SHAP for feature‑impact insights  
4. Use the simulator to test a custom applicant
""")
    st.info("Upload files to begin!")

# ═════════════  ⚖️  FAIRNESS  ═════════════
with tab_bias:
    st.header("Group‑wise Fairness Metrics")

    if base_csv is None:
        st.warning("Upload at least a Baseline CSV.")
    else:
        df_base = load_preds(base_csv, raw_df)
        df_deb = load_preds(deb_csv, raw_df) if deb_csv else None

        # Determine which sensitive features are available
        sens_avail = [c for c in ["gender", "race", "region"] if c in df_base.columns]
        if not sens_avail:
            st.error("No sensitive features (`gender`, `race`, `region`) present.")
            st.stop()

        sens = st.selectbox("Sensitive feature", sens_avail)

        # Small helper
        def metric_frame(df):
            df = df.dropna(subset=[sens, "y_true", "y_pred"])
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
        # ---------- Baseline ----------
        with col1:
            st.markdown("### Baseline")
            mf_b = metric_frame(df_base)
            st.dataframe(mf_b.by_group)
            fig, ax = plt.subplots(figsize=(5,3))
            mf_b.by_group.T.plot(kind="bar", ax=ax)
            ax.set_title(f"Baseline by {sens}")
            st.pyplot(fig)

        # ---------- Debiased ----------
        with col2:
            if df_deb is not None:
                st.markdown("### Debiased")
                mf_d = metric_frame(df_deb)
                st.dataframe(mf_d.by_group)
                fig, ax = plt.subplots(figsize=(5,3))
                mf_d.by_group.T.plot(kind="bar", color="salmon", ax=ax)
                ax.set_title(f"Debiased by {sens}")
                st.pyplot(fig)
            else:
                st.info("Upload debiased CSV to compare.")

        # ---------- Performance plots ----------
        st.markdown("---")
        st.subheader("Performance Curves")
        perf_cols = st.columns(3)

        for name, df_show, slot in [("Baseline", df_base, 0), ("Debiased", df_deb, 1)]:
            if df_show is None: continue
            with perf_cols[slot]:
                # CM
                cm = confusion_matrix(df_show["y_true"], df_show["y_pred"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title(f"{name} CM"); st.pyplot(fig)

                # ROC
                fpr, tpr, _ = roc_curve(df_show["y_true"], df_show["y_prob"])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
                ax.set_title(f"{name} ROC"); st.pyplot(fig)

                # PR
                prec, rec, _ = precision_recall_curve(df_show["y_true"], df_show["y_prob"])
                fig, ax = plt.subplots()
                ax.plot(rec, prec); ax.set_title(f"{name} PR")
                st.pyplot(fig)

# ═════════════  🔍  SHAP  ═════════════
with tab_shap:
    st.header("Global Feature Importance (SHAP)")
    if not enable_shap:
        st.info("Enable SHAP in sidebar.")
    else:
        try:
            model = joblib.load("results/model_xgb.pkl")
            model_feats = model.get_booster().feature_names
            df_enc = pd.get_dummies(raw_df, drop_first=True)
            df_enc = df_enc.reindex(columns=model_feats, fill_value=0)

            explainer = shap.Explainer(model)
            shap_vals = explainer(df_enc)

            fig = plt.figure()
            shap.summary_plot(shap_vals, df_enc, show=False)
            st.pyplot(fig); plt.clf()
        except Exception as e:
            st.error(f"SHAP failed: {e}")

# ═════════════  🧪  SIMULATOR  ═════════════
with tab_sim:
    st.header("Loan‑Approval Simulator")
    if not enable_sim:
        st.info("Enable simulator in sidebar.")
    else:
        # --- Input widgets ---
        left, right = st.columns(2)
        with left:
            age       = st.slider("Age", 18, 70, 30)
            income    = st.number_input("Income", 10000, 200000, 50000, 1000)
            credit    = st.slider("Credit Score", 300, 850, 650)
        with right:
            loan_amt  = st.number_input("Loan Amount", 1000, 100000, 15000, 500)
            gender    = st.radio("Gender", ["Male","Female"])
            race      = st.selectbox("Race", ["White","Black","Asian","Hispanic","Other"])
            region    = st.selectbox("Region", ["Urban","Rural","Suburban"])

        user = {
            "age": age, "income": income, "loan_amount": loan_amt,
            "credit_score": credit, f"gender_{gender}":1,
            f"race_{race}":1, f"region_{region}":1
        }
        X_user = pd.DataFrame([user])

        # Baseline model
        try:
            m_base = joblib.load("results/model_xgb.pkl")
            for col in m_base.get_booster().feature_names:
                if col not in X_user: X_user[col] = 0
            X_user = X_user[m_base.get_booster().feature_names]
            p_b = m_base.predict_proba(X_user)[0,1]
            st.success(f"Baseline → {'✅ Approved' if p_b>=0.5 else '❌ Rejected'}  (p={p_b:.2f})")
        except Exception as e:
            st.error(f"Baseline model error: {e}")

        # Debiased model
        deb_path = Path("results/model_debiased_xgb.pkl")
        if deb_path.exists():
            try:
                m_deb = joblib.load(deb_path)
                for col in m_deb.get_booster().feature_names:
                    if col not in X_user: X_user[col] = 0
                X_user_d = X_user[m_deb.get_booster().feature_names]
                p_d = m_deb.predict_proba(X_user_d)[0,1]
                st.info   (f"Debiased →  {'✅ Approved' if p_d>=0.5 else '❌ Rejected'}  (p={p_d:.2f})")
            except Exception as e:
                st.error(f"Debiased model error: {e}")
        else:
            st.info("Debiased model not found → upload debiased CSV to compare.")

# ───────────── Footer ─────────────
st.markdown("---")
st.caption("© FairLoans · HacktheFest AI Bias Bounty 2025")
