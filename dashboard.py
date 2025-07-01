# ──────────────────────────────────────────────────────────────
#  dashboard.py · FairLoans Interactive Fairness Dashboard
#  (robust to any user‑supplied prediction CSVs)
# ──────────────────────────────────────────────────────────────
import os, joblib, shap, pandas as pd, streamlit as st, seaborn as sns, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, false_positive_rate, false_negative_rate

# ───────────── Page setup ─────────────
st.set_page_config(page_title="FairLoans Dashboard", layout="wide")
st.title("📊 FairLoans – Bias & Performance Dashboard")

# ───────────── Sidebar ────────────────
st.sidebar.header("📂  Upload prediction CSVs")
base_csv = st.sidebar.file_uploader("Baseline predictions", type="csv")
deb_csv  = st.sidebar.file_uploader("Debiased predictions", type="csv")
st.sidebar.markdown("---")
enable_shap = st.sidebar.checkbox("🔍  Show SHAP explainability", value=False)
enable_sim  = st.sidebar.checkbox("🧪  Enable input simulator",   value=True)

REQ = {"y_true", "y_pred", "y_prob"}         # mandatory columns

# ───────────── Raw dataset (for sensitive cols & SHAP) ──────
DATA_PATH = Path("data/loan_dataset.csv")
if not DATA_PATH.exists():
    st.error("`data/loan_dataset.csv` not found.") ; st.stop()
raw_df = pd.read_csv(DATA_PATH)

# ───────────── Robust CSV loader ─────────────────────────────
def load_preds(file):
    """Ensure required columns, convert labels, attach sensitive cols."""
    df = pd.read_csv(file)

    missing = REQ - set(df.columns)
    if missing:
        st.error(f"❌ CSV missing column(s): {missing}") ; st.stop()

    # Map string labels → 1/0
    for col in ["y_true", "y_pred"]:
        if df[col].dtype == object:
            df[col] = df[col].str.strip().map({"Approved": 1, "Denied": 0})

    # Coerce to numeric
    for col in ["y_true", "y_pred", "y_prob"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaNs in any key col
    n0 = len(df)
    df.dropna(subset=list(REQ), inplace=True)
    if len(df) < n0:
        st.warning(f"{n0-len(df)} rows dropped due to NaNs in y_true/y_pred/y_prob")

    # Attach sensitive cols if missing
    for col in ["gender", "race", "region"]:
        if col in raw_df.columns and col not in df.columns:
            df[col] = raw_df[col]

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
2. Choose a sensitive feature (gender, race, region) to inspect group metrics  
3. Toggle SHAP for feature‑importance insights  
4. Use the Simulator to test a custom applicant
""")
    st.info("Ready when your CSVs are!")

# ═════════════  ⚖️  FAIRNESS  ═════════════
with tab_bias:
    st.header("Group‑wise Fairness Metrics")

    if base_csv is None:
        st.warning("Upload at least a Baseline CSV.")
    else:
        df_base = load_preds(base_csv)
        df_deb  = load_preds(deb_csv) if deb_csv else None

        sens_cols = [c for c in ["gender", "race", "region"] if c in df_base.columns]
        if not sens_cols:
            st.error("No sensitive columns (`gender`, `race`, `region`) found.") ; st.stop()

        sens = st.selectbox("Sensitive feature", sens_cols)

        def mf(df):
            if df.empty: return None
            return MetricFrame(
                y_true=df["y_true"],
                y_pred=df["y_pred"],
                sensitive_features=df[sens],
                metrics={
                    "Accuracy": accuracy_score,
                    "Selection Rate": selection_rate,
                    "FPR": false_positive_rate,
                    "FNR": false_negative_rate,
                },
            )

        col1, col2 = st.columns(2, gap="large")
        for label, data, col in [("Baseline", df_base, col1), ("Debiased", df_deb, col2)]:
            if data is None: continue
            with col:
                st.markdown(f"### {label}")
                m = mf(data)
                if m is None:
                    st.warning("No valid rows.") ; continue
                st.dataframe(m.by_group)
                fig, ax = plt.subplots(figsize=(5,3))
                m.by_group.T.plot(kind="bar", ax=ax)
                ax.set_title(f"{label} by {sens}")
                st.pyplot(fig)

        # ─── Performance curves ───
        st.markdown("---"); st.subheader("Performance Curves")
        for label, data in [("Baseline", df_base), ("Debiased", df_deb)]:
            if data is None: continue
            st.markdown(f"**{label}**")
            # CM
            cm = confusion_matrix(data["y_true"], data["y_pred"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            if data["y_prob"].notna().all():
                # ROC
                fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr); ax.plot([0,1],[0,1],'--')
                ax.set_title("ROC Curve"); st.pyplot(fig)

                # PR
                prec, rec, _ = precision_recall_curve(data["y_true"], data["y_prob"])
                fig, ax = plt.subplots()
                ax.plot(rec, prec); ax.set_title("Precision‑Recall"); st.pyplot(fig)
            else:
                st.warning(f"{label}: y_prob missing/invalid – ROC & PR skipped")

# ═════════════  🔍  SHAP  ═════════════
with tab_shap:
    st.header("Global Feature Importance (SHAP)")
    if not enable_shap:
        st.info("Enable SHAP checkbox in sidebar.")
    else:
        try:
            model = joblib.load("results/model_xgb.pkl")
            feats = model.get_booster().feature_names
            df_enc = pd.get_dummies(raw_df, drop_first=True).reindex(columns=feats, fill_value=0)
            explainer = shap.Explainer(model)
            shap_vals = explainer(df_enc)
            fig = plt.figure(); shap.summary_plot(shap_vals, df_enc, show=False); st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP error: {e}")

# ═════════════  🧪  SIMULATOR  ═════════════
with tab_sim:
    st.header("Loan‑Approval Simulator")
    if not enable_sim:
        st.info("Enable simulator in sidebar.")
    else:
        left, right = st.columns(2)
        with left:
            age    = st.slider("Age", 18, 75, 30)
            income = st.number_input("Income", 10_000, 250_000, 50_000, 1_000)
            credit = st.slider("Credit Score", 300, 850, 650)
        with right:
            loan_amt = st.number_input("Loan Amount", 1_000, 500_000, 15_000, 500)
            gender   = st.radio("Gender", ["Male","Female"])
            race     = st.selectbox("Race", ["White","Black","Asian","Hispanic","Other"])
            region   = st.selectbox("Region", ["Urban","Rural","Suburban"])

        sample = {
            "age": age, "income": income, "loan_amount": loan_amt, "credit_score": credit,
            f"gender_{gender}":1, f"race_{race}":1, f"region_{region}":1
        }
        X_user = pd.DataFrame([sample])

        def show_pred(model, label, style):
            names = (model.get_booster().feature_names 
                     if hasattr(model, "get_booster") else model.feature_names_in_)
            for c in names:
                if c not in X_user: X_user[c] = 0
            X_user_ = X_user[names]
            prob = model.predict_proba(X_user_)[0,1]
            msg  = f"{label} → {'✅ Approved' if prob>=0.5 else '❌ Rejected'}  (p={prob:.2f})"
            getattr(st, style)(msg)

        try:
            show_pred(joblib.load("results/model_xgb.pkl"), "Baseline", "success")
        except Exception as e:
            st.error(f"Baseline model error: {e}")

        deb_path = Path("results/model_debiased_xgb.pkl")
        if deb_path.exists():
            try:   show_pred(joblib.load(deb_path), "Debiased", "info")
            except Exception as e:
                st.error(f"Debiased model error: {e}")
        else:
            st.info("Debiased model not found.")

# ───────────── Footer ─────────────
st.markdown("---")
st.caption("© FairLoans · HacktheFest AI Bias Bounty 2025")
