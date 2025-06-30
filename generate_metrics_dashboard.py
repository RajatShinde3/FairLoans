"""
generate_metrics_dashboard.py
─────────────────────────────
Creates ROC, PR, confusion‑matrix plots + a markdown report
for baseline and debiased predictions.
"""
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

BASELINE = "results/baseline_predictions.csv"
DEBIASED = "results/debiased_predictions.csv"  # create later in 03_mitigate
OUT_DIR  = "results"
REPORT   = os.path.join(OUT_DIR, "metrics_report.md")
os.makedirs(OUT_DIR, exist_ok=True)

def load(path):
    df = pd.read_csv(path)
    return df["y_true"], df["y_pred"], df.get("y_prob")

def plot_roc(y, p, tag):
    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC · {tag}"); plt.legend()
    plt.savefig(f"{OUT_DIR}/roc_{tag}.png"); plt.close()

def plot_pr(y, p, tag):
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve · {tag}")
    plt.savefig(f"{OUT_DIR}/pr_{tag}.png"); plt.close()

def plot_cm(y, pred, tag):
    cm = confusion_matrix(y, pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix · {tag}")
    plt.savefig(f"{OUT_DIR}/cm_{tag}.png"); plt.close()

def add_report(y, pred, tag):
    rep = classification_report(y, pred, digits=2)
    with open(REPORT, "a") as f:
        f.write(f"### {tag} Classification Report\n\n```\n{rep}\n```\n\n")

def run_for(tag, path):
    y, pred, prob = load(path)
    add_report(y, pred, tag)
    if prob is not None: 
        plot_roc(y, prob, tag.lower())
        plot_pr(y, prob, tag.lower())
    plot_cm(y, pred, tag.lower())

if __name__ == "__main__":
    open(REPORT, "w").write("# Model Evaluation Report\n\n")
    for tag, path in [("Baseline", BASELINE), ("Debiased", DEBIASED)]:
        if os.path.exists(path) and os.path.getsize(path):
            run_for(tag, path)
    print(f"✅  Metrics + plots saved in {OUT_DIR}")
