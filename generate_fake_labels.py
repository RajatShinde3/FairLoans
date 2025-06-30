# generate_fake_labels.py

import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/loan_access_dataset.csv"
OUTPUT_PATH = "data/loan_dataset.csv"

def generate_fake_labels():
    df = pd.read_csv(INPUT_PATH)

    # Rule-based label generation (simulate approval criteria)
    def approve(row):
        if row["Credit_Score"] >= 700 and row["Income"] >= 50000 and row["Loan_Amount"] < 300000:
            return 1
        elif row["Credit_Score"] >= 650 and row["Loan_Amount"] < 150000:
            return 1
        else:
            return 0

    df["loan_approved"] = df.apply(approve, axis=1)

    # Save for use in training/dashboard
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[✅] Fake labels generated and saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_fake_labels()
