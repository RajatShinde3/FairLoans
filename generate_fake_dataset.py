# generate_fake_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

df = pd.DataFrame({
    "age":           np.random.randint(21, 70, 500),
    "income":        np.random.randint(20_000, 120_000, 500),
    "loan_amount":   np.random.randint(1_000, 50_000, 500),
    "credit_score":  np.random.randint(300, 850, 500),
    "gender":        np.random.choice(["Male", "Female"], 500),
    "race":          np.random.choice(["White", "Black", "Asian", "Hispanic", "Other"], 500),
    "region":        np.random.choice(["Urban", "Rural", "Suburban"], 500),
    "loan_approved": np.random.choice([0, 1], 500, p=[0.4, 0.6]),
})

df.to_csv(DATA_DIR / "loan_dataset.csv", index=False)
print("✅  Synthetic loan_dataset.csv generated → data/loan_dataset.csv")
