# src/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna(method='ffill', inplace=True)

    if 'loan_approved' not in df.columns:
        raise ValueError("Target column 'loan_approved' not found in dataset.")

    df.dropna(subset=['loan_approved'], inplace=True)

    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']

    encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return train_test_split(X, y, test_size=0.2, random_state=42), encoders
