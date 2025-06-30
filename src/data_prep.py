import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(data_path):
    print(f"[📂] Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Optional: fill missing values forward
    df.fillna(method='ffill', inplace=True)

    # Check for expected target column
    if 'loan_approved' not in df.columns:
        raise ValueError("[❌] Target column 'loan_approved' not found in dataset.\n[ℹ️] Make sure your dataset has a 'loan_approved' column as the target.")

    # Map Approved/Denied to 1/0 if necessary
    if df['loan_approved'].dtype == object:
        df['loan_approved'] = df['loan_approved'].str.strip().map({
            "Approved": 1,
            "Denied": 0
        })

    # Final check to make sure it's numeric now
    if df['loan_approved'].isnull().any():
        raise ValueError("[❌] Unable to convert target values to 0/1. Check your 'loan_approved' values.")

    # Features to use for modeling
    features = ['age', 'income', 'loan_amount', 'credit_score', 'gender', 'race', 'zip_code_group']
    for col in features:
        if col not in df.columns:
            raise ValueError(f"[❌] Missing required feature column: '{col}'")

    X = df[features]
    y = df['loan_approved']

    # Encode categorical features
    X = pd.get_dummies(X)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return (X_train, X_test, y_train, y_test), None
