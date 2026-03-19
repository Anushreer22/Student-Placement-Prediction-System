import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

RAW_PATH = "data/raw/student_placement_2026.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"
ENCODER_PATH = "models/label_encoders.pkl"


def load_data(path=RAW_PATH):
    df = pd.read_csv(path)
    print(f"[INFO] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def inspect_data(df):
    print("\n--- Shape ---")
    print(df.shape)
    print("\n--- Columns ---")
    print(df.columns.tolist())
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\n--- Target Distribution ---")
    if "placement_status" in df.columns:
        print(df["placement_status"].value_counts())


def clean_data(df):
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[INFO] Dropped {before - len(df)} duplicate rows")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Drop columns with >50% missing
    threshold = 0.5
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    if drop_cols:
        print(f"[INFO] Dropping high-missing columns: {drop_cols}")
        df.drop(columns=drop_cols, inplace=True)

    return df


def handle_missing(df):
    df = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    print("[INFO] Missing values handled")
    return df


def encode_categoricals(df, fit=True, encoders=None):
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Remove targets from encoding
    targets = ["placement_status", "salary_package_lpa"]
    cat_cols = [c for c in cat_cols if c not in targets]

    if fit:
        encoders = {}

    for col in cat_cols:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            df[col] = le.transform(df[col].astype(str))

    print(f"[INFO] Encoded columns: {cat_cols}")

    # Encode target classification label if present
    if "placement_status" in df.columns and df["placement_status"].dtype == "object":
        if fit:
            le_target = LabelEncoder()
            df["placement_status"] = le_target.fit_transform(df["placement_status"])
            encoders["placement_status"] = le_target
        else:
            df["placement_status"] = encoders["placement_status"].transform(df["placement_status"])

    return df, encoders


def save_processed(df, path=PROCESSED_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Processed data saved to {path}")


def save_encoders(encoders, path=ENCODER_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(encoders, path)
    print(f"[INFO] Encoders saved to {path}")


def run_preprocessing():
    df = load_data()
    inspect_data(df)
    df = clean_data(df)
    df = handle_missing(df)
    df, encoders = encode_categoricals(df, fit=True)
    save_processed(df)
    save_encoders(encoders)
    print("\n[DONE] Preprocessing complete.")
    return df, encoders


if __name__ == "__main__":
    run_preprocessing()
